from pathlib import Path
import json
import gym_pusht  # noqa: F401
import gym_aloha  # Import for ALOHA environment
import gymnasium as gym
import imageio
import numpy as np
import torch
import copy

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.utils.utils import get_safe_torch_device


# Helper: Recreate an env and replay actions to bring it to the current state
def clone_env_from_action_buffer(env_name, action_buffer, seed=42):
    # Use gym.make with appropriate params based on environment type
    if "aloha" in env_name.lower():
        new_env = gym.make(
            env_name,
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=400,
        )
    else:  # PushT environment
        new_env = gym.make(
            env_name,
            obs_type="state",
            max_episode_steps=300,
        )

    state, _ = new_env.reset(seed=seed)
    for action in action_buffer:
        state, reward, terminated, truncated, info = new_env.step(action)
        if terminated or truncated:
            break
    return new_env, state


# Configuration - choose environment to use
use_aloha = (
    False  # Set to True to use ALOHA with ACT, False to use PushT with Diffusion
)

n_episodes = 10
n_rollouts = 10
horizon = 100

# Baseline
# n_rollouts = 1
# horizon = 400

# Set environment and policy configuration based on choice
if use_aloha:
    env_name = "gym_aloha/AlohaInsertion-v0"
    output_directory = Path("outputs/eval/example_aloha_act_best_of_n_max_reward")
    # output_directory = Path("outputs/eval/example_aloha_act_baseline")  # Baseline
    pretrained_policy_path = "lerobot/act_aloha_sim_insertion_human"
else:
    env_name = "gym_pusht/PushT-v0"
    output_directory = Path("outputs/eval/example_pusht_diffusion_best_of_n_max_reward")
    pretrained_policy_path = "lerobot/diffusion_pusht"

output_directory.mkdir(parents=True, exist_ok=True)
device = get_safe_torch_device("cuda")

# Create policy based on environment choice
if use_aloha:
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
else:
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.to(device)

# Initialize the environment with correct parameters from EnvConfig
if use_aloha:
    env = gym.make(
        env_name,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=400,
    )
else:
    env = gym.make(
        env_name,
        obs_type="state",
        max_episode_steps=300,
    )

print(f"Environment: {env_name}")
print(f"Policy: {type(policy).__name__}")
print(f"Environment observation space: {env.observation_space}")
print(f"Environment action space: {env.action_space}")


# At the beginning of your script, add this to print more policy information
def print_policy_config(policy):
    print(f"Policy config: {policy.config}")
    if hasattr(policy, "config") and hasattr(policy.config, "image_features"):
        print(f"Policy image features: {policy.config.image_features}")
    if hasattr(policy, "config") and hasattr(policy.config, "input_features"):
        print(f"Policy input features: {policy.config.input_features}")


# After loading the policy, add:
print_policy_config(policy)

rewards = []
max_rewards = []
avg_rewards = []  # New tracker for average rewards per timestep
successes = []

# Run episodes
for episode in range(n_episodes):
    print(f"Starting episode {episode+1}/{n_episodes}...")
    policy.reset()
    full_state, info = env.reset(seed=42)
    overall_reward = 0.0
    overall_frames = [env.render()]

    # Global action buffer for replaying the env state
    global_action_buffer = []
    done = False
    segment_count = 0

    while not done:
        # Store rollout results
        rollout_results = []
        for rollout in range(n_rollouts):
            clone, clone_state = clone_env_from_action_buffer(
                env_name, global_action_buffer, seed=42
            )
            policy_clone = copy.deepcopy(policy)
            rollout_reward = 0.0
            rollout_frames = []
            rollout_action_buffer = []
            max_instant_reward = -float("inf")
            finished_flag = False
            success_flag = False

            # Run the cloned env for a fixed horizon
            for local_step in range(horizon):
                # Prepare observation for policy
                if use_aloha:
                    # For ACT policy with ALOHA - handle the nested observation structure
                    # The 'top' image is under the 'pixels' key
                    img = clone_state["pixels"]["top"]
                    img_tensor = (
                        torch.from_numpy(img)
                        .to(torch.float32)
                        .permute(2, 0, 1)  # HWC to CHW
                        .unsqueeze(0)
                        .to(device)
                    ) / 255.0  # Normalize to [0,1]

                    # Get the agent_pos from the observation
                    agent_pos = clone_state["agent_pos"]
                    state_tensor = (
                        torch.from_numpy(agent_pos)
                        .to(torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )

                    observation = {
                        "observation.images.top": img_tensor,
                        "observation.state": state_tensor,
                    }
                else:
                    # For Diffusion policy with PushT
                    agent_pos = np.array(clone_state[:2])
                    state_tensor = (
                        torch.from_numpy(agent_pos)
                        .to(torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )

                    # Get image for Diffusion policy
                    pixel_img = clone.unwrapped._render(visualize=False)
                    image_tensor = torch.from_numpy(pixel_img).to(torch.float32) / 255.0
                    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

                    observation = {
                        "observation.state": state_tensor,
                        "observation.image": image_tensor,
                    }
                    if policy.config.image_features:
                        observation["observation.images"] = image_tensor.unsqueeze(1)

                # Get action from policy
                with torch.inference_mode():
                    action = policy_clone.select_action(observation)
                numpy_action = action.squeeze(0).to("cpu").numpy()
                rollout_action_buffer.append(numpy_action)

                # Step environment
                clone_state, reward, terminated, truncated, info = clone.step(
                    numpy_action
                )
                rollout_reward += reward
                rollout_frames.append(clone.render())

                if reward > max_instant_reward:
                    max_instant_reward = reward
                if terminated or truncated:
                    finished_flag = True
                    success_flag = info.get("is_success", False)
                    break

            # Calculate average reward per timestep for this rollout
            steps_executed = len(rollout_action_buffer)
            avg_reward = rollout_reward / steps_executed if steps_executed > 0 else 0.0

            rollout_results.append(
                (
                    max_instant_reward,
                    rollout_reward,
                    avg_reward,  # Add average reward to the result tuple
                    rollout_frames,
                    rollout_action_buffer,
                    finished_flag,
                    success_flag,
                    clone,
                    policy_clone,
                )
            )

        # Select the rollout with the highest max instantaneous reward
        best_rollout = max(rollout_results, key=lambda x: x[0])
        (
            best_max_reward,
            best_segment_reward,
            best_avg_reward,  # Extract the average reward
            best_segment_frames,
            best_segment_actions,
            best_segment_finished,
            best_segment_success,
            best_rollout_clone,
            best_policy_clone,
        ) = best_rollout

        overall_reward += best_segment_reward
        overall_frames.extend(best_segment_frames)
        segment_count += 1
        print(
            f"Episode {episode+1} - Segment {segment_count} complete: max reward/timestep = {best_max_reward:.2f}, "
            f"avg reward/timestep = {best_avg_reward:.2f}, "  # Add avg reward to the print statement
            f"cumulative reward = {best_segment_reward:.2f}, overall reward = {overall_reward:.2f}, "
            f"finished = {best_segment_finished}, success = {best_segment_success}"
        )

        # Update the global action buffer with the best segment's actions
        global_action_buffer.extend(best_segment_actions)

        # Update the global policy's internal state using the best policy clone
        if hasattr(policy, "_queues"):
            policy._queues = best_policy_clone._queues
        if hasattr(policy, "_action_queue"):
            policy._action_queue = best_policy_clone._action_queue

        # Clean up clones from this round
        for _, _, _, _, _, _, _, clone_env_inst, _ in rollout_results:
            if clone_env_inst is not best_rollout_clone:
                clone_env_inst.close()

        if best_segment_finished:
            done = True

    successes.append(best_segment_success)
    rewards.append(overall_reward)
    max_rewards.append(best_max_reward)  # Store max reward instead of overall reward
    avg_rewards.append(best_avg_reward)  # Store the average reward per timestep

    # Save video of the episode
    fps = env.metadata.get("render_fps", 30)  # Default to 30 fps if not specified
    video_path = output_directory / f"rollout_episode_{episode}.mp4"
    imageio.mimsave(str(video_path), np.stack(overall_frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")
    best_rollout_clone.close()

print(f"Success rate: {sum(successes) / n_episodes}")
print(f"Cumulative reward: {sum(rewards) / n_episodes}")
print(f"Max instantaneous reward: {sum(max_rewards) / n_episodes}")
print(f"Average reward per timestep: {sum(avg_rewards) / n_episodes}")

# Compile per-episode statistics and aggregated metrics
eval_info = {
    "per_episode": [
        {
            "episode_ix": i,
            "sum_reward": rewards[i],
            "max_instantaneous_reward": max_rewards[i],
            "avg_reward_per_timestep": avg_rewards[i],
            "success": successes[i],
            "seed": 42,  # fixed seed used for each episode
        }
        for i in range(n_episodes)
    ],
    "aggregated": {
        "avg_sum_reward": sum(rewards) / n_episodes,
        "avg_max_instantaneous_reward": sum(max_rewards) / n_episodes,
        "avg_reward_per_timestep": sum(avg_rewards) / n_episodes,
        "pc_success": (sum(successes) / n_episodes) * 100,
    },
}

json_path = output_directory / "eval_info.json"
with open(json_path, "w") as f:
    json.dump(eval_info, f, indent=2)
print(f"Evaluation info saved to '{json_path}'.")

env.close()
