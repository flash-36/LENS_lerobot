from pathlib import Path
import json
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch
import copy

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


# Helper: Recreate an env and replay actions to bring it to the current state.
def clone_env_from_action_buffer(action_buffer, seed=42):
    new_env = gym.make("gym_pusht/PushT-v0", obs_type="state", max_episode_steps=300)
    state, _ = new_env.reset(seed=seed)
    for action in action_buffer:
        state, reward, terminated, truncated, info = new_env.step(action)
        if terminated or truncated:
            break
    return new_env, state


n_episodes = 10
n_rollouts = 5
horizon = 96

output_directory = Path("outputs/eval/example_pusht_diffusion_best_of_n_max_reward")
output_directory.mkdir(parents=True, exist_ok=True)

device = "cuda"
pretrained_policy_path = "lerobot/diffusion_pusht"
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.to(device)

# Initialize the main environment once for the episode.
env = gym.make("gym_pusht/PushT-v0", obs_type="state", max_episode_steps=300)

print(policy.config.input_features)
print(env.observation_space)
print(policy.config.output_features)
print(env.action_space)

rewards = []
max_rewards = []
successes = []

# We continue using the policy copies approach (without explicit warmup) for smooth trajectories.
for episode in range(n_episodes):
    print(f"Starting episode {episode+1}/{n_episodes}...")
    policy.reset()
    full_state, info = env.reset(seed=42)
    overall_reward = 0.0
    overall_frames = [env.render()]

    # Global action buffer for replaying the env state.
    global_action_buffer = []
    done = False
    segment_count = 0

    while not done:
        # We'll store rollout results as a tuple:
        # (max_instant_reward, cumulative_reward, rollout_frames, rollout_action_buffer,
        #  finished_flag, success_flag, clone, policy_clone)
        rollout_results = []
        for rollout in range(n_rollouts):
            clone, clone_state = clone_env_from_action_buffer(
                global_action_buffer, seed=42
            )
            policy_clone = copy.deepcopy(policy)
            rollout_reward = 0.0
            rollout_frames = []
            rollout_action_buffer = []
            steps_executed = 0
            max_instant_reward = -float("inf")
            finished_flag = False  # Indicates if the env actually terminated/truncated.
            success_flag = False  # Indicates if the termination was successful.
            # Run the cloned env for a fixed horizon.
            for local_step in range(horizon):
                current_clone_state = clone.unwrapped.get_obs()
                agent_pos = np.array(current_clone_state[:2])
                pixel_img = clone.unwrapped._render(visualize=False)

                # Prepare tensors.
                state_tensor = (
                    torch.from_numpy(agent_pos).to(torch.float32).unsqueeze(0)
                )
                image_tensor = torch.from_numpy(pixel_img).to(torch.float32) / 255.0
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
                state_tensor = state_tensor.to(device, non_blocking=True)
                image_tensor = image_tensor.to(device, non_blocking=True)

                observation = {
                    "observation.state": state_tensor,
                    "observation.image": image_tensor,
                }
                if policy.config.image_features:
                    observation["observation.images"] = image_tensor.unsqueeze(1)
                with torch.inference_mode():
                    action = policy_clone.select_action(observation)
                numpy_action = action.squeeze(0).to("cpu").numpy()
                rollout_action_buffer.append(numpy_action)

                clone_state, reward, terminated, truncated, info = clone.step(
                    numpy_action
                )
                rollout_reward += reward
                rollout_frames.append(clone.render())
                steps_executed += 1
                if reward > max_instant_reward:
                    max_instant_reward = reward
                if terminated or truncated:
                    finished_flag = True
                    success_flag = info.get("is_success", False)
                    break

            rollout_results.append(
                (
                    max_instant_reward,
                    rollout_reward,
                    rollout_frames,
                    rollout_action_buffer,
                    finished_flag,
                    success_flag,
                    clone,
                    policy_clone,
                )
            )

        # Select the rollout with the highest max instantaneous reward.
        best_rollout = max(rollout_results, key=lambda x: x[0])
        (
            best_max_reward,
            best_segment_reward,
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
            f"cumulative reward = {best_segment_reward:.2f}, overall reward = {overall_reward:.2f}, "
            f"finished = {best_segment_finished}, success = {best_segment_success}"
        )

        # Update the global action buffer with the best segment's actions.
        global_action_buffer.extend(best_segment_actions)

        # Update the global policy's internal state using the best policy clone.
        policy._queues = best_policy_clone._queues

        # Clean up clones from this round.
        for _, _, _, _, _, _, clone_env_inst, _ in rollout_results:
            if clone_env_inst is not best_rollout_clone:
                clone_env_inst.close()

        if best_segment_finished:
            done = True

    successes.append(best_segment_success)
    rewards.append(overall_reward)
    max_rewards.append(overall_reward)
    fps = env.metadata["render_fps"]
    video_path = output_directory / f"rollout_episode_{episode}.mp4"
    imageio.mimsave(str(video_path), np.stack(overall_frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")
    best_rollout_clone.close()

print(f"Success rate: {sum(successes) / n_episodes}")
print(f"Average sum reward: {sum(rewards) / n_episodes}")
print(f"Average max reward: {sum(max_rewards) / n_episodes}")

# Compile per-episode statistics and aggregated metrics.
eval_info = {
    "per_episode": [
        {
            "episode_ix": i,
            "sum_reward": rewards[i],
            "max_reward": max_rewards[i],
            "success": successes[i],
            "seed": 42,  # fixed seed used for each episode
        }
        for i in range(n_episodes)
    ],
    "aggregated": {
        "avg_sum_reward": sum(rewards) / n_episodes,
        "avg_max_reward": sum(max_rewards) / n_episodes,
        "pc_success": (sum(successes) / n_episodes) * 100,
    },
}

json_path = output_directory / "eval_info.json"
with open(json_path, "w") as f:
    json.dump(eval_info, f, indent=2)
print(f"Evaluation info saved to '{json_path}'.")

env.close()
