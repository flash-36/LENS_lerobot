import json
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

n_episodes = 10
# Create a directory to store videos and evaluation info
output_directory = Path("outputs/eval/example_pusht_diffusion_orig")
output_directory.mkdir(parents=True, exist_ok=True)

device = "cuda"

# Load the pretrained policy (hub or local)
pretrained_policy_path = "lerobot/diffusion_pusht"
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="state",
    max_episode_steps=300,
)

# Verify feature shapes (optional)
print(policy.config.input_features)
print(env.observation_space)
print(policy.config.output_features)
print(env.action_space)

rewards = []
max_rewards = []
successes = []

for episode in range(n_episodes):
    print(f"Starting episode {episode+1}/{n_episodes}...")
    policy.reset()
    full_state, info = env.reset(seed=42)
    rewards_episode = []
    frames_episode = []
    frames_episode.append(env.render())

    step = 0
    done = False
    while not done:
        # Prepare observation: agent position and rendered image.
        agent_pos = np.array(full_state[:2])
        pixel_img = env.unwrapped._render(visualize=False)

        state_tensor = torch.from_numpy(agent_pos).to(torch.float32).unsqueeze(0)
        image_tensor = torch.from_numpy(pixel_img).to(torch.float32) / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

        # Move tensors to the correct device.
        state_tensor = state_tensor.to(device, non_blocking=True)
        image_tensor = image_tensor.to(device, non_blocking=True)

        observation = {
            "observation.state": state_tensor,
            "observation.image": image_tensor,
        }

        with torch.inference_mode():
            action = policy.select_action(observation)

        numpy_action = action.squeeze(0).to("cpu").numpy()

        full_state, reward, terminated, truncated, info = env.step(numpy_action)

        rewards_episode.append(reward)
        frames_episode.append(env.render())

        # Use boolean "or" for clarity.
        done = terminated or truncated
        step += 1

    # Accumulate episode rewards.
    rewards.append(sum(rewards_episode))
    max_rewards.append(max(rewards_episode))
    # Fix success criterion: if the environment provided an "is_success" flag, use it; otherwise, default to False.
    success = info.get("is_success", False)
    successes.append(success)

    print(
        f"Episode {episode+1} - Success: {success} - Sum Reward: {sum(rewards_episode)} - Max Reward: {max(rewards_episode)}"
    )

    # Save the video for this episode.
    fps = env.metadata["render_fps"]
    video_path = output_directory / f"rollout_episode_{episode}.mp4"
    imageio.mimsave(str(video_path), np.stack(frames_episode), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")

# Print overall evaluation metrics.
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

# Save the evaluation info in JSON format.
json_path = output_directory / "eval_info.json"
with open(json_path, "w") as f:
    json.dump(eval_info, f, indent=2)
print(f"Evaluation info saved to '{json_path}'.")

env.close()
