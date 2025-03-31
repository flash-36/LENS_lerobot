import json
from pathlib import Path

import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch
from torch import Tensor, nn
import einops

from lerobot.common.envs.factory import make_env
from lerobot.configs.eval import EvalPipelineConfig

from lerobot.common.policies.act.modeling_act import ACTPolicy
import os
os.environ["MUJOCO_GL"] = "egl"
device = "cuda"


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img = torch.from_numpy(img)

            # sanity check that images are channel last
            # print(img.shape)
            h, w, c = img.shape

            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "h w c -> 1 c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float().unsqueeze()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float().unsqueeze(0)
    return return_observations




n_episodes = 1
# Create a directory to store videos and evaluation info
output_dir = Path("outputs/eval/test_set_state")
output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda"

# Load the pretrained policy 
pretrained_policy_path = "outputs/train/19-02-39_aloha_act/checkpoints/100000/pretrained_model"
policy = ACTPolicy.from_pretrained(pretrained_policy_path)

env = gym.make(
    "gym_aloha/AlohaInsertion-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=400,
)

observation, info = env.reset(seed=42)

Frames = [] 
for t in range(200):
    # print("observation",observation)
    observation = preprocess_observation(observation)
    # observation = {key: observation[key].to(device, non_blocking=True) for key in observation}
    # print("observation",observation)

    with torch.inference_mode():
        action = policy.select_action(observation)

    numpy_action = action.squeeze(0).numpy()
    observation, reward, terminated, truncated, info = env.step(numpy_action)
    Frames.append(env.render())

fps = env.metadata["render_fps"]
video_path = output_dir / "test1_.mp4"
imageio.mimsave(str(video_path), np.stack(Frames), fps=fps)   
env_pose,env_bx,nev_vel = env.get_state()
new_env = gym.make(
    "gym_aloha/AlohaInsertion-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=400,
)
new_env.reset(seed=42)
new_env.set_state(env_pose,env_bx,nev_vel)

for t in range(200,600):
    observation = preprocess_observation(observation)
    with torch.inference_mode():
            action = policy.select_action(observation)
    numpy_action = action.squeeze(0).numpy()
    observation, reward,terminated, truncated, info = new_env.step(numpy_action)
    Frames.append(new_env.render()) 

fps = env.metadata["render_fps"]
video_path = output_dir / "test1.mp4"
imageio.mimsave(str(video_path), np.stack(Frames), fps=fps)


