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

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import os
import time
import logging

os.environ["MUJOCO_GL"] = "egl"
device = "cuda"

task = "AlohaInsertion-v0"
DEFAULT_FEATURES = {
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "seed": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
}

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




n_episodes = 2
# Create a directory to store videos and evaluation info
root = Path("outputs/record/test_dataset_1")
# root.mkdir(parents=True, exist_ok=True)

device = "cuda"

# Load the pretrained policy 
pretrained_policy_path = "outputs/train/19-02-39_aloha_act/checkpoints/100000/pretrained_model"
policy = ACTPolicy.from_pretrained(pretrained_policy_path)

env = gym.make(
    "gym_aloha/AlohaInsertion-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=400,
)

episode_time_s = 10

num_cameras = sum([1 if "image" in key else 0 for key in env.observation_space])
num_image_writer_processes= 0
num_image_writer_threads_per_camera= 4


image_keys = [key for key in env.observation_space if "image" in key]
obs_key = [key for key in env.observation_space if "agent_pos" in key]
print("obs_key",obs_key)
# state_keys_dict = env.state_keys
features = DEFAULT_FEATURES

fps = env.metadata["render_fps"]
video_path = root / "test1.mp4"
repo_id = "lerobot/aloha_test_dataset"
# add image keys to features
for key in image_keys:
            shape = env.observation_space[key].shape
            if not key.startswith("observation.image."):
                key = "observation.image." + key
            features[key] = {"dtype": "video", "names": ["channels", "height", "width"], "shape": shape}

for key in obs_key:
            shape = env.observation_space[key].shape
            if not key.startswith("observation.state."):
                 key = "observation.state."+ key        
            features[key] = {
                "dtype": "float32",
                "names": None,
                "shape": shape,
            }

features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

dataset = LeRobotDataset.create(
            repo_id,
            fps,
            root=root,
            features=features,
            use_videos=True,
            image_writer_processes=num_image_writer_processes,
            image_writer_threads=num_image_writer_threads_per_camera * num_cameras,
)

recorded_episodes = 0
Frames = [] 

while True:
    
        # if events is None:
        #     events = {"exit_early": False}

        if episode_time_s is None:
            episode_time_s = float("inf")

        timestamp = 0
        start_episode_t = time.perf_counter()
        
        policy.reset()
        seed = np.random.randint(0, 1e5)
        observation, info = env.reset(seed=seed)
        Frames = [] 
        while timestamp < episode_time_s:
            start_loop_t = time.perf_counter()

            observation = preprocess_observation(observation)
            # observation = {key: observation[key].to(device, non_blocking=True) for key in observation}
            # print("observation",observation)

            with torch.inference_mode():
                action = policy.select_action(observation)

            numpy_action = action.squeeze(0).numpy()
            observation, reward, terminated, truncated, info = env.step(numpy_action)
            Frames.append(env.render())



            success = info.get("is_success", False)
            env_timestamp = info.get("timestamp", dataset.episode_buffer["size"] / fps)
            # print("action", torch.from_numpy(numpy_action))
            frame = {
                "action": torch.from_numpy(numpy_action),
                "next.reward": reward,
                "next.success": success,
                "seed": seed,
                "timestamp": env_timestamp,
            }

            for key in image_keys:
                if not key.startswith("observation.image"):
                    frame["observation.image." + key] = observation[key]
                else:
                    frame[key] = observation[key]

            for key in obs_key:
                if not key.startswith("observation.image"):
                    frame["observation.state."+ key] = torch.from_numpy(observation[key])
                else:
                    frame[key] = torch.from_numpy(observation[key])

            dataset.add_frame(frame)


            timestamp = time.perf_counter() - start_episode_t
        #     if events["exit_early"] or terminated:
        #         events["exit_early"] = False
        #         break

        # if events["rerecord_episode"]:
        #     events["rerecord_episode"] = False
        #     events["exit_early"] = False
        #     dataset.clear_episode_buffer()
        #     continue

        dataset.save_episode(task=task)
        video_path = root/ f"rollout_episode_{recorded_episodes}.mp4"
        imageio.mimsave(str(video_path), np.stack(Frames), fps=fps)
        recorded_episodes += 1
        Frames = [] 
        if recorded_episodes >= n_episodes:
            break
        else:
            logging.info("Waiting for a few seconds before starting next episode recording...")

run_compute_stats = True
dataset.consolidate(run_compute_stats)




