#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

from collections import deque, defaultdict
import copy
import json

from lerobot.common.policies.calql.modeling_calql import CalQLPolicy
from lerobot.common.policies.calql.configuration_calql import CalQLConfig
from safetensors.torch import load_file 
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType, NormalizationMode
import matplotlib.pyplot as plt


# def update_policy(
#     train_metrics: MetricsTracker,
#     policy: PreTrainedPolicy,
#     batch: Any,
#     next_observation: Any,
#     optimizer: Optimizer,
#     grad_clip_norm: float,
#     grad_scaler: GradScaler,
#     lr_scheduler=None,
#     use_amp: bool = False,
#     lock=None,
# ) -> tuple[MetricsTracker, dict]:
#     start_time = time.perf_counter()
#     device = get_device_from_parameters(policy)
#     policy.train()
#     with torch.autocast(device_type=device.type) if use_amp else nullcontext():
#         loss, output_dict = policy.forward(batch, next_observation)
#         # TODO(rcadene): policy.unnormalize_outputs(out_dict)
#     grad_scaler.scale(loss).backward()

#     # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
#     grad_scaler.unscale_(optimizer)

#     grad_norm = torch.nn.utils.clip_grad_norm_(
#         policy.parameters(),
#         grad_clip_norm,
#         error_if_nonfinite=False,
#     )

#     # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
#     # although it still skips optimizer.step() if the gradients contain infs or NaNs.
#     with lock if lock is not None else nullcontext():
#         grad_scaler.step(optimizer)
#     # Updates the scale for next iteration.
#     grad_scaler.update()

#     optimizer.zero_grad()

#     # Step through pytorch scheduler at every batch instead of epoch
#     if lr_scheduler is not None:
#         lr_scheduler.step()

#     if has_method(policy, "update"):
#         # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
#         policy.update()

#     train_metrics.loss = loss.item()
#     train_metrics.grad_norm = grad_norm.item()
#     train_metrics.lr = optimizer.param_groups[0]["lr"]
#     train_metrics.update_s = time.perf_counter() - start_time
#     return train_metrics, output_dict

def load_calql_policy(model_dir: str, device: str = "cuda"):
    # Load config
    with open(f"{model_dir}/config.json", "r") as f:
        config_dict = json.load(f)
    config_dict.pop("type", None)  # remove registry key

    norm_map = config_dict["normalization_mapping"]
    config_dict["normalization_mapping"] = {
        k: NormalizationMode[v] for k, v in norm_map.items()
    }

    # Convert input_features to PolicyFeature objects
    input_feats = config_dict["input_features"]
    config_dict["input_features"] = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=v["shape"])
        for k, v in input_feats.items()
    }

    # Convert output_features to PolicyFeature objects
    output_feats = config_dict["output_features"]
    config_dict["output_features"] = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=v["shape"])
        for k, v in output_feats.items()
    }

    config = CalQLConfig(**config_dict)

    # Load model
    policy = CalQLPolicy(config).to(device)
    state_dict = load_file(f"{model_dir}/model.safetensors", device=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    print(cfg)
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    print(dataset[0]["observation.image"].size())
    next_observation = []
  

    calql_policy = load_calql_policy("./outputs/train/2025-04-12/20-43-19_pusht_calql/checkpoints/last/pretrained_model")

    # print(dataset[1]['timestamp'])
    #================================================================
    # episode = 0

    

    # for i in range (len(dataset)):
    #     temp = {}
    #     if i + 1 < len(dataset) and dataset[i + 1]['episode_index'] == dataset[i]['episode_index']:
    #         temp["observation.image"] = copy.deepcopy(dataset[i + 1]['observation.image'])
    #         temp["observation.state"] = copy.deepcopy(dataset[i + 1]['observation.state'])
    #         temp["next_action"] = copy.deepcopy(dataset[i + 1]['action'])
    #     else:
    #         temp["observation.image"] = torch.zeros_like(dataset[0]['observation.image'])
    #         temp["observation.state"] = torch.zeros_like(dataset[0]['observation.state'])
    #         temp["next_action"] = torch.zeros_like(dataset[0]['action'])
        
    #     # dataset[i].add_column('next_observation', copy.deepcopy(temp))
    #     next_observation.append(temp)
    # print("here")
        
    #     print(dataset[i].keys())

    # print(type(next_observation[0]["observation.image"]))

    # filename = 'pusht_next_obs.json'

    # # Save the dictionary to a file
    # with open(filename, 'w') as file:
    #     json.dump(next_observation, file)

    # # Load the dictionary from the file
    # with open(filename, 'r') as file:
    #     loaded_dict = json.load(file)



    # def add_next_observation(example, idx):
    #     if idx + 1 < len(dataset) and dataset[idx + 1]['episode_index'] == example['episode_index']:
    #         return {
    #             "next_observation": {
    #                 "observation.image": copy.deepcopy(dataset[idx + 1]['observation.image']),
    #                 "observation.state": copy.deepcopy(dataset[idx + 1]['observation.state']),
    #             }
    #         }
    #     else:
    #         return {
    #             "next_observation": {
    #                 "observation.image": None,
    #                 "observation.state": None,
    #             }
    #         }

    # dataset = dataset.map(add_next_observation, with_indices=True)
    # print(dataset[0].keys()) 



    #================================================================
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    # print(cfg.batch_size)
    # print(len(dataloader))
    # this = next(iter(dataloader))
    # for key, value in this.items():
    #     # print(f"{key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    #     print(value)
    policy.train()
    # print(len(dataloader.dataset))

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    q_values_by_episode = defaultdict(list)

    logging.info("Start offline training on a fixed dataset")
    print("Start offline training on a fixed dataset")
    for _ in range(len(dataloader)):
        batch = next(dl_iter)
        if max(batch["episode_index"]) < 2:
            

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            batch_size = batch["episode_index"].shape[0]

            for i in range(batch_size):
                episode_idx = batch["episode_index"][i].item()

                obs = {
                    "observation.images": batch["observation.image"][i].unsqueeze(0),  # [1, C, H, W]
                    "observation.state": batch["observation.state"][i, 0, :].unsqueeze(0),  # [1, 2]

                }

                

                action = batch["action"][i].unsqueeze(0)
                action = action[:, 0, :] 
                # print(batch["action"])
                # print(batch["observation.state"])
                 
                with torch.no_grad():
                    q_val = calql_policy.q_net(obs, action)  # [1, 1]
                
                q = q_val.item()

                img = obs["observation.images"].squeeze(0).permute(1, 2, 0).cpu()  # shape: [96, 96, 3]
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Q value {q:.4f}, Episode {episode_idx}, Step {_}")
                plt.savefig(f"q_values_step_{_}.png")
                plt.close()  # Close the figure to free memory
                print(f"Step {_}, Episode {episode_idx}, Index {i}: Q = {q:.4f}")
        else:
            break


    # for ep_idx, q_vals in q_values_by_episode.items():
    #     if ep_idx < 10:
    #         plt.plot(q_vals, label=f"Episode {ep_idx}")
    #     else:
    #         break

    # plt.xlabel("Step")
    # plt.ylabel("Q-value")
    # plt.title("Q-values per Step by Episode")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
if __name__ == "__main__":
    init_logging()
    train()
