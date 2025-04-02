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
from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("calql")
@dataclass
class CalQLConfig(PreTrainedConfig):
    """Configuration class for the Calibrated Q-Learning policy.

    Defaults are configured for training on Aloha tasks like "insertion".

    Args:
        hidden_dim: Dimension of hidden layers in the network.
        discount: Discount factor for the Q-learning update.
        cal_alpha: Weight for the calibration loss.
        target_update_freq: Frequency to update the target network (in training steps).
    """

    # Input / output structure
    n_obs_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture
    hidden_dim: int = 256

    # Vision backbone (similar to ACT policy)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Training
    discount: float = 0.99
    cal_alpha: float = 0.5  # weight for the calibration loss
    target_update_freq: int = 1000  # frequency to update target network (in steps)

    # Optimizer
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.robot_state_feature:
            raise ValueError(
                "You must provide at least one image or robot state among the inputs."
            )
        if not self.action_feature:
            raise ValueError("Action feature must be provided in output_features.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
