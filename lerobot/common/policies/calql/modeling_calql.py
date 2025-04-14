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
"""Calibrated Q-Learning Policy for LeRobot"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.calql.configuration_calql import CalQLConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy

from lerobot.configs.types import DictLike, FeatureType, PolicyFeature

class CalQLNetwork(nn.Module):
    """
    Q-network for visual-proprioceptive control with calibration.
    """

    def __init__(self, config: CalQLConfig):
        """
        Initialize the Q-network.

        Args:
            config: Configuration for the Q-network
        """
        super(CalQLNetwork, self).__init__()

        # Extract key configuration
        self.hidden_dim = config.hidden_dim

        # Get input/output dimensions
        self.image_features = config.image_features
        self.robot_state_feature = config.robot_state_feature
 
        self.action_feature = config.action_feature
        # Visual encoder (for images)
        if self.image_features:
            # Use a ResNet backbone similar to ACT policy
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[
                    False,
                    False,
                    config.replace_final_stride_with_dilation,
                ],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Extract features from the final layer (layer4)
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )

            # Projection layer to convert backbone features to the hidden dimension
            self.img_feat_proj = nn.Conv2d(
                backbone_model.fc.in_features, self.hidden_dim, kernel_size=1
            )

            # Global average pooling to get a fixed-size representation
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            # Calculate the feature dimension after flattening the pooled output
            with torch.no_grad():
                # Use a dummy tensor to calculate output size
                dummy = torch.zeros(1, 3, 224, 224)  # Assuming standard ResNet input
                feat_map = self.backbone(dummy)["feature_map"]
                feat_proj = self.img_feat_proj(feat_map)
                feat_pool = self.global_pool(feat_proj)
                self.img_feat_dim = feat_pool.view(1, -1).size(1)

        # Proprioceptive encoder
        if self.robot_state_feature:
            proprio_dim = self.robot_state_feature.shape[0]
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )

        # Calculate combined feature size
        self.combined_feature_size = 0
        if self.image_features:
            # For each camera view, we'll have an img_feat_dim sized feature
            self.combined_feature_size += self.img_feat_dim * len(self.image_features)

        if self.robot_state_feature:
            self.combined_feature_size += self.hidden_dim

        # Action dimensions (for output Q-values)
        if not self.action_feature:
            raise ValueError("Action feature must be provided")

        # For discrete actions, shape is (num_actions,)
        self.action_dim = self.action_feature.shape[0]

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.combined_feature_size + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),  # <-- single Q-value
        )

    def forward(self, batch: dict[str, Tensor], action: Tensor) -> Tensor:
        """
        Forward pass through the Q-network for continuous actions.

        Args:
            batch: Dict with observations (images, states, etc.)
            action: Tensor of shape [B, action_dim], continuous actions

        Returns:
            Q-value for each (obs, action) pair: shape [B, 1]
        """
        features = []
        
        # Process visual observations
        if self.image_features:
            if "observation.images" in batch:
                image_list = batch["observation.images"]
            else:
                image_list = [batch[name] for name in self.image_features if name in batch]
            device = next(self.parameters()).device  # ensures match with model

            for img in image_list:
                if img is None:
                    continue
                img = img.to(device)
                feat_map = self.backbone(img)["feature_map"]
                feat_proj = self.img_feat_proj(feat_map)
                feat_pool = self.global_pool(feat_proj)
                visual_features = feat_pool.view(feat_pool.size(0), -1)
                features.append(visual_features)

        # Process proprioception
        if self.robot_state_feature and "observation.state" in batch:
            state = batch["observation.state"].to(next(self.parameters()).device)
            proprio_features = self.proprio_encoder(state)
            if proprio_features.dim() == 1:
                proprio_features = proprio_features.view(1, -1)
            features.append(proprio_features)

        # Combine all obs features
        action = action.to(device)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # features = [f.view(f.size(0), -1) if f.dim() > 2 else f for f in features]

        combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]

        # ⬇️ NEW: Concatenate action into obs representation
        obs_action = torch.cat([combined, action], dim=1)

        # ⬇️ NEW: Q-head now outputs a single Q-value
        q_value = self.q_head(obs_action)  # shape: [B, 1]

        return q_value



class CalQLPolicy(PreTrainedPolicy):
    """
    Calibrated Q-Learning Policy
    """

    config_class = CalQLConfig
    name = "calql"

    def __init__(
        self,
        config: CalQLConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Initialize the CalQL policy.

        Args:
            config: Policy configuration
            dataset_stats: Dataset statistics for normalization
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Main Q-network
        self.q_net = CalQLNetwork(config)

        # Target Q-network (used for stable training)
        self.target_q_net = CalQLNetwork(config)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Training steps counter
        self.training_steps = 0

        # Keep track of last experience for online learning
        self.last_observation = None
        self.last_action = None
        self.epsilon = 0.1  # For epsilon-greedy exploration

    def get_optim_params(self) -> list:
        # Similar to ACT, use different learning rates for backbone and other parameters
        if self.config.image_features:
            return [
                {
                    "params": [
                        p
                        for n, p in self.q_net.named_parameters()
                        if not n.startswith("backbone") and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.q_net.named_parameters()
                        if n.startswith("backbone") and p.requires_grad
                    ],
                    "lr": self.config.optimizer_lr_backbone,
                },
            ]
        else:
            return [{"params": [p for p in self.q_net.parameters() if p.requires_grad]}]

    def reset(self):
        """Reset the policy state when the environment is reset."""
        self.last_observation = None
        self.last_action = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], num_samples: int = 32) -> Tensor:
        """
        Select the best action by sampling from the action space and picking the one with the highest Q-value.

        Args:
            batch: Dictionary of observations
            num_samples: Number of candidate actions to sample

        Returns:
            Selected continuous action (Tensor of shape [action_dim])
        """
        self.eval()

        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Prepare image list if needed
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = [
                batch[key] for key in self.config.image_features if key in batch
            ]

        device = next(self.parameters()).device
        batch = {
            k: [x.to(device) for x in v] if isinstance(v, list) else v.to(device)
            for k, v in batch.items()
            }

        action_dim = self.q_net.action_dim

        # Sample candidate actions in [-1, 1]
        candidate_actions = torch.rand((num_samples, action_dim), device=device) * 2 - 1

        # Repeat observation for each candidate action
        repeated_batch = {
            k: v.expand(num_samples, *v.shape[1:]) if v.ndim >= 2 else v.repeat(num_samples)
            for k, v in batch.items()
        }

        with torch.no_grad():
            q_values = self.q_net(repeated_batch, candidate_actions)  # shape [num_samples, 1]
            best_idx = q_values.argmax(dim=0).item()
            best_action = candidate_actions[best_idx]  # shape: [action_dim]

        self.last_observation = batch
        self.last_action = best_action

        # Optional: unnormalize if needed
        return self.unnormalize_outputs({"action": best_action.unsqueeze(0)})["action"].squeeze(0)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Training forward pass to compute loss for Q-learning.

        Args:
            batch: Dictionary containing observations, actions, rewards, next observations, and dones

        Returns:
            Loss and loss metrics dictionary
        """
        # Normalize inputs
        batch = self.normalize_inputs(batch)

        #===============extract next obervation=====================================
        next_observations = dict()
        next_observations = {
        "observation.state": batch["observation.state"][:, 1],
        "observation.image": batch["observation.image"][:, 1]
        }
        batch["observation.state"] = batch["observation.state"][:, 0] 
        batch["observation.image"] = batch["observation.image"][:, 0]

        next_actions = batch["action"][:, 1]
        batch["action"] = batch["action"][:, 0]
        #========================================================================


        # Collect image inputs into a list (similar to ACT)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [
                batch[key] for key in self.config.image_features if key in batch
            ]

            # Also handle next_observation images if present
            if "next_observation" in batch:
                next_obs = dict(batch["next_observation"])
                next_obs["observation.images"] = [
                    next_obs[key]
                    for key in self.config.image_features
                    if key in next_obs
                ]
                batch["next_observation"] = next_obs

        # Extract batch components
        # (batch should contain 'observation', 'action', 'reward', 'next_observation', 'done')
        

        observations = batch
        actions = batch["action"].long()  # Action indices
        rewards = batch["next.reward"]
        dones = batch["next.done"]
        current_q_values = self.q_net(observations, actions)

        
        # Target Q-values (using the target network)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_observations, next_actions)
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
            target_q_values = (
                rewards.unsqueeze(1)
                + self.config.discount * ((1.0 - dones.float().unsqueeze(1))) * max_next_q
            )

        # Standard Bellman error loss (MSE)
        bellman_loss = F.mse_loss(current_q_values, target_q_values)

        # Calibration loss (penalize overestimation)
        calibration_loss = self.config.cal_alpha * torch.mean(
            torch.relu(current_q_values - target_q_values) ** 2
        )

        # Combined loss
        loss = bellman_loss + calibration_loss

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            logging.info(f"Updated target network at step {self.training_steps}")

        # Return loss and metrics
        loss_dict = {
            "bellman_loss": bellman_loss.item(),
            "calibration_loss": calibration_loss.item(),
            "total_loss": loss.item(),
        }

        return loss, loss_dict
