#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer Plus Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.act.modeling_act import ACT, ACTTemporalEnsembler
from lerobot.common.policies.actplus.configuration_actplus import ACTPlusConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class ACTPlus(ACT):
    """
    Action Chunking Transformer Plus: An extension of ACT with additional heads.
    Adds an observation prediction head that predicts future encoder tokens.
    """

    def __init__(self, config: ACTPlusConfig):
        super().__init__(config)
        self.config = config

        # Compute how many encoder tokens per observation: latent + robot + env + image patches
        num_tokens = 0  # latent token skipped
        if config.robot_state_feature:
            num_tokens += 1
        if config.env_state_feature:
            num_tokens += 1

        if config.image_features:
            # probe the backbone once to find Hf×Wf of its last feature map
            first_key = next(iter(config.image_features))
            c, h, w = config.input_features[first_key].shape  # raw image shape
            dummy = torch.zeros(1, c, h, w)  # on CPU is fine
            with torch.no_grad():
                fmap = self.backbone(dummy)["feature_map"]  # (1, C, Hf, Wf)
            hf, wf = fmap.shape[-2:]
            num_tokens += hf * wf * len(config.image_features)

        self.num_encoder_tokens = num_tokens

        # observation‑prediction head
        self.observation_head = nn.Linear(
            config.dim_model, num_tokens * config.dim_model
        )

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """A forward pass through the Action Chunking Transformer Plus (with optional VAE encoder).

        This overrides the forward method from ACT to expose the decoder_out for additional heads.

        Args:
            batch: Same structure as in ACT.forward()

        Returns:
            actions: (B, chunk_size, action_dim) batch of action sequences
            latent_params: Tuple containing the latent PDF's parameters (mean, log(σ²))
            observation_preds: Output from the observation prediction head
        """
        if self.config.use_vae and self.training:
            assert (
                "action" in batch
            ), "actions must be provided when using the variational objective in training mode."

        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                # Get current state (index 0 in time dimension if it exists)
                state = batch["observation.state"]
                if len(state.shape) > 2:  # If it has a time dimension
                    state = state[:, 0]  # Take the first timestep
                robot_state_embed = self.vae_encoder_robot_state_input_proj(state)
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(
                batch["action"]
            )  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [
                    cls_embed,
                    robot_state_embed,
                    action_embed,
                ]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[
                0
            ]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], dtype=torch.float32
            ).to(batch["observation.state"].device)

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )

        # Robot state token - use current observation (index 0)
        if self.config.robot_state_feature:
            state = batch["observation.state"]
            if len(state.shape) > 2:  # If it has a time dimension
                state = state[:, 0]  # Take the first timestep
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(state))

        # Environment state token - use current observation (index 0)
        if self.config.env_state_feature:
            env_state = batch["observation.environment_state"]
            if len(env_state.shape) > 2:  # If it has a time dimension
                env_state = env_state[:, 0]  # Take the first timestep
            encoder_in_tokens.append(self.encoder_env_state_input_proj(env_state))

        # Camera observation features and positional embeddings - use current observation (index 0)
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            # For a list of images, the H and W may vary but H*W is constant.
            images = batch["observation.images"]
            for img in images:
                if len(img.shape) > 4:  # If it has a time dimension [B, T, C, H, W]
                    img = img[:, 0]  # Take the first timestep
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        # Apply action prediction head
        actions = self.action_head(decoder_out)

        # Apply observation prediction head, then reshape to (B, S, N, D)
        flat_preds = self.observation_head(decoder_out)  # (B, S, N*D)
        observation_preds = flat_preds.view(
            flat_preds.size(0),
            flat_preds.size(1),
            self.num_encoder_tokens,
            self.config.dim_model,
        )
        return actions, (mu, log_sigma_x2), observation_preds


class ACTPlusPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Plus Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)

    Extends ACTPolicy with an observation prediction head.
    """

    config_class = ACTPlusConfig
    name = "actplus"

    def __init__(
        self,
        config: ACTPlusConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Use ACTPlus instead of ACT
        self.model = ACTPlus(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [
                batch[key] for key in self.config.image_features
            ]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            # For ACTPlus, the model returns actions, latent parameters, and observation predictions
            # But we still just want the actions (index 0)
            actions = self.model(batch)[0]  # Take just the actions
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # For ACTPlus, we only need the actions (first return value)
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # Unnormalize the actions
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _process_observation_for_encoder(self, batch, timestep=0):
        """
        Process observations at a specific timestep through encoder components.

        This mimics the processing in the ACTPlus.forward method, but for a specific timestep.

        Args:
            batch: The batch containing observations with time dimension
            timestep: The timestep to process (0 = current, 1+ = future)

        Returns:
            encoder_tokens: Processed encoder tokens for the given timestep
        """
        # This function contains similar code to ACTPlus.forward, but focused only on
        # processing observations at a specific timestep

        device = next(iter(batch.values())).device

        # Get batch size from any observation tensor
        if "observation.images" in batch:
            if isinstance(batch["observation.images"], list):
                batch_size = batch["observation.images"][0].shape[0]
            else:
                batch_size = batch["observation.images"].shape[0]
        elif "observation.environment_state" in batch:
            batch_size = batch["observation.environment_state"].shape[0]
        elif "observation.state" in batch:
            batch_size = batch["observation.state"].shape[0]
        else:
            raise ValueError("No valid observation keys found in batch")

        # Initialize list to collect encoder tokens
        encoder_tokens = []

        # Process robot state token
        if self.config.robot_state_feature and "observation.state" in batch:
            state = batch["observation.state"]
            if (
                len(state.shape) > 2 and state.shape[1] > timestep
            ):  # If it has time dimension and enough timesteps
                state = state[:, timestep]  # Take the specified timestep
                state_token = self.model.encoder_robot_state_input_proj(state)
                encoder_tokens.append(state_token)

        # Process environment state token
        if self.config.env_state_feature and "observation.environment_state" in batch:
            env_state = batch["observation.environment_state"]
            if (
                len(env_state.shape) > 2 and env_state.shape[1] > timestep
            ):  # If it has time dimension and enough timesteps
                env_state = env_state[:, timestep]  # Take the specified timestep
                env_state_token = self.model.encoder_env_state_input_proj(env_state)
                encoder_tokens.append(env_state_token)

        # Process image tokens the same way as in forward: patch-wise flattening
        if self.config.image_features and "observation.images" in batch:
            all_cam_features = []
            for img in batch["observation.images"]:
                # select timestep if temporal
                if img.ndim == 5:
                    if img.shape[1] <= timestep:
                        continue
                    img_t = img[:, timestep]
                else:
                    img_t = img
                # backbone + projection
                cam_feat_map = self.model.backbone(img_t)["feature_map"]
                cam_proj = self.model.encoder_img_feat_input_proj(cam_feat_map)
                # flatten patches: (B, D, H, W) -> (H*W, B, D)
                cam_patches = einops.rearrange(cam_proj, "b c h w -> (h w) b c")
                all_cam_features.append(cam_patches)
            # extend encoder tokens with each patch as a separate token
            if all_cam_features:
                patches = torch.cat(
                    all_cam_features, dim=0
                )  # shape (num_patches, B, D)
                encoder_tokens.extend(patches)

        # If we have collected any tokens, stack them
        if encoder_tokens:
            encoder_tokens = torch.stack(encoder_tokens, dim=0)
            return encoder_tokens

        # Return None if no valid tokens were processed
        return None

    def _compute_observation_prediction_loss(self, observation_preds, batch):
        """
        Compute the loss for observation prediction by processing future observations
        through the same encoder components used in the forward pass.

        Args:
            observation_preds: Tensor of shape (B, S, N, D)
            batch: Input batch containing observations with time dimension

        Returns:
            loss: Observation prediction loss
            loss_dict: Dictionary with loss details
        """
        loss_dict = {}
        obs_loss = 0.0
        n_valid_timesteps = 0
        max_future = observation_preds.size(1)
        for t in range(1, max_future + 1):
            future_tokens = self._process_observation_for_encoder(batch, timestep=t)
            if future_tokens is None:
                continue
            # Convert (N, B, D) -> (B, N, D)
            target = future_tokens.permute(1, 0, 2)
            pred = observation_preds[:, t - 1, :, :]
            t_loss = F.mse_loss(pred, target)
            obs_loss += t_loss
            n_valid_timesteps += 1
            loss_dict[f"obs_loss_t{t}"] = t_loss.item()
        if n_valid_timesteps > 0:
            obs_loss = obs_loss / n_valid_timesteps
            loss_dict["obs_loss"] = obs_loss.item()
            return obs_loss, loss_dict
        return 0.0, loss_dict

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [
                batch[key] for key in self.config.image_features
            ]

        batch = self.normalize_targets(batch)

        # For ACTPlus, model returns actions, latent params, observation predictions, and current tokens
        (
            actions_hat,
            (mu_hat, log_sigma_x2_hat),
            observation_preds,
        ) = self.model(batch)

        # Calculate L1 loss for primary actions
        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        # Initialize loss_dict with the action loss
        loss_dict = {"l1_loss": l1_loss.item()}

        # Total loss starts with action loss
        total_loss = l1_loss

        # Add observation prediction loss - passing just the predictions and batch
        # The function will handle processing future observations properly
        obs_loss, obs_loss_dict = self._compute_observation_prediction_loss(
            observation_preds, batch
        )

        if obs_loss > 0:
            loss_dict.update(obs_loss_dict)
            total_loss = total_loss + self.config.observation_loss_weight * obs_loss

        # Handle VAE loss if enabled
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (
                    -0.5
                    * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())
                )
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            total_loss = total_loss + mean_kld * self.config.kl_weight

        return total_loss, loss_dict
