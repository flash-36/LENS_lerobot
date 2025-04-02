#!/usr/bin/env python3

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

"""Train Calibrated Q Learning Policy on Aloha robot tasks."""

import logging
import os
from pathlib import Path

from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_logging
from lerobot.configs.parser import wrap
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.train import (
    train as train_base,
)  # Import the train function directly
from lerobot.common.policies.calql import CalQLConfig


@wrap()
def train(cfg: TrainPipelineConfig):
    """Train a CalQL policy on an Aloha task."""
    # Initialize logging
    init_logging()

    # If no policy is provided, create a CalQL policy config
    if cfg.policy is None:
        logging.info(
            f"No policy provided, creating a CalQL policy for {cfg.env.type}/{cfg.env.task}"
        )

        # Policy features will be inferred from environment
        input_features = {}
        output_features = {}

        # For Aloha, the action is discrete with 14 dimensions
        output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(14,))

        # Create CalQL config
        cfg.policy = CalQLConfig(
            input_features=input_features,
            output_features=output_features,
        )

    # Make sure we're using a CalQL policy
    if cfg.policy.type != "calql":
        logging.warning(
            f"Expected CalQL policy type, got {cfg.policy.type}. Switching to CalQL."
        )
        old_policy = cfg.policy

        # Transfer some parameters if possible
        cfg.policy = CalQLConfig(
            input_features=old_policy.input_features,
            output_features=old_policy.output_features,
            device=old_policy.device,
        )

    # Use the standard training function
    train_base(cfg)


if __name__ == "__main__":
    train()
