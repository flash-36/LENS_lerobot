import logging
from contextlib import nullcontext

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn

from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
    set_global_seed,
)
from lerobot.configs.eval import EvalPipelineConfig



def evaluate_policy(cfg: EvalPipelineConfig, policy: PreTrainedPolicy, observation_init):
    """
    Evaluates a given policy in the aloha_gym environment starting from a specific state.

    Parameters:
        observation_init: A dictionary of tensors. The initial state to set in the environment.
        policy: The policy must be a PyTorch nn module.



    Returns:
        float: The total reward obtained during the rollout.
    """
    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(cfg.seed)

    log_output_dir(cfg.output_dir)

    logging.info("Making environment.")
    env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        env_cfg=cfg.env,
    )
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    policy.eval()


    env.set_state(observation_init)  # Set the given initial state
    total_reward = 0
    done = False
    step = 0

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
      while not done:

        if step == 0:
           observation = observation_init
        else:
            observation = preprocess_observation(observation)


        observation = {key: observation[key].to(device, non_blocking=True) for key in observation}
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Convert to CPU / numpy.
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated | truncated | done
        step += 1

    env.close()
    print(total_reward)
    logging.info("End of eval")

    return total_reward

if __name__ == "__main__":
    init_logging()
    device = get_safe_torch_device(cfg.device, log=True)
    
    logging.info("Making policy.")
    cfg: EvalPipelineConfig
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        env_cfg=cfg.env,
    )
    evaluate_policy()
