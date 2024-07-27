import warnings
import time
import os
from functools import partial
from termcolor import colored
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import gymnasium as gym
import gin

from accelerate import Accelerator
from accelerate.utils import gather_object


def stack_list_array_dicts(list_: list[dict[np.ndarray]], axis=0):
    out = {}
    for t in list_:
        for k, v in t.items():
            if k in out:
                out[k].append(v)
            else:
                out[k] = [v]
    return {k: np.stack(v, axis=axis) for k, v in out.items()}


def amago_warning(msg: str, category=None):
    warnings.warn(colored(f"{msg}", "green"), category=category)


def avg_over_accelerate(data: dict[str, int | float]):
    merged_stats = gather_object([data])
    output = {}
    for device in merged_stats:
        for k, v in device.items():
            if k not in output:
                output[k] = []
            if isinstance(v, Iterable):
                output[k].extend(v)
            else:
                output[k].append(v)
    output = {k: np.array(v).mean() for k, v in output.items()}
    return output


def masked_avg(tensor: torch.Tensor, mask: torch.Tensor):
    return (tensor * mask).sum() / (mask.sum() + 1e-5)


def _get_constant_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def call_async_env(env: gym.vector.VectorEnv, method_name: str, *args, **kwargs):
    env.call_async(method_name, *args, **kwargs)
    return env.call_wait()


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gin_as_wandb_config() -> dict:
    config = gin.operative_config_str()
    lines = config.split("\n")
    params = [l.split("=") for l in lines if (not l.startswith("#") and "=" in l)]
    params_dict = {k.strip(): v.strip() for k, v in params}
    return params_dict


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        try:
            param = p.grad.data
        except AttributeError:
            continue
        else:
            param_norm = param.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def retry_load_checkpoint(ckpt_path, map_location, tries: int = 10):
    if not os.path.exists(ckpt_path):
        amago_warning("Skipping checkpoint load; file not found.")
        return

    ckpt, attempts = None, 0
    while attempts < tries:
        attempts += 1
        try:
            ckpt = torch.load(ckpt_path, map_location=map_location)
        except Exception as e:
            amago_warning(
                f"Error loading checkpoint. {'Retrying...' if attempts < tries else 'Failed'}"
            )
            time.sleep(1)
            continue
        else:
            break

    return ckpt
