import warnings
import time
import os
from functools import partial
from typing import Iterable

import gin
import numpy as np
import gymnasium as gym
from termcolor import colored
from accelerate.utils import gather_object
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW


class AmagoWarning(Warning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def amago_warning(msg: str, category=AmagoWarning):
    """Print a warning message in green, usually to warn about unintuitive hparam settings at startup."""
    warnings.warn(colored(f"{msg}", "green"), category=category)


@gin.configurable
class AdamWRel(AdamW):
    """A variant of AdamW with timestep resets.

    Implementation of the optimizer discussed in "Adam on Local Time: Addressing Nonstationarity
    in RL with Relative Adam Timesteps", Ellis et al., 2024.
    (https://openreview.net/pdf?id=biAqUbAuG7). Treats optimization of an RL policy as a sequence of stationary supervised learning stages,
    and resets Adam's timestep variable accordingly.

    Args:
        reset_interval: Number of gradient steps between resets of Adam's time / step count tracker. Must be configured by gin.

    Keyword Args:
        Follows the main Adam.
    """

    def __init__(
        self,
        params,
        reset_interval: int = gin.REQUIRED,
        lr: float = 1e-3,
        betas: tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.reset_interval = int(reset_interval)
        amago_warning(
            f"Using AdamW with non-stationary timestep resets every {self.reset_interval} steps."
        )
        self.global_step = 0

    def step(self, closure=None):
        loss = super().step(closure)
        self.global_step += 1
        if self.global_step % self.reset_interval == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p in self.state:
                        self.state[p]["step"] *= 0
        return loss


def stack_list_array_dicts(
    list_: list[dict[np.ndarray]], axis=0, cat: bool = False
) -> dict[str, np.ndarray]:
    """Stack a list of dictionaries of numpy arrays.

    Args:
        list_: List of dictionaries of numpy arrays.
        axis: Axis to stack along.
        cat: Whether to concatenate along an existing axis instead of stacking along a new one.
    """
    out = {}
    for t in list_:
        for k, v in t.items():
            if k in out:
                out[k].append(v)
            else:
                out[k] = [v]
    f = np.concatenate if cat else np.stack
    return {k: f(v, axis=axis) for k, v in out.items()}


def split_dict(dict_: dict[str, np.ndarray], axis=0) -> list[dict[str, np.ndarray]]:
    """Split a dictionary of numpy arrays into a list of dictionaries of numpy arrays.

    Inverse of `stack_list_array_dicts`.

    Args:
        dict_: Dictionary of numpy arrays.
        axis: Axis to split along.
    """
    unstacked = {k: split_batch(v, axis=axis) for k, v in dict_.items()}
    out = None
    for k, vs in unstacked.items():
        if out is None:
            out = [{k: v} for v in vs]
        else:
            for i, v in enumerate(vs):
                out[i][k] = v
    return out


def split_batch(arr: np.ndarray, axis: int) -> list[np.ndarray]:
    # this split does the same thing as `np.unstack` without requiring numpy 2.1+.
    # split seems to be much slower than unstack if the array is not in cpu memory.
    return np.split(np.ascontiguousarray(arr), arr.shape[axis], axis=axis)


def _func_over_accelerate(
    data: dict[str, int | float], func: callable
) -> dict[str, int | float]:
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
    output = {k: func(v) for k, v in output.items()}
    return output


def avg_over_accelerate(data: dict[str, int | float]) -> dict[str, int | float]:
    """Average a dictionary of ints or floats over all devices.

    Args:
        data: Dictionary of ints or floats.
    """
    return _func_over_accelerate(data, lambda x: np.array(x).mean())


def sum_over_accelerate(data: dict[str, int | float]) -> dict[str, int | float]:
    """Sum a dictionary of ints or floats over all devices.

    Args:
        data: Dictionary of ints or floats.
    """
    return _func_over_accelerate(data, lambda x: np.array(x).sum())


def masked_avg(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a tensor over a mask.

    Args:
        tensor: Tensor to average.
        mask: Mask to average over. False where indices should be ignored.
    """
    mask = mask.float()
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
    """Get a constant learning rate schedule with a warmup period."""
    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def call_async_env(env: gym.vector.VectorEnv, method_name: str, *args, **kwargs):
    """Convenience that calls a method over (async) parallel envs and waits for the results."""
    env.call_async(method_name, *args, **kwargs)
    return env.call_wait()


def count_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a pytorch module.

    Args:
        model: Pytorch module.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gin_as_wandb_config() -> dict:
    """Convert the active gin config to a dictionary for convenient logging to wandb."""
    config = gin.operative_config_str()
    lines = config.split("\n")
    params = [l.split("=") for l in lines if (not l.startswith("#") and "=" in l)]
    params_dict = {k.strip(): v.strip() for k, v in params}
    return params_dict


def get_grad_norm(model: nn.Module) -> float:
    """Get the (L2) norm of the gradients for a pytorch module.

    Args:
        model: Pytorch module.
    """
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
    """Load a model checkpoint with a retry loop in case of async read/write issues

    Args:
        ckpt_path: Path to the checkpoint file.
        map_location: Device map location for the checkpoint.
        tries: Number of tries to load the checkpoint before giving up.

    Returns:
        ckpt: torch.load() result. None if load failed.
    """
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
