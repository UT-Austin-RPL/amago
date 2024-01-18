import numpy as np
from torch import nn

from .loading import MAGIC_PAD_VAL


def stack_list_array_dicts(list_: list[dict[np.ndarray]], axis=0):
    out = {}
    for t in list_:
        for k, v in t.items():
            if k in out:
                out[k].append(v)
            else:
                out[k] = [v]
    return {k: np.stack(v, axis=axis) for k, v in out.items()}


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
