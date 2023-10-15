import random

import matplotlib.pyplot as plt
import numpy as np
import wandb
from torch import nn

from .loading import MAGIC_PAD_VAL


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


def init_plt():
    plt.switch_backend("agg")
    plt.style.use("fivethirtyeight")


def q_curve_plot(loss_info: dict, num_curves: int = 3) -> wandb.Image:
    q_seq_keys = sorted([k for k in loss_info.keys() if "q_seq_mean" in k])
    q_std_keys = sorted([k for k in loss_info.keys() if "q_seq_std" in k])

    mask = loss_info["mask"].detach().cpu().float().numpy()
    batch, lengthp1, *_ = mask.shape
    length = lengthp1 - 1
    real_return = loss_info["real_return"].detach().cpu().float().numpy()
    num_curves = min(batch, 3)
    random_indices = random.sample(list(range(batch)), num_curves)

    images = {}
    for mean_key, std_key in zip(q_seq_keys, q_std_keys):
        q_mean = loss_info[mean_key].cpu().float().numpy()
        q_std = loss_info[std_key].cpu().float().numpy()

        fig = plt.figure()
        ax = plt.axes()
        x = np.arange(length)
        mean_colors = ["#ff8605", "#ffe205", "#d10209"]  # orange, yellow, red
        std_colors = ["#facba2", "#fff67a", "#fa9194"]  # orange, yellow, red
        for i, idx in enumerate(random_indices):
            m = mask[idx].squeeze(1)[:length].astype(bool)
            mean = q_mean[idx][m, 0]
            std = q_std[idx][m, 0]
            return_ = real_return[idx][m, 0]
            ax.plot(x[m], mean, color=mean_colors[i], linewidth=0.75)
            ax.plot(x[m], return_, color=mean_colors[i], linestyle="dashed")
            ax.fill_between(
                x[m], mean - 2 * std, mean + 2 * std, color=std_colors[i], alpha=0.7
            )
        ax.set_ylabel("Q(s, a, g)")
        ax.set_xlabel("Timestep")
        ax.set_facecolor("white")
        plt.tight_layout()
        img = wandb.Image(fig)
        gamma = mean_key.split("gamma=")[-1].strip()
        images[f"Q Curve (Gamma = {float(gamma):.3f})"] = img
        plt.close()

    return images
