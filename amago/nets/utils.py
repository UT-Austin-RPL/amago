from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import gin

from amago.loading import MAGIC_PAD_VAL


def symlog(x: torch.Tensor | float) -> torch.Tensor | float:
    not_torch = not isinstance(x, torch.Tensor)
    if not_torch:
        assert isinstance(x, int | float)
        x = torch.Tensor([x])
    out = torch.sign(x) * torch.log(abs(x) + 1)
    if not_torch:
        out = out.item()
    return out


def symexp(x: torch.Tensor | float) -> torch.Tensor | float:
    not_torch = not isinstance(x, torch.Tensor)
    if not_torch:
        assert isinstance(x, int | float)
        x = torch.Tensor([x])
    out = torch.sign(x) * (torch.exp(abs(x)) - 1)
    if not_torch:
        out = out.item()
    return out


def add_activation_log(
    root_key: str, activation: torch.Tensor, log_dict: Optional[dict] = None
):
    if log_dict is None:
        return
    with torch.no_grad():
        log_dict[f"activation-{root_key}-max"] = activation.max()
        log_dict[f"activation-{root_key}-min"] = activation.min()
        log_dict[f"activation-{root_key}-std"] = activation.std()
        log_dict[f"activation-{root_key}-mean"] = activation.mean()


@gin.configurable
class InputNorm(nn.Module):
    def __init__(self, dim, beta=1e-4, init_nu=1.0, skip: bool = False):
        super().__init__()
        self.skip = skip
        self.mu = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.nu = nn.Parameter(torch.ones(dim) * init_nu, requires_grad=False)
        self.beta = beta
        self._t = nn.Parameter(torch.ones((1,)), requires_grad=False)
        self.pad_val = MAGIC_PAD_VAL

    @property
    def sigma(self):
        sigma_ = torch.sqrt(self.nu - self.mu**2 + 1e-5)
        return torch.nan_to_num(sigma_).clamp(1e-3, 1e6)

    def normalize_values(self, val):
        """
        Normalize the input with instability protection.

        This function has to normalize lots of elements that are
        not well distributed (terminal signals, rewards, some
        parts of the state).
        """
        if self.skip:
            return val
        sigma = self.sigma
        normalized = ((val - self.mu) / sigma).clamp(-1e4, 1e4)
        not_nan = ~torch.isnan(normalized)
        stable = (sigma > 0.01).expand_as(not_nan)
        use_norm = torch.logical_and(stable, not_nan)
        output = torch.where(use_norm, normalized, (val - torch.nan_to_num(self.mu)))
        return output

    def denormalize_values(self, val):
        if self.skip:
            return val
        sigma = self.sigma
        denormalized = (val * sigma) + self.mu
        stable = (sigma > 0.01).expand_as(denormalized)
        output = torch.where(stable, denormalized, (val + torch.nan_to_num(self.mu)))
        return output

    def masked_stats(self, val):
        # make sure the padding value doesn't impact statistics
        mask = (~((val == self.pad_val).all(-1, keepdim=True))).float()
        sum_ = (val * mask).sum((0, 1))
        square_sum = ((val * mask) ** 2).sum((0, 1))
        total = mask.sum((0, 1))
        mean = sum_ / total
        square_mean = square_sum / total
        return mean, square_mean

    def update_stats(self, val):
        self._t += 1
        old_sigma = self.sigma
        old_mu = self.mu
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        mean, square_mean = self.masked_stats(val)
        self.mu.data = (1.0 - beta_t) * self.mu + (beta_t * mean)
        self.nu.data = (1.0 - beta_t) * self.nu + (beta_t * square_mean)

    def forward(self, x, denormalize=False):
        if denormalize:
            return self.denormalize_values(x)
        else:
            return self.normalize_values(x)


class SlowAdaptiveRational(nn.Module):
    """
    A slow non-cuda version of "Adaptive Rational Activations to Boost Deep Reinforcement Learning",
    Delfosse et al., 2021 (https://arxiv.org/pdf/2102.09407.pdf).
    Uses Leaky ReLU init.
    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        # hardcoded to leaky relu version
        degrees = (6, 4)
        num_init = [
            0.029792778657264946,
            0.6183735264987601,
            2.323309062531321,
            3.051936237265109,
            1.4854203263828845,
            0.2510244961111299,
        ]

        den_init = [
            -1.1419548357285474,
            4.393159974992486,
            0.8714712309957245,
            0.34719662339598834,
        ]
        self.numerator = nn.Parameter(torch.Tensor(num_init), requires_grad=trainable)
        self.denominator = nn.Parameter(torch.Tensor(den_init), requires_grad=trainable)
        self.num_d, self.den_d = degrees
        self.max_d = max(degrees)

    def forward(self, x):
        pows = torch.linalg.vander(x, N=self.max_d + 1)
        num = (self.numerator * pows[..., : self.num_d]).sum(-1)
        den = (self.denominator * pows[..., 1 : 1 + self.den_d]).abs().sum(-1) + 1
        return num / den


def activation_switch(activation: str) -> callable:
    if activation == "leaky_relu":
        return F.leaky_relu
    elif activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "adaptive":
        return SlowAdaptiveRational()
    else:
        raise ValueError(f"Unrecognized `activation` func: {activation}")
