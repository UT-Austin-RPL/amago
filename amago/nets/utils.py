"""
Miscellaneous utilities for neural network modules.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import gin

from amago.loading import MAGIC_PAD_VAL


def symlog(x: torch.Tensor | float) -> torch.Tensor | float:
    """Symmetric log transform.

    Applies sign(x) * log(|x| + 1) to the input. This transform is useful for
    rescaling ~unbounded ranges to a suitable range for network inputs/outputs.

    Args:
        x: Input tensor or scalar value.

    Returns:
        symlog(x) as a Tensor if x is a Tensor, otherwise symlog(x) as a float.
    """
    not_torch = not isinstance(x, torch.Tensor)
    if not_torch:
        assert isinstance(x, int | float)
        x = torch.Tensor([x])
    out = torch.sign(x) * torch.log(abs(x) + 1)
    if not_torch:
        out = out.item()
    return out


def symexp(x: torch.Tensor | float) -> torch.Tensor | float:
    """Symmetric exponential transform.

    Applies sign(x) * (exp(|x|) - 1) to the input. This is the inverse of the
    symmetric log transform.

    Args:
        x: Input tensor or scalar value.

    Returns:
        symexp(x) as a Tensor if x is a Tensor, otherwise symexp(x) as a float.
    """
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
    """Add activation statistics to a logging dictionary.

    Logs the maximum, minimum, standard deviation, and mean of the activation
    tensor under the key prefix "activation-{root_key}-".

    Args:
        root_key: Prefix for the log keys.
        activation: Tensor to compute statistics from.
        log_dict: Dictionary to add statistics to. If None, no logging is
            performed.
    """
    if log_dict is None:
        return
    with torch.no_grad():
        log_dict[f"activation-{root_key}-max"] = activation.max()
        log_dict[f"activation-{root_key}-min"] = activation.min()
        log_dict[f"activation-{root_key}-std"] = activation.std()
        log_dict[f"activation-{root_key}-mean"] = activation.mean()


@gin.configurable(denylist=["skip"])
class InputNorm(nn.Module):
    """Moving-average feature normalization.

    Normalizes input features using a moving average of their statistics. This
    helps stabilize training by keeping the input distribution relatively
    constant.

    Args:
        dim: Dimension of the input feature.

    Keyword Args:
        beta: Smoothing parameter for the moving average. Defaults to 1e-4.
        init_nu: Initial value for the moving average of the squared feature
            values. Defaults to 1.0.
        skip (no gin): Whether to skip normalization. Defaults to False. Cannot be
            configured via gin (disable input norm in the TstepEncoder config).
    """

    def __init__(self, dim, beta=1e-4, init_nu=1.0, skip: bool = False):
        super().__init__()
        self.skip = skip
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("nu", torch.ones(dim) * init_nu)
        self.register_buffer("_t", torch.ones((1,)))
        self.beta = beta
        self.pad_val = MAGIC_PAD_VAL

    @property
    def sigma(self):
        sigma_ = torch.sqrt(self.nu - self.mu**2 + 1e-5)
        return torch.nan_to_num(sigma_).clamp(1e-3, 1e6)

    def normalize_values(self, val: torch.Tensor) -> torch.Tensor:
        if self.skip:
            return val
        sigma = self.sigma
        normalized = ((val - self.mu) / sigma).clamp(-1e4, 1e4)
        not_nan = ~torch.isnan(normalized)
        stable = (sigma > 0.01).expand_as(not_nan)
        use_norm = torch.logical_and(stable, not_nan)
        output = torch.where(use_norm, normalized, (val - torch.nan_to_num(self.mu)))
        return output

    def denormalize_values(self, val: torch.Tensor) -> torch.Tensor:
        if self.skip:
            return val
        sigma = self.sigma
        denormalized = (val * sigma) + self.mu
        stable = (sigma > 0.01).expand_as(denormalized)
        output = torch.where(stable, denormalized, (val + torch.nan_to_num(self.mu)))
        return output

    def masked_stats(self, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # make sure the padding value doesn't impact statistics
        mask = (~((val == self.pad_val).all(-1, keepdim=True))).float()
        sum_ = (val * mask).sum((0, 1))
        square_sum = ((val * mask) ** 2).sum((0, 1))
        total = mask.sum((0, 1))
        mean = sum_ / total
        square_mean = square_sum / total
        return mean, square_mean

    def update_stats(self, val: torch.Tensor) -> None:
        self._t += 1
        old_sigma = self.sigma
        old_mu = self.mu
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        mean, square_mean = self.masked_stats(val)
        self.mu.data = (1.0 - beta_t) * self.mu + (beta_t * mean)
        self.nu.data = (1.0 - beta_t) * self.nu + (beta_t * square_mean)

    def forward(self, x: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
        if denormalize:
            return self.denormalize_values(x)
        else:
            return self.normalize_values(x)


@gin.configurable
class SlowAdaptiveRational(nn.Module):
    """Adaptive Rational Activation.

    A slow non-cuda version of "Adaptive Rational Activations to Boost Deep
    Reinforcement Learning", Delfosse et al., 2021
    (https://arxiv.org/pdf/2102.09407.pdf). Hardcoded to the Leaky Relu version.

    Keyword Args:
        trainable: Whether to train the parameters of the activation. Defaults to
            True.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pows = torch.linalg.vander(x, N=self.max_d + 1)
        num = (self.numerator * pows[..., : self.num_d]).sum(-1)
        den = (self.denominator * pows[..., 1 : 1 + self.den_d]).abs().sum(-1) + 1
        return num / den


def activation_switch(activation: str) -> callable:
    """Quick switch for the activation function.

    Args:
        activation: The activation function name. Options are:
            - "leaky_relu" (Leaky ReLU)
            - "relu" (ReLU)
            - "gelu" (GeLU)
            - "adaptive" (SlowAdaptiveRational)

    Returns:
        The activation function (callable).

    Raises:
        ValueError: If the activation function name is not recognized.
    """
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
