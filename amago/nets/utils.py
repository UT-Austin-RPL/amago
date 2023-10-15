import torch
from torch import nn
from torch.nn import functional as F
import gin

from amago.loading import MAGIC_PAD_VAL


def activation_switch(activation: str) -> callable:
    choices = {"leaky_relu": F.leaky_relu, "relu": F.relu, "gelu": F.gelu}
    assert activation in choices
    return choices[activation]


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
