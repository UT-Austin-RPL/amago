import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange
from einops.layers.torch import EinMix as Mix
import numpy as np
import gin

from .ff import FFBlock, MLP
from .utils import activation_switch


class _TanhTransform(pyd.transforms.Transform):
    # Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-0.99, 0.99))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class _SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    # Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [_TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def ContinuousActionDist(
    vec,
    log_std_low=-5.0,
    log_std_high=2.0,
    d_action=None,
):
    mu, log_std = vec.chunk(2, dim=-1)
    log_std = torch.tanh(log_std)
    log_std = log_std_low + 0.5 * (log_std_high - log_std_low) * (log_std + 1)
    std = log_std.exp()
    dist = _SquashedNormal(mu, std)
    return dist


class _Categorical(pyd.Categorical):
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)


def DiscreteActionDist(vec):
    dist = _Categorical(logits=vec)
    probs = dist.probs
    # note: this clip helps stability but is something to keep in mind,
    # especially on some toy envs where the optimal policy is very
    # deterministic. action sampling can be turned off in the `learning.Experiment`.
    clip_probs = probs.clamp(0.001, 0.99)
    safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
    safe_dist = _Categorical(probs=safe_probs)
    return safe_dist


@gin.configurable(denylist=["state_dim", "action_dim", "discrete"])
class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        n_layers: int = 2,
        d_hidden: int = 256,
        activation: str = "leaky_relu",
        # dist-specific args
        log_std_low: float = -5.0,
        log_std_high: float = 2.0,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        d_output = action_dim if discrete else 2 * action_dim
        self.num_gammas = len(gammas)
        d_output *= self.num_gammas

        self.base = MLP(
            d_inp=state_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            d_output=d_output,
            dropout_p=dropout_p,
            activation=activation,
        )
        self.discrete = discrete
        self.action_dim = action_dim
        self._log_std_low = log_std_low
        self._log_std_high = log_std_high

    def forward(self, state):
        dist_params = self.base(state)
        dist_params = rearrange(
            dist_params, "b ... (g f) -> b ... g f", g=self.num_gammas
        )
        if self.discrete:
            return DiscreteActionDist(dist_params)
        else:
            return ContinuousActionDist(
                dist_params,
                log_std_high=self._log_std_high,
                log_std_low=self._log_std_low,
                d_action=self.action_dim,
            )


@gin.configurable(denylist=["state_dim", "action_dim", "discrete", "num_critics"])
class NCritics(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        num_critics: int = 4,
        d_hidden: int = 256,
        n_layers: int = 2,
        dropout_p: float = 0.0,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.discrete = discrete
        self.num_critics = num_critics
        inp_dim = state_dim
        self.num_gammas = len(gammas)
        self.gammas = gammas
        if not discrete:
            inp_dim += action_dim + 1
            out_dim = 1
        else:
            out_dim = self.num_gammas * action_dim

        # https://colab.research.google.com/drive/1WKgs2EPUm6UPP0j4aua5kroI18oFZ2a3?usp=sharing
        self.inp_layer = Mix(
            "b l d_in -> b l c d_out",
            weight_shape="c d_in d_out",
            bias_shape="c d_out",
            c=num_critics,
            d_in=inp_dim,
            d_out=d_hidden,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.core_layers = nn.ModuleList(
            [
                Mix(
                    "b l c d_in -> b l c d_out",
                    weight_shape="c d_in d_out",
                    bias_shape="c d_out",
                    c=num_critics,
                    d_in=d_hidden,
                    d_out=d_hidden,
                )
                for _ in range(n_layers - 1)
            ]
        )

        self.output_layer = Mix(
            "b l c d_in -> b l c d_out",
            weight_shape="c d_in d_out",
            bias_shape="c d_out",
            c=num_critics,
            d_in=d_hidden,
            d_out=out_dim,
        )
        self.activation = activation_switch(activation)

    def __len__(self):
        return self.num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        if self.discrete:
            # now clipped inside of Discrete dist
            clip_action = action
            inp = state
        else:
            assert action.dim() == 4
            B, L, G, D = action.shape
            state = repeat(state, "b l d -> (b g) l d", g=self.num_gammas)
            action = rearrange(action, "b l g d -> (b g) l d")
            gammas = self.gammas.to(action.device).log() * 10.0
            gammas = repeat(gammas, "g -> (b g) l 1", b=B, l=L)
            # clip to remove DPG incentive to push actions to [-1, 1] border
            clip_action = action.clamp(-0.999, 0.999)
            inp = torch.cat((state, gammas, clip_action), dim=-1)
        phis = self.dropout(self.activation(self.inp_layer(inp)))
        for layer in self.core_layers:
            phis = self.dropout(self.activation(layer(phis)))
        outputs = self.output_layer(phis)
        if self.discrete:
            outputs = rearrange(outputs, "b l c (g o) -> b l c g o", g=self.num_gammas)
            outputs = (outputs * clip_action.unsqueeze(2)).sum(-1, keepdims=True)
        else:
            outputs = rearrange(outputs, "(b g) l c o -> b l c g o", g=self.num_gammas)
        return outputs, phis


@gin.configurable(denylist=["enabled"])
class PopArtLayer(nn.Module):
    def __init__(
        self,
        gammas: int,
        beta: float = 5e-4,
        init_nu: float = 100.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(gammas, 1), requires_grad=False)
        self.nu = nn.Parameter(torch.ones(gammas, 1) * init_nu, requires_grad=False)
        self.beta = beta
        self.w = nn.Parameter(torch.ones((gammas, 1)), requires_grad=False)
        self.b = nn.Parameter(torch.zeros((gammas, 1)), requires_grad=False)
        self._t = nn.Parameter(torch.ones((gammas, 1)), requires_grad=False)
        self._stable = False
        self.enabled = enabled

    @property
    def sigma(self):
        return (torch.sqrt(self.nu - self.mu**2)).clamp(1e-4, 1e6)

    def normalize_values(self, val):
        if not self.enabled:
            return val
        return ((val - self.mu) / self.sigma).to(val.dtype)

    def to(self, device):
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        self.mu = self.mu.to(device)
        self.nu = self.nu.to(device)
        return self

    def update_stats(self, val, mask):
        if not self.enabled:
            return
        assert val.shape == mask.shape
        self._t += 1
        old_sigma = self.sigma.data.clone()
        old_mu = self.mu.data.clone()
        # Use adaptive step size to reduce reliance on initialization (pg 13)
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        # dims are Batch, Length, 1, Gammas, 1
        total = mask.sum((0, 1, 2))
        mean = (val * mask).sum((0, 1, 2)) / total
        square_mean = ((val * mask) ** 2).sum((0, 1, 2)) / total
        self.mu.data = (1.0 - beta_t) * self.mu + beta_t * mean
        self.nu.data = (1.0 - beta_t) * self.nu + beta_t * square_mean
        self.w.data *= old_sigma / self.sigma
        self.b.data = ((old_sigma * self.b) + old_mu - self.mu) / (self.sigma)

    def forward(self, x, normalized=True):
        if not self.enabled:
            return x
        normalized_out = (self.w * x) + self.b
        if normalized:
            return normalized_out.to(x.dtype)
        else:
            return ((self.sigma * normalized_out) + self.mu).to(x.dtype)
