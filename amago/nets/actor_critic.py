import math
from typing import Optional, Type
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange
from einops.layers.torch import EinMix as Mix
import gin

from .ff import FFBlock, MLP
from .utils import activation_switch, symlog, symexp
from .policy_dists import (
    Discrete,
    PolicyDistribution,
    TanhGaussian,
    DiscreteLikeContinuous,
)
from amago.utils import amago_warning


@gin.configurable
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
        dropout_p: float = 0.0,
        continuous_dist_type: Optional[Type[PolicyDistribution]] = TanhGaussian,
    ):
        super().__init__()
        # determine policy output
        self.num_gammas = len(gammas)
        dist_type = Discrete if discrete else continuous_dist_type
        self.policy_dist = dist_type(d_action=action_dim)
        assert isinstance(self.policy_dist, PolicyDistribution)
        assert self.policy_dist.is_discrete == discrete
        self.actions_differentiable = self.policy_dist.actions_differentiable
        d_output = self.policy_dist.input_dimension * self.num_gammas

        # build base network
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

    def forward(self, state):
        dist_params = self.base(state)
        dist_params = rearrange(
            dist_params, "b ... (g f) -> b ... g f", g=self.num_gammas
        )
        return self.policy_dist(dist_params)


class _EinMixEnsemble(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        inp_dim: int,
        d_hidden: int,
        n_layers: int,
        out_dim: int,
        activation: str,
        dropout_p: float,
    ):
        super().__init__()
        self.inp_layer = Mix(
            "b l d_in -> b l c d_out",
            weight_shape="c d_in d_out",
            bias_shape="c d_out",
            c=ensemble_size,
            d_in=inp_dim,
            d_out=d_hidden,
        )
        self.core_layers = nn.ModuleList(
            [
                Mix(
                    "b l c d_in -> b l c d_out",
                    weight_shape="c d_in d_out",
                    bias_shape="c d_out",
                    c=ensemble_size,
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
            c=ensemble_size,
            d_in=d_hidden,
            d_out=out_dim,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.activation = activation_switch(activation)

    def forward(self, inp):
        phis = self.dropout(self.activation(self.inp_layer(inp)))
        for layer in self.core_layers:
            phis = self.dropout(self.activation(layer(phis)))
        outputs = self.output_layer(phis)
        return outputs, phis


@lru_cache
def gammas_as_input_seq(
    gammas: torch.Tensor, batch_size: int, length: int
) -> torch.Tensor:
    gammas_rep = symlog(1.0 / (1.0 - gammas).clamp(1e-6, 1.0))
    gammas_rep = repeat(gammas_rep, "g -> (b g) l 1", b=batch_size, l=length)
    return gammas_rep


@gin.configurable
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
        self.net = _EinMixEnsemble(
            ensemble_size=num_critics,
            inp_dim=inp_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            out_dim=out_dim,
            dropout_p=dropout_p,
            activation=activation,
        )

    def __len__(self):
        return self.num_critics

    @torch.compile
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        assert action.dim() == 5
        K, B, L, G, D = action.shape
        if self.discrete:
            assert K == 1
            inp = state
        else:
            state = repeat(state, "b l d -> (k b g) l d", k=K, g=self.num_gammas)
            action = rearrange(action, "k b l g d -> (k b g) l d", k=K)
            gammas_rep = gammas_as_input_seq(self.gammas, K * B, L).to(action.device)
            clip_action = action.clamp(-0.999, 0.999)
            inp = torch.cat((state, gammas_rep, clip_action), dim=-1)
        outputs, _ = self.net(inp)
        if self.discrete:
            outputs = rearrange(
                outputs, "b l c (g o) -> 1 b l c g o", g=self.num_gammas
            )
            outputs = (outputs * action.unsqueeze(3)).sum(-1, keepdims=True)
        else:
            outputs = rearrange(
                outputs, "(k b g) l c o -> k b l c g o", k=K, g=self.num_gammas
            )
        return outputs


@gin.configurable
class NCriticsTwoHot(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gammas: torch.Tensor,
        num_critics: int = 4,
        d_hidden: int = 256,
        n_layers: int = 2,
        dropout_p: float = 0.0,
        activation: str = "leaky_relu",
        # see "Breaking Multi-Task Barrier.." Appendix A
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        output_bins: int = 128,
        use_symlog: bool = True,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.num_gammas = len(gammas)
        self.action_dim = action_dim
        self.gammas = gammas
        inp_dim = state_dim + action_dim + 1
        out_dim = output_bins
        self.num_bins = output_bins
        if min_return is None or max_return is None:
            amago_warning(
                "amago.nets.actor_critic.NCriticsTwoHot.min_return/max_return have not been set manually, and default to extreme values."
            )
        min_return = min_return or -100_000
        max_return = max_return or 100_000
        self.transform_values = symlog if use_symlog else lambda x: x
        self.invert_bins = symexp if use_symlog else lambda x: x
        assert min_return < max_return
        self.bin_vals = torch.linspace(
            self.transform_values(min_return),
            self.transform_values(max_return),
            output_bins,
        )
        self.net = _EinMixEnsemble(
            ensemble_size=num_critics,
            inp_dim=inp_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            out_dim=out_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

    def __len__(self):
        return self.num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        assert action.dim() == 4
        B, L, G, D = action.shape
        assert G == self.num_gammas
        state = repeat(state, "b l d -> (b g) l d", g=self.num_gammas)
        action = rearrange(action.clamp(-0.999, 0.999), "b l g d -> (b g) l d")
        gammas_rep = gammas_as_input_seq(self.gammas, B, L).to(action.device)
        inp = torch.cat((state, gammas_rep, action), dim=-1)
        outputs, phis = self.net(inp)
        outputs = rearrange(outputs, "(b g) l c o -> b l c g o", g=self.num_gammas)
        val_dist = pyd.Categorical(logits=outputs)
        clip_probs = val_dist.probs.clamp(1e-6, 0.999)
        safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
        safe_dist = pyd.Categorical(probs=safe_probs)
        return safe_dist, phis

    def bin_dist_to_raw_vals(self, bin_dist: pyd.Categorical):
        assert isinstance(bin_dist, pyd.Categorical)
        probs = bin_dist.probs
        bin_vals = self.bin_vals.to(probs.device, dtype=probs.dtype)
        exp_val = (probs * bin_vals).sum(-1, keepdims=True)
        return self.invert_bins(exp_val)

    def raw_vals_to_labels(self, raw_td_target):
        # raw scalar --> symlog --> two hot encoding
        # (github: danijar/dreamerv3/jaxutils.py)
        symlog_td_target = self.transform_values(raw_td_target)
        bin_vals = self.bin_vals.to(symlog_td_target.device)
        # below and above are indices of the bins directly above and below the scaled value
        below = ((bin_vals <= symlog_td_target).sum(-1) - 1).clamp(0, self.num_bins - 1)
        above = (self.num_bins - (bin_vals > symlog_td_target).sum(-1)).clamp(
            0, self.num_bins - 1
        )
        equal = (below == above).unsqueeze(-1)
        bin_vals = bin_vals.view(-1)
        # represent distance of scaled value to above and below as a % of the total gap between bins
        dist_to_below = torch.where(
            equal, 1, abs(bin_vals[below].unsqueeze(-1) - symlog_td_target)
        )
        dist_to_above = torch.where(
            equal, 1, abs(bin_vals[above].unsqueeze(-1) - symlog_td_target)
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        # create two hot encoded target where the labels are two consecutive bins with values that sum to 1
        target = (
            F.one_hot(below, num_classes=self.num_bins) * weight_below
            + F.one_hot(above, num_classes=self.num_bins) * weight_above
        )
        return target


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
        self.register_buffer("mu", torch.zeros(gammas, 1))
        self.register_buffer("nu", torch.ones(gammas, 1) * init_nu)
        self.register_buffer("w", torch.ones((gammas, 1)))
        self.register_buffer("b", torch.zeros((gammas, 1)))
        self.register_buffer("_t", torch.ones((gammas, 1)))
        self.beta = beta
        self.enabled = enabled

    @property
    def sigma(self):
        inner = (self.nu - self.mu**2).clamp(1e-4, 1e8)
        return torch.sqrt(inner).clamp(1e-4, 1e6)

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
