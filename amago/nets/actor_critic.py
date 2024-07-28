import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange
from einops.layers.torch import EinMix as Mix
import gin

from .ff import FFBlock, MLP
from .utils import activation_switch, symlog, symexp
from amago.utils import amago_warning


class _TanhWrappedDistribution(pyd.Distribution):
    """
    This is copied directly from Robomimic
    (https://github.com/ARISE-Initiative/robomimic/blob/b5d2aa9902825c6c652e3b08b19446d199b49590/robomimic/models/distributions.py),
    which orginally based it on rlkit and CQL
    (https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/distributions.py#L6).
    """

    def __init__(self, base_dist, scale=1.0, epsilon=1e-6):
        super().__init__(validate_args=False)
        self.base_dist = base_dist
        self.scale = scale
        self.tanh_epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        value = value / self.scale
        if pre_tanh_value is None:
            one_plus_x = (1.0 + value).clamp(min=self.tanh_epsilon)
            one_minus_x = (1.0 - value).clamp(min=self.tanh_epsilon)
            pre_tanh_value = 0.5 * torch.log(one_plus_x / one_minus_x)
        lp = self.base_dist.log_prob(pre_tanh_value)
        tanh_lp = torch.log(1 - value * value + self.tanh_epsilon)
        if len(lp.shape) == len(tanh_lp.shape):
            return lp - tanh_lp
        else:
            # unsqueeze for shape compatability with existing code
            return (lp - tanh_lp.sum(-1)).unsqueeze(-1)

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        z = self.base_dist.sample(sample_shape=sample_shape).detach()

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        z = self.base_dist.rsample(sample_shape=sample_shape)

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev


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
        return isinstance(other, _TanhTransform)

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


def _TanhGMM(means, stds, logits):
    comp = pyd.Independent(pyd.Normal(loc=means, scale=stds), 1)
    mix = pyd.Categorical(logits=logits)
    dist = pyd.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
    dist = _TanhWrappedDistribution(base_dist=dist, scale=1.0)
    return dist


def ContinuousActionDist(
    vec,
    kind: str,
    log_std_low: float = -5.0,
    log_std_high: float = 2.0,
    d_action: Optional[int] = None,
    gmm_modes: Optional[int] = None,
):
    assert kind in ["normal", "gmm", "multibinary"]

    if kind == "normal":
        mu, log_std = vec.chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = log_std_low + 0.5 * (log_std_high - log_std_low) * (log_std + 1)
        std = log_std.exp()
        dist = _SquashedNormal(mu, std)
    elif kind == "gmm":
        assert d_action is not None and gmm_modes is not None
        idx = gmm_modes * d_action
        means = rearrange(vec[..., :idx], "... g (m p) -> ... g m p", m=gmm_modes)
        log_std = rearrange(
            vec[..., idx : 2 * idx], "... g (m p) -> ... g m p", m=gmm_modes
        )
        log_std = log_std_low + 0.5 * (log_std_high - log_std_low) * (log_std + 1)
        stds = log_std.exp()
        logits = vec[..., 2 * idx :]
        dist = _TanhGMM(means=means, stds=stds, logits=logits)
    elif kind == "multibinary":
        dist = pyd.Bernoulli(logits=vec)
    return dist


class _Categorical(pyd.Categorical):
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)


class DiscreteLikeContinuous:
    def __init__(self, categorical: _Categorical):
        self.dist = categorical

    @property
    def probs(self):
        return self.dist.probs

    @property
    def logits(self):
        return self.dist.logits

    def entropy(self):
        return self.dist.entropy()

    def log_prob(self, action):
        return self.dist.log_prob(action.argmax(-1)).unsqueeze(-1)

    def sample(self, *args, **kwargs):
        samples = self.dist.sample(*args, **kwargs)
        action = (
            F.one_hot(samples, num_classes=self.probs.shape[-1]).squeeze(-2).float()
        )
        return action


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
        dropout_p: float = 0.0,
        # dist-specific args
        cont_dist_kind: str = "normal",
        log_std_low: float = -5.0,
        log_std_high: float = 2.0,
        gmm_modes: int = 5,
    ):
        super().__init__()
        if discrete:
            d_output = action_dim
            self.actions_differentiable = True
        elif cont_dist_kind == "normal":
            d_output = 2 * action_dim
            self.actions_differentiable = True
        elif cont_dist_kind == "gmm":
            d_output = gmm_modes * 2 * action_dim + gmm_modes
            self.actions_differentiable = False
        elif cont_dist_kind == "multibinary":
            d_output = action_dim
            self.actions_differentiable = False
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
        self._gmm_modes = gmm_modes
        self._cont_dist_kind = cont_dist_kind

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
                kind=self._cont_dist_kind,
                log_std_high=self._log_std_high,
                log_std_low=self._log_std_low,
                d_action=self.action_dim,
                gmm_modes=self._gmm_modes,
            )


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


@gin.configurable(
    denylist=["state_dim", "action_dim", "discrete", "gammas", "num_critics"]
)
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
        outputs, phis = self.net(inp)
        if self.discrete:
            outputs = rearrange(outputs, "b l c (g o) -> b l c g o", g=self.num_gammas)
            outputs = (outputs * clip_action.unsqueeze(2)).sum(-1, keepdims=True)
        else:
            outputs = rearrange(outputs, "(b g) l c o -> b l c g o", g=self.num_gammas)
        return outputs, phis


@gin.configurable(denylist=["state_dim", "action_dim", "gammas", "num_critics"])
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
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        output_bins: int = 128,
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
        assert min_return < max_return
        self.bin_vals = torch.linspace(
            symlog(min_return), symlog(max_return), output_bins
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
        action = rearrange(action.clamp(-0.995, 0.995), "b l g d -> (b g) l d")
        gammas = (
            torch.arange(self.num_gammas, dtype=action.dtype, device=action.device)
            / self.num_gammas
        )
        gammas = repeat(gammas, "g -> (b g) l 1", b=B, l=L)
        inp = torch.cat((state, gammas, action), dim=-1)
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
        return symexp(exp_val)

    def raw_vals_to_labels(self, raw_td_target):
        # raw scalar --> symlog --> two hot encoding
        # (github: danijar/dreamerv3/jaxutils.py)
        symlog_td_target = symlog(raw_td_target)
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
        self.mu = nn.Parameter(torch.zeros(gammas, 1), requires_grad=False)
        self.nu = nn.Parameter(torch.ones(gammas, 1) * init_nu, requires_grad=False)
        self.beta = beta
        self.w = nn.Parameter(torch.ones((gammas, 1)), requires_grad=False)
        self.b = nn.Parameter(torch.zeros((gammas, 1)), requires_grad=False)
        self._t = nn.Parameter(torch.ones((gammas, 1)), requires_grad=False)
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
