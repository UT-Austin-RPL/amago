import math
from typing import Optional
from abc import ABC, abstractmethod

import gin
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange


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
        # this property is only used to pick actions from this dist when sampling is disabled
        return torch.tanh(self.base_dist.mean) * self.scale


class _ClippedDistribution(pyd.Distribution):
    """Clips the output of a distribution to be within [-1, 1] when sampling during environment interaction."""

    def __init__(self, base_dist):
        super().__init__(validate_args=False)
        self.base_dist = base_dist

    def log_prob(self, *args, **kwargs):
        # Unsqueeze the last dimension of the base distribution for compatibility with existing code
        return self.base_dist.log_prob(*args, **kwargs).unsqueeze(-1)

    def sample(self, *args, **kwargs):
        return self.base_dist.sample(*args, **kwargs).clamp(-1.0, 1.0)

    def rsample(self, *args, **kwargs):
        # Don't clip the rsample because it would mess up the gradients
        return self.base_dist.rsample(*args, **kwargs)

    @property
    def mean(self):
        return self.base_dist.mean


class _TanhTransform(pyd.transforms.Transform):
    # Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, clip_on_inverse: tuple[float, float], cache_size=1):
        super().__init__(cache_size=cache_size)
        self.clip_inv_low, self.clip_inv_high = clip_on_inverse
        assert -1.0 <= self.clip_inv_low < self.clip_inv_high <= 1.0

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, _TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(self.clip_inv_low, self.clip_inv_high))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class _SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    # Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    def __init__(self, loc, scale, clip_on_tanh_inverse: tuple[float, float]):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [_TanhTransform(clip_on_inverse=clip_on_tanh_inverse)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class _ClippedNormal(pyd.Normal):
    """Clips the output of a normal distribution to be within [-1, 1] when sampling during environment interaction.
    We don't clip the rsample because it would mess up the gradients."""
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).clamp(-1.0, 1.0)


def _TanhGMM(means, stds, logits, tanh_means_only):
    if tanh_means_only:
        means = torch.tanh(means)
    comp = pyd.Independent(pyd.Normal(loc=means, scale=stds), 1)
    mix = pyd.Categorical(logits=logits)
    dist = pyd.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
    if tanh_means_only:
        dist = _ClippedDistribution(dist)
    else:
        dist = _TanhWrappedDistribution(base_dist=dist, scale=1.0)
    return dist


class _Categorical(pyd.Categorical):
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)


class PolicyDistribution(ABC):
    def __init__(self, d_action: int):
        super().__init__()
        self.d_action = d_action

    def __call__(self, vec: torch.Tensor) -> pyd.Distribution:
        # looks like a nn.Module but doesn't break old checkpoints
        return self.forward(vec)

    @property
    @abstractmethod
    def actions_differentiable(self) -> bool:
        # (can you use -Q(s, a ~ pi) as an actor loss?)
        raise NotImplementedError

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        # used to decide how the critic predicts vals
        raise NotImplementedError

    @property
    @abstractmethod
    def input_dimension(self) -> int:
        # used to determine the output of the actor network
        raise NotImplementedError

    @abstractmethod
    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        raise NotImplementedError


@gin.configurable
class Discrete(PolicyDistribution):
    def __init__(
        self,
        d_action: int,
        # clips and renoramlizes actor probs with the intention
        # of stabilizing value computation in the critic and
        # protecting against total entropy collapse. These
        # defaults are now thought to be too conservative.
        clip_prob_low: float = 0.001,
        clip_prob_high: float = 0.99,
    ):
        super().__init__(d_action)
        self.clip_prob_low = clip_prob_low
        self.clip_prob_high = clip_prob_high

    @property
    def actions_differentiable(self):
        return True

    @property
    def is_discrete(self):
        return True

    @property
    def input_dimension(self):
        return self.d_action

    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        dist = _Categorical(logits=vec)
        probs = dist.probs
        clip_probs = probs.clamp(self.clip_prob_low, self.clip_prob_high)
        safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
        safe_dist = _Categorical(probs=safe_probs)
        return safe_dist


class _Continuous(PolicyDistribution):
    def __init__(
        self, d_action: int, std_low: float, std_high: float, std_activation: str
    ):
        super().__init__(d_action)
        assert 0 < std_low < (std_high or float("inf"))
        self.std_low = std_low
        self.std_high = std_high
        self.std_activation = std_activation

    @property
    def is_discrete(self):
        return False

    def std_from_network_output(self, raw_std: torch.Tensor):
        # maps the network's output value to a valid standard deviation
        # for the policy distribution. There are many ways to do this.
        if self.std_activation == "tanh":
            # this version shows up more in off-policy RL codebases
            tanh_scale = torch.tanh(raw_std)
            log_std_low = math.log(self.std_low)
            log_std_high = math.log(self.std_high)
            log_std = log_std_low + 0.5 * (log_std_high - log_std_low) * (
                tanh_scale + 1
            )
            return log_std.exp()
        elif self.std_activation == "softplus":
            # this version is used by robomimic for robot IL
            std = F.softplus(raw_std) + self.std_low
            if self.std_high is not None:
                std = std.clamp(max=self.std_high)
            return std
        else:
            raise ValueError(
                f"Invalid strategy: {self.std_activation}. Must be 'tanh' or 'softplus'."
            )


@gin.configurable
class TanhGaussian(_Continuous):
    def __init__(
        self,
        d_action: int,
        std_low: float = math.exp(-5.0),
        std_high: float = math.exp(2.0),
        std_activation: str = "tanh",  # or "softplus"
        # clips the actions before computing dist.log_prob(action);
        # these values can be quite unstable on actions far from the
        # current policy (offline RL, IL), and at the (-1, 1) border.
        clip_actions_on_log_prob: tuple[float, float] = (-0.99, 0.99),
        tanh_means_only: bool = False,  # only apply tanh to means instead of the whole distribution
    ):
        super().__init__(
            d_action=d_action,
            std_low=std_low,
            std_high=std_high,
            std_activation=std_activation,
        )
        self.clip_actions_on_log_prob = clip_actions_on_log_prob
        self.tanh_means_only = tanh_means_only

    @property
    def actions_differentiable(self):
        return True

    @property
    def input_dimension(self):
        return 2 * self.d_action

    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        mu, raw_std = vec.chunk(2, dim=-1)
        std = self.std_from_network_output(raw_std)
        if self.tanh_means_only:
            mu = torch.tanh(mu)
            dist = _ClippedNormal(mu, std)
        else:
            dist = _SquashedNormal(
                mu, std, clip_on_tanh_inverse=self.clip_actions_on_log_prob
            )
        return dist


@gin.configurable
class GMM(_Continuous):
    def __init__(
        self,
        d_action: int,
        gmm_modes: int = 5,
        std_low: float = 1e-4,
        std_high: Optional[float] = None,
        std_activation: str = "softplus",  # or "tanh"
        tanh_means_only: bool = True,  # only apply tanh to means instead of the whole distribution
    ):
        super().__init__(
            d_action=d_action,
            std_low=std_low,
            std_high=std_high,
            std_activation=std_activation,
        )
        self.gmm_modes = gmm_modes
        self.tanh_means_only = tanh_means_only

    @property
    def actions_differentiable(self):
        return False

    @property
    def input_dimension(self):
        return 2 * self.gmm_modes * self.d_action + self.gmm_modes

    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        idx = self.gmm_modes * self.d_action
        means = rearrange(vec[..., :idx], "... g (m p) -> ... g m p", m=self.gmm_modes)
        raw_std = rearrange(
            vec[..., idx : 2 * idx], "... g (m p) -> ... g m p", m=self.gmm_modes
        )
        stds = self.std_from_network_output(raw_std)
        logits = vec[..., 2 * idx :]
        dist = _TanhGMM(
            means=means, stds=stds, logits=logits, tanh_means_only=self.tanh_means_only
        )
        return dist


@gin.configurable
class Multibinary(PolicyDistribution):
    def __init__(self, d_action: int):
        super().__init__(d_action)

    @property
    def actions_differentiable(self):
        return False

    @property
    def is_discrete(self):
        return False

    @property
    def input_dimension(self):
        return self.d_action

    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        dist = pyd.Bernoulli(logits=vec)
        return dist


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
