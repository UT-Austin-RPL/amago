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


def _TanhGMM(means, stds, logits):
    comp = pyd.Independent(pyd.Normal(loc=means, scale=stds), 1)
    mix = pyd.Categorical(logits=logits)
    dist = pyd.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
    dist = _TanhWrappedDistribution(base_dist=dist, scale=1.0)
    return dist


class _Categorical(pyd.Categorical):
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)


def tanh_bounded_std(log_std: torch.Tensor, lower_limit: float, upper_limit: float):
    log_std = torch.tanh(log_std)
    log_std = lower_limit + 0.5 * (upper_limit - lower_limit) * (log_std + 1)
    return log_std.exp()


class PolicyDistribution(nn.Module, ABC):
    def __init__(self, d_action: int):
        super().__init__()
        self.d_action = d_action

    @property
    @abstractmethod
    def actions_differentiable(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_dimension(self) -> int:
        raise NotImplementedError

    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        raise NotImplementedError


@gin.configurable
class Discrete(PolicyDistribution):
    def __init__(
        self, d_action: int, clip_prob_low: float = 0.001, clip_prob_high: float = 0.99
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
        clip_probs = probs.clamp(min_prob, max_prob)
        safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
        safe_dist = _Categorical(probs=safe_probs)
        return safe_dist


@gin.configurable
class TanhGaussian(PolicyDistribution):
    def __init__(
        self,
        d_action: int,
        log_std_low: float = -5.0,
        log_std_high: float = 2.0,
        clip_actions_on_log_prob: tuple[float, float] = (-0.99, 0.99),
    ):
        super().__init__(d_action)
        assert log_std_low < log_std_high
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.clip_actions_on_log_prob = clip_actions_on_log_prob

    @property
    def actions_differentiable(self):
        return True

    @property
    def is_discrete(self):
        return False

    @property
    def input_dimension(self):
        return 2 * self.d_action

    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        mu, log_std = vec.chunk(2, dim=-1)
        std = tanh_bounded_std(log_std, self.log_std_low, self.log_std_high)
        dist = _SquashedNormal(
            mu, std, clip_on_tanh_inverse=self.clip_actions_on_log_prob
        )
        return dist


@gin.configurable
class GMM(PolicyDistribution):
    def __init__(
        self,
        d_action: int,
        gmm_modes: int = 5,
        log_std_low: float = -5.0,
        log_std_high: float = 2.0,
    ):
        super().__init__(d_action)
        self.gmm_modes = gmm_modes
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high

    @property
    def actions_differentiable(self):
        return False

    @property
    def is_discrete(self):
        return False

    @property
    def input_dimension(self):
        return 2 * self.gmm_modes * self.d_action + self.gmm_modes

    def forward(self, vec: torch.Tensor) -> pyd.Distribution:
        idx = self.gmm_modes * self.d_action
        means = rearrange(vec[..., :idx], "... g (m p) -> ... g m p", m=self.gmm_modes)
        log_std = rearrange(
            vec[..., idx : 2 * idx], "... g (m p) -> ... g m p", m=self.gmm_modes
        )
        stds = tanh_bounded_std(log_std, self.log_std_low, self.log_std_high)
        logits = vec[..., 2 * idx :]
        dist = _TanhGMM(means=means, stds=stds, logits=logits)
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
