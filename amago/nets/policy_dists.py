import math
from typing import Optional

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


class _Categorical(pyd.Categorical):
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)


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


@gin.configurable
def DiscreteActionDist(vec, min_prob: float = 0.001, max_prob: float = 0.99):
    dist = _Categorical(logits=vec)
    probs = dist.probs
    # note: this clip helps stability but is something to keep in mind,
    # especially on some toy envs where the optimal policy is very
    # deterministic. action sampling can be turned off in the `learning.Experiment`.
    clip_probs = probs.clamp(min_prob, max_prob)
    safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
    safe_dist = _Categorical(probs=safe_probs)
    return safe_dist
