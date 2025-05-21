import math
from typing import Optional, Callable
from abc import ABC, abstractmethod

import gin
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from einops import rearrange

from amago.nets.utils import add_activation_log


class _TanhWrappedDistribution(pyd.Distribution):
    """This is copied directly from Robomimic.

    https://github.com/ARISE-Initiative/robomimic/blob/b5d2aa9902825c6c652e3b08b19446d199b49590/robomimic/models/distributions.py

    Originally based on rlkit and CQL:
    https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/distributions.py#L6
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
    """Utility for mapping torch distributions to [-1, 1].

    Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    """

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
    """A multivariate normal distribution with a tanh transform.

    Sampled actions lie in the standard continuous action space of [-1, 1].

    Credit: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    """

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
    """Gaussian Mixture Model with a tanh transform."""
    comp = pyd.Independent(pyd.Normal(loc=means, scale=stds), 1)
    mix = pyd.Categorical(logits=logits)
    dist = pyd.MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
    dist = _TanhWrappedDistribution(base_dist=dist, scale=1.0)
    return dist


class _Categorical(pyd.Categorical):
    """Categorical distribution that samples discrete actions with an action dim (shape[-1]) of 1."""

    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)


class _ShiftedBeta(pyd.TransformedDistribution):
    """Create a Beta distribution on [0,1], then y=2*x - 1 to shift to [-1, 1]"""

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        clip_actions_on_log_prob: tuple[float, float],
    ):
        base = pyd.Beta(concentration1=alpha, concentration0=beta)
        self.clip_actions_on_log_prob = clip_actions_on_log_prob
        super().__init__(base, [pyd.AffineTransform(loc=-1.0, scale=2.0)])

    def log_prob(self, value):
        clipped_value = value.clamp(*self.clip_actions_on_log_prob)
        return super().log_prob(clipped_value)

    @property
    def mean(self):
        return 2.0 * self.base_dist.mean - 1.0


class PolicyOutput(ABC):
    """Abstract base class for mapping network outputs to a distribution over actions.

    Actor networks use a PolicyOutput to produce a distribution over actions
    that is compatible with the Agent's loss function.

    Pretends to be a torch.nn.Module (`forward` == `__call__`) but is not. Has no
    parameters and can be swapped without breaking checkpoints.

    Args:
        d_action: Dimension of the action space.
    """

    def __init__(self, d_action: int):
        super().__init__()
        self.d_action = d_action

    def __call__(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> pyd.Distribution:
        # looks like a nn.Module but doesn't break old checkpoints
        return self.forward(vec, log_dict=log_dict)

    @property
    @abstractmethod
    def actions_differentiable(self) -> bool:
        """Does the output distribution have `rsample`?

        Used to answer the question: "can we optimize -Q(s, a ~ pi) as an actor
        loss?"
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """Whether the action space is discrete."""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_dimension(self) -> int:
        """Required input dimension for this policy distribution.

        This is used to determine the output of the actor network. How many values
        does the actor network need to produce to parameterize this policy
        distribution?
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> pyd.Distribution:
        """Maps the output of the actor network to a distribution over actions.

        Args:
            vec: Output of the actor network
            log_dict: If None, this is not a log step and any log value computation can be skipped. If provided, any data added added to the dict will be automatically logged. Defaults to None.

        Returns:
            A torch.distributions.Distribution that at least has a `log_prob()` and
            `sample()`, and would be expected to have `rsample()` if
            `self.actions_differentiable` is True.
        """
        raise NotImplementedError


@gin.configurable
class Discrete(PolicyOutput):
    """Generates a categorical distribution over actions.

    Returns a thin wrapper around torch `Categorical` that unsqueezes the last dimension of `sample()` actions to be 1.

    Args:
        d_action: Dimension of the action space.

    Keyword Args:
        clip_prob_low: Clips action probabilities to this value before
            renormalizing. Default is 0.001.
        clip_prob_high: Clips action probabilities to this value before
            renormalizing. Default is 0.99, which is left for backwards
            compatibility but is now thought to be too conservative. .999 or 1.0
            is fine.
    """

    def __init__(
        self,
        d_action: int,
        clip_prob_low: float = 0.001,
        clip_prob_high: float = 0.99,
    ):
        super().__init__(d_action)
        self.clip_prob_low = clip_prob_low
        self.clip_prob_high = clip_prob_high

    @property
    def actions_differentiable(self) -> bool:
        return True

    @property
    def is_discrete(self) -> bool:
        return True

    @property
    def input_dimension(self) -> int:
        return self.d_action

    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> _Categorical:
        dist = _Categorical(logits=vec)
        probs = dist.probs
        clip_probs = probs.clamp(self.clip_prob_low, self.clip_prob_high)
        safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
        safe_dist = _Categorical(probs=safe_probs)
        if log_dict is not None:
            add_activation_log("Discrete-probs", probs, log_dict)
        return safe_dist


@gin.configurable
class DiscreteLikeContinuous:
    """Wrapper around `Categorical` used by `MultiTaskAgent`.

    Lets us use discrete actions in a continuous actor-critic setup where the
    critic takes action vectors as input and outputs a scalar value.

    Categorial --> OneHotCategorical + `rsample()` as F.gumbel_softmax with straight-through `hard` sampling.

    Args:
        categorical: A `Categorical` distribution.

    Keyword Args:
        gumbel_softmax_temperature: Temperature for the Gumbel-Softmax trick that enables `rsample()`. Default is 0.5.
    """

    def __init__(
        self,
        categorical: pyd.Categorical | _Categorical,
        gumbel_softmax_temperature: float = 0.5,
    ):
        self.dist = pyd.OneHotCategorical(logits=categorical.logits)
        self.gumbel_softmax_temperature = gumbel_softmax_temperature

    @property
    def probs(self) -> torch.Tensor:
        return self.dist.probs

    @property
    def logits(self) -> torch.Tensor:
        return self.dist.logits

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def sample(self, *args, **kwargs) -> torch.Tensor:
        return self.dist.sample(*args, **kwargs)

    def rsample(self) -> torch.Tensor:
        return F.gumbel_softmax(
            self.logits, tau=self.gumbel_softmax_temperature, hard=True, dim=-1
        )


def tanh_bounded_positive(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    tanh_scale = torch.tanh(x)
    log_low = math.log(low)
    log_high = math.log(high)
    log_std = log_low + 0.5 * (log_high - log_low) * (tanh_scale + 1)
    return log_std.exp()


def softplus_bounded_positive(
    x: torch.Tensor, low: float, high: Optional[float] = None
) -> torch.Tensor:
    return (F.softplus(x) + low).clamp(max=high or float("inf"))


StdActivation = Callable[[torch.Tensor, float, float], torch.Tensor]


@gin.configurable
class TanhGaussian(PolicyOutput):
    """Generates a multivariate normal with a tanh transform to sample in [-1, 1].

    Args:
        d_action: Dimension of the action space.

    Keyword Args:
        std_low: Minimum standard deviation. Default is exp(-5.0).
        std_high: Maximum standard deviation. Default is exp(2.0).
        std_activation: Activation function to produce a valid standard deviation from the raw network
            output.
        clip_actions_on_log_prob: Tuple of floats that clips the actions before
            computing dist.log_prob(action). Adresses numerical stability issues
            when computing log_probs at or near saturation points of Tanh
            transforms. Default is (-0.99, 0.99).
    """

    def __init__(
        self,
        d_action: int,
        std_low: float = math.exp(-5.0),
        std_high: float = math.exp(2.0),
        std_activation: StdActivation = tanh_bounded_positive,
        clip_actions_on_log_prob: tuple[float, float] = (-0.99, 0.99),
    ):
        super().__init__(d_action=d_action)
        self.std_activation = std_activation
        self.std_low = std_low
        self.std_high = std_high
        self.clip_actions_on_log_prob = clip_actions_on_log_prob

    @property
    def actions_differentiable(self) -> bool:
        return True

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def input_dimension(self) -> int:
        return 2 * self.d_action

    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> _SquashedNormal:
        mu, raw_std = vec.chunk(2, dim=-1)
        std = self.std_activation(raw_std, self.std_low, self.std_high)
        dist = _SquashedNormal(
            mu, std, clip_on_tanh_inverse=self.clip_actions_on_log_prob
        )
        if log_dict is not None:
            add_activation_log("TanhGaussian-mu", mu, log_dict)
            add_activation_log("TanhGaussian-std", std, log_dict)
            add_activation_log("TanhGaussian-raw_std", raw_std, log_dict)
        return dist


@gin.configurable
class GMM(PolicyOutput):
    """Generates a Gaussian Mixture Model with a tanh transform.

    A more expressive policy than TanhGaussian, but output does not support `rsample()`
    or the DPG -Q(s, a ~ pi) loss. Often used in offline or imitation learning
    (IL) settings. Heavily based on robomimic's robot IL setup.

    Args:
        d_action: Dimension of the action space.

    Keyword Args:
        gmm_modes: Number of modes in the GMM. Default is 5.
        std_low: Minimum standard deviation. Default is 1e-4.
        std_high: Maximum standard deviation. Default is None.
        std_activation: Activation function to produce a std from the raw network
            output.
    """

    def __init__(
        self,
        d_action: int,
        gmm_modes: int = 5,
        std_low: float = 1e-4,
        std_high: Optional[float] = None,
        std_activation: StdActivation = softplus_bounded_positive,
    ):
        super().__init__(d_action)
        self.gmm_modes = gmm_modes
        self.std_activation = std_activation
        self.std_low = std_low
        self.std_high = std_high

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def actions_differentiable(self) -> bool:
        return False

    @property
    def input_dimension(self) -> int:
        return 2 * self.gmm_modes * self.d_action + self.gmm_modes

    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> _TanhWrappedDistribution:
        idx = self.gmm_modes * self.d_action
        means = rearrange(vec[..., :idx], "... g (m p) -> ... g m p", m=self.gmm_modes)
        raw_std = rearrange(
            vec[..., idx : 2 * idx], "... g (m p) -> ... g m p", m=self.gmm_modes
        )
        stds = self.std_activation(raw_std, self.std_low, self.std_high)
        logits = vec[..., 2 * idx :]
        dist = _TanhGMM(means=means, stds=stds, logits=logits)
        if log_dict is not None:
            add_activation_log("GMM-means", means, log_dict)
            add_activation_log("GMM-raw_std", raw_std, log_dict)
            add_activation_log("GMM-stds", stds, log_dict)
            add_activation_log("GMM-logits", logits, log_dict)
        return dist


@gin.configurable
class Beta(PolicyOutput):
    """Generates a Beta distribution rescaled to [-1, 1].

    Args:
        d_action: Dimension of the action space.

    Keyword Args:
        alpha_low: Minimum value of alpha. Default is 1e-4.
        alpha_high: Maximum value of alpha. Default is None.
        beta_low: Minimum value of beta. Default is 1e-4.
        beta_high: Maximum value of beta. Default is None.
        std_activation: Activation function to produce a valid standard deviation from the raw network
            output.
        clip_actions_on_log_prob: Tuple of floats that clips the actions before
            computing dist.log_prob(action). Adresses numerical stability issues
            when computing log_probs at the boundary of the action space.
    """

    def __init__(
        self,
        d_action: int,
        alpha_low: float = 1e-4,
        alpha_high: Optional[float] = None,
        beta_low: float = 1e-4,
        beta_high: Optional[float] = None,
        std_activation: StdActivation = softplus_bounded_positive,
        clip_actions_on_log_prob: tuple[float, float] = (-0.99, 0.99),
    ):
        super().__init__(d_action)
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.beta_low = beta_low
        self.beta_high = beta_high
        self.std_activation = std_activation
        self.clip_actions_on_log_prob = clip_actions_on_log_prob

    @property
    def actions_differentiable(self) -> bool:
        return True

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def input_dimension(self) -> int:
        return 2 * self.d_action

    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> _ShiftedBeta:
        alpha_input, beta_input = vec.chunk(2, dim=-1)
        alpha = self.std_activation(alpha_input, self.alpha_low, self.alpha_high)
        beta = self.std_activation(beta_input, self.beta_low, self.beta_high)
        dist = _ShiftedBeta(
            alpha, beta, clip_actions_on_log_prob=self.clip_actions_on_log_prob
        )
        if log_dict is not None:
            add_activation_log("Beta-alpha", alpha, log_dict)
            add_activation_log("Beta-beta", beta, log_dict)
        return dist


@gin.configurable
class Multibinary(PolicyOutput):
    """Multi-binary action space support."""

    def __init__(self, d_action: int):
        super().__init__(d_action)

    @property
    def actions_differentiable(self) -> bool:
        return False

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def input_dimension(self) -> int:
        return self.d_action

    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> pyd.Bernoulli:
        dist = pyd.Bernoulli(logits=vec)
        return dist
