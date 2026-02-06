"""
Actor and critic output modules.
"""

import contextlib
from typing import Optional, Type
import math
import random
from functools import lru_cache
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange
from einops.layers.torch import EinMix as Mix
import gin

from amago.nets.ff import MLP, Normalization
from amago.nets.utils import activation_switch, symlog, symexp
from amago.nets.policy_dists import (
    Discrete,
    PolicyOutput,
    TanhGaussian,
)
from amago.utils import amago_warning


class BaseActorHead(nn.Module, ABC):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        continuous_dist_type: Type[PolicyOutput],
        discrete_dist_type: Type[PolicyOutput] = Discrete,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.gammas = gammas
        self.num_gammas = len(self.gammas)
        # determine policy output
        dist_type = discrete_dist_type if self.discrete else continuous_dist_type
        self.policy_dist = dist_type(d_action=self.action_dim)
        assert isinstance(self.policy_dist, PolicyOutput)
        assert self.policy_dist.is_discrete == self.discrete
        self.actions_differentiable = self.policy_dist.actions_differentiable

    def forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ) -> pyd.Distribution:
        """Compute an action distribution from a state representation.

        Args:
            state: The "state" sequence (the output of the TrajEncoder) (Batch, Length, state_dim)

        Returns:
            The action distribution. Type varies according to the output of `PolicyOutput`
            (e.g. `Discrete` or `TanhGaussian`). Always a pytorch distribution (e.g., `Categorical`)
            where sampled actions would have shape (Batch, Length, Gammas, action_dim).
        """
        dist_params = self.actor_network_forward(
            state=state, log_dict=log_dict, straight_from_obs=straight_from_obs
        )
        assert dist_params.ndim == 4
        assert dist_params.shape[-2:] == (
            self.num_gammas,
            self.policy_dist.input_dimension,
        )
        return self.policy_dist(dist_params, log_dict=log_dict)

    @abstractmethod
    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


@gin.configurable
class Actor(BaseActorHead):
    """Actor output head architecture.

    A (small) MLP that maps the output of the TrajEncoder to a distribution over actions.

    Args:
        state_dim: Dimension of the "state" space (which is the output of the TrajEncoder)
        action_dim: Dimension of the action space
        discrete: Whether the action space is discrete
        gammas: List of gamma values to use for the multi-gamma actor

    Keyword Args:
        n_layers: Number of layers in the MLP. Defaults to 2.
        d_hidden: Dimension of hidden layers in the MLP. Defaults to 256.
        activation: Activation function to use in the MLP. Defaults to "leaky_relu".
        dropout_p: Dropout rate to use in the MLP. Defaults to 0.0.
        continuous_dist_type: Type of continuous distribution to use if applicable. Must be a
            :py:class:`~amago.nets.policy_dists.PolicyOutput`. Defaults to :py:class:`~amago.nets.policy_dists.TanhGaussian`.
    """

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
        continuous_dist_type: Type[PolicyOutput] = TanhGaussian,
        discrete_dist_type: Type[PolicyOutput] = Discrete,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            continuous_dist_type=continuous_dist_type,
            discrete_dist_type=discrete_dist_type,
        )
        # build base network
        self.base = MLP(
            d_inp=state_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            d_output=self.policy_dist.input_dimension * self.num_gammas,
            dropout_p=dropout_p,
            activation=activation,
        )

    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        dist_params = self.base(state)
        dist_params = rearrange(
            dist_params, "b ... (g f) -> b ... g f", g=self.num_gammas
        )
        return dist_params


@gin.configurable
class ResidualActor(BaseActorHead):
    """Actor output head with residual blocks.

    Based on BRO https://arxiv.org/pdf/2405.16158v1,
    which recommends similar hparams to our exsiting defaults.

    Args:
        state_dim: Dimension of the "state" space (which is the output of the TrajEncoder)
        action_dim: Dimension of the action space
        discrete: Whether the action space is discrete
        gammas: List of gamma values to use for the multi-gamma actor

    Keyword Args:
        feature_dim: Dimension of the embedding between residual blocks (analogous to d_model in a Transformer). Defaults to 256.
        residual_ff_dim: Hidden dimension of residual blocks (analogous to d_ff in a Transformer). Defaults to 512.
        residual_blocks: Number of residual blocks. Defaults to 2.
        activation: Activation function to use in the MLPs. Defaults to "leaky_relu".
        normalization: Normalization to use in the residual blocks. Defaults to "layer" (LayerNorm).
        dropout_p: Dropout rate to use in the initial linear layers. Defaults to 0.0.
        continuous_dist_type: Type of continuous distribution to use if applicable. Must be a
            :py:class:`~amago.nets.policy_dists.PolicyOutput`. Defaults to :py:class:`~amago.nets.policy_dists.TanhGaussian`.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        feature_dim: int = 256,
        residual_ff_dim: int = 512,
        residual_blocks: int = 2,
        activation: str = "leaky_relu",
        normalization: str = "layer",
        dropout_p: float = 0.0,
        continuous_dist_type: Type[PolicyOutput] = TanhGaussian,
        discrete_dist_type: Type[PolicyOutput] = Discrete,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            continuous_dist_type=continuous_dist_type,
            discrete_dist_type=discrete_dist_type,
        )
        self.inp = MLP(
            d_inp=state_dim,
            d_hidden=feature_dim,
            n_layers=1,
            d_output=feature_dim,
            dropout_p=dropout_p,
            activation=activation,
        )

        class _Lambda(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x):
                return self.func(x)

        residual_block = lambda: nn.Sequential(
            nn.Linear(feature_dim, residual_ff_dim),
            Normalization(normalization, residual_ff_dim),
            _Lambda(activation_switch(activation)),
            nn.Linear(residual_ff_dim, feature_dim),
            Normalization(normalization, feature_dim),
        )

        self.residual_blocks = nn.ModuleList(
            [residual_block() for _ in range(residual_blocks)]
        )

        self.out = nn.Linear(
            feature_dim, self.policy_dist.input_dimension * self.num_gammas
        )

    @torch.compile
    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, D = state.shape
        x = self.inp(state)
        for block in self.residual_blocks:
            x = x + block(x)
        out = self.out(x)
        params = rearrange(out, "b l (g f) -> b l g f", g=self.num_gammas)
        return params


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


class BaseCriticHead(nn.Module, ABC):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        num_critics: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.gammas = gammas
        self.num_gammas = len(self.gammas)
        self.num_critics = num_critics

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, log_dict: Optional[dict] = None
    ) -> torch.Tensor:
        assert state.ndim == 3
        assert action.ndim == 5
        out = self.critic_network_forward(state=state, action=action, log_dict=log_dict)
        return out

    @abstractmethod
    def critic_network_forward(
        self, state: torch.Tensor, action: torch.Tensor, log_dict: Optional[dict] = None
    ) -> torch.Tensor:
        raise NotImplementedError


@gin.configurable
class NCritics(BaseCriticHead):
    """Critic output head architecture.

    A (small) ensemble of MLPs that maps the state and action to a value estimate.

    Args:
        state_dim: Dimension of the "state" space (which is the output of the TrajEncoder)
        action_dim: Dimension of the action space
        discrete: Whether the action space is discrete
        gammas: List of gamma values to use for the multi-gamma critic

    Keyword Args:
        num_critics: Number of critics in the ensemble. Defaults to 4.
        d_hidden: Dimension of hidden layers in the MLP. Defaults to 256.
        n_layers: Number of layers in the MLP. Defaults to 2.
        dropout_p: Dropout rate to use in the MLP. Defaults to 0.0.
        activation: Activation function to use in the MLP. Defaults to "leaky_relu".
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        num_critics: int,
        d_hidden: int = 256,
        n_layers: int = 2,
        dropout_p: float = 0.0,
        activation: str = "leaky_relu",
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            num_critics=num_critics,
        )
        inp_dim = self.state_dim
        if not self.discrete:
            inp_dim += self.action_dim + 1
            out_dim = 1
        else:
            out_dim = self.num_gammas * self.action_dim
        self.net = _EinMixEnsemble(
            ensemble_size=self.num_critics,
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
    def critic_network_forward(
        self, state: torch.Tensor, action: torch.Tensor, log_dict: Optional[dict] = None
    ) -> torch.Tensor:
        """Compute a value estimate from a state and action.

        Args:
            state: The "state" sequence (the output of the TrajEncoder). Has shape
                (Batch, Length, state_dim).
            action: The action sequence. Has shape (K, Batch, Length, Gammas, action_dim),
                where K is a dimension denoting multiple action samples from the same state
                (can be 1, but must exist). Discrete actions are expected to be one-hot vectors.

        Returns:
            The value estimate with shape (K, Batch, Length, num_critics, Gammas, 1).
        """
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
class NCriticsTwoHot(BaseCriticHead):
    """Critic output head architecture.

    A (small) ensemble of MLPs that maps the state and action to a value estimate in the form
    of a categorical distribution over bins.

    Args:
        state_dim: Dimension of the "state" space (which is the output of the TrajEncoder)
        action_dim: Dimension of the action space
        gammas: List of gamma values to use for the multi-gamma critic

    Keyword Args:
        num_critics: Number of critics in the ensemble. Defaults to 4.
        d_hidden: Dimension of hidden layers in the MLP. Defaults to 256.
        n_layers: Number of layers in the MLP. Defaults to 2.
        dropout_p: Dropout rate to use in the MLP. Defaults to 0.0.
        activation: Activation function to use in the MLP. Defaults to "leaky_relu".
        min_return: Minimum return value. If not set, defaults to a very negative value (-100_000).
        max_return: Maximum return value. If not set, defaults to a very positive value (100_000).
        output_bins: Number of bins in the categorical distribution. Defaults to 128.
        use_symlog: Whether to use a symlog transformation on the value estimates. Defaults to True.
        label_type: How to construct target distributions from scalar values. Options:
            - "twohot": Two-hot encoding (Dreamer-V3). Puts mass on the two bins flanking the
              target value, weighted by proximity. (Default)
            - "hlgauss": Histogram Loss with Gaussian targets (Imani & White, 2018;
              Farebrother et al., 2024 "Stop Regressing"). Spreads mass across multiple bins
              using a Gaussian kernel centered at the target.
        sigma: Ratio of Gaussian standard deviation to bin width  for HL-Gauss targets.
        init_value: If set, initializes the network to predict this value at the start of training.

    Note:
        The default bin settings (wide range, lots of bins, symlog transformation) follow
        Dreamer-V3 in picking a range that does not demand domain-specific tuning. It may be
        more sample efficient to use tighter bounds, in which case the unintuitive spacing
        created by use_symlog may be turned off. However, note that the min_return and
        max_return do not compensate for Agent.reward_multiplier. For example, if the highest
        possible return in an env is 1, and reward multiplier is 10, then a tuned max_return might be
        10 but should never be 1. More discussion on bin settings in AMAGO-2 Appendix A.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gammas: torch.Tensor,
        discrete: bool,
        num_critics: int,
        d_hidden: int = 256,
        n_layers: int = 2,
        dropout_p: float = 0.0,
        activation: str = "leaky_relu",
        # see "Breaking Multi-Task Barrier.." Appendix A
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        output_bins: int = 128,
        use_symlog: bool = True,
        label_type: str = "twohot",
        sigma: float = 0.75,
        init_value: Optional[float] = None,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            num_critics=num_critics,
        )
        inp_dim = self.state_dim + self.action_dim + 1
        out_dim = output_bins
        self.num_bins = output_bins
        if min_return is None or max_return is None:
            amago_warning(
                "amago.nets.actor_critic.NCriticsTwoHot.min_return/max_return have not been set manually, and default to extreme values."
            )
        min_return = min_return if min_return is not None else -100_000
        max_return = max_return if max_return is not None else 100_000
        self.transform_values = symlog if use_symlog else lambda x: x
        self.invert_bins = symexp if use_symlog else lambda x: x
        assert min_return < max_return
        self.bin_vals = torch.linspace(
            self.transform_values(min_return),
            self.transform_values(max_return),
            output_bins,
        )
        assert label_type in ("twohot", "hlgauss"), f"Unknown label_type: {label_type}"
        self.label_type = label_type
        self.sigma = sigma
        self.net = _EinMixEnsemble(
            ensemble_size=self.num_critics,
            inp_dim=inp_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            out_dim=out_dim,
            activation=activation,
            dropout_p=dropout_p,
        )

        if init_value is not None:
            if not (min_return <= init_value <= max_return):
                raise ValueError(
                    f"init_value={init_value} is outside the bin support "
                    f"[{min_return}, {max_return}]"
                )
            init_bin_idx = (
                (self.bin_vals - self.transform_values(init_value))
                .abs()
                .argmin()
                .item()
            )
            edge_margin = min(init_bin_idx, output_bins - 1 - init_bin_idx)
            if edge_margin < 12:
                amago_warning(
                    f"init_value={init_value} is only {edge_margin} bins from the edge "
                    f"of the support. The initialization Gaussian may be truncated, "
                    f"slightly biasing the initial prediction inward. Consider widening "
                    f"[min_return, max_return] or adjusting init_value."
                )
            self._initialize_to_value(init_value)

    def __len__(self):
        return self.num_critics

    def _initialize_to_value(self, init_value: float):
        """Initialize the output layer bias so the network's initial prediction
        is centered near init_value.
        """
        init_transformed = self.transform_values(init_value)
        bin_width = self.bin_vals[1] - self.bin_vals[0]
        sigma_init = 4.0 * bin_width
        logits = -((self.bin_vals - init_transformed) ** 2) / (2.0 * sigma_init**2)
        output_layer = self.net.output_layer
        with torch.no_grad():
            output_layer.bias.data[..., :] = logits

    @torch.compile
    def critic_network_forward(
        self, state: torch.Tensor, action: torch.Tensor, log_dict: Optional[dict] = None
    ) -> pyd.Categorical:
        """Compute a categorical distribution over bins from a state and action.

        Args:
            state: The "state" sequence (the output of the TrajEncoder). Has shape
                (Batch, Length, state_dim).
            action: The action sequence. Has shape (K, Batch, Length, Gammas, action_dim),
                where K is a dimension denoting multiple action samples from the same state
                (can be 1, but must exist). Discrete actions are expected to be one-hot vectors.

        Returns:
            The categorical distribution over bins with shape (K, Batch, Length, num_critics, output_bins).
        """
        K, B, L, G, D = action.shape
        assert G == self.num_gammas
        state = repeat(state, "b l d -> (k b g) l d", k=K, g=self.num_gammas)
        action = rearrange(action.clamp(-0.999, 0.999), "k b l g d -> (k b g) l d")
        gammas_rep = gammas_as_input_seq(self.gammas, K * B, L).to(action.device)
        inp = torch.cat((state, gammas_rep, action), dim=-1)
        outputs, phis = self.net(inp)
        outputs = rearrange(
            outputs, "(k b g) l c o -> k b l c g o", k=K, g=self.num_gammas
        )
        return pyd.Categorical(logits=outputs)

    def bin_dist_to_raw_vals(
        self, bin_dist: pyd.Categorical, temperature: float = 1.0
    ) -> torch.Tensor:
        """Convert a categorical distribution over bins to a scalar.

        Args:
            bin_dist: The categorical distribution over bins (output of `forward`).
            temperature: Temperature for softening the distribution.

        Returns:
            The scalar value.
        """
        assert isinstance(bin_dist, pyd.Categorical)
        if temperature == 1.0:
            probs = bin_dist.probs
        else:
            probs = torch.softmax(bin_dist.logits / temperature, dim=-1)
        bin_vals = self.bin_vals.to(probs.device, dtype=probs.dtype)
        exp_val = (probs * bin_vals).sum(-1, keepdims=True)
        return self.invert_bins(exp_val)

    def raw_vals_to_labels(self, raw_td_target: torch.Tensor) -> torch.Tensor:
        """Convert a scalar to a categorical target distribution over bins.

        Dispatches to two-hot or HL-Gauss encoding based on ``self.label_type``.

        Args:
            raw_td_target: The scalar value.

        Returns:
            A categorical distribution over bins (sums to 1 along the last dim).
        """
        if self.label_type == "hlgauss":
            return self._hlgauss_labels(raw_td_target)
        return self._twohot_labels(raw_td_target)

    def _twohot_labels(self, raw_td_target: torch.Tensor) -> torch.Tensor:
        """Two-hot encoding (Dreamer-V3).

        Puts probability mass on exactly two bins flanking the target value,
        weighted by proximity. Torch port of `dreamerv3/jaxutils.py`.
        """
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

    def _hlgauss_labels(self, raw_td_target: torch.Tensor) -> torch.Tensor:
        """HL-Gauss: Histogram Loss with Gaussian targets.

        Reference: Imani & White (2018); Farebrother et al. (2024)
        "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
        (https://arxiv.org/abs/2403.03950)
        """
        symlog_target = self.transform_values(raw_td_target)
        bin_vals = self.bin_vals.to(symlog_target.device)
        bin_width = bin_vals[1] - bin_vals[0]
        sigma_abs = self.sigma * bin_width
        edges = torch.cat([bin_vals[:1] - bin_width / 2, bin_vals + bin_width / 2])
        z = (edges - symlog_target) / sigma_abs
        cdf_vals = 0.5 * (1.0 + torch.erf(z * (1.0 / math.sqrt(2.0))))
        bin_probs = cdf_vals[..., 1:] - cdf_vals[..., :-1]
        bin_probs = bin_probs / bin_probs.sum(-1, keepdim=True).clamp(min=1e-8)
        return bin_probs


@gin.configurable(denylist=["enabled"])
class PopArtLayer(nn.Module):
    """PopArt value normalization.

    Shifts value estimates according to a moving average and helps the outputs of the
    critic to compensate for the distribution shift.
    (https://arxiv.org/abs/1809.04474)

    Args:
        gammas: Number of gamma values in the critic.

    Keyword Args:
        beta: The beta parameter for the moving average. Defaults to 5e-4.
        init_nu: The initial nu parameter. Defaults to 100.0 following a
            recommendation in the PopArt paper.
        enabled (no gin): If False, this layer is a no-op. Defaults to True. Cannot be
            configured by gin. Instead, use `Agent.use_popart`.
    """

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
    def sigma(self) -> torch.Tensor:
        inner = (self.nu - self.mu**2).clamp(1e-4, 1e8)
        return torch.sqrt(inner).clamp(1e-4, 1e6)

    def normalize_values(self, val: torch.Tensor) -> torch.Tensor:
        """Get normalized (Q) values"""
        if not self.enabled:
            return val
        return ((val - self.mu) / self.sigma).to(val.dtype)

    def to(self, device):
        """Move to another torch device."""
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        self.mu = self.mu.to(device)
        self.nu = self.nu.to(device)
        return self

    def update_stats(self, val: torch.Tensor, mask: torch.Tensor) -> None:
        """Update the moving average statistics.

        Args:
            val: The value estimate.
            mask: A mask that is 0 where value estimates should be ignored (e.g., from padded timesteps).
        """
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

    def forward(self, x: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """Modify the value estimate according to the PopArt layer.

        Applies normalization or denormalization to value estimates using PopArt's moving average statistics.
        When normalized=True, scales and shifts values using the current statistics to normalize them.
        When normalized=False, maps normalized values back to the original scale of the environment.

        Args:
            x: Value estimate to modify
            normalized: Whether to normalize (True) or denormalize (False) the values

        Returns:
            Modified value estimate in either normalized or denormalized form
        """
        if not self.enabled:
            return x
        normalized_out = (self.w * x) + self.b
        if normalized:
            return normalized_out.to(x.dtype)
        else:
            return ((self.sigma * normalized_out) + self.mu).to(x.dtype)
