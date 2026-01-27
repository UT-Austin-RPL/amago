"""
Map trajectory data to a sequence of timestep embeddings.
"""

from abc import ABC, abstractmethod
from typing import Optional, Type
import math

import gymnasium as gym
import gin
import torch
from torch import nn

from amago.nets.utils import InputNorm, add_activation_log, symlog, activation_switch
from amago.nets import ff, cnn


###############################
## TstepEncoder Registration ##
###############################

_TSTEP_ENCODER_REGISTRY: dict[str, type] = {}


def register_tstep_encoder(name: str):
    """Decorator to register a TstepEncoder class under a shortcut name.

    Args:
        name: The shortcut name to register the encoder under (e.g., "ff", "cnn").

    Example:
        @gin.configurable
        @register_tstep_encoder("my_encoder")
        class MyCustomTstepEncoder(TstepEncoder):
            ...
    """

    def decorator(cls):
        if name in _TSTEP_ENCODER_REGISTRY:
            raise ValueError(
                f"TstepEncoder '{name}' is already registered to {_TSTEP_ENCODER_REGISTRY[name]}. "
                f"Cannot re-register to {cls}."
            )
        _TSTEP_ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def get_tstep_encoder_cls(name: str) -> type:
    """Look up a registered TstepEncoder class by its shortcut name."""
    if name not in _TSTEP_ENCODER_REGISTRY:
        available = list(_TSTEP_ENCODER_REGISTRY.keys())
        raise KeyError(
            f"TstepEncoder '{name}' is not registered. Available: {available}"
        )
    return _TSTEP_ENCODER_REGISTRY[name]


def list_registered_tstep_encoders() -> list[str]:
    """Return a list of all registered TstepEncoder shortcut names."""
    return list(_TSTEP_ENCODER_REGISTRY.keys())


class TstepEncoder(nn.Module, ABC):
    """Abstract base class for Timestep Encoders.

    Timestep (Tstep) Encoders fuse a dict observation and tensor of extra trajectory
    data (previous actions & rewards) into a single embedding per timestep,
    creating a sequence of [Batch, Length, TstepEncoder.emb_dim] embeddings
    that becomes the input for the main sequence model (TrajEncoder).

    Note:
        Should operate on each timestep of the input sequences independently.
        Sequence modeling should be left to the TrajEncoder. This is not enforced
        during training but would break at inference, as the TstepEncoder currently
        has no hidden state.

    Args:
        obs_space: Environment observation space.
        rl2_space: A gym space declaring the shape of previous action and reward
            features. This is created by the AMAGOEnv wrapper.
        hide_rl2s: Whether to ignore the previous action and reward features (but
            otherwise keep the same parameter count and layer dimensions).
        hide_rewards: Whether to ignore the reward features (but otherwise keep the
            same parameter count and layer dimensions).
    """

    def __init__(
        self,
        obs_space: gym.Space,
        rl2_space: gym.Space,
        hide_rl2s: bool = False,
        hide_rewards: bool = False,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.rl2_space = rl2_space
        self.hide_rl2s = hide_rl2s
        self.hide_rewards = hide_rewards

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        if self.hide_rewards or self.hide_rl2s:
            rl2s = rl2s.clone()
            if self.hide_rewards:
                rl2s[..., 0].zero_()
            elif self.hide_rl2s:
                rl2s.zero_()
        out = self.inner_forward(obs, rl2s, log_dict=log_dict)
        return out

    @abstractmethod
    def inner_forward(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """Override to implement the network forward pass.
        Args:
            obs: dict of {key : torch.Tensor w/ shape (Batch, Length) + self.obs_space[key].shape}
            rl2s: previous actions and rewards features, which might be ignored. Organized here for meta-RL problems.
            log_dict: If provided, we are tracking extra metrics for a logging step, and should add any wandb metrics here.

        Returns:
            torch.Tensor w/ shape (Batch, Length, self.emb_dim)
        """
        pass

    @property
    @abstractmethod
    def emb_dim(self) -> int:
        """The output dimension of the TstepEncoder.

        This is used to determine the input dimension of the TrajEncoder.
        Returns:
            int, the output dimension of the TstepEncoder. If inner_forward outputs shape (Batch, Length, emb_dim), this should return emb_dim.
        """
        pass


@gin.configurable
@register_tstep_encoder("ff")
class FFTstepEncoder(TstepEncoder):
    """A simple MLP-based TstepEncoder.

    Useful when observations are dicts of 1D arrays.

    Args:
        obs_space: Environment observation space.
        rl2_space: A gym space declaring the shape of previous action and reward
            features. This is created by the AMAGOEnv wrapper.

    Keyword Args:
        n_layers: Number of layers in the MLP. Defaults to 2.
        d_hidden: Dimension of the hidden layers. Defaults to 512.
        d_output: Dimension of the output. Defaults to 256.
        norm: Normalization layer to use. See `nets.ff.Normalization` for options.
            Defaults to "layer".
        activation: Activation function to use. See `nets.utils.activation_switch`
            for options. Defaults to "leaky_relu".
        hide_rl2s: Whether to ignore the previous action and reward features (but
            otherwise keep the same parameter count and layer dimensions).
        normalize_inputs: Whether to normalize the input features. See
            `nets.utils.InputNorm`.
        specify_obs_keys: If provided, only use these keys from the observation
            space. If None, every key in the observation is used. Multi-modal
            observations are handled by flattening and concatenating values in a
            consistent order (alphabetical by key). Defaults to None.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        rl2_space: gym.Space,
        n_layers: int = 2,
        d_hidden: int = 512,
        d_output: int = 256,
        norm: str = "layer",
        activation: str = "leaky_relu",
        hide_rl2s: bool = False,
        normalize_inputs: bool = True,
        specify_obs_keys: Optional[list[str]] = None,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space, hide_rl2s=hide_rl2s)
        if specify_obs_keys is None:
            self.obs_keys = sorted(list(obs_space.keys()))
        else:
            self.obs_keys = sorted(specify_obs_keys)
        flat_obs_shape = sum(
            math.prod(self.obs_space[key].shape) for key in self.obs_keys
        )
        in_dim = flat_obs_shape + self.rl2_space.shape[-1]
        self.in_norm = InputNorm(in_dim, skip=not normalize_inputs)
        self.base = ff.MLP(
            d_inp=in_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            d_output=d_output,
            activation=activation,
        )
        self.out_norm = ff.Normalization(norm, d_output)
        self._emb_dim = d_output

    def _cat_flattened_obs(self, obs):
        # B, L, dim_0, ... -> B L D; in fixed order
        arrs = []
        for key in self.obs_keys:
            a = obs[key]
            if a.ndim == 2:
                a = a.unsqueeze(-1)
            arrs.append(a.flatten(start_dim=2))
        return torch.cat(arrs, dim=-1)

    def inner_forward(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        flat_obs = self._cat_flattened_obs(obs)
        flat_obs_rl2 = torch.cat((flat_obs.float(), rl2s), dim=-1)
        if self.training:
            self.in_norm.update_stats(flat_obs_rl2)
        flat_obs_rl2 = self.in_norm(flat_obs_rl2)
        prenorm = self.base(flat_obs_rl2)
        out = self.out_norm(prenorm)
        return out

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
@register_tstep_encoder("cnn")
class CNNTstepEncoder(TstepEncoder):
    """A simple CNN-based TstepEncoder.

    Useful for pixel-based environments. Currently only supports the case where
    observations are a single image without additional state arrays.

    Args:
        obs_space: Environment observation space.
        rl2_space: A gym space declaring the shape of previous action and reward
            features. This is created by the AMAGOEnv wrapper.

    Keyword Args:
        cnn_type: The type of `nets.cnn.CNN` to use. Defaults to
            `nets.cnn.NatureishCNN` (the small DQN CNN).
        channels_first: Whether the image is in channels-first format. Defaults
            to False.
        img_features: Linear map the output of the CNN to this many features.
            Defaults to 256.
        rl2_features: Linear map the previous action and reward to this many
            features. Defaults to 12.
        d_output: The output dimension of a layer that fuses the img_features and
            rl2_features. Defaults to 256.
        out_norm: The normalization layer to use. See `nets.ff.Normalization` for
            options. Defaults to "layer".
        activation: The activation function to use. See
            `nets.utils.activation_switch` for options. Defaults to "leaky_relu".
        hide_rl2s: Whether to ignore the previous action and reward features (but
            otherwise keep the same parameter count and layer dimensions).
        drqv2_aug: Quick-apply the default DrQv2 image augmentation. Applies
            random crops to `aug_pct_of_batch`% of every batch during training.
            Currently requires square images. Defaults to False.
        aug_pct_of_batch: The percentage of every batch to apply DrQv2
            augmentation to, if `drqv2_aug` is True. Defaults to 0.75.
        obs_key: The key in the observation space that contains the image.
            Defaults to "observation", which is the default created by AMAGOEnv
            when the original observation space is not a dict.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        rl2_space: gym.Space,
        cnn_type: Type[cnn.CNN] = cnn.NatureishCNN,
        channels_first: bool = False,
        img_features: int = 256,
        rl2_features: int = 12,
        d_output: int = 256,
        out_norm: str = "layer",
        activation: str = "leaky_relu",
        hide_rl2s: bool = False,
        drqv2_aug: bool = False,
        aug_pct_of_batch: float = 0.75,
        obs_key: str = "observation",
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space, hide_rl2s=hide_rl2s)
        self.data_aug = (
            cnn.DrQv2Aug(4, channels_first=channels_first) if drqv2_aug else lambda x: x
        )
        self.using_aug = drqv2_aug
        obs_shape = self.obs_space[obs_key].shape
        self.cnn = cnn_type(
            img_shape=obs_shape,
            channels_first=channels_first,
            activation=activation,
        )
        img_out_dim = self.cnn(self.cnn.blank_img).shape[-1]
        self.img_features = nn.Linear(img_out_dim, img_features)
        self.rl2_features = nn.Linear(rl2_space.shape[-1], rl2_features)
        mlp_in = img_features + rl2_features
        self.merge = nn.Linear(mlp_in, d_output)
        self.out_norm = ff.Normalization(out_norm, d_output)
        self.obs_key = obs_key
        self._emb_dim = d_output
        assert 0 <= aug_pct_of_batch <= 1, "aug_pct_of_batch must be between 0 and 1"
        self.aug_pct_of_batch = aug_pct_of_batch
        self.activation = activation_switch(activation)

    def inner_forward(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        log_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        img = obs[self.obs_key].float()
        B, L, *_ = img.shape
        if self.using_aug and self.training:
            og_split = max(min(math.ceil(B * (1.0 - self.aug_pct_of_batch)), B - 1), 0)
            aug = self.data_aug(img[og_split:, ...])
            img = torch.cat((img[:og_split, ...], aug), dim=0)
        img = (img / 128.0) - 1.0
        img_rep = self.cnn(img, flatten=True, from_float=True)
        add_activation_log("cnn_out", img_rep, log_dict)
        img_rep = self.img_features(img_rep)
        add_activation_log("img_features", img_rep, log_dict)
        rl2s_rep = self.rl2_features(symlog(rl2s))
        inp = self.activation(torch.cat((img_rep, rl2s_rep), dim=-1))
        merge = self.merge(inp)
        add_activation_log("tstep_encoder_prenorm", merge, log_dict)
        out = self.out_norm(merge)
        return out

    @property
    def emb_dim(self):
        return self._emb_dim
