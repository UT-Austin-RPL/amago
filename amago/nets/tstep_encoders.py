from abc import ABC, abstractmethod
from typing import Optional
import math

import torch
from torch import nn
from einops import rearrange
import gin

from amago.nets.goal_embedders import FFGoalEmb, TokenGoalEmb
from amago.nets.utils import InputNorm, add_activation_log, symlog
from amago.nets import ff, cnn


class TstepEncoder(nn.Module, ABC):
    def __init__(self, obs_space, rl2_space):
        super().__init__()
        self.obs_space = obs_space
        self.rl2_space = rl2_space

    def forward(self, obs, rl2s, log_dict: Optional[dict] = None):
        out = self.inner_forward(obs, rl2s, log_dict=log_dict)
        return out

    @abstractmethod
    def inner_forward(self, obs, rl2s, log_dict: Optional[dict] = None):
        pass

    @property
    @abstractmethod
    def emb_dim(self):
        pass


@gin.configurable
class FFTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        rl2_space,
        n_layers: int = 2,
        d_hidden: int = 512,
        d_output: int = 256,
        norm: str = "layer",
        activation: str = "leaky_relu",
        hide_rl2s: bool = False,
        normalize_inputs: bool = True,
        obs_key: str = "observation",
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        flat_obs_shape = math.prod(self.obs_space[obs_key].shape)
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
        self.hide_rl2s = hide_rl2s
        self.obs_key = obs_key

    def inner_forward(self, obs, rl2s, log_dict: Optional[dict] = None):
        # multi-modal envs that do not use the default `observation` key need their own custom encoders.
        obs = obs[self.obs_key]
        B, L, *_ = obs.shape
        if self.hide_rl2s:
            rl2s = rl2s * 0
        flat_obs_rl2 = torch.cat((obs.view(B, L, -1).float(), rl2s), dim=-1)
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
class CNNTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        rl2_space,
        cnn_type=cnn.NatureishCNN,
        channels_first: bool = False,
        img_features: int = 384,
        rl2_features: int = 12,
        d_output: int = 384,
        out_norm: str = "layer",
        activation: str = "leaky_relu",
        skip_rl2_norm: bool = False,
        hide_rl2s: bool = False,
        drqv2_aug: bool = False,
        obs_key: str = "observation",
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.data_aug = (
            cnn.DrQv2Aug(4, channels_first=channels_first) if drqv2_aug else lambda x: x
        )
        obs_shape = self.obs_space[obs_key].shape
        self.cnn = cnn_type(
            img_shape=obs_shape,
            channels_first=channels_first,
            activation=activation,
        )
        img_feature_dim = self.cnn(
            torch.zeros((1, 1) + obs_shape, dtype=torch.uint8)
        ).shape[-1]
        self.img_features = nn.Linear(img_feature_dim, img_features)

        self.rl2_norm = InputNorm(self.rl2_space.shape[-1], skip=skip_rl2_norm)
        self.rl2_features = nn.Linear(rl2_space.shape[-1], rl2_features)

        mlp_in = img_features + rl2_features
        self.merge = nn.Linear(mlp_in, d_output)
        self.out_norm = ff.Normalization(out_norm, d_output)
        self.hide_rl2s = hide_rl2s
        self.obs_key = obs_key
        self._emb_dim = d_output

    def inner_forward(self, obs, rl2s, log_dict: Optional[dict] = None):
        # multi-modal envs that do not use the default `observation` key need their own custom encoders.
        img = obs[self.obs_key].float()
        B, L, *_ = img.shape
        if self.training:
            og_split = max(min(math.ceil(B * 0.25), B - 1), 0)
            aug = self.data_aug(img[og_split:, ...])
            img = torch.cat((img[:og_split, ...], aug), dim=0)
        img = (img / 128.0) - 1.0
        img_rep = self.cnn(img, flatten=True, from_float=True)
        add_activation_log("cnn_out", img_rep, log_dict)
        img_rep = self.img_features(img_rep)
        add_activation_log("img_features", img_rep, log_dict)

        rl2s = symlog(rl2s)
        if self.training:
            self.rl2_norm.update_stats(rl2s)
        rl2s_norm = self.rl2_norm(rl2s)
        if self.hide_rl2s:
            rl2s_norm = rl2s_norm * 0
        rl2s_rep = self.rl2_features(rl2s_norm)

        inp = torch.cat((img_rep, rl2s_rep), dim=-1)
        merge = self.merge(inp)
        add_activation_log("tstep_encoder_prenorm", merge, log_dict)
        out = self.out_norm(merge)
        return out

    @property
    def emb_dim(self):
        return self._emb_dim
