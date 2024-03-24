from abc import ABC, abstractmethod
import warnings

import gin

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from amago.nets.utils import activation_switch
from amago.nets.ff import Normalization


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("leaky_relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


@gin.configurable
class CNN(nn.Module, ABC):
    def __init__(
        self,
        img_shape: tuple[int, int, int],
        channels_first: bool,
        activation: str,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.channels_first = channels_first
        self.activation = activation_switch(activation)

    @abstractmethod
    def conv_forward(self, imgs):
        pass

    def forward(self, obs, from_float: bool = False, flatten: bool = True):
        assert obs.ndim == 5
        if not from_float:
            assert obs.dtype == torch.uint8
            obs = (obs.float() / 128.0) - 1.0
        if not self.channels_first:
            B, L, H, W, C = obs.shape
            img = rearrange(obs, "b l h w c -> (b l) c h w")
        else:
            B, L, C, H, W = obs.shape
            img = rearrange(obs, "b l c h w -> (b l) c h w")
        features = self.conv_forward(img)
        if flatten:
            features = rearrange(features, "(b l) c h w -> b l (c h w)", l=L)
        else:
            features = rearrange(features, "(b l) c h w -> b l c h w", l=L)
        return features


class DrQCNN(CNN):
    def __init__(self, img_shape: tuple[int], channels_first: bool, activation: str):
        super().__init__(
            img_shape, channels_first=channels_first, activation=activation
        )
        C = img_shape[0] if self.channels_first else img_shape[-1]
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.apply(weight_init)

    def conv_forward(self, imgs):
        x = self.activation(self.conv1(imgs))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        return x


@gin.configurable(allowlist=["channels", "kernels", "strides"])
class NatureishCNN(CNN):
    def __init__(
        self,
        img_shape: tuple[int],
        channels_first: bool,
        activation: str,
        channels: list[int] = [32, 64, 64],
        kernels: list[int] = [8, 4, 3],
        strides: list[int] = [4, 2, 1],
    ):
        assert len(channels) == 3 and len(kernels) == 3 and len(strides) == 3
        super().__init__(
            img_shape, channels_first=channels_first, activation=activation
        )
        C = img_shape[0] if self.channels_first else img_shape[-1]
        self.conv1 = nn.Conv2d(
            C, channels[0], kernel_size=kernels[0], stride=strides[0]
        )
        self.conv2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=kernels[1], stride=strides[1]
        )
        self.conv3 = nn.Conv2d(
            channels[1], channels[2], kernel_size=kernels[2], stride=strides[2]
        )
        self.apply(weight_init)

    def conv_forward(self, imgs):
        x = self.activation(self.conv1(imgs))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return x


@gin.configurable(allowlist=["cnn_block_depths", "post_group_norm"])
class IMPALAishCNN(CNN):
    def __init__(
        self,
        img_shape: tuple[int],
        channels_first: bool,
        activation: str,
        cnn_block_depths: list[int] = [16, 32, 32],
        post_group_norm: bool = True,
    ):
        super().__init__(
            img_shape, channels_first=channels_first, activation=activation
        )

        class _ResidualBlock(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                self.conv1 = nn.Conv2d(
                    depth, depth, kernel_size=3, stride=1, padding="same"
                )
                self.conv2 = nn.Conv2d(
                    depth, depth, kernel_size=3, stride=1, padding="same"
                )
                self.activation = activation_switch(activation)
                self.norm = nn.GroupNorm(4, depth) if post_group_norm else lambda i: i

            def forward(self, x):
                xp = self.conv1(self.activation(x))
                xp = self.conv2(self.activation(xp))
                xp = self.norm(xp)
                return x + xp

        class _IMPALAConvBlock(nn.Module):
            def __init__(self, inp_c: int, depth: int):
                super().__init__()
                self.conv = nn.Conv2d(
                    inp_c, depth, kernel_size=3, stride=1, padding="same"
                )
                self.pool = nn.MaxPool2d(3, stride=2)
                self.residual1 = _ResidualBlock(depth)
                self.residual2 = _ResidualBlock(depth)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = self.residual1(x)
                x = self.residual2(x)
                return x

        channels = [img_shape[0 if channels_first else -1]] + cnn_block_depths
        blocks = []
        for inp, out in zip(channels, channels[1:]):
            blocks.append(_IMPALAConvBlock(inp, out))
        self.blocks = nn.ModuleList(blocks)

    def conv_forward(self, imgs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            imgs = block(imgs)
        return imgs
