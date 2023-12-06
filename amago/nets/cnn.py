from abc import ABC, abstractmethod
import warnings

import gin

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from amago.nets.utils import activation_switch


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
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
        data_aug: bool = False,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.channels_first = channels_first
        self.activation = activation_switch(activation)
        _i = (9, 84, 84)
        if data_aug:
            augs = [TranslationAug(_i, 4), BatchWiseImgColorJitterAug(_i)]
        else:
            augs = [IdentityAug()]
        self.data_aug = DataAugGroup(augs)

    @abstractmethod
    def conv_forward(self, imgs):
        pass

    def forward(self, obs):
        assert obs.dtype == torch.uint8
        assert obs.ndim == 5
        obs = obs / 255.0
        obs = torch.cat(self.data_aug(obs.split(3, dim=2)), axis=2)
        if not self.channels_first:
            B, L, H, W, C = obs.shape
            img = rearrange(obs, "b l h w c -> (b l) c h w")
        else:
            B, L, C, H, W = obs.shape
            img = rearrange(obs, "b l c h w -> (b l) c h w")
        features = self.conv_forward(img)
        out = rearrange(features, "(b l) c h w -> b l (c h w)", l=L)
        return out


from libero.lifelong.models.modules.data_augmentation import *


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


class NatureishCNN(CNN):
    def __init__(self, img_shape: tuple[int], channels_first: bool, activation):
        super().__init__(
            img_shape, channels_first=channels_first, activation=activation
        )
        C = img_shape[0] if self.channels_first else img_shape[-1]
        self.conv1 = nn.Conv2d(C, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.apply(weight_init)

    def conv_forward(self, imgs):
        x = self.activation(self.conv1(imgs))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return x
