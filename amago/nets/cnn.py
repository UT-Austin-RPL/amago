from abc import ABC, abstractmethod
import warnings
from typing import Callable

import gin

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from amago.nets.utils import activation_switch


@gin.configurable(allowlist=["pad"])
class DrQv2Aug(nn.Module):
    def __init__(self, channels_first : bool, pad : int = 4):
        super().__init__()
        self.pad = pad
        self.channels_first = channels_first

    def forward(self, imgs):
        if self.channels_first:
            B, C, H, W = imgs.shape
        else:
            B, H, W, C = imgs.shape
        assert H == W and self.channels_first, "not sure if this works yet"
        padding = tuple([self.pad] * 4)
        x = F.pad(imgs, padding, "replicate")
        eps = 1. / (H + 2 * self.pad)
        arange = torch.linspace(-1. + eps, 1. - eps, H + 2 * self.pad, device=imgs.device, dtype=imgs.dtype)[:H]
        arange = arange.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(B, 1, 1, 2), device=imgs.device, dtype=imgs.dtype)
        shift *= 2. / (H + 2 * self.pad)
        grid = base_grid + shift
        
        out = F.grid_sample(imgs, grid, padding_mode="zeros", align_corners=False)
        return out



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
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class CNN(nn.Module, ABC):
    def __init__(self, img_shape: tuple[int, int, int], channels_first: bool, aug_Cls : Callable | None, activation : str):
        super().__init__()
        self.img_shape = img_shape
        self.channels_first = channels_first
        self.aug = aug_Cls(channels_first=channels_first) if aug_Cls else lambda x : x
        self.activation = activation_switch(activation)

    @abstractmethod
    def conv_forward(self, imgs):
        pass

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, obs):
        assert obs.dtype == torch.uint8
        assert obs.ndim == 5
        if not self.channels_first:
            B, L, H, W, C = obs.shape
            img = rearrange(obs, "b l h w c -> (b l) c h w")
        else:
            B, L, C, H, W = obs.shape
            img = rearrange(obs, "b l c h w -> (b l) c h w")
        img = (img / 128.0) - 1.0
        if self.training:
            img = self.aug(img)
        features = self.conv_forward(img)


        out = rearrange(features, "(b l) c h w -> (b l) (c h w)", l=L)
        out = rearrange(out, "(b l) f -> b l f", l=L)
        #out = rearrange(features, "(b l) c h w -> b l (c h w)", l=L)
        return out



class DrQCNN(CNN):
    def __init__(self, img_shape, channels_first, **kwargs):
        super().__init__(img_shape, channels_first=channels_first, **kwargs)
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
    def __init__(self, img_shape, channels_first, **kwargs):
        super().__init__(img_shape, channels_first=channels_first, **kwargs)
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
