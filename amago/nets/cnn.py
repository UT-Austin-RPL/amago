from abc import ABC, abstractmethod

import gin

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from amago.nets.utils import activation_switch


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


class CNN(nn.Module, ABC):
    """Abstract base class for built-in CNN architectures.

    Args:
        img_shape: Shape of the image (H, W, C) or (C, H, W).
        channels_first: Whether the image is in channels-first format.
        activation: Activation function to use. See
            `amago.nets.utils.activation_switch`.
    """

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

    @property
    def blank_img(self):
        return torch.zeros((1, 1) + self.img_shape, dtype=torch.uint8)

    @torch.compile
    def forward(
        self, obs: torch.Tensor, from_float: bool = False, flatten: bool = True
    ) -> torch.FloatTensor:
        """Produce a feature map from an image.

        Args:
            obs: Image tensor of shape (Batch, Len, H, W, C) for channels_last or
                (Batch, Len, C, H, W) for channels_first. Can be uint8 or float
                dtype.
            from_float: If False, assumes obs has pixel values in [0, 255],
                casts to float, and scales to [-1, 1]. If True, assumes obs is
                already in desired range/dtype. Defaults to False.
            flatten: If True, flatten output activations into feature array for
                linear layers. Defaults to True.

        Returns:
            Feature tensor of shape (B, L, out_dim) if flatten=True, otherwise
            (B, L, C, H, W) matching CNN output shape.
        """
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


class DrQv2Aug(nn.Module):
    """Pad+Random Crop image augmentation from DrQv2.

    https://github.com/facebookresearch/drqv2/blob/main/drqv2.py

    Args:
        pad: Number of pixels to pad on each side of the image.
        channels_first: Whether the image is in channels-first format.
    """

    def __init__(self, pad: int, channels_first: bool):
        super().__init__()
        self.pad = pad
        self.channels_first = channels_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pad+random crop augmentation to an image.

        Args:
            x: Image tensor of shape (Batch, Len, H, W, C) for channels_last or
                (Batch, Len, C, H, W) for channels_first. Currently, H must equal
                W.

        Returns:
            Augmented image with the same shape as the input.
        """
        if self.channels_first:
            B, L, c, h, w = x.shape
            x = rearrange(x, "b l c h w -> (b l) c h w")
        else:
            B, L, h, w, c = x.shape
            x = rearrange(x, "b l h w c -> (b l) c h w")
        b, *_ = x.shape
        assert h == w

        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(b, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        out = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

        if self.channels_first:
            out = rearrange(out, "(b l) c h w -> b l c h w", b=B, l=L)
        else:
            out = rearrange(out, "(b l) c h w -> b l h w c", b=B, l=L)

        return out


class DrQCNN(CNN):
    """CNN architecture from DrQ-v2.

    https://arxiv.org/abs/2107.09645

    Args:
        img_shape: Shape of the image (H, W, C) or (C, H, W).
        channels_first: Whether the image is in channels-first format.
        activation: Activation function to use. See
            `amago.nets.utils.activation_switch`.
    """

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

    def conv_forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(imgs))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        return x


@gin.configurable
class GridworldCNN(CNN):
    """Tiny CNN architecture useful for gridworld map features.

    Args:
        img_shape: Shape of the image (H, W, C) or (C, H, W).
        channels_first: Whether the image is in channels-first format.
        activation: Activation function to use. See
            `amago.nets.utils.activation_switch`.

    Keyword Args:
        channels: List of 3 ints representing the number of output channels for
            each convolutional layer. Defaults to [16, 32, 48].
        kernels: List of 3 ints representing the kernel size for each
            convolutional layer. Defaults to [2, 2, 2].
        strides: List of 3 ints representing the stride for each convolutional
            layer. Defaults to [1, 1, 1].
    """

    def __init__(
        self,
        img_shape: tuple[int],
        channels_first: bool,
        activation: str,
        channels: list[int] = [16, 32, 48],
        kernels: list[int] = [2, 2, 2],
        strides: list[int] = [1, 1, 1],
    ):
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

    def conv_forward(self, imgs):
        x = self.activation(self.conv1(imgs))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return x


@gin.configurable
class NatureishCNN(CNN):
    """Customizable version of the small CNN architecture from DQN.

    Args:
        img_shape: Shape of the image (H, W, C) or (C, H, W).
        channels_first: Whether the image is in channels-first format.
        activation: Activation function to use. See
            `amago.nets.utils.activation_switch`.

    Keyword Args:
        channels: List of 3 ints representing the number of output channels for
            each convolutional layer. Defaults to [32, 64, 64].
        kernels: List of 3 ints representing the kernel size for each
            convolutional layer. Defaults to [8, 4, 3].
        strides: List of 3 ints representing the stride for each convolutional
            layer. Defaults to [4, 2, 1].
    """

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


@gin.configurable
class IMPALAishCNN(CNN):
    """CNN architecture from IMPALA.

    Args:
        img_shape: Shape of the image (H, W, C) or (C, H, W).
        channels_first: Whether the image is in channels-first format.
        activation: Activation function to use. See
            `amago.nets.utils.activation_switch`.

    Keyword Args:
        cnn_block_depths: List of ints representing the number of output
            channels for each convolutional block. Length defines the number of
            residual blocks. Defaults to [16, 32, 32].
        post_group_norm: Whether to use group normalization after each
            convolutional block. Defaults to True.
    """

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
                self.norm = nn.GroupNorm(4, depth) if post_group_norm else nn.Identity()

            def forward(self, x):
                xp = self.conv1(self.activation(x))
                xp = self.conv2(self.activation(xp))
                return self.norm(x + xp)

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
