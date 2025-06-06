"""
Feed-forward network components.
"""

import torch
from torch import nn

from amago.nets.utils import activation_switch


class Normalization(nn.Module):
    """Quick-switch between different normalization methods.

    Args:
        method: Normalization method to use. Options are: "layer", "batch",
            "rmsnorm", "unitball", "unitball-detach", "none". "unitball" is
            (x / ||x||), "unitball-detach" is (x / ||x||.detach()). "none" is a
            no-op and the rest are standard LayerNorm, BatchNorm, RMSNorm.
        d_model: Expected dimension of the input to normalize (scalar). Operates
            on the last dimensions of the input sequence.
    """

    def __init__(self, method: str, d_model: int):
        super().__init__()
        if not method in {
            "layer",
            "none",
            "unitball",
            "unitball-detach",
            "rmsnorm",
        }:
            raise ValueError(f"Invalid normalization method: {method}")
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        elif method == "unitball":
            self.norm = lambda x: x / (
                torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True) + 1e-5
            )
        elif method == "unitball-detach":
            self.norm = (
                lambda x: x
                / (
                    torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True) + 1e-5
                ).detach()
            )
        elif method == "rmsnorm":
            self.norm = _RMSNorm(size=d_model)
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py#L255

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py.
    BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class FFBlock(nn.Module):
    """Feed-forward block with a residual connection.

    Args:
        d_model: Dimension of the input.
        d_ff: Dimension of the hidden layer.
        dropout: Dropout rate.
        activation: Activation function.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.00,
        activation: str = "leaky_relu",
    ):
        super().__init__()

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_switch(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dropout(self.activation(self.ff1(x)))
        x1 = self.dropout(self.activation(self.ff2(x1)))
        return x + x1


class MLP(nn.Module):
    """Basic multi-layer feed-forward network.

    d_inp --> d_hidden --> ... --> d_hidden --> d_output

    Args:
        d_inp: Dimension of the input.
        d_hidden: Dimension of the hidden layer.
        n_layers: Number of non-output layers (including the input layer)
        d_output: Dimension of the output.
        activation: Activation function. See `amago.nets.utils.activation_switch`
            for options. Default is "leaky_relu".
        dropout_p: Dropout rate. Default is 0.0.
    """

    def __init__(
        self,
        d_inp: int,
        d_hidden: int,
        n_layers: int,
        d_output: int,
        activation: str = "leaky_relu",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert n_layers >= 1
        self.in_layer = nn.Linear(d_inp, d_hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.layers = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 1)]
        )
        self.out_layer = nn.Linear(d_hidden, d_output)
        self.activation = activation_switch(activation)

    def forward(self, x):
        x = self.dropout(self.activation(self.in_layer(x)))
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        x = self.out_layer(x)
        return x
