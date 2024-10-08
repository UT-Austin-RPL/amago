import torch
from torch import nn
import gin

from .utils import activation_switch


class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in [
            "layer",
            "batch",
            "rmsnorm",
            "unitball",
            "unitball-detach",
            "none",
        ]
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
            self.norm = RMSNorm(size=d_model)
        else:
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py#L255

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
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


# caution: this is probably too low of a level to be using gin-config
@gin.configurable
class FFBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.00, activation="leaky_relu"):
        super().__init__()

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_switch(activation)

    def forward(self, x):
        x1 = self.dropout(self.activation(self.ff1(x)))
        x1 = self.dropout(self.activation(self.ff2(x1)))
        return x + x1


# caution: this is probably too low of a level to be using gin-config
@gin.configurable
class MLP(nn.Module):
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
