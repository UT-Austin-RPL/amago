import torch
from torch import nn
import torch.nn.functional as F
import gin

from .utils import activation_switch


class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "batch", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        else:
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)


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

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        x = self.dropout(self.activation(self.in_layer(x)))
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        x = self.out_layer(x)
        return x
