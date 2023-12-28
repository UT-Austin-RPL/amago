from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
import gin

from amago.nets import ff, transformer, utils


class TrajEncoder(nn.Module, ABC):
    def __init__(self, tstep_dim: int, max_seq_len: int, horizon: int):
        super().__init__()
        self.tstep_dim = tstep_dim
        self.max_seq_len = max_seq_len
        self.horzion = horizon

    def reset_hidden_state(self, hidden_state, dones):
        return hidden_state

    def init_hidden_state(self, batch_size: int, device: torch.device):
        return None

    @abstractmethod
    def forward(self, seq: torch.Tensor, time_idxs: torch.Tensor, hidden_state=None):
        pass

    @abstractmethod
    def emb_dim(self):
        pass


@gin.configurable
class FFTrajEncoder(TrajEncoder):
    def __init__(
        self,
        tstep_dim,
        max_seq_len,
        horizon,
        d_model: int = 256,
        d_ff: int | None = None,
        n_layers: int = 1,
        dropout=0.0,
        activation="leaky_relu",
        norm="layer",
    ):
        super().__init__(tstep_dim, max_seq_len, horizon)
        d_ff = d_ff or d_model * 4
        self.traj_emb = nn.Linear(tstep_dim, d_model)
        self.traj_blocks = nn.ModuleList(
            [
                ff.FFBlock(d_model, d_ff, dropout=dropout, activation=activation)
                for _ in range(n_layers)
            ]
        )
        self.traj_last = nn.Linear(d_model, d_model)
        self.norm = ff.Normalization(norm, d_model)
        self.activation = utils.activation_switch(activation)
        self.dropout = nn.Dropout(dropout)
        self._emb_dim = d_model

    def init_hidden_state(self, batch_size: int, device: torch.device):
        # easy trick that will make the Agent chop off previous timesteps
        # from the sequence.
        return True

    def forward(self, seq, time_idxs=None, hidden_state=None):
        traj_emb = self.dropout(self.activation(self.traj_emb(seq)))
        for traj_block in self.traj_blocks:
            traj_emb = traj_block(traj_emb)
        traj_emb = self.traj_last(traj_emb)
        traj_emb = self.dropout(self.norm(traj_emb))
        return traj_emb, hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
class GRUTrajEncoder(TrajEncoder):
    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        horizon: int,
        d_hidden: int = 256,
        n_layers: int = 2,
        d_output: int = 256,
        norm: str = "layer",
    ):
        super().__init__(tstep_dim, max_seq_len, horizon)

        self.rnn = nn.GRU(
            input_size=tstep_dim,
            hidden_size=d_hidden,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.out = nn.Linear(d_hidden, d_output)
        self.out_norm = ff.Normalization(norm, d_output)
        self._emb_dim = d_output

    def reset_hidden_state(self, hidden_state, dones):
        assert hidden_state is not None
        hidden_state[:, dones] = 0.0
        return hidden_state

    def forward(self, seq, time_idxs=None, hidden_state=None):
        output_seq, new_hidden_state = self.rnn(seq, hidden_state)
        out = self.out_norm(self.out(output_seq))
        return out, new_hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
class TformerTrajEncoder(TrajEncoder):
    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        horizon: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 3,
        dropout_ff: float = 0.05,
        dropout_emb: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        activation: str = "leaky_relu",
        norm: str = "layer",
        attention: str = "flash",
        pos_emb: str = "learnable",
    ):
        super().__init__(tstep_dim, max_seq_len, horizon)
        self.tformer = transformer.Transformer(
            inp_dim=tstep_dim,
            max_pos_idx=horizon,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            layers=n_layers,
            dropout_emb=dropout_emb,
            dropout_ff=dropout_ff,
            dropout_attn=dropout_attn,
            dropout_qkv=dropout_qkv,
            activation=activation,
            attention=attention,
            norm=norm,
            pos_emb=pos_emb,
        )
        self.d_model = d_model

    def init_hidden_state(self, batch_size: int, device: torch.device):
        def make_cache():
            return transformer.Cache(
                device=device,
                dtype=torch.bfloat16,
                batch_size=batch_size,
                max_seq_len=self.max_seq_len,
                n_heads=self.tformer.n_heads,
                head_dim=self.tformer.head_dim,
            )

        hidden_state = transformer.TformerHiddenState(
            key_cache=[make_cache() for _ in range(self.tformer.n_layers)],
            val_cache=[make_cache() for _ in range(self.tformer.n_layers)],
            timesteps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
        )
        return hidden_state

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, transformer.TformerHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(self, seq, time_idxs, hidden_state=None):
        assert time_idxs is not None
        return self.tformer(seq, pos_idxs=time_idxs, hidden_state=hidden_state)

    @property
    def emb_dim(self):
        return self.d_model
