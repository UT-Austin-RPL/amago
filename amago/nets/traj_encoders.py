from abc import ABC, abstractmethod

import torch
from torch import nn
import gin

try:
    import flash_attn
except ImportError:
    flash_attn = None

from amago.nets import ff, transformer, utils
from amago.utils import amago_warning


class TrajEncoder(nn.Module, ABC):
    safe_serialization = True
    def __init__(self, tstep_dim: int, max_seq_len: int):
        super().__init__()
        self.tstep_dim = tstep_dim
        self.max_seq_len = max_seq_len

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
        d_model: int = 256,
        d_ff: int | None = None,
        n_layers: int = 1,
        dropout=0.0,
        activation="leaky_relu",
        norm="layer",
    ):
        super().__init__(tstep_dim, max_seq_len)
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
    safe_serialization = False
    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_hidden: int = 256,
        n_layers: int = 2,
        d_output: int = 256,
        norm: str = "layer",
    ):
        super().__init__(tstep_dim, max_seq_len)
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
class LSTMTrajEncoder(TrajEncoder):
    safe_serialization = False

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_hidden: int = 256,
        n_layers: int = 2,
        d_output: int = 256,
        norm: str = "layer",
    ):
        super().__init__(tstep_dim, max_seq_len)
        self.rnn = nn.LSTM(
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
        # tuple
        # 0 shape: (2, 12, 400)
        # 1 shape: (2, 12, 400)
        assert hidden_state is not None
        hidden_state[0][:, dones] = 0.0
        hidden_state[1][:, dones] = 0.0
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
        pos_emb: str = "fixed",
        causal: bool = True,
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
        head_scaling: bool = True,
        attention_type: type[transformer.SelfAttention] = transformer.FlashAttention,
    ):
        super().__init__(tstep_dim, max_seq_len)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_layers = n_layers
        if flash_attn is None and attention_type == transformer.FlashAttention:
            amago_warning(
                f"`flash_attn` is not installed; falling back to VanillaAttention"
            )
            attention_type = transformer.VanillaAttention
        self.attention_type = attention_type
        self.d_model = d_model

        def make_layer():
            return transformer.TransformerLayer(
                attention_layer=transformer.AttentionLayer(
                    self_attention=attention_type(causal=causal, dropout=dropout_attn),
                    d_model=self.d_model,
                    d_qkv=self.head_dim,
                    n_heads=self.n_heads,
                    dropout_qkv=dropout_qkv,
                    head_scaling=head_scaling,
                    sigma_reparam=sigma_reparam,
                ),
                d_model=self.d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
                sigma_reparam=sigma_reparam,
                normformer_norms=normformer_norms,
            )

        layers = [make_layer() for _ in range(self.n_layers)]
        self.tformer = transformer.Transformer(
            inp_dim=tstep_dim,
            d_model=self.d_model,
            layers=layers,
            dropout_emb=dropout_emb,
            norm=norm,
            pos_emb=pos_emb,
        )

    def init_hidden_state(self, batch_size: int, device: torch.device):
        def make_cache():
            dtype = (
                torch.bfloat16
                if self.attention_type == transformer.FlashAttention
                else torch.float32
            )
            return transformer.Cache(
                device=device,
                dtype=dtype,
                layers=self.n_layers,
                batch_size=batch_size,
                max_seq_len=self.max_seq_len,
                n_heads=self.n_heads,
                head_dim=self.head_dim,
            )

        hidden_state = transformer.TformerHiddenState(
            key_cache=make_cache(),
            val_cache=make_cache(),
            seq_lens=torch.zeros((batch_size,), dtype=torch.int32, device=device),
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


try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, norm: str):
        super().__init__()
        self.norm = ff.Normalization(norm, d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, seq):
        return seq + self.mamba(self.norm(seq))

    def step(self, seq, conv_state, ssm_state):
        res, new_conv_state, new_ssm_state = self.mamba.step(
            self.norm(seq), conv_state, ssm_state
        )
        return seq + res, new_conv_state, new_ssm_state


class _MambaHiddenState:
    def __init__(self, conv_states: list[torch.Tensor], ssm_states: list[torch.Tensor]):
        assert len(conv_states) == len(ssm_states)
        self.n_layers = len(conv_states)
        self.conv_states = conv_states
        self.ssm_states = ssm_states

    def reset(self, idxs):
        for i in range(self.n_layers):
            # hidden states are initialized to zero
            self.conv_states[i][idxs] = 0.0
            self.ssm_states[i][idxs] = 0.0

    def __getitem__(self, layer_idx: int):
        assert layer_idx < self.n_layers
        return self.conv_states[layer_idx], self.ssm_states[layer_idx]

    def __setitem__(self, layer_idx: int, conv_ssm: tuple[torch.Tensor]):
        conv, ssm = conv_ssm
        self.conv_states[layer_idx] = conv
        self.ssm_states[layer_idx] = ssm


@gin.configurable
class MambaTrajEncoder(TrajEncoder):
    """
    "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", Gu and Dao, 2023.

    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    https://github.com/johnma2006/mamba-minimal/tree/master
    """

    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 3,
        norm: str = "layer",
    ):
        super().__init__(tstep_dim, max_seq_len)

        assert (
            Mamba is not None
        ), "Missing Mamba installation (pip install amago[mamba])"
        self.inp = nn.Linear(tstep_dim, d_model)

        self.mambas = nn.ModuleList(
            [
                _MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = ff.Normalization(norm, d_model)
        self._emb_dim = d_model

    def init_hidden_state(self, batch_size: int, device: torch.device):
        conv_states, ssm_states = [], []
        for mamba_block in self.mambas:
            conv_state, ssm_state = mamba_block.mamba.allocate_inference_cache(
                batch_size, max_seqlen=self.max_seq_len
            )
            conv_states.append(conv_state)
            ssm_states.append(ssm_state)
        return _MambaHiddenState(conv_states, ssm_states)

    def reset_hidden_state(self, hidden_state, dones):
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, _MambaHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(self, seq, time_idxs=None, hidden_state=None):
        seq = self.inp(seq)
        if hidden_state is None:
            for mamba in self.mambas:
                seq = mamba(seq)
        else:
            assert not self.training
            assert isinstance(hidden_state, _MambaHiddenState)
            for i, mamba in enumerate(self.mambas):
                conv_state_i, ssm_state_i = hidden_state[i]
                seq, new_conv_state_i, new_ssm_state_i = mamba.step(
                    seq, conv_state=conv_state_i, ssm_state=ssm_state_i
                )
                hidden_state[i] = new_conv_state_i, new_ssm_state_i
        return self.out_norm(seq), hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim
