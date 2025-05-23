from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple

import torch
from torch import nn
import numpy as np
import gin

from amago.nets import ff, transformer, utils
from amago.utils import amago_warning


class TrajEncoder(nn.Module, ABC):
    """Abstract base class for trajectory encoders.

    An agent's "TrajEncoder" is the sequence model in charge of mapping the output
    of the "TstepEncoder" for each timestep of the trajectory to the latent
    dimension where actor-critic learning takes place. Because the actor and
    critic are feed-forward networks, this is the place to add long-term memory
    over previous timesteps.

    Note:
        It would *not* make sense for the sequence model defined here to be
        bi-directional or non-causal.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. Any inputs will have
            been trimmed to this length before reaching the TrajEncoder.
    """

    def __init__(self, tstep_dim: int, max_seq_len: int):
        super().__init__()
        self.tstep_dim = tstep_dim
        self.max_seq_len = max_seq_len

    @property
    @abstractmethod
    def emb_dim(self) -> int:
        """Defines the expected output dim of this model.

        Used to infer the input dim of actor/critics.

        Returns:
            int: The embedding dimension.
        """
        pass

    def init_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        """Hook to create an architecture-specific hidden state.

        Return value is passed as `TrajEncoder.forward(..., hidden_state=self.init_hidden_state(...))`
        when the agent begins to interact with the environment.

        Args:
            batch_size: Number of parallel environments.
            device: Device to store hidden state tensors (if applicable).

        Returns:
            Optional[Any]: Some hidden state object, or None if not applicable.
                Defaults to None.
        """
        return None

    def reset_hidden_state(
        self, hidden_state: Optional[Any], dones: np.ndarray
    ) -> Optional[Any]:
        """Hook to implement architecture-specific hidden state reset.

        Args:
            hidden_state: We only expect to see hidden states that were created
                by `self.init_hidden_state()`.
            dones: A bool array of shape (num_parallel_envs,) where True
                indicates the agent loop has finished this episode and expects
                the hidden state for this batch index to be reset.

        Returns:
            Optional[Any]: Architecture-specific hidden state. Defaults to a
                no-op: `new_hidden_state = hidden_state`.
        """
        return hidden_state

    @abstractmethod
    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[Any] = None,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Sequence model forward pass.

        Args:
            seq: [Batch, Num Timesteps, TstepDim]. TstepDim is defined by the
                output of the TstepEncoder.
            time_idxs: [Batch, Num Timesteps, 1]. A sequence of ints tying the
                input seq to the number of steps that have passed since the start
                of the episode. Can be used to compute position embeddings or
                other temporal features.
            hidden_state: Architecture-specific hidden state. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[Any]]: A tuple containing:
                - output_seq: [Batch, Timestep, self.emb_dim]. Output of our
                    seq2seq model.
                - new_hidden_state: Architecture-specific hidden state. Expected to
                    be `None` if input `hidden_state` is `None`. Otherwise, we
                    assume we are at inference time and that this `forward`
                    method has handled any updates to the hidden state that were
                    needed.
        """
        pass


@gin.configurable
class FFTrajEncoder(TrajEncoder):
    """Feed-forward (memory-free) trajectory encoder.

    A useful tool for applying AMAGO to standard MDPs and benchmarking general
    RL details/hyperparamters on common benchmarks. The feed-forward architecture
    is designed to be close to an attention-less Transformer (residual blocks,
    norm, dropout, etc.). This makes it easy to create perfect 1:1 ablations of
    "memory vs. no memory" by only changing the TrajEncoder and without touching
    the `max_seq_len`, which would have the side-effect of changing the effective
    batch size of actor-critic learning.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. Any inputs will have
            been trimmed to this length before reaching the TrajEncoder.

    Keyword Args:
        d_model: Dimension of the main residual stream and output. 1:1 with how
            this would be defined in a Transformer. Defaults to 256.
        d_ff: Hidden dim of the feed-forward network along each residual block.
            1:1 with how this would be defined in a Transformer. Defaults to
            `4 * d_model`.
        n_layers: Number of residual feed-forward blocks. 1:1 with how this would
            be defined in a Transformer. Defaults to 1.
        dropout: Dropout rate. Equivalent to the dropout paramter of feed-forward
            blocks in a Transformer, but is also applied to the first and last
            linear layers (inp --> d_model and d_model --> out). Defaults to 0.0.
        activation: Activation function. Defaults to "leaky_relu".
        norm: Normalization function. Defaults to "layer" (LayerNorm).
    """

    def __init__(
        self,
        tstep_dim,
        max_seq_len,
        d_model: int = 256,
        d_ff: Optional[int] = None,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "leaky_relu",
        norm: str = "layer",
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

    @torch.compile
    def _traj_blocks_forward(self, seq: torch.Tensor) -> torch.Tensor:
        traj_emb = self.dropout(self.activation(self.traj_emb(seq)))
        for traj_block in self.traj_blocks:
            traj_emb = traj_block(traj_emb)
        traj_emb = self.traj_last(traj_emb)
        traj_emb = self.dropout(self.norm(traj_emb))
        return traj_emb

    def forward(
        self, seq, time_idxs=None, hidden_state=None, log_dict: Optional[dict] = None
    ):
        return self._traj_blocks_forward(seq), hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
class GRUTrajEncoder(TrajEncoder):
    """RNN (GRU) Trajectory Encoder.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. Any inputs will have
            been trimmed to this length before reaching the TrajEncoder.

    Keyword Args:
        d_hidden: Dimension of the hidden state of the GRU. Defaults to 256.
        n_layers: Number of layers in the GRU. Defaults to 2.
        d_output: Dimension of the output linear layer after the GRU. Defaults to
            256.
        norm: Normalization applied after the final linear layer. Defaults to
            "layer" (LayerNorm).
    """

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

    def forward(
        self, seq, time_idxs=None, hidden_state=None, log_dict: Optional[dict] = None
    ):
        output_seq, new_hidden_state = self.rnn(seq, hidden_state)
        out = self.out_norm(self.out(output_seq))
        return out, new_hidden_state

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
class TformerTrajEncoder(TrajEncoder):
    r"""Transformer Trajectory Encoder.

    A pre-norm Transformer decoder-only model that processes sequences of timestep
    embeddings.

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. The max context length
            of the model during training.

    Keyword Args:
        d_model: Dimension of the main residual stream and output. Defaults to
            256.
        n_heads: Number of self-attention heads. Each head has dimension
            d_model/n_heads. Defaults to 8.
        d_ff: Dimension of feed-forward network in residual blocks. Defaults to
            4*d_model.
        n_layers: Number of Transformer layers. Defaults to 3.
        dropout_ff: Dropout rate for linear layers within Transformer. Defaults to
            0.05.
        dropout_emb: Dropout rate for input embedding (combined input sequence and
            position embeddings passed to Transformer). Defaults to 0.05.
        dropout_attn: Dropout rate for attention matrix. Defaults to 0.00.
        dropout_qkv: Dropout rate for query/key/value projections. Defaults to
            0.00.
        activation: Activation function. Defaults to "leaky_relu".
        norm: Normalization function. Defaults to "layer" (LayerNorm).
        pos_emb: Position embedding type. "fixed" (default) uses sinusoidal
            embeddings, "learned" uses trainable embeddings per timestep.
        causal: Whether to use causal attention mask. Defaults to True.
        sigma_reparam: Whether to use :math:`\sigma`-reparam feed-forward layers
            from https://arxiv.org/abs/2303.06296. Defaults to True.
        normformer_norms: Whether to use extra norm layers from NormFormer
            (https://arxiv.org/abs/2110.09456). Always uses pre-norm Transformer.
        head_scaling: Whether to use head scaling from NormFormer. Defaults to
            True.
        attention_type: Attention layer type. Defaults to
            transformer.FlashAttention. transformer.VanillaAttention provided as
            backup. New types can inherit from transformer.SelfAttention.
    """

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
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
        head_scaling: bool = True,
        attention_type: type[transformer.SelfAttention] = transformer.FlashAttention,
    ):
        super().__init__(tstep_dim, max_seq_len)
        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attention_type = attention_type
        self.d_model = d_model

        def make_layer():
            return transformer.TransformerLayer(
                attention_layer=transformer.AttentionLayer(
                    self_attention=attention_type(causal=True, dropout=dropout_attn),
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

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> transformer.TformerHiddenState:
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

    def reset_hidden_state(
        self, hidden_state: Optional[transformer.TformerHiddenState], dones: np.ndarray
    ) -> Optional[transformer.TformerHiddenState]:
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, transformer.TformerHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[transformer.TformerHiddenState] = None,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[transformer.TformerHiddenState]]:
        assert time_idxs is not None
        return self.tformer(seq, pos_idxs=time_idxs, hidden_state=hidden_state)

    @property
    def emb_dim(self) -> int:
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
    """Mamba Trajectory Encoder.

    Implementation of the Mamba architecture from "Mamba: Linear-Time Sequence
    Modeling with Selective State Spaces" (https://arxiv.org/abs/2312.00752).

    Args:
        tstep_dim: Dimension of the input timestep representation (last dim of
            the input sequence). Defined by the output of the TstepEncoder.
        max_seq_len: Maximum sequence length of the model. The max context length
            of the model during training.

    Keyword Args:
        d_model: Dimension of the main residual stream and output, analogous to
            the d_model in a Transformer. Defaults to 256.
        d_state: Dimension of the SSM in Mamba blocks. Defaults to 16.
        d_conv: Dimension of the convolution layer in Mamba blocks. Defaults to 4.
        expand: Expansion factor of the SSM. Defaults to 2.
        n_layers: Number of Mamba blocks. Defaults to 3.
        norm: Normalization function. Defaults to "layer" (LayerNorm).

    References:
        - https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
        - https://github.com/johnma2006/mamba-minimal/tree/master
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

    def init_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> _MambaHiddenState:
        conv_states, ssm_states = [], []
        for mamba_block in self.mambas:
            conv_state, ssm_state = mamba_block.mamba.allocate_inference_cache(
                batch_size, max_seqlen=self.max_seq_len
            )
            conv_states.append(conv_state)
            ssm_states.append(ssm_state)
        return _MambaHiddenState(conv_states, ssm_states)

    def reset_hidden_state(
        self, hidden_state: Optional[_MambaHiddenState], dones: np.ndarray
    ) -> Optional[_MambaHiddenState]:
        if hidden_state is None:
            return None
        assert isinstance(hidden_state, _MambaHiddenState)
        hidden_state.reset(idxs=dones)
        return hidden_state

    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: Optional[torch.Tensor] = None,
        hidden_state: Optional[_MambaHiddenState] = None,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[_MambaHiddenState]]:
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
