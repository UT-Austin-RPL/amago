import math
from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, einsum
import gin

from .utils import activation_switch
from amago.utils import amago_warning
from amago.nets.ff import Normalization


try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    amago_warning("Missing FlexAttention (torch >= 2.5)")
    create_block_mask = None
    flex_attention = None

try:
    import flash_attn
except ImportError:
    amago_warning("Missing FlashAttention (2.0) Install")
    flash_attn = None
else:
    torch.set_float32_matmul_precision("high")


class SelfAttention(nn.Module, ABC):
    def __init__(self, causal: bool = True, dropout: float = 0.0):
        super().__init__()
        self.causal = causal
        self.dropout = dropout

    @abstractmethod
    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        raise NotImplementedError


class VanillaAttention(SelfAttention):
    def __init__(self, causal: bool = True, dropout: float = 0.0):
        super().__init__(causal=causal, dropout=dropout)
        self.dropout = nn.Dropout(self.dropout)
        self._mask = None

    @torch.compile
    def _inference_with_cache(self, qkv, key_cache, val_cache, cache_seqlens):
        # fmt: off
        queries, keys, values = torch.unbind(qkv, dim=2)
        B, L, H, E = queries.shape
        assert L == 1
        scale = 1.0 / math.sqrt(E)
        # fill cache, trim sequences
        key_cache[:, cache_seqlens] = keys
        val_cache[:, cache_seqlens] = values
        end = cache_seqlens + 1
        max_len = end.max()
        k_cache = key_cache[:, :max_len]
        v_cache = val_cache[:, :max_len]
        # attention scores + masking
        scores = torch.einsum("blhe,blhe->blh", queries, k_cache)
        mask = torch.arange(max_len, device=cache_seqlens.device)[None, :] >= end[:, None]
        scores.masked_fill_(mask[:, :, None], -torch.inf)
        # output
        A = self.dropout(torch.softmax(scale * scores, dim=1))
        V = torch.einsum("blh,blhd->bhd", A, v_cache).unsqueeze(1)
        # fmt: on
        return V

    @torch.compiler.disable
    def _forward_without_cache(self, qkv):
        queries, keys, values = torch.unbind(qkv, dim=2)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self._mask is None or self._mask.shape != (B, 1, L, L):
            self._mask = torch.triu(
                torch.ones((B, 1, L, L), dtype=torch.bool, device=qkv.device),
                diagonal=1,
            )
        if self.causal:
            scores.masked_fill_(self._mask, -torch.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        if key_cache is None and val_cache is None or cache_seqlens is None:
            return self._forward_without_cache(qkv)
        else:
            assert not self.training
            return self._inference_with_cache(qkv, key_cache, val_cache, cache_seqlens)


@gin.configurable(allowlist=["window_size"])
class FlashAttention(SelfAttention):
    def __init__(
        self,
        causal: bool = True,
        dropout: float = 0.0,
        window_size: tuple[int, int] = (-1, -1),
    ):
        assert flash_attn is not None, "Missing flash attention 2 install."
        super().__init__(causal=causal, dropout=dropout)
        self.window_size = window_size

    @torch.compiler.disable
    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = qkv.to(torch.bfloat16)
        if key_cache is None or val_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.window_size,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k_cache=key_cache,
                v_cache=val_cache,
                cache_seqlens=cache_seqlens,
                k=k,
                v=v,
                causal=self.causal,
                window_size=self.window_size,
            )
        return out


class SigmaReparam(nn.Linear):
    """
    Updated version of SigmaReparam following the initialization strategy in the official code release.
    https://github.com/apple/ml-sigma-reparam/blob/fea4e359126f812bd3e0a12234c56330fe4b5fa2/vision/layers.py#L90
    https://github.com/ywchan2005/sigma-reparam-pytorch/blob/2a5676ac71f75567a09db4ecafc1a4d7bc135b8e/sigma_reparam.py#L5
    """

    def __init__(self, d_in, d_out, bias: bool = True):
        super().__init__(d_in, d_out, bias=bias)
        nn.init.trunc_normal_(self.weight, std=0.02)
        # init can be quite slow...
        u = torch.linalg.svd(self.weight.T, full_matrices=False)[-1][0].detach()
        v = torch.linalg.svd(self.weight, full_matrices=False)[-1][0].detach()
        self.register_buffer("u", u)
        self.register_buffer("v", v)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                u = (self.weight @ self.v).float()
                self.u.data = F.normalize(u, dim=0)
                v = (self.weight.T @ self.u).float()
                self.v.data = F.normalize(v, dim=0)
        sigma = einsum(self.u, self.weight, self.v, "d, d c , c->")
        W_hat = self.gamma / sigma * self.weight
        out = F.linear(x, W_hat, self.bias)
        return out


class SigmaReparamLegacyInit(nn.Module):
    """
    When I implemented SigmaReparam for AMAGOv1, the code had not been open-sourced and I only
    had https://arxiv.org/pdf/2303.06296.pdf to go on. This code follows the pseudocode in
    Appendix C. The initialization strategy results in unusually large initial output values.
    I assumed this was fine because it worked so well empirically (w/ flash attention).
    Finally looked into this for VanillaAttention, and the official code release clearly goes out
    of its way to fix this problem with a specific init...

    Leaving this original version here in case the large init happens to be helpful in some cases.
    It is not realistic to re-run all the experiments in both papers to find out for sure. Between
    the clear emphasis on numerical stability in the official code and my own experience with (rare)
    policy NaNs at init, I think the updated version should be the default.
    """

    def __init__(self, d_in, d_out, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(d_out), requires_grad=True) if bias else None
        u = torch.randn(d_out)
        self.register_buffer("u", u / u.norm(dim=0))
        v = torch.randn(d_in)
        self.register_buffer("v", v / v.norm(dim=0))
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # same as nn.Linear
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                u = (self.W @ self.v).float()
                self.u.data = u / u.norm(dim=0)
                v = (self.W.T @ self.u).float()
                self.v.data = v / v.norm(dim=0)
        sigma = einsum(self.u, self.W, self.v, "d, d c , c->")
        W_hat = self.gamma / sigma * self.W
        out = F.linear(x, W_hat, self.b)
        return out


@gin.configurable(allowlist=["head_scaling", "sigma_reparam"])
class AttentionLayer(nn.Module):
    def __init__(
        self,
        self_attention: SelfAttention,
        d_model: int,
        d_qkv: int,
        n_heads: int,
        dropout_qkv: float = 0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ):
        super().__init__()
        assert isinstance(self_attention, SelfAttention)
        self.self_attention = self_attention
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.qkv_projection = FF(d_model, 3 * d_qkv * n_heads, bias=False)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.out_projection = FF(d_qkv * n_heads, d_model)
        self.head_scaler = nn.Parameter(
            torch.ones(1, 1, n_heads, 1), requires_grad=head_scaling
        )
        self.n_heads = n_heads

    def forward(self, sequence, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = self.dropout_qkv(self.qkv_projection(sequence))
        qkv = rearrange(
            qkv,
            "batch len (three d_qkv heads) -> batch len three heads d_qkv",
            heads=self.n_heads,
            three=3,
        )
        out = self.head_scaler * self.self_attention(
            qkv=qkv,
            key_cache=key_cache,
            val_cache=val_cache,
            cache_seqlens=cache_seqlens,
        )
        out = rearrange(out, "batch len heads dim -> batch len (heads dim)")
        out = self.out_projection(out)
        return out


@gin.configurable(denylist=["activation", "norm", "dropout_ff"])
class TransformerLayer(nn.Module):
    """
    Pre-Norm Self-Attention
    """

    def __init__(
        self,
        attention_layer: AttentionLayer,
        d_model: int,
        d_ff: int,
        dropout_ff: float = 0.1,
        activation: str = "leaky_relu",
        norm: str = "layer",
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
    ):
        super().__init__()
        assert isinstance(attention_layer, AttentionLayer)
        self.attention_layer = attention_layer
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.ff1 = FF(d_model, d_ff)
        self.ff2 = FF(d_ff, d_model)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = (
            Normalization(method=norm, d_model=d_model)
            if normformer_norms
            else lambda x: x
        )
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = (
            Normalization(method=norm, d_model=d_ff)
            if normformer_norms
            else lambda x: x
        )
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = activation_switch(activation)

    @torch.compile
    def forward(self, self_seq, key_cache=None, val_cache=None, cache_seqlens=None):
        q1 = self.norm1(self_seq)  # pre-norm
        q1 = self.attention_layer(
            q1, key_cache=key_cache, val_cache=val_cache, cache_seqlens=cache_seqlens
        )
        q1 = self.norm2(q1)  # normformer extra norm 1
        self_seq = self_seq + q1
        q1 = self.norm3(self_seq)  # regular norm
        # normformer extra norm 2
        q1 = self.norm4(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        self_seq = self_seq + q1
        return self_seq


class Cache:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        layers: int,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
    ):
        self.data = torch.zeros(
            (layers, batch_size, max_seq_len, n_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.max_seq_len = max_seq_len
        # make silent bugs in k/v cache... much louder
        self.data[:] = torch.nan
        self.device = device

    def __len__(self):
        return self.data.shape[2]

    def roll_back(self, seq_lens):
        idxs = torch.where(seq_lens == self.max_seq_len)[0]
        roll = self.data[:, idxs, 1:].clone()
        self.data[:, idxs, :-1] = roll
        self.data[:, idxs, -1] = torch.nan  # no silent bugs
        return idxs


class TformerHiddenState:
    def __init__(self, key_cache: Cache, val_cache: Cache, seq_lens: torch.Tensor):
        assert seq_lens.dtype == torch.int32
        assert key_cache.device == val_cache.device
        self.n_layers = key_cache.data.shape[0]
        assert self.n_layers == val_cache.data.shape[0]
        self.key_cache = key_cache
        self.val_cache = val_cache
        self.seq_lens = seq_lens
        self.device = key_cache.device

    def reset(self, idxs):
        self.seq_lens[idxs] = 0

    def update(self):
        self.seq_lens += 1
        self.key_cache.roll_back(self.seq_lens)
        idxs = self.val_cache.roll_back(self.seq_lens)
        self.seq_lens[idxs] -= 1

    def __getitem__(self, layer_idx):
        assert layer_idx < self.n_layers
        return (
            self.key_cache.data[layer_idx],
            self.val_cache.data[layer_idx],
            self.seq_lens,
        )


class FixedPosEmb(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, pos_idxs: torch.LongTensor):
        B, L = pos_idxs.shape
        emb = torch.zeros(
            (B, L, self.d_model), device=pos_idxs.device, dtype=torch.float32
        )
        coeff = torch.exp(
            (
                torch.arange(0, self.d_model, 2, device=emb.device, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        )
        emb[..., 0::2] = torch.sin(pos_idxs.float().unsqueeze(-1) * coeff)
        emb[..., 1::2] = torch.cos(pos_idxs.float().unsqueeze(-1) * coeff)
        return emb


class Transformer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        d_model: int = 128,
        d_ff: int = 512,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        attention_type: type[SelfAttention] = FlashAttention,
        activation: str = "leaky_relu",
        norm: str = "layer",
        causal: bool = True,
    ):
        super().__init__()
        # embedding
        self.position_embedding = FixedPosEmb(d_model)
        self.inp = nn.Linear(inp_dim, d_model)
        self.dropout = nn.Dropout(dropout_emb)

        self.head_dim = d_model // n_heads
        assert self.head_dim in range(8, 129, 8)
        self.n_heads = n_heads
        self.n_layers = layers

        def make_layer():
            return TransformerLayer(
                attention_layer=AttentionLayer(
                    self_attention=attention_type(causal=causal, dropout=dropout_attn),
                    d_model=d_model,
                    d_qkv=self.head_dim,
                    n_heads=self.n_heads,
                    dropout_qkv=dropout_qkv,
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
            )

        self.layers = nn.ModuleList([make_layer() for _ in range(layers)])
        self.norm = Normalization(method=norm, d_model=d_model)
        self.d_model = d_model
        self._blank_hidden_state = [[None, None, None] for _ in range(self.n_layers)]

    @property
    def emb_dim(self):
        return self.d_model

    def preprocess_seq(self, seq, pos_idxs):
        pos_emb = self.position_embedding(pos_idxs.squeeze(-1))
        traj_emb = self.inp(seq)
        traj_emb = self.dropout(traj_emb + pos_emb)
        return traj_emb

    @torch.compile
    def training_forward(self, seq):
        for layer in self.layers:
            seq = layer(seq)
        return self.norm(seq)

    def inference_forward(self, seq, hidden_state):
        for i, layer in enumerate(self.layers):
            seq = layer(seq, *hidden_state[i])
        return self.norm(seq)

    def forward(self, seq, pos_idxs, hidden_state: Optional[TformerHiddenState] = None):
        traj_emb = self.preprocess_seq(seq, pos_idxs)
        if hidden_state is not None:
            assert not self.training
            traj_emb = self.inference_forward(traj_emb, hidden_state)
            hidden_state.update()
        else:
            assert self.training
            traj_emb = self.training_forward(traj_emb)
        return traj_emb, hidden_state
