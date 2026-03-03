"""
Custom Transformer components.
"""

import math
from functools import lru_cache
from typing import Optional, Iterable
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, einsum
import gin

from .utils import activation_switch
from amago.utils import amago_warning
from amago.nets.ff import Normalization

# Flex Attention
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
        and_masks,
    )
except ImportError:
    flex_attention = None

# Flash Attention 2
try:
    import flash_attn
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_with_kvcache
except ImportError:
    amago_warning("Missing FlashAttention (2.0) Install")
    flash_attn = None
else:
    torch.set_float32_matmul_precision("high")


class SelfAttention(nn.Module, ABC):
    """A base class for self-attention layers.

    Args:
        causal: Whether to use a causal mask.
        dropout: The dropout rate of the attention matrix.
    """

    def __init__(self, causal: bool = True, dropout: float = 0.0):
        super().__init__()
        self.causal = causal
        self.dropout = dropout

    @abstractmethod
    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        """Map queries keys and values to attention output.

        Should implement full training pass when key_cache/val_cache/cache_seqlens are
        None, and (cached) inference when provided.

        Args:
            qkv: A tensor of shape (batch_size, sequence_length, 3, num_heads, head_dim).
                Packed queries, keys, and values.

        Keyword Args:
            key_cache: A tensor of shape (batch_size, max_sequence_length, num_heads,
                head_dim).
            val_cache: A tensor of shape (batch_size, max_sequence_length, num_heads,
                head_dim).
            cache_seqlens: A tensor of shape (batch_size,) that defines the current index
                of the k/v cache.

        Returns:
            A tensor of shape (batch_size, sequence_length, num_heads, head_dim).
        """
        raise NotImplementedError


class VanillaAttention(SelfAttention):
    """Unoptimized self-attention in regular pytorch.

    Args:
        causal: Whether to use a causal mask.
        dropout: The dropout rate of the attention matrix.
    """

    def __init__(self, causal: bool, dropout: float):
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
        cache_idxs = torch.arange(key_cache.shape[0], device=key_cache.device)
        key_cache[cache_idxs, cache_seqlens] = keys[:, 0]
        val_cache[cache_idxs, cache_seqlens] = values[:, 0]
        end = cache_seqlens + 1
        max_len = end.max()
        k_cache = torch.nan_to_num(key_cache[:, :max_len])
        v_cache = torch.nan_to_num(val_cache[:, :max_len])
        # attention scores + masking
        scores = scale * torch.einsum("blhe,blhe->blh", queries, k_cache)
        mask = torch.arange(max_len, device=cache_seqlens.device)[None, :] >= end[:, None]
        scores.masked_fill_(mask[:, :, None], -torch.inf)
        # output
        A = self.dropout(torch.softmax(scores, dim=1))
        V = torch.einsum("blh,blhd->bhd", A, v_cache).unsqueeze(1)
        # fmt: on
        return V

    @torch.compile
    def _forward_without_cache(self, qkv, mask):
        queries, keys, values = torch.unbind(qkv, dim=2)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.causal:
            scores.masked_fill_(mask, -torch.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V

    @torch.compiler.disable
    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        if key_cache is None and val_cache is None or cache_seqlens is None:
            B, L, *_ = qkv.shape
            if self._mask is None or self._mask.shape != (B, 1, L, L):
                self._mask = torch.triu(
                    torch.ones((B, 1, L, L), dtype=torch.bool, device=qkv.device),
                    diagonal=1,
                )
            return self._forward_without_cache(qkv, self._mask)
        else:
            assert not self.training
            return self._inference_with_cache(qkv, key_cache, val_cache, cache_seqlens)


@gin.configurable
class FlashAttention(SelfAttention):
    """Optimized self-attention using flash_attn.

    Args:
        causal: Whether to use a causal mask.
        dropout: The dropout rate of the attention matrix.
        window_size: flash_attn's window_size parameter, which enables sliding window
            attention. Defaults to (-1, -1), which is standard full-length attention.
    """

    def __init__(
        self,
        causal: bool,
        dropout: float,
        window_size: tuple[int, int] = (-1, -1),
    ):
        assert (
            flash_attn is not None
        ), "Missing flash attention 2 install (pip install amago[flash])."
        super().__init__(causal=causal, dropout=dropout)
        self.window_size = window_size

    @torch.compiler.disable
    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = qkv.to(torch.bfloat16)
        if key_cache is None or val_cache is None or cache_seqlens is None:
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.window_size,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn_with_kvcache(
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


class FlexAttention(SelfAttention):
    """Experimental support for flex_attention (a recent pytorch feature).

    Allows custom sparse attention patterns using score_mod and mask_mod function.
    (https://pytorch.org/blog/flexattention/)
    (https://github.com/pytorch-labs/attention-gym)

    The main benefit of flex_attention for our purposes is a unified implementation
    of key/value cache inference for more complex attention patterns.

    Args:
        score_mod: A function that takes the batch_idx, head_idx, q_idx, kv_idx and
            computes a scalar score for the attention matrix entry between these
            locations.
        mask_mod: A function that takes the batch_idx, head_idx, q_idx, kv_idx and
            returns False if attention scores between these locations should be
            masked.
        causal: Whether to use a causal mask. If True, the causal mask is applied on
            top of the custom mask_mod. Defaults to True.
        dropout: The dropout rate of the attention matrix. Defaults to 0.0.
    """

    def __init__(
        self,
        score_mod: callable,
        mask_mod: callable,
        causal: bool,
        dropout: float,
    ):
        assert flex_attention is not None, "FlexAttention requires pytorch >= 2.5"
        if dropout > 0.0:
            amago_warning(
                "FlexAttention does not support attention dropout. Setting to 0."
            )
        super().__init__(causal=causal, dropout=0.0)

        def causal_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) if causal else True

        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.causal_mask = causal_mask

    @lru_cache
    def cached_training_mask(self, q_len: int, kv_len: int):
        return create_block_mask(
            and_masks(self.mask_mod, self.causal_mask),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
        )

    def kv_cache_score_mod(self, cache_seqlens):
        def _kv_cache_score_mod(score, b, h, q_idx, kv_idx):
            q_idx_rel = q_idx + cache_seqlens[b]
            base = self.score_mod(score, b, h, q_idx_rel, kv_idx)
            return base

        return _kv_cache_score_mod

    def kv_cache_mask_mod(self, cache_seqlens):
        def _kv_cache_mask_mod(b, h, q_idx, kv_idx):
            q_idx_rel = q_idx + cache_seqlens[b]
            base = self.mask_mod(b, h, q_idx_rel, kv_idx)
            base = base & (kv_idx <= cache_seqlens[b])
            if self.causal:
                return base & (q_idx_rel >= kv_idx)
            return base

        return _kv_cache_mask_mod

    @torch.compile
    def flex_attention(self, q, k, v, score_mod, block_mask):
        return flex_attention(q, k, v, score_mod, block_mask)

    @torch.compile
    def flex_attention_inf(self, q, k, v, score_mod, block_mask):
        # pretend this is a different function than training to keep
        # torch's compilation separate.
        return flex_attention(q, k, v, score_mod, block_mask)

    @torch.compiler.disable
    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        if key_cache is None or val_cache is None or cache_seqlens is None:
            assert self.training
            qkv = rearrange(qkv, "b l three h e -> b h three l e")
            *_, L, _ = qkv.shape
            q, k, v = torch.unbind(qkv, dim=2)
            mask = self.cached_training_mask(L, L)
            out = self.flex_attention(q, k, v, self.score_mod, mask)
            out = rearrange(out, "b h l e -> b l h e")
        else:
            assert not self.training
            q, k, v = torch.unbind(qkv, dim=2)
            cache_idxs = torch.arange(key_cache.shape[0], device=k.device)
            key_cache[cache_idxs, cache_seqlens] = k[:, 0]
            val_cache[cache_idxs, cache_seqlens] = v[:, 0]
            max_len = cache_seqlens.max() + 1
            q = rearrange(q, "b l h e -> b h l e")
            k_cache = torch.nan_to_num(
                rearrange(key_cache[:, :max_len], "b l h e -> b h l e")
            )
            v_cache = torch.nan_to_num(
                rearrange(val_cache[:, :max_len], "b l h e -> b h l e")
            )
            # TODO: custom constructor as potential speedup?
            # https://pytorch.org/blog/flexattention/#q-how-can-we-compute-blockmask-quicker
            inf_mask = create_block_mask(
                self.kv_cache_mask_mod(cache_seqlens),
                B=q.shape[0],
                H=None,
                Q_LEN=1,
                KV_LEN=max_len,
            )
            out = self.flex_attention_inf(
                q, k_cache, v_cache, self.kv_cache_score_mod(cache_seqlens), inf_mask
            )
            out = rearrange(out, "b h l e -> b l h e")
        return out


class VanillaFlexAttention(FlexAttention):
    """A sanity-check test of FlexAttention that should be equivalent to VanillaAttention."""

    def __init__(self, causal: bool, dropout: float):
        super().__init__(
            score_mod=lambda score, b, h, q_idx, kv_idx: score,
            mask_mod=lambda b, h, q_idx, kv_idx: True,
            causal=causal,
            dropout=dropout,
        )


@gin.configurable
class SlidingWindowFlexAttention(FlexAttention):
    """A more useful test of FlexAttention that implements a sliding window pattern for long context lengths."""

    def __init__(
        self,
        causal: bool,
        dropout: float,
        window_size: int = gin.REQUIRED,
    ):
        def sliding_window_mask_mod(b, h, q_idx, kv_idx):
            window_mask = q_idx - kv_idx <= window_size
            return window_mask

        super().__init__(
            score_mod=lambda score, b, h, q_idx, kv_idx: score,
            mask_mod=sliding_window_mask_mod,
            causal=causal,
            dropout=dropout,
        )


@gin.configurable
class ClippedSlidingSinkAttention(FlexAttention):
    """
    Sliding-window attention with optional attention sink and logit clipping.
    """

    def __init__(
        self,
        causal: bool,
        dropout: float,
        window_size: int = gin.REQUIRED,
        logit_clip: float = 0.0,
        sink_size: int = 0,
        sink_bias: float = 0.0,
    ):
        assert window_size > 0, "window_size must be > 0"
        self.window_size = int(window_size)
        self.logit_clip = float(logit_clip) if logit_clip is not None else 0.0
        self.sink_size = int(sink_size)
        self.sink_bias = float(sink_bias)

        has_sink = self.sink_size > 0
        has_sink_bias = has_sink and (self.sink_bias != 0.0)
        clip_active = self.logit_clip > 0.0

        def sliding_window_with_sink_mask_mod(
            b: int, h: int, q_idx: int, kv_idx: int
        ) -> bool:
            dq = q_idx - kv_idx
            in_window = (dq >= 0) & (dq <= self.window_size)
            in_sink = (kv_idx < self.sink_size) if has_sink else False
            return in_window | in_sink

        def score_with_sink_and_clip(
            score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int
        ) -> torch.Tensor:
            if has_sink_bias and kv_idx < self.sink_size:
                score = score + score.new_tensor(self.sink_bias)
            if clip_active:
                score = torch.clamp(score, -self.logit_clip, self.logit_clip)
            return score

        super().__init__(
            score_mod=score_with_sink_and_clip,
            mask_mod=sliding_window_with_sink_mask_mod,
            causal=causal,
            dropout=dropout,
        )


@gin.configurable
class SigmaReparam(nn.Linear):
    """SigmaReparam nn.Linear alternative.

    https://github.com/apple/ml-sigma-reparam/blob/fea4e359126f812bd3e0a12234c56330fe4b5fa2/vision/layers.py#L90
    https://github.com/ywchan2005/sigma-reparam-pytorch/blob/2a5676ac71f75567a09db4ecafc1a4d7bc135b8e/sigma_reparam.py#L5

    SigmaReparam is an alternative to nn.Linear that can be used in Transformer blocks
    to stabilize attention scores. (https://arxiv.org/abs/2303.06296)

    Args:
        d_in: The input dimension of the layer.
        d_out: The output dimension of the layer.

    Keyword Args:
        bias: Whether to use a bias in the layer. Defaults to True.
        fast_init: Skip a SVD initialization step and use a simpler strategy. Mainly
            used for backward compatability with old results and as a hacky way to
            speed up init for large models when we'll be loading a pretrained
            checkoint soon anyway. Defaults to False.
    """

    def __init__(self, d_in, d_out, bias: bool = True, fast_init: bool = False):
        super().__init__(d_in, d_out, bias=bias)
        if not fast_init:
            nn.init.trunc_normal_(self.weight, std=0.02)
            u = torch.linalg.svd(self.weight.T, full_matrices=False)[-1][0].detach()
            v = torch.linalg.svd(self.weight, full_matrices=False)[-1][0].detach()
        else:
            # initialization from legacy version used in the original AMAGO paper.
            # This was a guess based on the sigma reparam pseudocode before the code was released,
            # and leads to large outputs early in training... though we never encountered any
            # real problems with this.
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            u = torch.randn(d_out, device=self.weight.device, dtype=self.weight.dtype)
            v = torch.randn(d_in, device=self.weight.device, dtype=self.weight.dtype)
            u = u / u.norm(dim=0)
            v = v / v.norm(dim=0)
        self.register_buffer("u", u)
        self.register_buffer("v", v)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        if self.training:
            # with torch.no_grad(): # does not compile w/ accelerate 1.0 DDP torch 2.5
            u = (self.weight @ self.v).float()
            self.u.data = F.normalize(u, dim=0)
            v = (self.weight.T @ self.u).float()
            self.v.data = F.normalize(v, dim=0)
        # detach instead...
        sigma = einsum(self.u.detach(), self.weight, self.v.detach(), "d, d c , c->")
        W_hat = self.gamma / sigma * self.weight
        out = F.linear(x, W_hat, self.bias)
        return out


class AttentionLayer(nn.Module):
    """Query, Key, Value --> Self-Attention --> Output Projection"""

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


class TransformerLayer(nn.Module):
    """Pre-Norm Self-Attention Layer"""

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
        self.d_model = d_model

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
    """A cache for key and value tensors."""

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
    """Helps manage the Cache hidden state during rollouts."""

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
    """Classic sinusoidal positional encoding."""

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


@gin.configurable
class LearnablePosEmb(nn.Module):
    """Learnable positional encoding.

    Creates a lookup table of d_model size embeddings for every timestep of the
    episode.

    Args:
        d_model: The dimension of the embeddings.

    Keyword Args:
        max_time_idx: The maximum timestep we'll need to learn an embedding for. So
            application-specific that it's gin.REQUIRED and therefore must be
            configured manually in the training script or its .gin files.
    """

    def __init__(self, d_model: int, max_time_idx: int = gin.REQUIRED):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=max_time_idx, embedding_dim=d_model
        )

    def forward(self, pos_idxs: torch.LongTensor):
        return self.embeddings(pos_idxs)


class Transformer(nn.Module):
    """Build a full Transformer model from a list of layers."""

    def __init__(
        self,
        inp_dim: int,
        d_model: int,
        layers: Iterable[nn.Module],
        dropout_emb: float = 0.05,
        norm: str = "layer",
        pos_emb: str = "fixed",
    ):
        super().__init__()
        if pos_emb == "fixed":
            self.position_embedding = FixedPosEmb(d_model)
        elif pos_emb == "learnable":
            self.position_embedding = LearnablePosEmb(d_model)
        else:
            raise ValueError(
                f"Unrecognized pos_emb: {pos_emb}. Options are 'fixed' or 'learnable'."
            )
        self.inp = nn.Linear(inp_dim, d_model)
        self.dropout = nn.Dropout(dropout_emb)
        assert all(l.d_model == d_model for l in layers)
        self.n_layers = len(layers)
        self.layers = nn.ModuleList(layers)
        self.norm = Normalization(method=norm, d_model=d_model)
        self.d_model = d_model

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
        """Transformer seq2seq

        Args:
            seq: The input sequence of shape (batch_size, seq_len, inp_dim).
            pos_idxs: The position indices of the input sequence of shape (batch_size, seq_len).
            hidden_state: The hidden state of the transformer.

        Returns:
            The output sequence of shape (batch_size, seq_len, d_model).
            The new hidden state of the transformer.
        """

        traj_emb = self.preprocess_seq(seq, pos_idxs)
        if hidden_state is not None:
            assert not self.training
            traj_emb = self.inference_forward(traj_emb, hidden_state)
            hidden_state.update()
        else:
            traj_emb = self.training_forward(traj_emb)
        return traj_emb, hidden_state
