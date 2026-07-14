# Derived from IndexTTS2's indextts/s2mel/modules/gpt_fast/model.py.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0. See ../LICENSE.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def find_multiple(value: int, divisor: int) -> int:
    return value if value % divisor == 0 else value + divisor - value % divisor


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model: int, norm: nn.Module) -> None:
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, inputs: Tensor, embedding: Tensor | None = None) -> Tensor:
        if embedding is None:
            return self.norm(inputs)
        weight, bias = torch.split(self.project_layer(embedding), self.d_model, dim=-1)
        return weight * self.norm(inputs) + bias


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int | None = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    has_cross_attention: bool = False
    context_dim: int = 0
    uvit_skip_connection: bool = False
    time_as_token: bool = False

    def __post_init__(self) -> None:
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            self.intermediate_size = find_multiple(int(2 * (4 * self.dim) / 3), 256)


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor) -> tuple[Tensor, Tensor]:
        if input_pos.shape[0] != k_val.shape[2]:
            raise ValueError("KV cache positions must match key sequence length")
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val
        return self.k_cache, self.v_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.freqs_cis: Tensor | None = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size: int, max_seq_length: int, use_kv_cache: bool = True) -> None:
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.project_layer.weight.dtype
        device = self.norm.project_layer.weight.device
        if not self.training and use_kv_cache:
            for block in self.layers:
                block.attention.kv_cache = KVCache(
                    max_batch_size,
                    max_seq_length,
                    self.config.n_local_heads,
                    head_dim,
                    dtype,
                ).to(device)
        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.head_dim,
            self.config.rope_base,
            dtype,
        ).to(device)
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool, device=device)
        )
        self.use_kv_cache = use_kv_cache
        if self.config.uvit_skip_connection:
            self.layers_emit_skip = [idx for idx in range(self.config.n_layer) if idx < self.config.n_layer // 2]
            self.layers_receive_skip = [idx for idx in range(self.config.n_layer) if idx > self.config.n_layer // 2]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        input_pos: Tensor | None = None,
        mask: Tensor | None = None,
        context: Tensor | None = None,
        context_input_pos: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if self.freqs_cis is None:
            raise RuntimeError("DiT caches are not initialized; call setup_caches before forward")
        if input_pos is None:
            raise ValueError("input_pos is required")
        if mask is None:
            mask = self.causal_mask[None, None, input_pos]
            if self.training or not self.use_kv_cache:
                mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        context_freqs_cis = self.freqs_cis[context_input_pos] if context is not None else None
        skip_inputs: list[Tensor] = []
        for idx, layer in enumerate(self.layers):
            skip_input = skip_inputs.pop() if idx in self.layers_receive_skip else None
            x = layer(
                x,
                c,
                input_pos,
                freqs_cis,
                mask,
                context,
                context_freqs_cis,
                cross_attention_mask,
                skip_input,
            )
            if idx in self.layers_emit_skip:
                skip_inputs.append(x)
        return self.norm(x, c)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.uvit_skip_connection = config.uvit_skip_connection
        self.time_as_token = config.time_as_token
        self.skip_in_linear = nn.Linear(config.dim * 2, config.dim) if config.uvit_skip_connection else None

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        context: Tensor | None = None,
        context_freqs_cis: Tensor | None = None,
        cross_attention_mask: Tensor | None = None,
        skip_in_x: Tensor | None = None,
    ) -> Tensor:
        del input_pos, context, context_freqs_cis, cross_attention_mask
        c = None if self.time_as_token else c
        if self.uvit_skip_connection and skip_in_x is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in_x], dim=-1))
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h, c))


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        if config.dim % config.n_head:
            raise ValueError("Transformer dim must be divisible by num_heads")
        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache: KVCache | None = None
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        batch, sequence, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        q = q.view(batch, sequence, self.n_head, self.head_dim)
        k = k.view(batch, sequence, self.n_local_heads, self.head_dim)
        v = v.view(batch, sequence, self.n_local_heads, self.head_dim)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if self.kv_cache is not None:
            if input_pos is None:
                raise ValueError("input_pos is required when using the KV cache")
            k, v = self.kv_cache.update(input_pos, k, v)
        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().view(batch, sequence, self.head_dim * self.n_head)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x.float() * torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float, dtype: torch.dtype) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: n_elem // 2].float() / n_elem))
    positions = torch.arange(seq_len, device=freqs.device)
    angles = torch.outer(positions, freqs)
    complex_freqs = torch.polar(torch.ones_like(angles), angles)
    return torch.stack([complex_freqs.real, complex_freqs.imag], dim=-1).to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs = freqs_cis.view(1, shaped.size(1), 1, shaped.size(3), 2)
    rotated = torch.stack(
        [
            shaped[..., 0] * freqs[..., 0] - shaped[..., 1] * freqs[..., 1],
            shaped[..., 1] * freqs[..., 0] + shaped[..., 0] * freqs[..., 1],
        ],
        dim=-1,
    )
    return rotated.flatten(3).type_as(x)
