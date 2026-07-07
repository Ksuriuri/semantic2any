from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Sinusoidal timestep embedding used by flow-matching decoders."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _as_tuple(value: int | Sequence[int], length: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * length
    value = tuple(int(v) for v in value)
    if len(value) == 1:
        return value * length
    if len(value) != length:
        raise ValueError(f"Expected {length} values, got {len(value)}")
    return value


def _resample_time(x: Tensor, target_len: int) -> Tensor:
    if x.size(1) == target_len:
        return x
    return F.interpolate(x.transpose(1, 2), size=target_len, mode="nearest").transpose(1, 2)


def _resample_padding_mask(mask: Tensor | None, target_len: int) -> Tensor | None:
    if mask is None or mask.size(1) == target_len:
        return mask
    mask_f = mask.float().unsqueeze(1)
    return F.interpolate(mask_f, size=target_len, mode="nearest").squeeze(1).bool()


class ZipFormerBlock(nn.Module):
    """A compact ZipFormer-style block with attention, convolution and FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        feedforward_dim: int,
        cnn_kernel: int,
        dropout: float = 0.1,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.use_conv = use_conv
        if use_conv:
            padding = cnn_kernel // 2
            self.conv = nn.Sequential(
                nn.Conv1d(dim, dim * 2, kernel_size=1),
                nn.GLU(dim=1),
                nn.Conv1d(dim, dim, kernel_size=cnn_kernel, padding=padding, groups=dim),
                nn.SiLU(),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)

        if self.use_conv:
            conv_in = self.conv_norm(x).transpose(1, 2)
            x = x + self.dropout(self.conv(conv_in).transpose(1, 2))

        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        return x


class TTSZipformer(nn.Module):
    """Small ZipVoice-inspired TTS ZipFormer.

    This intentionally mirrors the public ``TTSZipformer`` call shape from
    ZipVoice while keeping the dependency surface self-contained for this repo.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsampling_factor: int | Sequence[int] = (1, 2, 4, 2, 1),
        num_encoder_layers: int | Sequence[int] = (2, 2, 4, 4, 4),
        cnn_module_kernel: int | Sequence[int] = (31, 15, 7, 15, 31),
        encoder_dim: int = 512,
        feedforward_dim: int = 1536,
        num_heads: int = 4,
        query_head_dim: int = 32,
        pos_head_dim: int = 4,
        value_head_dim: int = 12,
        pos_dim: int = 48,
        dropout: float = 0.1,
        use_time_embed: bool = True,
        time_embed_dim: int = 192,
        use_guidance_scale_embed: bool = False,
        guidance_scale_embed_dim: int = 192,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        del query_head_dim, pos_head_dim, value_head_dim, pos_dim
        if isinstance(downsampling_factor, int):
            downsampling_factor = (downsampling_factor,)
        self.downsampling_factor = tuple(int(v) for v in downsampling_factor)
        n_stacks = len(self.downsampling_factor)
        num_encoder_layers = _as_tuple(num_encoder_layers, n_stacks)
        cnn_module_kernel = _as_tuple(cnn_module_kernel, n_stacks)

        self.in_proj = nn.Linear(in_dim, encoder_dim)
        self.out_norm = nn.LayerNorm(encoder_dim)
        self.out_proj = nn.Linear(encoder_dim, out_dim)
        self.use_time_embed = use_time_embed
        self.use_guidance_scale_embed = use_guidance_scale_embed

        if use_time_embed:
            self.time_embed = nn.Sequential(
                nn.Linear(time_embed_dim, encoder_dim),
                nn.SiLU(),
                nn.Linear(encoder_dim, encoder_dim),
            )
            self.time_embed_dim = time_embed_dim
        else:
            self.time_embed = None
            self.time_embed_dim = 0

        if use_guidance_scale_embed:
            self.guidance_scale_embed = nn.Sequential(
                nn.Linear(guidance_scale_embed_dim, encoder_dim),
                nn.SiLU(),
                nn.Linear(encoder_dim, encoder_dim),
            )
            self.guidance_scale_embed_dim = guidance_scale_embed_dim
        else:
            self.guidance_scale_embed = None
            self.guidance_scale_embed_dim = 0

        self.stacks = nn.ModuleList()
        for layers, kernel in zip(num_encoder_layers, cnn_module_kernel, strict=True):
            self.stacks.append(
                nn.ModuleList(
                    ZipFormerBlock(
                        dim=encoder_dim,
                        num_heads=num_heads,
                        feedforward_dim=feedforward_dim,
                        cnn_kernel=kernel,
                        dropout=dropout,
                        use_conv=use_conv,
                    )
                    for _ in range(layers)
                )
            )

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
        padding_mask: Tensor | None = None,
        guidance_scale: Tensor | None = None,
    ) -> Tensor:
        batch, frames = x.shape[:2]
        h = self.in_proj(x)

        if self.time_embed is not None:
            if t is None:
                t = torch.zeros(batch, device=x.device, dtype=x.dtype)
            if t.ndim == 0:
                t = t.expand(batch)
            time = timestep_embedding(t, self.time_embed_dim).to(dtype=h.dtype)
            h = h + self.time_embed(time).unsqueeze(1)

        if self.guidance_scale_embed is not None and guidance_scale is not None:
            if guidance_scale.ndim == 0:
                guidance_scale = guidance_scale.expand(batch)
            guidance = timestep_embedding(guidance_scale, self.guidance_scale_embed_dim).to(dtype=h.dtype)
            h = h + self.guidance_scale_embed(guidance).unsqueeze(1)

        base = h
        for factor, blocks in zip(self.downsampling_factor, self.stacks, strict=True):
            target_len = max(1, math.ceil(frames / factor))
            stack_h = _resample_time(h, target_len)
            stack_mask = _resample_padding_mask(padding_mask, target_len)
            for block in blocks:
                stack_h = block(stack_h, stack_mask)
            stack_h = _resample_time(stack_h, frames)
            h = (h + stack_h) * math.sqrt(0.5)

        h = h + base
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0)
        return self.out_proj(self.out_norm(h))
