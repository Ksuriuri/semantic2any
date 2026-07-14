# Derived from IndexTTS2's indextts/s2mel/modules/encodec.py.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0. See LICENSE.

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def _extra_padding(x: Tensor, kernel_size: int, stride: int, padding_total: int) -> int:
    frames = (x.shape[-1] - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return ideal_length - x.shape[-1]


def _pad1d(x: Tensor, paddings: tuple[int, int], mode: str) -> Tensor:
    left, right = paddings
    if mode != "reflect":
        return F.pad(x, paddings, mode)
    extra_right = 0
    if x.shape[-1] <= max(left, right):
        extra_right = max(left, right) - x.shape[-1] + 1
        x = F.pad(x, (0, extra_right))
    padded = F.pad(x, paddings, mode)
    return padded[..., : padded.shape[-1] - extra_right] if extra_right else padded


class NormConv1d(nn.Module):
    """IndexTTS2-compatible normalized Conv1d wrapper."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        **kwargs,
    ) -> None:
        super().__init__()
        if norm not in {"none", "weight_norm"}:
            raise ValueError(f"Unsupported DiT convolution normalization: {norm}")
        conv = nn.Conv1d(*args, **kwargs)
        self.conv = weight_norm(conv) if norm == "weight_norm" else conv
        self.norm = nn.Identity()
        self.norm_type = norm
        self.causal = causal

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


class SConv1d(nn.Module):
    """IndexTTS2-compatible same-length 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: Tensor) -> Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        effective_kernel = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel - stride
        extra_padding = _extra_padding(x, effective_kernel, stride, padding_total)
        if self.causal:
            x = _pad1d(x, (padding_total, extra_padding), self.pad_mode)
        else:
            right = padding_total // 2
            left = padding_total - right
            x = _pad1d(x, (left, right + extra_padding), self.pad_mode)
        return self.conv(x)
