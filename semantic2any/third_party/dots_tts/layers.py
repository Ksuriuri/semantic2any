"""Causal conv wrappers used by the dots.tts AudioVAE decoder."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        **_kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = int((kernel_size * dilation - dilation) / 2)
                self.left_padding = 0
        else:
            self.left_padding = 0

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )

    def forward(self, x):
        if self.causal and self.left_padding > 0:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super().forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        **_kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            if padding != 0:
                raise AssertionError("padding is not allowed in causal ConvTranspose1d.")
            if kernel_size != 2 * stride:
                raise AssertionError(
                    "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."
                )

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.causal = causal
        self.stride_size = stride

    def forward(self, x):
        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride_size]
        return x


__all__ = ["Conv1d", "ConvTranspose1d"]
