# Derived from IndexTTS2's indextts/s2mel/modules/wavenet.py.
# Licensed under the Apache License, Version 2.0. See LICENSE.

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .conv import SConv1d


def _fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, channels: int) -> Tensor:
    inputs = input_a + input_b
    return torch.tanh(inputs[:, :channels, :]) * torch.sigmoid(inputs[:, channels:, :])


class WN(nn.Module):
    """IndexTTS2 WaveNet output head."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("WaveNet kernel_size must be odd")
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.cond_layer = (
            SConv1d(gin_channels, 2 * hidden_channels * n_layers, 1, norm="weight_norm")
            if gin_channels
            else None
        )
        for index in range(n_layers):
            dilation = dilation_rate**index
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                SConv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    pad_mode="reflect",
                    norm="weight_norm",
                    causal=causal,
                )
            )
            res_skip_channels = 2 * hidden_channels if index < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                SConv1d(
                    hidden_channels,
                    res_skip_channels,
                    1,
                    norm="weight_norm",
                    causal=causal,
                )
            )

    def forward(self, x: Tensor, x_mask: Tensor, g: Tensor | None = None) -> Tensor:
        output = torch.zeros_like(x)
        conditioned = self.cond_layer(g) if g is not None and self.cond_layer is not None else None
        for index, (in_layer, res_skip_layer) in enumerate(
            zip(self.in_layers, self.res_skip_layers, strict=True)
        ):
            x_in = in_layer(x)
            if conditioned is None:
                conditioning = torch.zeros_like(x_in)
            else:
                offset = index * 2 * self.hidden_channels
                conditioning = conditioned[:, offset : offset + 2 * self.hidden_channels, :]
            activations = _fused_add_tanh_sigmoid_multiply(x_in, conditioning, self.hidden_channels)
            residual_skip = res_skip_layer(self.drop(activations))
            if index < self.n_layers - 1:
                x = (x + residual_skip[:, : self.hidden_channels, :]) * x_mask
                output = output + residual_skip[:, self.hidden_channels :, :]
            else:
                output = output + residual_skip
        return output * x_mask
