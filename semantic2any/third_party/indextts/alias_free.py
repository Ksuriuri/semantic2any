"""Pure-PyTorch anti-aliased activations from NVIDIA BigVGAN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = (
            0.5842 * (attenuation - 21) ** 0.4
            + 0.07886 * (attenuation - 21.0)
        )
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride=1,
        padding=True,
        padding_mode="replicate",
        kernel_size=12,
    ):
        super().__init__()
        if not 0 <= cutoff <= 0.5:
            raise ValueError("cutoff must be between zero and 0.5")
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, x):
        channels = x.shape[1]
        if self.padding:
            x = F.pad(
                x, (self.pad_left, self.pad_right), mode=self.padding_mode
            )
        return F.conv1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = (
            self.pad * self.stride + (self.kernel_size - self.stride) // 2
        )
        self.pad_right = (
            self.pad * self.stride
            + (self.kernel_size - self.stride + 1) // 2
        )
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            ),
        )

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio=2,
        down_ratio=2,
        up_kernel_size=12,
        down_kernel_size=12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))
