"""CAMPPlus speaker encoder used by IndexTTS.

Derived from 3D-Speaker via IndexTTS. Module names intentionally match the
upstream implementation so existing CAMPPlus checkpoints strict-load.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


class StatsPool(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True)
        return torch.cat([mean, std], dim=-1)


class TDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            if kernel_size % 2 != 1:
                raise ValueError("Equal padding requires an odd kernel size")
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        reduction=2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def seg_pooling(x, seg_len=100):
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        return y * self.sigmoid(self.linear2(context))


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Equal padding requires an odd kernel size")
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride,
            (kernel_size - 1) // 2 * dilation,
            dilation,
            bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x, use_reentrant=False)
        else:
            x = self.bn_function(x)
        return self.cam_layer(self.nonlinear2(x))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        for index in range(num_layers):
            self.add_module(
                f"tdnnd{index + 1}",
                CAMDenseTDNNLayer(
                    in_channels + index * out_channels,
                    out_channels,
                    bn_channels,
                    kernel_size,
                    stride,
                    dilation,
                    bias,
                    config_str,
                    memory_efficient,
                ),
            )

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.linear(self.nonlinear(x))


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if x.ndim == 2:
            x = self.linear(x.unsqueeze(-1)).squeeze(-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=(2, 2), m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(
            m_channels,
            m_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for block_stride in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class CAMPPlus(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=512,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
    ):
        super().__init__()
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    )
                ]
            )
        )
        channels = init_channels
        for index, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=True)
        ):
            self.xvector.add_module(
                f"block{index + 1}",
                CAMDenseTDNNBlock(
                    num_layers,
                    channels,
                    growth_rate,
                    bn_size * growth_rate,
                    kernel_size,
                    dilation=dilation,
                    config_str=config_str,
                    memory_efficient=memory_efficient,
                ),
            )
            channels += num_layers * growth_rate
            self.xvector.add_module(
                f"transit{index + 1}",
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2
        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module(
            "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.xvector(self.head(x.permute(0, 2, 1)))
