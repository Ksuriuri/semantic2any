"""NVIDIA BigVGAN inference implementation vendored through IndexTTS."""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .alias_free import Activation1d


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class Snake(nn.Module):
    def __init__(
        self,
        in_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=False,
    ):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        initial = (
            torch.zeros(in_features) * alpha
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha = nn.Parameter(initial, requires_grad=alpha_trainable)
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + torch.sin(x * alpha).pow(2) / (alpha + self.no_div_by_zero)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=False,
    ):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        initial = (
            torch.zeros(in_features) * alpha
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha = nn.Parameter(initial.clone(), requires_grad=alpha_trainable)
        self.beta = nn.Parameter(initial.clone(), requires_grad=alpha_trainable)
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + torch.sin(x * alpha).pow(2) / (beta + self.no_div_by_zero)


def _init_weights(module, mean=0.0, std=0.01):
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def _activation(kind, channels, h):
    if kind == "snake":
        return Activation1d(
            Snake(channels, alpha_logscale=h.snake_logscale)
        )
    if kind == "snakebeta":
        return Activation1d(
            SnakeBeta(channels, alpha_logscale=h.snake_logscale)
        )
    raise NotImplementedError(f"Unsupported BigVGAN activation: {kind}")


class AMPBlock1(nn.Module):
    def __init__(
        self,
        h,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        activation=None,
    ):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=value,
                        padding=_get_padding(kernel_size, value),
                    )
                )
                for value in dilation
            ]
        )
        self.convs1.apply(_init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(_init_weights)
        self.activations = nn.ModuleList(
            [
                _activation(activation, channels, h)
                for _ in range(len(self.convs1) + len(self.convs2))
            ]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, acts1, acts2, strict=True
        ):
            x = conv2(act2(conv1(act1(x)))) + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(nn.Module):
    def __init__(
        self,
        h,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        activation=None,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=value,
                        padding=_get_padding(kernel_size, value),
                    )
                )
                for value in dilation
            ]
        )
        self.convs.apply(_init_weights)
        self.activations = nn.ModuleList(
            [_activation(activation, channels, h) for _ in self.convs]
        )

    def forward(self, x):
        for conv, activation in zip(self.convs, self.activations, strict=True):
            x = conv(activation(x)) + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class BigVGAN(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        if use_cuda_kernel:
            raise ValueError(
                "The minimal vendored BigVGAN supports the portable PyTorch "
                "activation only; set use_cuda_kernel=False."
            )
        self.h = h
        self.h["use_cuda_kernel"] = False
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Incorrect resblock class: {h.resblock}")
        self.ups = nn.ModuleList()
        for index, (rate, kernel) in enumerate(
            zip(h.upsample_rates, h.upsample_kernel_sizes, strict=True)
        ):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**index),
                                h.upsample_initial_channel // (2 ** (index + 1)),
                                kernel,
                                rate,
                                padding=(kernel - rate) // 2,
                            )
                        )
                    ]
                )
            )
        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = h.upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(
                h.resblock_kernel_sizes, h.resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(
                    resblock_class(
                        h,
                        channels,
                        kernel,
                        dilation,
                        activation=h.activation,
                    )
                )
        final_channels = h.upsample_initial_channel // (2 ** len(self.ups))
        self.activation_post = _activation(h.activation, final_channels, h)
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(
                final_channels,
                1,
                7,
                1,
                padding=3,
                bias=self.use_bias_at_final,
            )
        )
        for upsampler in self.ups:
            upsampler.apply(_init_weights)
        self.conv_post.apply(_init_weights)
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        x = self.conv_pre(x)
        for index in range(self.num_upsamples):
            for upsampler in self.ups[index]:
                x = upsampler(x)
            summed = None
            for kernel_index in range(self.num_kernels):
                value = self.resblocks[
                    index * self.num_kernels + kernel_index
                ](x)
                summed = value if summed is None else summed + value
            x = summed / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)

    def remove_weight_norm(self):
        try:
            for upsamplers in self.ups:
                for layer in upsamplers:
                    remove_weight_norm(layer)
            for layer in self.resblocks:
                layer.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            pass

    def _save_pretrained(self, save_directory: Path):
        torch.save(
            {"generator": self.state_dict()},
            save_directory / "bigvgan_generator.pt",
        )
        with (save_directory / "config.json").open("w", encoding="utf-8") as file:
            json.dump(self.h, file, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        del strict, model_kwargs
        if os.path.isdir(model_id):
            config_file = os.path.join(model_id, "config.json")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            common = {
                "repo_id": model_id,
                "revision": revision,
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "token": token,
                "local_files_only": local_files_only,
            }
            config_file = hf_hub_download(filename="config.json", **common)
            model_file = hf_hub_download(
                filename="bigvgan_generator.pt", **common
            )
        with open(config_file, encoding="utf-8") as file:
            h = AttrDict(json.load(file))
        model = cls(h, use_cuda_kernel=use_cuda_kernel)
        checkpoint = torch.load(model_file, map_location=map_location)
        try:
            model.load_state_dict(checkpoint["generator"])
        except RuntimeError:
            model.remove_weight_norm()
            model.load_state_dict(checkpoint["generator"])
        return model


class _BigVGANModule:
    """Compatibility shim for callers that used ``bigvgan.BigVGAN``."""

    BigVGAN = BigVGAN


bigvgan = _BigVGANModule()

__all__ = ["AttrDict", "BigVGAN", "bigvgan"]
