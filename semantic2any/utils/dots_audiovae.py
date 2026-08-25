"""Frozen dots.tts AudioVAE encode / decode wrapper."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from semantic2any.third_party.dots_tts import (
    DEFAULT_DOTS_AUDIO_VAE_CONFIG,
    AudioVAE,
    AudioVAEConfig,
)

VOCODER_WEIGHTS_NAME = "vocoder.safetensors"
LATENT_STATS_NAME = "latent_stats.pt"
CONFIG_NAME = "config.json"
DEFAULT_DOTS_TTS_DIR = "checkpoints/dots-tts"
EXPECTED_SAMPLE_RATE = 48000
EXPECTED_HOP_SIZE = 1920
EXPECTED_LATENT_DIM = 128


def tiny_audio_vae_config() -> AudioVAEConfig:
    """Small causal AudioVAE for unit tests (hop = 4, latent = 8)."""
    return AudioVAEConfig(
        sample_rate=48000,
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        upsample_initial_channel=32,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        downsample_rates=[2, 2],
        downsample_channels=[12, 24, 48],
        activation="snakebeta",
        snake_logscale=True,
        latent_dim=8,
        causal=True,
        mi_num_layers=1,
        causal_encoder=True,
        use_bias_at_final=False,
        use_tanh_at_final=False,
        num_decoder_lookahead=2,
        num_encoder_lookahead=2,
    )


def is_vae_latent_target(cfg: Any) -> bool:
    target_cfg = getattr(cfg, "target", None)
    if target_cfg is None and isinstance(cfg, dict):
        target_cfg = cfg.get("target")
    kind = getattr(target_cfg, "type", None)
    if kind is None and isinstance(target_cfg, dict):
        kind = target_cfg.get("type")
    return str(kind or "mel").strip().lower() == "vae_latent"


def resolve_dots_tts_dir(cfg: Any) -> Path:
    paths = getattr(cfg, "paths", None)
    if paths is None and isinstance(cfg, dict):
        paths = cfg.get("paths")
    raw = getattr(paths, "dots_tts_dir", None)
    if raw is None and isinstance(paths, dict):
        raw = paths.get("dots_tts_dir")
    return Path(str(raw or DEFAULT_DOTS_TTS_DIR)).expanduser()


def _as_1d(stat: torch.Tensor, latent_dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(stat, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 1:
        return tensor.repeat(latent_dim)
    if tensor.numel() != latent_dim:
        raise ValueError(
            f"latent_stats length must be 1 or {latent_dim}, got {tensor.numel()}"
        )
    return tensor


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state):
        return state
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state.items()
    }


def _load_vocoder_config(path: Path) -> AudioVAEConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocoder = payload.get("vocoder", payload)
    if not isinstance(vocoder, dict):
        raise ValueError(f"No vocoder config object in {path}")
    return AudioVAEConfig.from_dict(vocoder)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = (
            payload["state_dict"]
            if isinstance(payload, dict) and "state_dict" in payload
            else payload
        )
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported vocoder checkpoint format: {path}")
    tensors = {
        str(key): value for key, value in state.items() if torch.is_tensor(value)
    }
    for prefix in ("vocoder.", "module.vocoder.", "module."):
        tensors = _strip_prefix(tensors, prefix)
    return tensors


def _load_latent_stats(path: Path, latent_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"latent_stats must be a dict, got {type(payload)!r}")
    if "mean" not in payload:
        raise KeyError(f"{path} is missing 'mean'")
    variance = payload.get("var", payload.get("variance", payload.get("std")))
    if variance is None:
        raise KeyError(f"{path} is missing 'var' / 'variance' / 'std'")
    mean = _as_1d(payload["mean"], latent_dim)
    variance = _as_1d(variance, latent_dim)
    if "std" in payload and "var" not in payload and "variance" not in payload:
        variance = variance.square()
    return mean, variance.clamp_min(1e-8)


class DotsAudioVAE(nn.Module):
    """Encode 48 kHz audio to normalized 128-d / 25 Hz latents and decode back."""

    def __init__(
        self,
        vocoder: AudioVAE,
        *,
        latent_mean: torch.Tensor | None = None,
        latent_var: torch.Tensor | None = None,
        enforce_released_shapes: bool = True,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder.eval()
        for parameter in self.vocoder.parameters():
            parameter.requires_grad_(False)
        hop_size = int(self.vocoder.hop_size)
        sample_rate = int(self.vocoder.sample_rate)
        latent_dim = int(self.vocoder.h.latent_dim)
        if enforce_released_shapes:
            if sample_rate != EXPECTED_SAMPLE_RATE:
                raise ValueError(
                    f"AudioVAE sample_rate must be {EXPECTED_SAMPLE_RATE}, got {sample_rate}"
                )
            if hop_size != EXPECTED_HOP_SIZE:
                raise ValueError(
                    f"AudioVAE hop_size must be {EXPECTED_HOP_SIZE}, got {hop_size}"
                )
            if latent_dim != EXPECTED_LATENT_DIM:
                raise ValueError(
                    f"AudioVAE latent_dim must be {EXPECTED_LATENT_DIM}, got {latent_dim}"
                )
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        mean = (
            torch.zeros(latent_dim)
            if latent_mean is None
            else _as_1d(latent_mean, latent_dim)
        )
        variance = (
            torch.ones(latent_dim)
            if latent_var is None
            else _as_1d(latent_var, latent_dim).clamp_min(1e-8)
        )
        self.register_buffer("latent_mean", mean.view(1, latent_dim, 1), persistent=True)
        self.register_buffer("latent_var", variance.view(1, latent_dim, 1), persistent=True)

    @classmethod
    def from_config(
        cls,
        config: AudioVAEConfig | None = None,
        *,
        latent_mean: torch.Tensor | None = None,
        latent_var: torch.Tensor | None = None,
        remove_weight_norm: bool = True,
        enforce_released_shapes: bool | None = None,
    ) -> DotsAudioVAE:
        config = config or DEFAULT_DOTS_AUDIO_VAE_CONFIG
        if enforce_released_shapes is None:
            enforce_released_shapes = (
                math.prod(int(rate) for rate in config.downsample_rates) == EXPECTED_HOP_SIZE
            )
        vocoder = AudioVAE(config)
        if remove_weight_norm:
            vocoder.remove_weight_norm()
        vocoder.eval()
        return cls(
            vocoder,
            latent_mean=latent_mean,
            latent_var=latent_var,
            enforce_released_shapes=enforce_released_shapes,
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> DotsAudioVAE:
        root = Path(path).expanduser()
        if not root.exists():
            raise FileNotFoundError(
                f"dots.tts AudioVAE directory does not exist: {root}. "
                "Download config.json, vocoder.safetensors, and latent_stats.pt "
                "from dots-studio/dots.tts-soar (or dots.tts-base) into this path."
            )
        config_path = root / CONFIG_NAME
        weights_path = root / VOCODER_WEIGHTS_NAME
        stats_path = root / LATENT_STATS_NAME
        missing = [
            str(item)
            for item in (config_path, weights_path, stats_path)
            if not item.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "dots.tts AudioVAE directory is missing "
                + ", ".join(missing)
                + ". Only the vocoder trio is required; do not download the 2B LLM."
            )
        config = _load_vocoder_config(config_path)
        vocoder = AudioVAE(config)
        state = _load_state_dict(weights_path)
        try:
            vocoder.load_state_dict(state, strict=True)
        except RuntimeError:
            vocoder.remove_weight_norm()
            vocoder.load_state_dict(state, strict=True)
        else:
            vocoder.remove_weight_norm()
        vocoder.eval()
        mean, variance = _load_latent_stats(stats_path, int(config.latent_dim))
        wrapper = cls(vocoder, latent_mean=mean, latent_var=variance)
        return wrapper.to(map_location)

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return (
            latent - self.latent_mean.to(device=latent.device, dtype=latent.dtype)
        ) / torch.sqrt(self.latent_var.to(device=latent.device, dtype=latent.dtype))

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * torch.sqrt(
            self.latent_var.to(device=latent.device, dtype=latent.dtype)
        ) + self.latent_mean.to(device=latent.device, dtype=latent.dtype)

    def _as_wave_batch(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if wav.ndim != 3 or wav.size(1) != 1:
            raise ValueError(
                f"waveform must be [T], [B, T], or [B, 1, T], got {tuple(wav.shape)}"
            )
        return wav.float()

    def _pad_to_hop(self, wav: torch.Tensor) -> torch.Tensor:
        remainder = wav.size(-1) % self.hop_size
        if remainder == 0:
            return wav
        return F.pad(wav, (0, self.hop_size - remainder))

    @torch.no_grad()
    def encode_mean(
        self,
        wav: torch.Tensor,
        *,
        normalize: bool = True,
        sample_lengths: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        """Return posterior-mean latents as [B, C, T], optionally stats-normalized."""
        wav = self._as_wave_batch(wav)
        if sample_lengths is None:
            lengths = [int(wav.size(-1))] * int(wav.size(0))
        else:
            lengths = [int(value) for value in sample_lengths]
            if len(lengths) != wav.size(0):
                raise ValueError("sample_lengths must match the waveform batch size")
        wav = self._pad_to_hop(wav)
        posterior = self.vocoder.extract_latents(wav, do_sample=False)
        mean, _log_std = torch.split(posterior, self.latent_dim, dim=1)
        keep = max(length // self.hop_size for length in lengths)
        if keep <= 0:
            raise ValueError(
                f"waveform is shorter than one VAE hop ({self.hop_size} samples)"
            )
        mean = mean[..., :keep]
        for index, length in enumerate(lengths):
            valid = length // self.hop_size
            if valid < keep:
                mean[index, :, valid:] = 0
        if normalize:
            mean = self.normalize(mean)
        return mean

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, *, normalized: bool = True) -> torch.Tensor:
        """Decode [B, C, T] or [B, T, C] latents to [B, 1, samples] at 48 kHz."""
        if latent.ndim != 3:
            raise ValueError(f"latent must be 3D, got {tuple(latent.shape)}")
        if latent.size(1) == self.latent_dim:
            latent_bt = latent
        elif latent.size(-1) == self.latent_dim:
            latent_bt = latent.transpose(1, 2)
        else:
            raise ValueError(
                f"latent channel count must be {self.latent_dim}, got {tuple(latent.shape)}"
            )
        if normalized:
            latent_bt = self.denormalize(latent_bt)
        return self.vocoder.inference_from_latents(latent_bt, do_sample=False)
