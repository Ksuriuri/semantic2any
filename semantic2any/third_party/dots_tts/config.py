"""AudioVAE hyperparameters, flattened from dots.tts without pydantic."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


def _as_int_lists(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return [list(int(x) for x in row) for row in value]
        return [int(x) for x in value]
    return list(value)


@dataclass
class AudioVAEConfig:
    sample_rate: int = 48000
    upsample_rates: list[int] = field(default_factory=list)
    upsample_kernel_sizes: list[int] = field(default_factory=list)
    upsample_initial_channel: int = 1536
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = field(default_factory=list)
    resblock_dilation_sizes: list[list[int]] = field(default_factory=list)
    downsample_rates: list[int] = field(default_factory=list)
    downsample_channels: list[int] = field(default_factory=list)
    activation: str = "snakebeta"
    snake_logscale: bool = True
    latent_dim: int = 128
    causal: bool = True
    mi_num_layers: int = 4
    causal_encoder: bool = True
    use_bias_at_final: bool = False
    use_tanh_at_final: bool = False
    num_decoder_lookahead: int = 2
    num_encoder_lookahead: int = 2
    mi_skip: bool = True

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AudioVAEConfig:
        payload = dict(data or {})
        known = {item.name for item in fields(cls)}
        aliases = {
            "upsample_kernel_sizes": "upsample_kernel_sizes",
            "upsample_kernel_size": "upsample_kernel_sizes",
        }
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            name = aliases.get(key, key)
            if name not in known:
                continue
            if name in {
                "upsample_rates",
                "upsample_kernel_sizes",
                "resblock_kernel_sizes",
                "downsample_rates",
                "downsample_channels",
            }:
                kwargs[name] = _as_int_lists(value)
            elif name == "resblock_dilation_sizes":
                kwargs[name] = _as_int_lists(value)
            else:
                kwargs[name] = value
        return cls(**kwargs)


DEFAULT_DOTS_AUDIO_VAE_CONFIG = AudioVAEConfig(
    sample_rate=48000,
    upsample_rates=[10, 6, 4, 2, 2, 2],
    upsample_kernel_sizes=[20, 12, 8, 4, 4, 4],
    upsample_initial_channel=1536,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    downsample_rates=[2, 2, 2, 4, 6, 10],
    downsample_channels=[12, 24, 48, 96, 192, 384, 768],
    activation="snakebeta",
    snake_logscale=True,
    latent_dim=128,
    causal=True,
    mi_num_layers=4,
    causal_encoder=True,
    use_bias_at_final=False,
    use_tanh_at_final=False,
    num_decoder_lookahead=2,
    num_encoder_lookahead=2,
)


__all__ = ["AudioVAEConfig", "DEFAULT_DOTS_AUDIO_VAE_CONFIG"]
