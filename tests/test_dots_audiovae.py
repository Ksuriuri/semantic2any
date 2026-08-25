from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from semantic2any.utils.dots_audiovae import (
    EXPECTED_HOP_SIZE,
    EXPECTED_LATENT_DIM,
    EXPECTED_SAMPLE_RATE,
    DotsAudioVAE,
    is_vae_latent_target,
    tiny_audio_vae_config,
)


def test_tiny_config_hop_and_roundtrip_shapes() -> None:
    config = tiny_audio_vae_config()
    assert int(torch.tensor(config.downsample_rates).prod()) == 4
    vae = DotsAudioVAE.from_config(config)
    assert vae.sample_rate == EXPECTED_SAMPLE_RATE
    assert vae.hop_size == 4
    assert vae.latent_dim == 8
    wav = torch.randn(2, 1, 48)
    latent = vae.encode_mean(wav)
    assert latent.shape == (2, 8, 12)
    decoded = vae.decode(latent)
    assert decoded.shape == (2, 1, 48)


def test_normalize_denormalize_is_invertible() -> None:
    vae = DotsAudioVAE.from_config(
        tiny_audio_vae_config(),
        latent_mean=torch.arange(8, dtype=torch.float32),
        latent_var=torch.linspace(0.5, 2.0, 8),
    )
    latent = torch.randn(2, 8, 5)
    restored = vae.denormalize(vae.normalize(latent))
    torch.testing.assert_close(restored, latent, atol=1e-5, rtol=1e-5)


def test_is_vae_latent_target() -> None:
    assert is_vae_latent_target(OmegaConf.create({"target": {"type": "vae_latent"}}))
    assert is_vae_latent_target({"target": {"type": "VAE_LATENT"}})
    assert not is_vae_latent_target(OmegaConf.create({}))
    assert not is_vae_latent_target(OmegaConf.create({"target": {"type": "mel"}}))


def test_from_pretrained_requires_vocoder_trio(tmp_path: Path) -> None:
    missing = tmp_path / "dots-tts"
    with pytest.raises(FileNotFoundError, match="vocoder"):
        DotsAudioVAE.from_pretrained(missing)
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        DotsAudioVAE.from_pretrained(missing)


def test_released_shape_constants() -> None:
    assert EXPECTED_SAMPLE_RATE == 48000
    assert EXPECTED_HOP_SIZE == 1920
    assert EXPECTED_LATENT_DIM == 128
