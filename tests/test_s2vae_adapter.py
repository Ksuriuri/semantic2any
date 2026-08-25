from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn

from semantic2any.utils.dots_audiovae import DotsAudioVAE, tiny_audio_vae_config
from semantic2any.utils.indextts_adapters import S2MelFeatureAdapter


class _FakeSemanticCodec(nn.Module):
    def extract(self, waveforms):
        outputs = []
        for waveform in waveforms:
            length = int(waveform.shape[-1]) if hasattr(waveform, "shape") else len(waveform)
            frames = max(1, int(round(length / 16000 * 50)))
            outputs.append(torch.randn(frames, 1024))
        return outputs


class _StubAudioVAE(nn.Module):
    sample_rate = 48000
    hop_size = 1920
    latent_dim = 128

    def encode_mean(self, wav, *, normalize=True, sample_lengths=None):
        del normalize
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        batch, _, samples = wav.shape
        if sample_lengths is None:
            sample_lengths = [samples] * batch
        keep = max(max(int(length) // self.hop_size for length in sample_lengths), 1)
        latent = torch.zeros(batch, self.latent_dim, keep, device=wav.device, dtype=wav.dtype)
        for index, length in enumerate(sample_lengths):
            valid = int(length) // self.hop_size
            if valid < keep:
                latent[index, :, valid:] = 0
        return latent


def _patch_sac(monkeypatch) -> None:
    import semantic2any.utils.semantic_codecs as codecs

    monkeypatch.setattr(codecs, "semantic_codec_type", lambda _cfg: "sac")
    monkeypatch.setattr(
        codecs, "build_semantic_codec", lambda _cfg, model_dir: _FakeSemanticCodec()
    )


def _vae_cfg(*, in_channels: int, extract_mel_in_worker: bool = False, tmp_path) -> OmegaConf:
    return OmegaConf.create(
        {
            "target": {"type": "vae_latent"},
            "paths": {"model_dir": str(tmp_path / "missing")},
            "semantic_codec": {"type": "sac"},
            "vocoder": {"model_id": "nvidia/bigvgan_v2_22khz_80band_256x"},
            "data": {
                "sample_rate_vae": 48000,
                "extract_mel_in_worker": extract_mel_in_worker,
            },
            "preprocess_params": {
                "sr": 22050,
                "spect_params": {
                    "n_fft": 1024,
                    "n_mels": 80,
                    "hop_length": 256,
                    "win_length": 1024,
                    "fmin": 0,
                    "fmax": None,
                },
            },
            "s2mel": {
                "dit_type": "DiT",
                "length_ratio": 0.5,
                "style_encoder": {"dim": 192},
                "DiT": {"in_channels": in_channels, "style_condition": False},
            },
        }
    )


def test_worker_mel_extraction_is_rejected(monkeypatch, tmp_path) -> None:
    _patch_sac(monkeypatch)
    cfg = _vae_cfg(in_channels=8, extract_mel_in_worker=True, tmp_path=tmp_path)
    vae = DotsAudioVAE.from_config(tiny_audio_vae_config())
    try:
        S2MelFeatureAdapter(cfg, audio_vae=vae)
    except ValueError as exc:
        assert "extract_mel_in_worker" in str(exc)
    else:
        raise AssertionError("expected extract_mel_in_worker to be rejected")


def test_in_channels_must_match_latent_dim(monkeypatch, tmp_path) -> None:
    _patch_sac(monkeypatch)
    cfg = _vae_cfg(in_channels=80, tmp_path=tmp_path)
    vae = DotsAudioVAE.from_config(tiny_audio_vae_config())
    try:
        S2MelFeatureAdapter(cfg, audio_vae=vae)
    except ValueError as exc:
        assert "in_channels" in str(exc)
    else:
        raise AssertionError("expected in_channels mismatch to be rejected")


def test_tiny_vae_encode_keeps_batch_key_mel(monkeypatch, tmp_path) -> None:
    _patch_sac(monkeypatch)
    cfg = _vae_cfg(in_channels=8, tmp_path=tmp_path)
    vae = DotsAudioVAE.from_config(tiny_audio_vae_config())
    adapter = S2MelFeatureAdapter(cfg, audio_vae=vae)
    waveform = torch.randn(1, 48)
    features = adapter.extract_utterance_features(
        ["clip.wav"],
        waveforms=[waveform],
        sample_rates=[48000],
    )
    assert features[0]["mel"].shape[0] == 8
    assert features[0]["mel"].shape[-1] == 12


def test_semantic_50hz_aligns_to_latent_25hz(monkeypatch, tmp_path) -> None:
    _patch_sac(monkeypatch)
    cfg = _vae_cfg(in_channels=128, tmp_path=tmp_path)
    adapter = S2MelFeatureAdapter(cfg, audio_vae=_StubAudioVAE())
    waveform = torch.randn(1, 48000)
    features = adapter.extract_utterance_features(
        ["clip.wav"],
        waveforms=[waveform],
        sample_rates=[48000],
    )
    latent = features[0]["mel"]
    semantic = features[0]["semantic"]
    assert latent.shape == (128, 25)
    assert semantic.shape[0] == 50
    assert adapter.acoustic_hop_size == 1920
    assert adapter.sample_rate_acoustic == 48000
    assert adapter.acoustic_channels == 128
