from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from scripts.infer_s2mel_zipformer import load_vocoder
from semantic2any.defaults import (
    DEFAULT_BIGVGAN_MODEL_ID,
    DEFAULT_MEL_CHANNELS,
    DEFAULT_MEL_HOP_LENGTH,
    DEFAULT_MEL_N_FFT,
    DEFAULT_MEL_SAMPLE_RATE,
    DEFAULT_MEL_WIN_LENGTH,
)


class _DefaultVocoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = {
            "sampling_rate": DEFAULT_MEL_SAMPLE_RATE,
            "num_mels": DEFAULT_MEL_CHANNELS,
            "n_fft": DEFAULT_MEL_N_FFT,
            "hop_size": DEFAULT_MEL_HOP_LENGTH,
            "win_size": DEFAULT_MEL_WIN_LENGTH,
        }

    def remove_weight_norm(self) -> None:
        pass


class DefaultAudioConfigTest(unittest.TestCase):
    def test_standard_configs_use_44khz_bigvgan(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative_path in ("configs/s2mel_zipformer.yaml", "configs/s2mel_dit.yaml"):
            with self.subTest(config=relative_path):
                cfg = OmegaConf.load(root / relative_path)
                self.assertEqual(cfg.vocoder.model_id, DEFAULT_BIGVGAN_MODEL_ID)
                self.assertEqual(cfg.preprocess_params.sr, DEFAULT_MEL_SAMPLE_RATE)
                self.assertEqual(
                    cfg.preprocess_params.spect_params.n_mels,
                    DEFAULT_MEL_CHANNELS,
                )
                self.assertEqual(
                    cfg.preprocess_params.spect_params.hop_length,
                    DEFAULT_MEL_HOP_LENGTH,
                )
                self.assertEqual(cfg.s2mel.DiT.in_channels, DEFAULT_MEL_CHANNELS)

    def test_inference_without_vocoder_config_uses_44khz_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"paths": {"model_dir": tmpdir}})
            with patch(
                "semantic2any.third_party.indextts.bigvgan.BigVGAN.from_pretrained",
                return_value=_DefaultVocoder(),
            ) as from_pretrained:
                load_vocoder(cfg, torch.device("cpu"), torch.float32)

        from_pretrained.assert_called_once_with(
            DEFAULT_BIGVGAN_MODEL_ID,
            cache_dir=None,
            local_files_only=False,
        )


if __name__ == "__main__":
    unittest.main()
