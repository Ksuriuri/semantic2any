from pathlib import Path

import torch
from omegaconf import OmegaConf

from semantic2any.third_party.indextts.bigvgan import AttrDict, BigVGAN
from semantic2any.third_party.indextts.campplus import CAMPPlus
from semantic2any.third_party.indextts.audio import (
    mel_spectrogram,
    mel_spectrogram_batch,
)
from semantic2any.third_party.indextts.maskgct import RepCodec
from semantic2any.utils.indextts_adapters import (
    S2MelFeatureAdapter,
    _uses_style_condition,
)


def test_runtime_files_do_not_import_external_indextts_package() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_files = (
        root / "semantic2any/utils/indextts_adapters.py",
        root / "semantic2any/utils/semantic_codecs.py",
        root / "scripts/infer_s2mel_zipformer.py",
    )
    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        assert "from indextts" not in source
        assert "import indextts" not in source


def test_variable_length_batched_mel_matches_single_waveforms() -> None:
    torch.manual_seed(1234)
    waveforms = [torch.randn(1, length) for length in (8192, 10001, 12288)]
    mel_args = {
        "n_fft": 2048,
        "num_mels": 128,
        "sampling_rate": 44100,
        "hop_size": 512,
        "win_size": 2048,
        "fmin": 0.0,
        "fmax": None,
        "center": False,
    }
    expected = [
        mel_spectrogram(waveform, **mel_args).squeeze(0)
        for waveform in waveforms
    ]
    actual = mel_spectrogram_batch(waveforms, batch_size=2, **mel_args)

    assert [mel.shape for mel in actual] == [mel.shape for mel in expected]
    for batched, single in zip(actual, expected, strict=True):
        torch.testing.assert_close(batched, single)


def test_campplus_state_dict_strict_round_trip() -> None:
    source = CAMPPlus(feat_dim=80, embedding_size=192)
    target = CAMPPlus(feat_dim=80, embedding_size=192)
    target.load_state_dict(source.state_dict(), strict=True)
    assert target.xvector.dense.linear.out_channels == 192


def test_repcodec_uses_published_maskgct_shape() -> None:
    codec = RepCodec()
    state = codec.state_dict()
    assert state["encoder.0.embed.weight"].shape == (384, 1024, 7)
    assert state["quantizer.quantizers.0.codebook.weight"].shape == (8192, 8)
    assert state["decoder.1.weight"].shape == (1024, 384)


def test_bigvgan_portable_runtime_forward() -> None:
    config = AttrDict(
        {
            "num_mels": 4,
            "upsample_initial_channel": 8,
            "upsample_rates": [2],
            "upsample_kernel_sizes": [4],
            "resblock": "1",
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3, 5]],
            "activation": "snakebeta",
            "snake_logscale": True,
        }
    )
    model = BigVGAN(config)
    output = model(torch.randn(1, 4, 4))
    assert output.shape == (1, 1, 8)


def test_no_style_config_does_not_require_campplus() -> None:
    cfg = OmegaConf.create(
        {
            "s2mel": {
                "dit_type": "ZipFormer",
                "ZipFormer": {"style_condition": False},
            }
        }
    )
    assert not _uses_style_condition(cfg)


def test_no_style_sac_does_not_require_model_asset_dir(
    monkeypatch, tmp_path: Path
) -> None:
    import semantic2any.utils.semantic_codecs as codecs

    monkeypatch.setattr(codecs, "semantic_codec_type", lambda _cfg: "sac")
    monkeypatch.setattr(
        codecs, "build_semantic_codec", lambda _cfg, model_dir: torch.nn.Identity()
    )
    cfg = OmegaConf.create(
        {
            "paths": {"model_dir": str(tmp_path / "missing")},
            "semantic_codec": {"type": "sac"},
            "data": {},
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
                "dit_type": "ZipFormer",
                "style_encoder": {"dim": 192},
                "DiT": {"in_channels": 80},
                "ZipFormer": {"style_condition": False},
            },
        }
    )
    adapter = S2MelFeatureAdapter(cfg)
    assert adapter.campplus_model is None
