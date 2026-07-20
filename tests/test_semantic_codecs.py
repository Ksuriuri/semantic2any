from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from semantic2any.data.s2mel_dataset import S2MelCollator
from semantic2any.third_party.sac_whisper.modeling import WhisperVQSemanticEncoder
from semantic2any.utils.checkpoint import load_compatible_checkpoint
from semantic2any.utils.semantic_codecs import (
    SACSemanticCodec,
    SEMANTIC_CODEC_SPECS,
    prepare_feature_metadata,
    resolve_semantic_codec_config,
)
from transformers import WhisperConfig


def _minimal_cfg():
    return OmegaConf.create(
        {
            "semantic_codec": {"type": "maskgct"},
            "data": {"semantic_fps": 50.0, "sample_rate_16k": 16000},
            "s2mel": {
                "length_regulator": {
                    "is_discrete": False,
                    "in_channels": 1024,
                }
            },
        }
    )


def test_codec_switch_resolves_rate_and_dimension():
    cfg = _minimal_cfg()
    maskgct = resolve_semantic_codec_config(cfg)
    assert maskgct.name == "maskgct"
    assert cfg.s2mel.length_regulator.in_channels == 1024
    assert cfg.data.semantic_fps == 50.0

    sac = resolve_semantic_codec_config(cfg, "sac")
    assert sac.name == "sac"
    assert cfg.s2mel.length_regulator.in_channels == 1280
    assert cfg.data.semantic_fps == 12.5
    assert cfg.s2mel.length_regulator.is_discrete is False


def test_minimal_whisper_vq_preserves_masked_lengths_and_embedding_shape():
    cfg = WhisperConfig(
        d_model=8,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        encoder_layers=2,
        num_mel_bins=4,
        max_source_positions=20,
    )
    for key, value in {
        "encoder_causal_convolution": True,
        "quantize_position": 2,
        "pooling_position": 2,
        "pooling_kernel_size": 4,
        "pooling_type": "avg",
        "quantize_vocab_size": 16,
        "quantize_causal_block_size": 8,
    }.items():
        setattr(cfg, key, value)
    model = WhisperVQSemanticEncoder(cfg).eval()
    ids, embeddings, mask = model(
        torch.randn(2, 4, 24),
        torch.tensor([[1] * 24, [1] * 16 + [0] * 8]),
    )
    assert ids.shape == (2, 3)
    assert embeddings.shape == (2, 3, 8)
    assert mask.sum(dim=1).tolist() == [3, 2]


class _FeatureBatch:
    def __init__(self, input_features: torch.Tensor, attention_mask: torch.Tensor):
        self.input_features = input_features
        self.attention_mask = attention_mask


class _FakeFeatureExtractor:
    hop_length = 1

    def __call__(self, waveforms, **_kwargs):
        maximum = max(len(waveform) for waveform in waveforms)
        features = torch.zeros(len(waveforms), 1, maximum)
        mask = torch.zeros(len(waveforms), maximum, dtype=torch.long)
        for index, waveform in enumerate(waveforms):
            features[index, 0, : len(waveform)] = torch.from_numpy(waveform)
            mask[index, : len(waveform)] = 1
        return _FeatureBatch(features, mask)


class _FakeSemanticEncoder(nn.Module):
    pooling_kernel_size = 1

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.conv1 = SimpleNamespace(stride=(1,))
        self.conv2 = SimpleNamespace(stride=(1,))

    def forward(self, input_features, attention_mask):
        batch, _, length = input_features.shape
        ids = torch.zeros(batch, length, dtype=torch.long, device=input_features.device)
        # Code 0 is represented by an all-zero embedding and must not be treated as padding.
        embeddings = torch.zeros(batch, length, 1280, device=input_features.device)
        return ids, embeddings, attention_mask.bool()


def test_sac_chunking_uses_explicit_mask_not_token_value():
    codec = SACSemanticCodec.__new__(SACSemanticCodec)
    nn.Module.__init__(codec)
    codec.feature_extractor = _FakeFeatureExtractor()
    codec.encoder = _FakeSemanticEncoder()
    codec._info = SEMANTIC_CODEC_SPECS["sac"]
    codec.max_chunk_samples = 4
    codec.chunk_batch_size = 2

    features = codec.extract([np.ones(10, dtype=np.float32)])[0]
    assert features.shape == (10, 1280)
    assert torch.count_nonzero(features) == 0


def test_precompute_metadata_rejects_codec_reuse(tmp_path):
    cfg = _minimal_cfg()
    resolve_semantic_codec_config(cfg, "maskgct")
    first = prepare_feature_metadata(tmp_path, cfg, overwrite=False)
    assert first["name"] == "maskgct"
    (tmp_path / "feats").mkdir()

    resolve_semantic_codec_config(cfg, "sac")
    with pytest.raises(ValueError, match="codec mismatch"):
        prepare_feature_metadata(tmp_path, cfg, overwrite=False)
    replaced = prepare_feature_metadata(tmp_path, cfg, overwrite=True)
    assert replaced["name"] == "sac"


def test_collator_rejects_manifest_codec_mismatch():
    collator = S2MelCollator(
        hop_length=256,
        sample_rate=22050,
        min_prompt_seconds=1.0,
        max_prompt_seconds=5.0,
        min_generated_frames=8,
        expected_semantic_codec="sac",
        expected_semantic_fingerprint="expected",
    )
    with pytest.raises(ValueError, match="manifest=maskgct"):
        collator([{"semantic_codec": "maskgct"}])


def test_checkpoint_dimension_mismatch_has_codec_error(tmp_path):
    class _Model:
        def __init__(self):
            self.models = nn.ModuleDict(
                {"length_regulator": nn.ModuleDict({"content_in_proj": nn.Linear(1280, 8)})}
            )

    checkpoint = tmp_path / "maskgct.pth"
    torch.save(
        {
            "net": {
                "length_regulator": {
                    "content_in_proj.weight": torch.zeros(8, 1024),
                    "content_in_proj.bias": torch.zeros(8),
                }
            }
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="checkpoint input dim=1024"):
        load_compatible_checkpoint(_Model(), checkpoint)
