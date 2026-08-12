"""IndexTTS-2.5 (EnhancedCodec) semantic code path.

The properties worth pinning are the ones MaskGCT never had: codes run at half
the feature rate, and the decode is context dependent, so decoding must happen
per unpadded utterance instead of over a padded batch.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from semantic2any.third_party.indextts.maskgct import RepCodec, build_semantic_codec
from semantic2any.utils.indextts_adapters import IndexTTSFeatureAdapter
from semantic2any.data.s2mel_dataset import (
    S2MelCollator,
    _record_has_singleton_semantic_budget,
)
from semantic2any.utils.semantic_codecs import (
    DEFAULT_SEMANTIC_CODEC,
    SEMANTIC_CODEC_FRAMES_PER_CODE,
    SEMANTIC_CODEC_SPECS,
    IndexTTS25CodeDecoder,
    IndexTTS25SemanticCodec,
    MaskGCTCodebookDecoder,
    SemanticCodecInfo,
    SemanticCodeDecoder,
    build_semantic_code_decoder,
    canonical_semantic_codec,
    semantic_codec_type,
    semantic_feature_fps,
    semantic_frames_per_code,
)

TINY_ARCH = {
    "codebook_size": 32,
    "hidden_size": 8,
    "codebook_dim": 4,
    "vocos_dim": 8,
    "vocos_intermediate_dim": 16,
    "vocos_num_layers": 2,
    "num_quantizers": 1,
    "downsample_scale": 2,
}


def _tiny_codec() -> RepCodec:
    torch.manual_seed(0)
    return RepCodec(**TINY_ARCH).eval()


def _bundle(codec: RepCodec) -> dict:
    state = {
        key: value.detach().clone()
        for key, value in codec.state_dict().items()
        if key.startswith(IndexTTS25SemanticCodec.DECODE_PREFIXES)
    }
    return {
        "codec_type": "indextts25",
        "frames_per_code": 2,
        "semantic_dim": TINY_ARCH["hidden_size"],
        "arch": dict(TINY_ARCH),
        "state_dict": state,
        "source_checkpoint": "unit-test",
        "source_checkpoint_sha256": "0" * 64,
    }


class RepCodecDecodeTest(unittest.TestCase):
    def test_quantize_halves_and_decode_doubles_the_rate(self) -> None:
        codec = _tiny_codec()
        feature = torch.randn(1, 20, TINY_ARCH["hidden_size"])
        with torch.no_grad():
            codes, _ = codec.quantize(feature)
            decoded = codec.decode(codes)
        self.assertEqual(tuple(codes.shape), (1, 10))
        self.assertEqual(tuple(decoded.shape), (1, 20, TINY_ARCH["hidden_size"]))

    def test_decode_accepts_two_and_three_dim_codes(self) -> None:
        codec = _tiny_codec()
        codes = torch.randint(0, TINY_ARCH["codebook_size"], (1, 6))
        with torch.no_grad():
            flat = codec.decode(codes)
            stacked = codec.decode(codes.unsqueeze(0))
        self.assertTrue(torch.equal(flat, stacked))
        with self.assertRaises(ValueError):
            codec.decode(codes[0])[0]

    def test_downsample_scale_override_must_agree_with_the_config(self) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({**TINY_ARCH, "downsample_scale": 1})
        with self.assertRaises(ValueError):
            build_semantic_codec(cfg, downsample_scale=2)
        codec = build_semantic_codec(OmegaConf.create({**TINY_ARCH}), downsample_scale=2)
        self.assertEqual(codec.downsample_scale, 2)

    def test_padding_changes_the_decode_so_slicing_must_come_first(self) -> None:
        codec = _tiny_codec()
        codes = torch.randint(1, TINY_ARCH["codebook_size"], (1, 12))
        with torch.no_grad():
            clean = codec.decode(codes[:, :6])
            padded = codec.decode(
                torch.cat([codes[:, :6], torch.zeros(1, 6, dtype=torch.long)], dim=-1)
            )
        self.assertFalse(torch.allclose(clean, padded[:, :12], atol=1e-4))


class DecoderArtifactTest(unittest.TestCase):
    def test_bundle_roundtrip_matches_the_source_codec(self) -> None:
        codec = _tiny_codec()
        codes = torch.randint(0, TINY_ARCH["codebook_size"], (11,))
        with torch.no_grad():
            expected = codec.decode(codes.reshape(1, -1))[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "indextts25_decoder.pt"
            torch.save(_bundle(codec), path)
            decoder = build_semantic_code_decoder(path, expected_codec="indextts25")
            self.assertIsInstance(decoder, IndexTTS25CodeDecoder)
            self.assertEqual(decoder.frames_per_code, 2)
            actual = decoder.decode_sequences([codes])[0]
        self.assertEqual(tuple(actual.shape), (22, TINY_ARCH["hidden_size"]))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_decode_sequences_is_independent_of_the_other_sequences(self) -> None:
        codec = _tiny_codec()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.pt"
            torch.save(_bundle(codec), path)
            decoder = build_semantic_code_decoder(path)
            short = torch.randint(0, TINY_ARCH["codebook_size"], (4,))
            long = torch.randint(0, TINY_ARCH["codebook_size"], (17,))
            alone = decoder.decode_sequences([short])[0]
            together = decoder.decode_sequences([long, short, long])[1]
        self.assertTrue(torch.equal(alone, together))

    def test_lookup_artifact_still_selects_the_maskgct_decoder(self) -> None:
        lookup = torch.zeros(8192, SEMANTIC_CODEC_SPECS["maskgct"].semantic_dim)
        lookup[7] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "maskgct_lookup.pt"
            torch.save({"lookup": lookup}, path)
            decoder = build_semantic_code_decoder(path, expected_codec="maskgct")
            self.assertIsInstance(decoder, MaskGCTCodebookDecoder)
            self.assertEqual(decoder.frames_per_code, 1)
            decoded = decoder.decode_sequences([torch.tensor([7, 0]), torch.tensor([7])])
            self.assertEqual([item.shape[0] for item in decoded], [2, 1])
            self.assertEqual(decoded[0][0, 0].item(), 1.0)
            with self.assertRaises(ValueError):
                build_semantic_code_decoder(path, expected_codec="indextts25")

    def test_bundle_rejects_a_mismatched_config_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.pt"
            torch.save(_bundle(_tiny_codec()), path)
            with self.assertRaises(ValueError):
                build_semantic_code_decoder(path, expected_codec="maskgct")

    def test_bundle_checksum_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.pt"
            torch.save(_bundle(_tiny_codec()), path)
            with self.assertRaises(ValueError):
                IndexTTS25CodeDecoder(path, expected_sha256="f" * 64)

    def test_there_is_no_codebook_lookup_for_a_context_dependent_decoder(self) -> None:
        with self.assertRaises(TypeError):
            IndexTTS25SemanticCodec.codebook_lookup(object())


class WorkerBatchRateTest(unittest.TestCase):
    """finalize_worker_paired_batch must convert code counts to frame counts."""

    @staticmethod
    def _adapter(frames_per_code: int) -> IndexTTSFeatureAdapter:
        class Stub(SemanticCodeDecoder):
            def __init__(self) -> None:
                super().__init__()
                self.frames_per_code = frames_per_code

            def forward(self, codes: torch.Tensor) -> torch.Tensor:
                repeated = codes.float().repeat_interleave(frames_per_code, dim=-1)
                return repeated.unsqueeze(-1)

        adapter = IndexTTSFeatureAdapter.__new__(IndexTTSFeatureAdapter)
        nn.Module.__init__(adapter)
        adapter.semantic_mean = torch.zeros(1)
        adapter.semantic_backend = None
        adapter.semantic_decoder = Stub()
        return adapter

    @staticmethod
    def _batch() -> dict:
        return {
            "semantic": torch.tensor([[1, 2, 3, 4, 5]]),
            "semantic_lens": torch.tensor([5]),
            "prompt_semantic_lens": torch.tensor([2]),
            "mel": torch.zeros(1, 4, 6),
            "mel_lens": torch.tensor([6]),
        }

    def test_maskgct_rate_is_unchanged(self) -> None:
        out = self._adapter(1).finalize_worker_paired_batch(self._batch())
        self.assertEqual(out["semantic_lens"].tolist(), [5])
        self.assertEqual(out["prompt_semantic_lens"].tolist(), [2])
        self.assertEqual(out["semantic"][0, :, 0].tolist(), [1, 2, 3, 4, 5])

    def test_indextts25_lengths_are_scaled_and_the_seam_is_not_decoded_across(self) -> None:
        out = self._adapter(2).finalize_worker_paired_batch(self._batch())
        self.assertEqual(out["semantic_lens"].tolist(), [10])
        self.assertEqual(out["prompt_semantic_lens"].tolist(), [4])
        self.assertEqual(
            out["semantic"][0, :, 0].tolist(),
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        )

    def test_prompt_length_must_split_the_pair(self) -> None:
        batch = self._batch()
        batch["prompt_semantic_lens"] = torch.tensor([5])
        with self.assertRaises(ValueError):
            self._adapter(2).finalize_worker_paired_batch(batch)

    def test_worker_batches_must_carry_the_prompt_split(self) -> None:
        batch = self._batch()
        del batch["prompt_semantic_lens"]
        with self.assertRaises(ValueError):
            self._adapter(2).finalize_worker_paired_batch(batch)


class SelectorTest(unittest.TestCase):
    def test_indextts25_is_the_default_and_maskgct_stays_selectable(self) -> None:
        self.assertEqual(DEFAULT_SEMANTIC_CODEC, "indextts25")
        self.assertEqual(semantic_codec_type(None), "indextts25")
        self.assertEqual(semantic_codec_type({}), "indextts25")
        self.assertEqual(
            semantic_codec_type({"semantic_codec": {"type": "maskgct"}}), "maskgct"
        )
        with self.assertRaises(ValueError):
            semantic_codec_type({"semantic_codec": {"type": "nope"}})

    def test_codes_are_25hz_but_features_stay_at_50hz(self) -> None:
        info = SEMANTIC_CODEC_SPECS["indextts25"]
        self.assertEqual(info.semantic_fps, 25.0)
        self.assertEqual(info.semantic_dim, 1024)
        self.assertFalse(info.is_discrete)
        self.assertEqual(semantic_frames_per_code("indextts25"), 2)
        self.assertEqual(semantic_feature_fps(info), 50.0)
        self.assertEqual(semantic_feature_fps(SEMANTIC_CODEC_SPECS["maskgct"]), 50.0)

    def test_every_codec_declares_its_code_rate(self) -> None:
        self.assertEqual(
            set(SEMANTIC_CODEC_SPECS), set(SEMANTIC_CODEC_FRAMES_PER_CODE)
        )

    def test_semantic_codec_info_fields_are_frozen(self) -> None:
        # fingerprint() hashes these fields and every existing manifest carries
        # the fingerprint of exactly this field set; adding one invalidates them.
        self.assertEqual(
            [field.name for field in dataclasses.fields(SemanticCodecInfo)],
            [
                "name",
                "semantic_dim",
                "semantic_fps",
                "sample_rate",
                "is_discrete",
                "source_model",
                "tokenizer_model",
                "revision",
            ],
        )
        maskgct = SEMANTIC_CODEC_SPECS["maskgct"]
        self.assertEqual(
            (maskgct.semantic_dim, maskgct.semantic_fps, maskgct.sample_rate),
            (1024, 50.0, 16000),
        )


class ManifestSpellingTest(unittest.TestCase):
    """The code-generation workers stamp "indextts2.5" and semantic_frame_rate."""

    @staticmethod
    def _collator(expected: str) -> S2MelCollator:
        collator = S2MelCollator.__new__(S2MelCollator)
        collator.expected_semantic_codec = expected
        collator.expected_semantic_fingerprint = None
        return collator

    def test_dotted_spelling_is_the_same_codec(self) -> None:
        for spelling in ("indextts2.5", "IndexTTS-2.5", "indextts_2.5", "EnhancedCodec"):
            self.assertEqual(canonical_semantic_codec(spelling), "indextts25")
        self.assertEqual(canonical_semantic_codec("maskgct"), "maskgct")
        self.assertEqual(canonical_semantic_codec(""), "")

    def test_worker_manifests_validate_against_the_config_selector(self) -> None:
        collator = self._collator("indextts25")
        collator._validate_precomputed_metadata([{"semantic_codec": "indextts2.5"}])
        with self.assertRaises(ValueError):
            collator._validate_precomputed_metadata([{"semantic_codec": "maskgct"}])

    def test_a_batch_may_mix_spellings_of_one_codec(self) -> None:
        collator = self._collator("indextts25")
        collator.max_audio_seconds = None
        records = [
            {
                "semantic_lookup_path": "a.pt",
                "semantic_lookup_sha256": "0" * 64,
                "semantic_codec": spelling,
            }
            for spelling in ("indextts2.5", "indextts25")
        ]
        metadata = collator._semantic_code_batch_metadata(records)
        self.assertEqual(metadata["semantic_lookup_path"], "a.pt")

    def test_semantic_frame_rate_is_read_when_semantic_fps_is_absent(self) -> None:
        # 100 codes at 25 Hz is 4 s of audio: enough for 1 s + 2 s, and the
        # 50 Hz default would wrongly call it 2 s and drop the record.
        record = {"semantic_code_length": 100, "semantic_frame_rate": 25.0}
        self.assertTrue(
            _record_has_singleton_semantic_budget(
                record, min_prompt_seconds=1.0, min_target_seconds=2.0
            )
        )
        self.assertFalse(
            _record_has_singleton_semantic_budget(
                {"semantic_code_length": 100},
                min_prompt_seconds=1.0,
                min_target_seconds=2.0,
            )
        )

    def test_semantic_fps_wins_over_semantic_frame_rate(self) -> None:
        record = {
            "semantic_code_length": 100,
            "semantic_fps": 50.0,
            "semantic_frame_rate": 25.0,
        }
        self.assertFalse(
            _record_has_singleton_semantic_budget(
                record, min_prompt_seconds=1.0, min_target_seconds=2.0
            )
        )


if __name__ == "__main__":
    unittest.main()
