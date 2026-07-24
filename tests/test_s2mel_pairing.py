from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from semantic2any.data.s2mel_dataset import (
    LengthBucketBatchSampler,
    S2MelCollator,
    S2MelInMemoryDataset,
    S2MelSpeakerPairedDataset,
    collate_paired_features,
    trim_paired_feature_lengths,
)
from semantic2any.models.s2mel_model import Semantic2MelModel
from semantic2any.utils.indextts_adapters import IndexTTSFeatureAdapter


def _feature(
    mel_frames: int,
    semantic_frames: int,
    value: float,
    mel_channels: int = 80,
) -> dict[str, torch.Tensor]:
    return {
        "mel": torch.full((mel_channels, mel_frames), value),
        "semantic": torch.full((semantic_frames, 1), value),
        "style": torch.full((192,), value),
    }


class _IdentityRegulator(nn.Module):
    def forward(self, x, ylens, n_quantizers=None, f0=None, xlens=None):
        del n_quantizers, f0
        out = x.new_zeros(x.size(0), int(ylens.max().item()), x.size(-1))
        for index, (xlen, ylen) in enumerate(zip(xlens, ylens, strict=True)):
            xlen_value = int(xlen.item())
            ylen_value = int(ylen.item())
            out[index, :ylen_value] = F.interpolate(
                x[index : index + 1, :xlen_value].transpose(1, 2),
                size=ylen_value,
                mode="nearest",
            ).transpose(1, 2)[0]
        return out, ylens, None, None, None


class PairedLengthTest(unittest.TestCase):
    def test_trims_prompt_before_target(self) -> None:
        lengths = trim_paired_feature_lengths(
            10,
            25,
            10,
            25,
            hop_length=1,
            sample_rate=1,
            max_pair_seconds=30,
            min_prompt_seconds=3,
            min_generated_frames=1,
        )
        self.assertEqual(lengths, (5, 25, 5, 25))

        lengths = trim_paired_feature_lengths(
            5,
            30,
            5,
            30,
            hop_length=1,
            sample_rate=1,
            max_pair_seconds=30,
            min_prompt_seconds=3,
            min_generated_frames=1,
        )
        self.assertEqual(lengths, (3, 27, 3, 27))

    def test_rejects_prompt_shorter_than_minimum(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer than the required"):
            trim_paired_feature_lengths(
                2,
                10,
                2,
                10,
                hop_length=1,
                sample_rate=1,
                max_pair_seconds=30,
                min_prompt_seconds=3,
                min_generated_frames=1,
            )


class SpeakerPairDatasetTest(unittest.TestCase):
    def test_pairs_distinct_utterances_from_same_speaker(self) -> None:
        dataset = S2MelInMemoryDataset(
            [
                {"id": "a", "speaker_id": "s1", "duration": 4.0},
                {"id": "b", "speaker_id": "s1", "duration": 2.0},
                {"id": "c", "speaker_id": "s1", "duration": 5.0},
                {"id": "d", "speaker_id": "singleton", "duration": 8.0},
            ]
        )
        paired = S2MelSpeakerPairedDataset(
            dataset,
            min_prompt_seconds=3.0,
            max_prompt_seconds=20.0,
            min_target_seconds=3.0,
            max_target_seconds=30.0,
            hop_length=1,
            sample_rate=1,
        )

        self.assertEqual(len(paired), 3)
        for index in range(len(paired)):
            item = paired[index]
            self.assertEqual(item["prompt"]["speaker_id"], item["target"]["speaker_id"])
            if item["singleton_split"]:
                self.assertEqual(item["prompt"]["id"], item["target"]["id"])
                expected_seconds = item["target"]["duration"]
            else:
                self.assertNotEqual(item["prompt"]["id"], item["target"]["id"])
                self.assertGreaterEqual(item["prompt"]["duration"], 3.0)
                expected_seconds = item["target"]["duration"] + min(
                    item["prompt"]["duration"],
                    20.0,
                )
            self.assertEqual(paired.estimated_sample_seconds(index), expected_seconds)

        self.assertEqual(paired.paired_target_count, 2)
        self.assertEqual(paired.singleton_target_count, 1)
        self.assertEqual(paired.too_short_target_count, 1)

    def test_singleton_must_fit_both_minimum_segments(self) -> None:
        dataset = S2MelInMemoryDataset(
            [
                {"id": "short", "speaker_id": "short", "duration": 5.9},
                {
                    "id": "short-code",
                    "speaker_id": "short-code",
                    "duration": 6.01,
                    "semantic_code_length": 299,
                    "semantic_fps": 50.0,
                },
                {"id": "usable", "speaker_id": "usable", "duration": 6.0},
            ]
        )
        paired = S2MelSpeakerPairedDataset(
            dataset,
            min_prompt_seconds=3.0,
            max_prompt_seconds=20.0,
            min_target_seconds=3.0,
            max_target_seconds=30.0,
            hop_length=1,
            sample_rate=1,
        )

        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["target"]["id"], "usable")
        self.assertTrue(paired[0]["singleton_split"])
        self.assertEqual(paired.unusable_target_count, 2)

    def test_overlong_audio_can_be_prompt_but_not_target(self) -> None:
        dataset = S2MelInMemoryDataset(
            [
                {"id": "long", "speaker_id": "s1", "duration": 40.0},
                {"id": "target", "speaker_id": "s1", "duration": 10.0},
            ]
        )
        paired = S2MelSpeakerPairedDataset(
            dataset,
            min_prompt_seconds=3.0,
            max_prompt_seconds=20.0,
            min_target_seconds=3.0,
            max_target_seconds=30.0,
            hop_length=1,
            sample_rate=1,
        )

        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["prompt"]["id"], "long")
        self.assertEqual(paired[0]["target"]["id"], "target")
        self.assertEqual(paired.overlong_target_count, 1)


class LengthBucketBatchSamplerTest(unittest.TestCase):
    def test_groups_world_size_batches_from_same_bucket(self) -> None:
        lengths = [4, 5, 6, 7, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
        sampler = LengthBucketBatchSampler(
            lengths,
            batch_size=2,
            world_size=2,
            boundaries=[10, 20, 40],
            seed=1234,
            drop_last=True,
            shuffle=True,
        )
        batches = list(sampler)

        self.assertEqual(len(batches) % 2, 0)
        for start in range(0, len(batches), 2):
            group = batches[start : start + 2]
            bucket_ids = {
                sampler._bucket_id(lengths[index])
                for batch in group
                for index in batch
            }
            self.assertEqual(len(bucket_ids), 1)

        flattened = [index for batch in batches for index in batch]
        self.assertEqual(len(flattened), len(set(flattened)))


class PairedFeatureTest(unittest.TestCase):
    def test_supports_128_band_bigvgan_mels(self) -> None:
        batch = collate_paired_features(
            [_feature(10, 10, 1.0, mel_channels=128)],
            [_feature(20, 20, 2.0, mel_channels=128)],
            hop_length=1,
            sample_rate=1,
            max_pair_seconds=30,
            min_prompt_seconds=3,
            min_generated_frames=1,
        )

        self.assertEqual(tuple(batch["mel"].shape), (1, 128, 30))

    def test_collates_prompt_then_target_and_uses_prompt_style(self) -> None:
        batch = collate_paired_features(
            [_feature(10, 10, 1.0)],
            [_feature(25, 25, 2.0)],
            hop_length=1,
            sample_rate=1,
            max_pair_seconds=30,
            min_prompt_seconds=3,
            min_generated_frames=1,
        )

        self.assertEqual(batch["mel_lens"].tolist(), [30])
        self.assertEqual(batch["prompt_lens"].tolist(), [5])
        self.assertEqual(batch["semantic_lens"].tolist(), [30])
        self.assertEqual(batch["prompt_semantic_lens"].tolist(), [5])
        self.assertTrue(torch.all(batch["mel"][0, :, :5] == 1))
        self.assertTrue(torch.all(batch["mel"][0, :, 5:30] == 2))
        self.assertTrue(torch.all(batch["style"][0] == 1))

    def test_model_regulates_pair_segments_independently(self) -> None:
        model = Semantic2MelModel.__new__(Semantic2MelModel)
        nn.Module.__init__(model)
        model.models = nn.ModuleDict({"length_regulator": _IdentityRegulator()})
        semantic = torch.tensor([[[1.0], [1.0], [2.0], [2.0]]])

        mu = model.build_paired_condition(
            semantic,
            mel_lens=torch.tensor([6]),
            prompt_lens=torch.tensor([3]),
            semantic_lens=torch.tensor([4]),
            prompt_semantic_lens=torch.tensor([2]),
        )

        self.assertTrue(torch.all(mu[0, :3] == 1))
        self.assertTrue(torch.all(mu[0, 3:6] == 2))


class PairedAudioExtractionTest(unittest.TestCase):
    @staticmethod
    def _adapter() -> IndexTTSFeatureAdapter:
        class Decoder(nn.Module):
            def forward(self, codes):
                return codes.float().unsqueeze(-1)

        adapter = IndexTTSFeatureAdapter.__new__(IndexTTSFeatureAdapter)
        nn.Module.__init__(adapter)
        adapter.semantic_mean = torch.zeros(1)
        adapter.semantic_backend = None
        adapter.semantic_decoder = Decoder()
        adapter.max_audio_seconds = 30.0
        adapter.min_prompt_seconds = 3.0
        adapter.max_prompt_seconds = 20.0
        adapter.min_pair_prompt_seconds = 3.0
        adapter.min_target_seconds = 3.0
        adapter.max_target_seconds = 30.0
        adapter.min_generated_frames = 1
        adapter.sample_rate_mel = 1
        adapter.sample_rate_16k = 1
        adapter.feature_batch_size = 16
        adapter.use_style_condition = False
        adapter.style_dim = 192
        adapter.mel_args = {"hop_size": 1}
        adapter.mel_spectrogram = lambda audio, **_: torch.zeros(
            1, 2, audio.size(-1)
        )
        adapter._resample_waveform_batch = lambda waveforms, _rates, target_rates: {
            rate: waveforms for rate in target_rates
        }
        return adapter

    def test_paired_prompt_and_code_are_truncated_to_twenty_seconds(self) -> None:
        adapter = self._adapter()
        batch = adapter.extract_paired_from_audio_paths(
            ["prompt.flac"],
            ["target.flac"],
            prompt_waveforms=[torch.zeros(1, 30)],
            prompt_sample_rates=[1],
            target_waveforms=[torch.zeros(1, 10)],
            target_sample_rates=[1],
            singleton_splits=[False],
            prompt_semantic_codes=torch.arange(30).unsqueeze(0),
            prompt_semantic_code_lens=torch.tensor([30]),
            target_semantic_codes=torch.arange(10).unsqueeze(0),
            target_semantic_code_lens=torch.tensor([10]),
        )

        self.assertEqual(batch["prompt_lens"].tolist(), [20])
        self.assertEqual(batch["mel_lens"].tolist(), [30])
        self.assertEqual(batch["prompt_semantic_lens"].tolist(), [20])
        self.assertEqual(batch["semantic_lens"].tolist(), [30])

    def test_singleton_split_chooses_an_exact_semantic_frame_boundary(self) -> None:
        adapter = self._adapter()
        with patch(
            "semantic2any.utils.indextts_adapters.random.randint",
            return_value=4,
        ) as randint:
            batch = adapter.extract_paired_from_audio_paths(
                ["singleton.flac"],
                ["singleton.flac"],
                prompt_waveforms=[torch.zeros(1, 10)],
                prompt_sample_rates=[1],
                target_waveforms=[torch.zeros(1, 10)],
                target_sample_rates=[1],
                singleton_splits=[True],
                prompt_semantic_codes=torch.arange(10).unsqueeze(0),
                prompt_semantic_code_lens=torch.tensor([10]),
                target_semantic_codes=torch.arange(10).unsqueeze(0),
                target_semantic_code_lens=torch.tensor([10]),
            )

        randint.assert_called_once_with(3, 7)
        self.assertEqual(batch["prompt_lens"].tolist(), [4])
        self.assertEqual(batch["prompt_semantic_lens"].tolist(), [4])
        self.assertEqual(batch["mel_lens"].tolist(), [10])
        self.assertEqual(batch["semantic"][0, :, 0].tolist(), list(range(10)))

    def test_paired_and_singleton_items_share_one_batch(self) -> None:
        adapter = self._adapter()
        with patch(
            "semantic2any.utils.indextts_adapters.random.randint",
            return_value=4,
        ):
            batch = adapter.extract_paired_from_audio_paths(
                ["prompt.flac", "singleton.flac"],
                ["target.flac", "singleton.flac"],
                prompt_waveforms=[torch.zeros(1, 30), torch.zeros(1, 10)],
                prompt_sample_rates=[1, 1],
                target_waveforms=[torch.zeros(1, 10), torch.zeros(1, 10)],
                target_sample_rates=[1, 1],
                singleton_splits=[False, True],
                prompt_semantic_codes=torch.stack(
                    [torch.arange(30), torch.arange(30)]
                ),
                prompt_semantic_code_lens=torch.tensor([30, 10]),
                target_semantic_codes=torch.stack(
                    [torch.arange(30), torch.arange(30)]
                ),
                target_semantic_code_lens=torch.tensor([10, 10]),
            )

        self.assertEqual(batch["prompt_lens"].tolist(), [20, 4])
        self.assertEqual(batch["prompt_semantic_lens"].tolist(), [20, 4])
        self.assertEqual(batch["mel_lens"].tolist(), [30, 10])
        self.assertEqual(batch["semantic_lens"].tolist(), [30, 10])

    def test_worker_paired_mel_batch_finalizes_semantic_codes(self) -> None:
        collator = S2MelCollator(
            hop_length=1,
            sample_rate=1,
            min_prompt_seconds=3.0,
            max_prompt_seconds=20.0,
            min_pair_prompt_seconds=3.0,
            min_generated_frames=1,
            min_target_seconds=3.0,
            max_target_seconds=30.0,
            extract_mel_in_worker=True,
        )
        collator._maybe_limit_prompt_bandwidth = lambda waveform: waveform
        collator._mel_from_waveform = lambda waveform: torch.zeros(128, waveform.size(-1))
        batch = {
            "singleton_splits": [False, True],
            "records": [{"id": "paired"}, {"id": "singleton"}],
            "prompt_semantic_codes": torch.stack([torch.arange(30), torch.arange(30)]),
            "prompt_semantic_code_lens": torch.tensor([30, 10]),
            "target_semantic_codes": torch.stack([torch.arange(30), torch.arange(30)]),
            "target_semantic_code_lens": torch.tensor([10, 10]),
        }
        with patch("semantic2any.data.s2mel_dataset.random.randint", return_value=4):
            collator._attach_paired_worker_features(
                batch,
                prompt_waveforms=[torch.zeros(1, 30), torch.zeros(1, 10)],
                prompt_sample_rates=[1, 1],
                target_waveforms=[torch.zeros(1, 10), torch.zeros(1, 10)],
                target_sample_rates=[1, 1],
            )

        self.assertTrue(batch["worker_precomputed_mel"])
        self.assertEqual(batch["prompt_lens"].tolist(), [20, 4])
        self.assertEqual(batch["prompt_semantic_lens"].tolist(), [20, 4])
        self.assertEqual(batch["mel_lens"].tolist(), [30, 10])
        self.assertEqual(batch["semantic_lens"].tolist(), [30, 10])
        self.assertFalse(torch.is_floating_point(batch["semantic"]))

        adapter = self._adapter()
        finalized = adapter.finalize_worker_paired_batch(batch)
        self.assertTrue(torch.is_floating_point(finalized["semantic"]))
        self.assertEqual(tuple(finalized["semantic"].shape), (2, 30, 1))
        self.assertEqual(finalized["semantic"][0, :30, 0].tolist(), list(range(20)) + list(range(10)))
        self.assertEqual(finalized["semantic"][1, :10, 0].tolist(), list(range(10)))


if __name__ == "__main__":
    unittest.main()
