from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from semantic2any.data.s2mel_dataset import (
    S2MelInMemoryDataset,
    S2MelSpeakerPairedDataset,
    collate_paired_features,
    trim_paired_feature_lengths,
)
from semantic2any.models.s2mel_model import Semantic2MelModel


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
                {"id": "d", "speaker_id": "singleton", "duration": 5.0},
            ]
        )
        paired = S2MelSpeakerPairedDataset(
            dataset,
            min_prompt_seconds=3.0,
            hop_length=1,
            sample_rate=1,
        )

        self.assertEqual(len(paired), 3)
        for index in range(len(paired)):
            item = paired[index]
            self.assertEqual(item["prompt"]["speaker_id"], item["target"]["speaker_id"])
            self.assertNotEqual(item["prompt"]["id"], item["target"]["id"])
            self.assertGreaterEqual(item["prompt"]["duration"], 3.0)

    def test_uses_fallback_prompt_pool_for_singleton_validation_speaker(self) -> None:
        train = S2MelInMemoryDataset(
            [{"id": "train", "speaker_id": "s1", "duration": 4.0}]
        )
        valid = S2MelInMemoryDataset(
            [{"id": "valid", "speaker_id": "s1", "duration": 2.0}]
        )
        paired = S2MelSpeakerPairedDataset(
            valid,
            min_prompt_seconds=3.0,
            hop_length=1,
            sample_rate=1,
            fallback_prompt_dataset=train,
        )

        self.assertEqual(len(paired), 1)
        self.assertEqual(paired.fallback_target_count, 1)
        self.assertEqual(paired[0]["prompt"]["id"], "train")


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


if __name__ == "__main__":
    unittest.main()
