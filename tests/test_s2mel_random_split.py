from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from semantic2any.data.s2mel_dataset import (
    S2MelCollator,
    S2MelJsonlDataset,
    choose_prompt_len,
)
from semantic2any.utils.indextts_adapters import IndexTTSFeatureAdapter


class RandomPromptSplitTest(unittest.TestCase):
    def test_leaves_minimum_prompt_and_target_duration(self) -> None:
        with patch(
            "semantic2any.data.s2mel_dataset.random.randint",
            return_value=5,
        ) as randint:
            prompt_len = choose_prompt_len(
                10,
                hop_length=1,
                sample_rate=1,
                min_prompt_seconds=3,
                max_prompt_seconds=None,
                min_generated_frames=1,
                min_target_seconds=3,
            )

        self.assertEqual(prompt_len, 5)
        randint.assert_called_once_with(3, 7)

    def test_rejects_audio_that_cannot_fit_both_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            choose_prompt_len(
                5,
                hop_length=1,
                sample_rate=1,
                min_prompt_seconds=3,
                max_prompt_seconds=None,
                min_generated_frames=1,
                min_target_seconds=3,
            )


class PromptStyleExtractionTest(unittest.TestCase):
    def test_style_uses_only_the_random_prompt_prefix(self) -> None:
        style_input_lengths: list[int] = []
        adapter = IndexTTSFeatureAdapter.__new__(IndexTTSFeatureAdapter)
        torch.nn.Module.__init__(adapter)
        adapter.semantic_mean = torch.zeros(1)
        adapter.max_audio_seconds = 30.0
        adapter.sample_rate_mel = 1
        adapter.sample_rate_16k = 1
        adapter.mel_args = {"hop_size": 1}
        adapter.min_prompt_seconds = 3.0
        adapter.max_prompt_seconds = None
        adapter.min_target_seconds = 3.0
        adapter.min_generated_frames = 1
        adapter.feature_batch_size = 16
        adapter._resampler_cache = {}
        adapter.mel_spectrogram = lambda audio, **kwargs: torch.zeros(
            1, 2, audio.size(-1)
        )
        adapter._style_from_audio = lambda audio: (
            style_input_lengths.append(audio.size(-1)) or torch.zeros(192)
        )
        adapter._semantic_from_waveforms = lambda waveforms: [
            torch.zeros(len(waveform), 4) for waveform in waveforms
        ]

        with (
            patch(
                "semantic2any.utils.indextts_adapters._load_audio",
                return_value=(torch.zeros(1, 10), 1),
            ),
            patch(
                "semantic2any.utils.indextts_adapters.random.randint",
                return_value=4,
            ),
        ):
            batch = IndexTTSFeatureAdapter.extract_random_split_from_audio_paths(
                adapter,
                ["sample.flac"],
            )

        self.assertEqual(style_input_lengths, [4])
        self.assertEqual(batch["prompt_lens"].tolist(), [4])
        self.assertEqual(batch["mel_lens"].tolist(), [10])
        self.assertEqual(batch["prompt_semantic_lens"].tolist(), [4])


class BatchedResamplerTest(unittest.TestCase):
    def test_batches_variable_lengths_and_reuses_cached_kernel(self) -> None:
        adapter = IndexTTSFeatureAdapter.__new__(IndexTTSFeatureAdapter)
        torch.nn.Module.__init__(adapter)
        adapter.semantic_mean = torch.zeros(1)
        adapter._resampler_cache = {}
        waveforms = [torch.arange(8).view(1, -1).float(), torch.arange(4).view(1, -1).float()]

        first = adapter._resample_waveform_batch(waveforms, [4, 4], (2,))
        cached_resampler = next(iter(adapter._resampler_cache.values()))
        second = adapter._resample_waveform_batch(waveforms, [4, 4], (2,))

        self.assertEqual([item.size(-1) for item in first[2]], [4, 2])
        self.assertEqual([item.size(-1) for item in second[2]], [4, 2])
        self.assertEqual(len(adapter._resampler_cache), 1)
        self.assertIs(next(iter(adapter._resampler_cache.values())), cached_resampler)


class WorkerAudioDecodeTest(unittest.TestCase):
    def test_collator_decodes_mixes_down_and_trims_audio(self) -> None:
        collator = S2MelCollator(
            hop_length=1,
            sample_rate=1,
            min_prompt_seconds=3,
            max_prompt_seconds=None,
            min_generated_frames=1,
            decode_audio_in_worker=True,
            max_audio_seconds=6,
        )
        stereo = torch.stack([torch.ones(10), torch.full((10,), 3.0)])
        with patch(
            "semantic2any.data.s2mel_dataset.torchaudio.load",
            return_value=(stereo, 1),
        ):
            batch = collator([{"audio_path": "sample.flac"}])

        self.assertEqual(batch["audio_sample_rates"], [1])
        self.assertEqual(tuple(batch["audio_waveforms"][0].shape), (1, 6))
        self.assertTrue(torch.all(batch["audio_waveforms"][0] == 2))

    def test_collator_skips_undecodable_audio_and_keeps_batch_aligned(self) -> None:
        collator = S2MelCollator(
            hop_length=1,
            sample_rate=1,
            min_prompt_seconds=3,
            max_prompt_seconds=None,
            min_generated_frames=1,
            decode_audio_in_worker=True,
            skip_audio_errors=True,
        )

        def load(path: str) -> tuple[torch.Tensor, int]:
            if path == "bad.flac":
                raise RuntimeError("invalid audio")
            return torch.ones(1, 8), 1

        records = [
            {"id": "good", "audio_path": "good.flac"},
            {"id": "bad", "audio_path": "bad.flac"},
        ]
        with (
            patch("semantic2any.data.s2mel_dataset.torchaudio.load", side_effect=load),
            self.assertWarnsRegex(RuntimeWarning, "bad.flac"),
        ):
            batch = collator(records)

        self.assertEqual(batch["audio_paths"], ["good.flac"])
        self.assertEqual([record["id"] for record in batch["records"]], ["good"])
        self.assertEqual(len(batch["audio_waveforms"]), 1)
        self.assertEqual(batch["audio_sample_rates"], [1])

    def test_collator_drops_whole_pair_when_one_side_is_undecodable(self) -> None:
        collator = S2MelCollator(
            hop_length=1,
            sample_rate=1,
            min_prompt_seconds=3,
            max_prompt_seconds=None,
            min_generated_frames=1,
            decode_audio_in_worker=True,
            skip_audio_errors=True,
        )

        def load(path: str) -> tuple[torch.Tensor, int]:
            if path == "bad-target.flac":
                raise RuntimeError("invalid audio")
            return torch.ones(1, 8), 1

        records = [
            {
                "id": "good-pair",
                "prompt": {"audio_path": "good-prompt.flac"},
                "target": {"audio_path": "good-target.flac"},
            },
            {
                "id": "bad-pair",
                "prompt": {"audio_path": "other-prompt.flac"},
                "target": {"audio_path": "bad-target.flac"},
            },
        ]
        with (
            patch("semantic2any.data.s2mel_dataset.torchaudio.load", side_effect=load),
            self.assertWarnsRegex(RuntimeWarning, "bad-target.flac"),
        ):
            batch = collator(records)

        self.assertEqual(batch["prompt_audio_paths"], ["good-prompt.flac"])
        self.assertEqual(batch["target_audio_paths"], ["good-target.flac"])
        self.assertEqual([record["id"] for record in batch["records"]], ["good-pair"])
        self.assertEqual(len(batch["prompt_audio_waveforms"]), 1)
        self.assertEqual(len(batch["target_audio_waveforms"]), 1)


class JsonlDirectoryDatasetTest(unittest.TestCase):
    def test_loads_all_manifests_and_resolves_relative_audio_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (root / "ears").mkdir()
            (root / "expresso").mkdir()
            manifests = {
                "ears.jsonl": {
                    "id": "ears-1",
                    "audio_path": "../ears/one.flac",
                    "duration": 7.0,
                },
                "expresso.jsonl": {
                    "id": "expresso-1",
                    "audio_path": "../expresso/two.flac",
                    "duration": 8.0,
                },
            }
            for name, record in manifests.items():
                (metadata_dir / name).write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = S2MelJsonlDataset(metadata_dir)

            self.assertEqual(len(dataset), 2)
            records = {dataset[index]["id"]: dataset[index] for index in range(len(dataset))}
            self.assertEqual(
                Path(records["ears-1"]["audio_path"]),
                metadata_dir / "../ears/one.flac",
            )
            self.assertEqual(
                Path(records["expresso-1"]["audio_path"]),
                metadata_dir / "../expresso/two.flac",
            )


if __name__ == "__main__":
    unittest.main()
