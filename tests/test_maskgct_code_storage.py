from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from scripts.precompute_maskgct_codes import AudioCollator, _load_jsonl
from semantic2any.data.s2mel_dataset import S2MelCollator, S2MelJsonlDataset


def _collator() -> S2MelCollator:
    return S2MelCollator(
        hop_length=1,
        sample_rate=1,
        min_prompt_seconds=1.0,
        max_prompt_seconds=2.0,
        min_generated_frames=1,
        expected_semantic_codec="maskgct",
        expected_semantic_fingerprint="fingerprint",
    )


def test_jsonl_resolves_and_reads_binary_semantic_code_range(tmp_path):
    codes_path = tmp_path / "codes.bin"
    np.asarray([99, 1, 2, 3, 88], dtype="<u2").tofile(codes_path)
    lookup_path = tmp_path / "lookup.pt"
    lookup_path.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "audio_path": "sample.flac",
                "semantic_code_path": "codes.bin",
                "semantic_code_offset": 1,
                "semantic_code_length": 3,
                "semantic_lookup_path": "lookup.pt",
                "semantic_lookup_sha256": "sha",
                "semantic_codec": "maskgct",
                "semantic_fingerprint": "fingerprint",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = S2MelJsonlDataset(manifest)

    batch = _collator()([dataset[0]])

    assert batch["has_semantic_codes"] is True
    assert batch["semantic_codes"].tolist() == [[1, 2, 3]]
    assert batch["semantic_code_lens"].tolist() == [3]
    assert batch["semantic_lookup_path"] == str(lookup_path)


def test_collator_rejects_mixed_semantic_code_records(tmp_path):
    codes_path = tmp_path / "codes.bin"
    np.asarray([1], dtype="<u2").tofile(codes_path)
    coded = {
        "audio_path": "coded.flac",
        "semantic_code_path": str(codes_path),
        "semantic_code_offset": 0,
        "semantic_code_length": 1,
        "semantic_lookup_path": str(tmp_path / "lookup.pt"),
        "semantic_lookup_sha256": "sha",
        "semantic_codec": "maskgct",
        "semantic_fingerprint": "fingerprint",
    }
    plain = {"audio_path": "plain.flac"}

    with pytest.raises(ValueError, match="with and without semantic codes"):
        _collator()([coded, plain])


def test_collator_rejects_binary_range_past_end(tmp_path):
    codes_path = tmp_path / "codes.bin"
    np.asarray([1], dtype="<u2").tofile(codes_path)
    record = {
        "audio_path": "sample.flac",
        "semantic_code_path": str(codes_path),
        "semantic_code_offset": 1,
        "semantic_code_length": 1,
        "semantic_lookup_path": str(tmp_path / "lookup.pt"),
        "semantic_lookup_sha256": "sha",
        "semantic_codec": "maskgct",
        "semantic_fingerprint": "fingerprint",
    }

    with pytest.raises(ValueError, match="exceeds"):
        _collator()([record])


def test_skipped_audio_keeps_semantic_codes_aligned(tmp_path, monkeypatch):
    codes_path = tmp_path / "codes.bin"
    np.asarray([7, 8], dtype="<u2").tofile(codes_path)

    def record(name: str, offset: int) -> dict:
        return {
            "id": name,
            "audio_path": f"{name}.flac",
            "semantic_code_path": str(codes_path),
            "semantic_code_offset": offset,
            "semantic_code_length": 1,
            "semantic_lookup_path": str(tmp_path / "lookup.pt"),
            "semantic_lookup_sha256": "sha",
            "semantic_codec": "maskgct",
            "semantic_fingerprint": "fingerprint",
        }

    def load(path: str):
        if path == "bad.flac":
            raise RuntimeError("bad audio")
        return torch.ones(1, 4), 1

    monkeypatch.setattr(
        "semantic2any.data.s2mel_dataset.torchaudio.load",
        load,
    )
    collator = _collator()
    collator.decode_audio_in_worker = True
    collator.skip_audio_errors = True

    with pytest.warns(RuntimeWarning, match="bad.flac"):
        batch = collator([record("good", 0), record("bad", 1)])

    assert [item["id"] for item in batch["records"]] == ["good"]
    assert batch["semantic_codes"].tolist() == [[7]]


def test_resume_repairs_only_a_partial_final_journal_line(tmp_path):
    journal = tmp_path / "manifest.jsonl"
    committed = b'{"source_index":0}\n'
    journal.write_bytes(committed + b'{"source_index":')

    rows = _load_jsonl(journal, repair_partial_tail=True)

    assert rows[0]["source_index"] == 0
    assert journal.read_bytes() == committed


def test_audio_collator_resamples_before_returning_numpy(monkeypatch):
    monkeypatch.setattr(
        "scripts.precompute_maskgct_codes.torchaudio.load",
        lambda _: (torch.arange(8).view(1, -1).float(), 4),
    )
    calls = []

    def resample(audio, source_rate, target_rate):
        calls.append((source_rate, target_rate))
        return torch.ones(1, 32)

    monkeypatch.setattr(
        "scripts.precompute_maskgct_codes.torchaudio.functional.resample",
        resample,
    )

    records, waveforms, failures = AudioCollator(10.0, False)(
        [{"audio_path": "sample.wav"}]
    )

    assert [record["audio_path"] for record in records] == ["sample.wav"]
    assert failures == []
    assert calls == [(4, 16000)]
    assert len(waveforms) == 1
    assert waveforms[0].shape == (32,)
    assert waveforms[0].dtype == np.float32


def test_audio_collator_records_decode_errors_and_continues(monkeypatch):
    def fail(_):
        raise RuntimeError("corrupt audio")

    monkeypatch.setattr(
        "scripts.precompute_maskgct_codes.torchaudio.load",
        fail,
    )

    with pytest.warns(RuntimeWarning, match="bad.wav"):
        records, waveforms, failures = AudioCollator(10.0, True)(
            [{"audio_path": "bad.wav", "_source_index": 3}]
        )

    assert records == []
    assert waveforms == []
    assert len(failures) == 1
    assert failures[0]["source_index"] == 3
    assert failures[0]["error"] == "corrupt audio"
