from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.split_s2mel_validation import allocate_validation_counts, main


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_validation_allocation_keeps_exact_requested_size() -> None:
    counts = {
        Path("manifest.shard00000.jsonl"): 700_000,
        Path("manifest.shard00001.jsonl"): 713_343,
    }

    allocated = allocate_validation_counts(counts, 1000)

    assert sum(allocated.values()) == 1000


def test_split_preserves_semantic_code_metadata_and_resolves_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    code_root = tmp_path / "maskgct-codes"
    manifests = code_root / "manifests"
    codes = code_root / "codes"
    output = code_root / "splits" / "seed1234_valid2"
    manifests.mkdir(parents=True)
    codes.mkdir()
    (codes / "codes.bin").write_bytes(b"\x00\x00")
    (code_root / "maskgct_lookup.pt").write_bytes(b"lookup")

    records = []
    for index in range(6):
        record = {
            "id": f"item-{index}",
            "audio_path": str(tmp_path / f"item-{index}.flac"),
            "speaker_id": f"speaker-{index}",
            "duration": 8.0,
            "semantic_code_path": "../codes/codes.bin",
            "semantic_code_offset": index,
            "semantic_code_length": 10,
            "semantic_lookup_path": "../maskgct_lookup.pt",
            "semantic_lookup_sha256": "checksum",
            "semantic_fingerprint": "fingerprint",
        }
        records.append(record)
    for shard in range(2):
        shard_records = records[shard::2]
        (manifests / f"manifest.shard{shard:05d}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in shard_records),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_s2mel_validation.py",
            "--metadata-dir",
            str(manifests),
            "--output-dir",
            str(output),
            "--valid-size",
            "2",
            "--seed",
            "1234",
        ],
    )
    main()

    train = _read_jsonl(output / "train.jsonl")
    valid = _read_jsonl(output / "valid.jsonl")
    assert len(train) == 4
    assert len(valid) == 2
    assert {record["id"] for record in train}.isdisjoint(
        record["id"] for record in valid
    )
    for record in train + valid:
        assert record["semantic_code_path"] == str(
            (codes / "codes.bin").resolve()
        )
        assert record["semantic_lookup_path"] == str(
            (code_root / "maskgct_lookup.pt").resolve()
        )
        assert record["semantic_code_offset"] in range(6)
        assert record["semantic_code_length"] == 10
        assert record["semantic_lookup_sha256"] == "checksum"
        assert record["semantic_fingerprint"] == "fingerprint"
