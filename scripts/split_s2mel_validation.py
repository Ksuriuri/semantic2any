#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


PATH_KEYS = (
    "audio_path",
    "mel_path",
    "semantic_path",
    "style_path",
    "semantic_code_path",
    "semantic_lookup_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic, source-stratified s2mel train/validation manifests."
    )
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def has_semantic_codes(record: dict[str, Any]) -> bool:
    return bool(
        record.get("semantic_code_path")
        and record.get("semantic_code_length")
        and record.get("semantic_lookup_path")
        and record.get("semantic_lookup_sha256")
    )


def allocate_validation_counts(
    counts: dict[Path, int], valid_size: int
) -> dict[Path, int]:
    total = sum(counts.values())
    if valid_size <= 0 or valid_size >= total:
        raise ValueError(f"valid_size must be between 1 and {total - 1}, got {valid_size}")

    exact = {path: count * valid_size / total for path, count in counts.items()}
    allocated = {path: math.floor(value) for path, value in exact.items()}
    remaining = valid_size - sum(allocated.values())
    order = sorted(
        counts,
        key=lambda path: (-(exact[path] - allocated[path]), path.name),
    )
    for path in order[:remaining]:
        allocated[path] += 1
    return allocated


def resolve_record_paths(record: dict, manifest_path: Path) -> dict:
    record = dict(record)
    for key in PATH_KEYS:
        value = record.get(key)
        if not isinstance(value, str) or not value or "://" in value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        record[key] = str(path.resolve())
    return record


def load_manifest_records(manifest_path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    skipped_missing_codes = 0
    with manifest_path.open("r", encoding="utf-8") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            record = resolve_record_paths(json.loads(line), manifest_path)
            records.append(record)
            if not has_semantic_codes(record):
                skipped_missing_codes += 1
    return records, skipped_missing_codes


def main() -> None:
    args = parse_args()
    metadata_dir = args.metadata_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifests = sorted(metadata_dir.glob("*.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No JSONL manifests found under {metadata_dir}")

    loaded = {path: load_manifest_records(path) for path in manifests}
    any_coded = any(
        has_semantic_codes(record)
        for records, _ in loaded.values()
        for record in records
    )
    selected: dict[Path, list[dict[str, Any]]] = {}
    skipped_missing_codes = 0
    for path, (records, missing) in loaded.items():
        if any_coded:
            kept = [record for record in records if has_semantic_codes(record)]
            skipped_missing_codes += missing
            selected[path] = kept
        else:
            selected[path] = records

    counts = {path: len(records) for path, records in selected.items()}
    total = sum(counts.values())
    if total == 0:
        raise ValueError(
            f"No usable records found under {metadata_dir}"
            + (" after filtering records without semantic codes" if any_coded else "")
        )
    allocations = allocate_validation_counts(counts, args.valid_size)
    rng = random.Random(args.seed)
    validation_indices = {
        path: set(rng.sample(range(counts[path]), allocations[path]))
        for path in manifests
        if counts[path] > 0
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    summary_path = output_dir / "split_summary.json"
    train_tmp = train_path.with_suffix(".jsonl.tmp")
    valid_tmp = valid_path.with_suffix(".jsonl.tmp")
    summary_tmp = summary_path.with_suffix(".json.tmp")

    source_summary = {}
    total_train = 0
    total_valid = 0
    try:
        with (
            train_tmp.open("w", encoding="utf-8") as train_file,
            valid_tmp.open("w", encoding="utf-8") as valid_file,
        ):
            for manifest_path in manifests:
                source_train = 0
                source_valid = 0
                for record_index, record in enumerate(selected[manifest_path]):
                    encoded = json.dumps(
                        record, ensure_ascii=False, separators=(",", ":")
                    )
                    if record_index in validation_indices.get(manifest_path, ()):
                        valid_file.write(encoded + "\n")
                        source_valid += 1
                    else:
                        train_file.write(encoded + "\n")
                        source_train += 1

                source_summary[manifest_path.name] = {
                    "total": counts[manifest_path],
                    "train": source_train,
                    "valid": source_valid,
                    "skipped_missing_semantic_codes": (
                        loaded[manifest_path][1] if any_coded else 0
                    ),
                }
                total_train += source_train
                total_valid += source_valid

        summary = {
            "seed": args.seed,
            "metadata_dir": str(metadata_dir),
            "train_manifest": str(train_path),
            "valid_manifest": str(valid_path),
            "train_records": total_train,
            "valid_records": total_valid,
            "require_semantic_codes": any_coded,
            "skipped_missing_semantic_codes": skipped_missing_codes if any_coded else 0,
            "sources": source_summary,
        }
        summary_tmp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        train_tmp.replace(train_path)
        valid_tmp.replace(valid_path)
        summary_tmp.replace(summary_path)
    finally:
        train_tmp.unlink(missing_ok=True)
        valid_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
