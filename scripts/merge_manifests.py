#!/usr/bin/env python3
"""Merge sync metadata (local audio paths + text) with code manifests.

The maskGCT code manifests keep the GCS-relative audio_path, which does not
resolve locally.  This script produces per-dataset manifests whose rows have
the local metadata fields AND the semantic code fields, with absolute
audio_path / semantic_code_path, ready for split + training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


OUTPUT_ROOT = Path(
    "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-v2"
)
METADATA_DIR = OUTPUT_ROOT / "metadata"
CODE_MANIFESTS_DIR = OUTPUT_ROOT / "maskgct-codes" / "manifests"
MERGED_DIR = OUTPUT_ROOT / "maskgct-codes" / "manifests_merged"

SEMANTIC_FIELDS = (
    "semantic_code_path",
    "semantic_code_offset",
    "semantic_code_length",
    "semantic_frame_rate",
    "codebook_size",
    "status",
    "completed_at",
    "metadata_path",
    # Stamped by the sync so the pulled codes are trainable as-is; dropping any
    # of these here would send training back to re-encoding audio every step.
    "semantic_lookup_path",
    "semantic_lookup_sha256",
    "semantic_codec",
    "semantic_codebooks",
    "semantic_fps",
)


def main() -> int:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    code_manifests = sorted(CODE_MANIFESTS_DIR.glob("*.jsonl"))
    for code_path in code_manifests:
        dataset = code_path.stem
        metadata_path = METADATA_DIR / f"{dataset}.jsonl"
        if not metadata_path.is_file():
            print(f"SKIP {dataset}: no metadata {metadata_path}", flush=True)
            continue

        meta_by_id: dict[str, dict] = {}
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = row.get("id")
                if isinstance(rid, str) and rid:
                    meta_by_id[rid] = row
        print(
            f"{dataset}: metadata rows {len(meta_by_id)}",
            flush=True,
        )

        matched = 0
        missing_meta = 0
        merged_path = MERGED_DIR / f"{dataset}.jsonl"
        with code_path.open("r", encoding="utf-8") as src, \
                merged_path.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                code_row = json.loads(line)
                rid = code_row.get("id")
                meta = meta_by_id.get(rid) if isinstance(rid, str) else None
                if meta is None:
                    missing_meta += 1
                    continue
                merged = dict(meta)
                # Absolute local audio path (resolve symlinks for podcasts).
                merged["audio_path"] = str(
                    (metadata_path.parent / meta["audio_path"]).resolve()
                )
                for field in SEMANTIC_FIELDS:
                    if field in code_row:
                        merged[field] = code_row[field]
                dst.write(json.dumps(merged, ensure_ascii=False) + "\n")
                matched += 1
        print(
            f"{dataset}: merged {matched}, code-without-metadata {missing_meta}",
            flush=True,
        )
    print("MERGE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
