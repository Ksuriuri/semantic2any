#!/usr/bin/env python3
"""Read MaskGCT semantic codes back out of the packed shards.

Layout, one pair of files per audio tar, under
``preprocessed/<dataset>/features/maskGCT_codes/``::

    <dataset>-NNNNNN.u2.bin    all codes of that tar, concatenated, little-endian uint16
    <dataset>-NNNNNN.jsonl     one row per sample, with offset/length into the .bin

A row's codes are ``blob[offset : offset + length]`` in *code units* (not bytes),
so a memmap of dtype ``<u2`` is indexed directly with the numbers in the index.

Read it locally, not over the network. Measured on ``vctk-000000`` (2000 samples):

    local mmap random access   1.5 us/sample
    GCS ranged read (per row)  513 ms/sample

That is ~340,000x, and it is why the training path should copy the shard down
once (whole-blob download was 1.94 s) and then mmap it, rather than issuing a
ranged GET per sample. One request per shard instead of one per utterance is the
entire point of the packed layout.

Usage::

    # local shard
    python features/read_codes.py --index /data/vctk-000000.jsonl

    # straight from the bucket (downloads the shard once into --cache-dir)
    python features/read_codes.py --dataset vctk --shard vctk-000000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CODE_DTYPE = "<u2"
DEFAULT_ROOT = "noiz-taiwan-audio-data/preprocessed"
FEATURE_DIR = "features/maskGCT_codes"


class CodesShard:
    """One ``.u2.bin`` plus its index, with codes looked up by sample id.

    The blob is memory-mapped, so opening a shard costs nothing and only the
    pages actually touched are read.  Rows whose ``status`` is not ``encoded``
    (e.g. ``skipped_short_audio``) carry no offset and are not in ``rows``.
    """

    def __init__(self, index_path: str | Path, blob_path: str | Path | None = None):
        index_path = Path(index_path)
        if blob_path is None:
            # The index names its own blob, which keeps a shard relocatable.
            first = None
            with index_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        first = json.loads(line)
                        break
            if first is None:
                raise ValueError(f"{index_path} is empty")
            blob_path = index_path.parent / first["semantic_code_path"]
        self.blob_path = Path(blob_path)
        self.codes = np.memmap(self.blob_path, dtype=CODE_DTYPE, mode="r")

        self.rows: dict[str, dict] = {}
        self.skipped: list[dict] = []
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("semantic_code_offset") is None:
                    self.skipped.append(row)
                else:
                    self.rows[row["id"]] = row

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, sample_id: str) -> np.ndarray:
        row = self.rows[sample_id]
        start = row["semantic_code_offset"]
        return self.codes[start : start + row["semantic_code_length"]]

    def items(self):
        for sample_id in self.rows:
            yield sample_id, self[sample_id]


def fetch_shard(dataset: str, shard: str, cache_dir: Path,
                root: str = DEFAULT_ROOT, token: str = "gcs-key.json") -> Path:
    """Download one shard's index and blob, once, and return the index path."""
    import gcsfs

    fs = gcsfs.GCSFileSystem(token=token)
    cache_dir.mkdir(parents=True, exist_ok=True)
    remote = f"{root}/{dataset}/{FEATURE_DIR}"
    index_path = cache_dir / f"{shard}.jsonl"
    blob_path = cache_dir / f"{shard}.u2.bin"
    if not index_path.exists():
        fs.get(f"{remote}/{shard}.jsonl", str(index_path))
    if not blob_path.exists():
        fs.get(f"{remote}/{shard}.u2.bin", str(blob_path))
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", help="path to a local <shard>.jsonl")
    parser.add_argument("--dataset", help="fetch from the bucket instead")
    parser.add_argument("--shard", help="e.g. vctk-000000")
    parser.add_argument("--cache-dir", default="/tmp/maskgct-codes-cache", type=Path)
    parser.add_argument("--show", type=int, default=3, help="print this many samples")
    args = parser.parse_args()

    if args.index:
        index_path = Path(args.index)
    elif args.dataset and args.shard:
        index_path = fetch_shard(args.dataset, args.shard, args.cache_dir)
    else:
        parser.error("pass --index, or both --dataset and --shard")

    shard = CodesShard(index_path)
    print(f"shard={index_path.name} encoded={len(shard)} skipped={len(shard.skipped)} "
          f"codes={shard.codes.size} ({shard.codes.size / 50 / 3600:.2f} audio-hours)")
    for sample_id, codes in list(shard.items())[: args.show]:
        row = shard.rows[sample_id]
        print(f"  {sample_id}\n    duration={row['duration']:.2f}s codes={codes.size} "
              f"({codes.size / row['semantic_frame_rate']:.2f}s at "
              f"{row['semantic_frame_rate']:g} Hz) first10={codes[:10].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
