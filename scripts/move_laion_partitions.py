#!/usr/bin/env python3
"""Move flat laion_emolia FLACs into pXXX partition subdirectories.

The previous extraction wrote ~4.5M files into one flat directory, which
saturates ext4 htree hash buckets (ENOSPC with free disk).  This script
groups files by shard number // 1000 so each partition stays small.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


FLAT_DIR = Path(
    "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-v2/audios/laion_emolia"
)
NAME_RE = re.compile(r"^laion_emolia-(\d+)__")


def main() -> int:
    moved = 0
    skipped = 0
    errors = 0
    by_partition: dict[str, int] = {}
    leftovers: list[str] = []
    with os.scandir(FLAT_DIR) as it:
        for entry in it:
            name = entry.name
            if entry.is_dir(follow_symlinks=False):
                continue
            match = NAME_RE.match(name)
            if not match:
                leftovers.append(name)
                continue
            partition = f"p{int(match.group(1)) // 1000:03d}"
            target_dir = FLAT_DIR / partition
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                os.rename(FLAT_DIR / name, target_dir / name)
                moved += 1
                by_partition[partition] = by_partition.get(partition, 0) + 1
            except OSError as exc:
                errors += 1
                if errors <= 10:
                    print(f"ERROR {name}: {exc}", flush=True)
            if moved and moved % 500_000 == 0:
                print(f"progress moved={moved} errors={errors}", flush=True)
    print(
        json_dump(
            {
                "moved": moved,
                "skipped_dirs": skipped,
                "errors": errors,
                "partitions": len(by_partition),
                "per_partition": dict(
                    sorted(by_partition.items(), key=lambda kv: kv[0])
                ),
                "leftovers": leftovers[:50],
                "leftover_count": len(leftovers),
            }
        ),
        flush=True,
    )
    return 0 if errors == 0 else 1


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
