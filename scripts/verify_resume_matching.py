#!/usr/bin/env python3
"""Verify resume basename matching against one GCS tar before moving files."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

import gcsfs

sys.path.insert(0, "/mnt/data_sdd/hhy/noiz-tts/semantic2any/scripts")
from sync_s2mel_train_data import (  # noqa: E402
    SOURCE_PREFIX,
    flat_output_name,
)


OUTPUT_ROOT = Path(
    "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-v2"
)
DATASET = "laion_emolia"
SHARD = "laion_emolia-000000"


def main() -> int:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/mnt/data_sdd/hhy/SpeechData/gcs-key.json"
    )
    os.environ["GOOGLE_CLOUD_PROJECT"] = "noiz-430406"
    fs = gcsfs.GCSFileSystem(project="noiz-430406")
    tar_key = f"{SOURCE_PREFIX}/{DATASET}/audio/{SHARD}.tar"
    tar_relative = f"audio/{SHARD}.tar"

    # expected basenames from the metadata tmp
    expected: set[str] = set()
    tmp = OUTPUT_ROOT / "metadata" / f"{DATASET}.jsonl.tmp"
    prefix = SHARD + "__"
    with tmp.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            row = json.loads(line)
            basename = PurePosixPath(row["audio_path"]).name
            if basename.startswith(prefix):
                expected.add(basename)
    print(f"expected basenames for {SHARD}: {len(expected)}", flush=True)

    # compute candidates by streaming the tar exactly like resume mode
    candidates: dict[str, str] = {}
    occurrences: dict[str, int] = defaultdict(int)
    with fs.open(tar_key, "rb") as raw_file:
        import tarfile

        with tarfile.open(fileobj=raw_file, mode="r|*") as tar_file:
            for member in tar_file:
                occurrence = occurrences[member.name]
                occurrences[member.name] += 1
                if not member.isfile():
                    continue
                member_basename = PurePosixPath(member.name).name
                candidate = flat_output_name(
                    tar_relative,
                    member.name,
                    member_basename,
                    occurrence,
                )
                candidates[candidate] = member.name
    print(f"candidates computed: {len(candidates)}", flush=True)

    missing = sorted(expected - set(candidates))
    extra = sorted(set(candidates) - expected)
    print(f"expected missing from candidates: {len(missing)}", flush=True)
    for name in missing[:5]:
        print(f"  MISSING {name}", flush=True)
    print(f"candidates not in expected (should be ~0): {len(extra)}", flush=True)
    for name in extra[:5]:
        print(f"  EXTRA {name}", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
