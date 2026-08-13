#!/usr/bin/env python3
"""Finish laion_emolia maskGCT codes download+manifest after audio is done.

Standalone version of download_maskgct_codes that builds selected ids from
the finalized metadata JSONL (avoids re-running plan/extract/verify), scans
the GCS code shards, then downloads the needed bins in parallel.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs

sys.path.insert(0, "/mnt/data_sdd/hhy/noiz-tts/semantic2any/scripts")
from sync_s2mel_train_data import (  # noqa: E402
    SOURCE_PREFIX,
    log,
    retry,
)


OUTPUT_ROOT = Path(
    "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-v2"
)
DATASET = "laion_emolia"
KEY_FILE = Path("/mnt/data_sdd/hhy/SpeechData/gcs-key.json")
ATTEMPTS = 4
WORKERS = 8


def main() -> int:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_FILE)
    os.environ["GOOGLE_CLOUD_PROJECT"] = "noiz-430406"
    fs = gcsfs.GCSFileSystem(project="noiz-430406")

    metadata_path = OUTPUT_ROOT / "metadata" / f"{DATASET}.jsonl"
    if not metadata_path.is_file():
        raise SystemExit(f"missing {metadata_path}")
    selected_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rid = row.get("id")
            if isinstance(rid, str) and rid:
                selected_ids.add(rid)
    log(
        "codes_selected_ids_loaded",
        dataset=DATASET,
        selected_ids=len(selected_ids),
    )

    codes_root = OUTPUT_ROOT / "maskgct-codes"
    codes_root.mkdir(parents=True, exist_ok=True)
    bins_dir = codes_root / "bins" / DATASET
    bins_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = codes_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    codes_prefix = f"{SOURCE_PREFIX}/{DATASET}/features/maskGCT_codes"
    all_files = retry(
        f"ls {codes_prefix}",
        lambda: fs.ls(codes_prefix),
        ATTEMPTS,
    )
    bin_files = sorted(f for f in all_files if f.endswith(".u2.bin"))
    jsonl_files = sorted(f for f in all_files if f.endswith(".jsonl"))
    log(
        "codes_listed",
        dataset=DATASET,
        bin_files=len(bin_files),
        jsonl_files=len(jsonl_files),
    )
    if not bin_files or not jsonl_files:
        raise SystemExit("no code shards on GCS")

    def scan_one(jsonl_path: str) -> tuple[int, list[str], set[str]]:
        local_ids: set[str] = set()
        local_records: list[str] = []
        scanned = 0
        try:
            with retry(
                f"open {jsonl_path}",
                lambda p=jsonl_path: fs.open(p, "r"),
                ATTEMPTS,
            ) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    scanned += 1
                    row = json.loads(line)
                    record_id = row.get("id")
                    if record_id in selected_ids:
                        code_path = row.get("semantic_code_path", "")
                        if code_path:
                            local_ids.add(code_path)
                        local_records.append(line)
        except Exception as exc:
            log(
                "maskgct_jsonl_error",
                dataset=DATASET,
                path=jsonl_path,
                error=str(exc),
            )
        return scanned, local_records, local_ids

    needed_bins: set[str] = set()
    per_shard_records: dict[str, list[str]] = {}
    total_scanned = 0
    scan_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(scan_one, jsonl_path): jsonl_path
            for jsonl_path in jsonl_files
        }
        for future in as_completed(futures):
            jsonl_path = futures[future]
            scanned, records, ids = future.result()
            with scan_lock:
                total_scanned += scanned
                needed_bins.update(ids)
                per_shard_records[jsonl_path] = records
                done = len(per_shard_records)
            if done % 200 == 0 or done == len(jsonl_files):
                log(
                    "codes_scan_progress",
                    dataset=DATASET,
                    done=done,
                    total=len(jsonl_files),
                )
    filtered_records: list[str] = []
    for jsonl_path in jsonl_files:
        filtered_records.extend(per_shard_records[jsonl_path])
    log(
        "maskgct_scan",
        dataset=DATASET,
        total_scanned=total_scanned,
        filtered_records=len(filtered_records),
        needed_bins=len(needed_bins),
    )

    needed_bin_files = [
        bin_file for bin_file in bin_files
        if bin_file.rsplit("/", 1)[-1] in needed_bins
    ]
    log(
        "codes_needed_bins",
        dataset=DATASET,
        needed_bin_files=len(needed_bin_files),
    )

    done_count = 0
    lock = threading.Lock()

    def download_one(bin_file: str) -> None:
        nonlocal done_count
        bin_name = bin_file.rsplit("/", 1)[-1]
        target = bins_dir / bin_name
        if target.is_file() and target.stat().st_size > 0:
            with lock:
                done_count += 1
            return
        tmp = target.with_name(f".{target.name}.tmp")
        try:
            with retry(
                f"download {bin_file}",
                lambda p=bin_file: fs.open(p, "rb"),
                ATTEMPTS,
            ) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(target)
            with lock:
                done_count += 1
                if done_count % 100 == 0 or done_count == len(needed_bin_files):
                    log(
                        "codes_bins_progress",
                        dataset=DATASET,
                        done=done_count,
                        total=len(needed_bin_files),
                    )
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            log(
                "maskgct_bin_error",
                dataset=DATASET,
                file=bin_name,
                error=str(exc),
            )

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(download_one, bin_file)
            for bin_file in needed_bin_files
        ]
        for future in as_completed(futures):
            future.result()

    manifest_path = manifests_dir / f"{DATASET}.jsonl"
    manifest_tmp = manifest_path.with_name(f"{manifest_path.name}.tmp")
    with manifest_tmp.open("w", encoding="utf-8") as f:
        for line in filtered_records:
            row = json.loads(line)
            bin_name = row.get("semantic_code_path", "")
            row["semantic_code_path"] = str(bins_dir / bin_name)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_tmp.replace(manifest_path)
    log(
        "maskgct_manifest_written",
        dataset=DATASET,
        records=len(filtered_records),
        path=str(manifest_path),
    )
    log(
        "MASKGCT_CODES_COMPLETE",
        dataset=DATASET,
        output=str(codes_root),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
