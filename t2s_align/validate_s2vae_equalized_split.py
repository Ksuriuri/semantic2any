#!/usr/bin/env python3
"""Validate a stored-code/stored-latent S2VAE training manifest."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def identity(row):
    dataset = str(row.get("dataset") or "")
    language = str(row.get("language") or "")
    parts = str(row.get("id") or "").split("/")
    speaker = parts[-2] if len(parts) >= 3 else str(row.get("speaker_id") or "")
    return dataset, language, speaker


def latent_path(code_path: Path) -> Path:
    dataset = code_path.parent.name
    stem = code_path.name[:-7] if code_path.name.endswith(".u2.bin") else code_path.stem
    return code_path.parent.parent.parent / "dotstts-latents" / dataset / f"{stem}.f16.bin"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()

    held = set()
    with args.valid.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                held.add(identity(json.loads(line)))

    rows = Counter()
    seconds = Counter()
    dataset_seconds = defaultdict(Counter)
    groups = defaultdict(lambda: [0, 0.0])
    overlap = Counter()
    missing = Counter()
    out_of_bounds = Counter()
    size_cache = {}
    reservoir = []
    with args.manifest.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            rate = int(row["sample_rate"])
            duration = float(row["duration"])
            rows[rate] += 1
            seconds[rate] += duration
            dataset_seconds[rate][str(row["dataset"])] += duration
            ident = identity(row)
            # The training sampler pairs rows by the manifest's speaker_id.
            # Keep the ID-path identity only for holdout leakage detection,
            # where it deliberately normalizes historical speaker_id variants.
            group = groups[(
                rate,
                str(row.get("dataset") or ""),
                str(row.get("language") or ""),
                str(row.get("speaker_id") or ""),
            )]
            group[0] += 1
            group[1] = max(group[1], duration)
            if ident in held:
                overlap[rate] += 1

            code = Path(row["semantic_code_path"])
            latent = latent_path(code)
            for kind, path in (("code", code), ("latent", latent)):
                if path not in size_cache:
                    size_cache[path] = path.stat().st_size if path.is_file() else -1
                if size_cache[path] < 0:
                    missing[kind] += 1
            code_end = (int(row["semantic_code_offset"]) + int(row["semantic_code_length"])) * 2
            latent_end = (int(row["vae_latent_offset"]) + int(row["vae_latent_length"])) * 128 * 2
            if code_end > size_cache[code]:
                out_of_bounds["code"] += 1
            if latent_end > size_cache[latent]:
                out_of_bounds["latent"] += 1

            score = int.from_bytes(hashlib.blake2b(str(row["id"]).encode(), digest_size=8).digest(), "big")
            item = (-score, row)
            if len(reservoir) < args.samples:
                heapq.heappush(reservoir, item)
            elif item > reservoir[0]:
                heapq.heapreplace(reservoir, item)
            if line_no % 500_000 == 0:
                print(f"[Scan] {line_no:,}", flush=True)

    bad_groups = sum(1 for count, longest in groups.values() if count < 2 or longest < 3.0)
    sample_errors = []
    for _score, row in reservoir:
        code = Path(row["semantic_code_path"])
        latent = latent_path(code)
        try:
            c = np.memmap(code, mode="r", dtype="<u2")
            cs = int(row["semantic_code_offset"])
            cn = int(row["semantic_code_length"])
            v = np.memmap(latent, mode="r", dtype="<f2")
            vs = int(row["vae_latent_offset"]) * 128
            vn = int(row["vae_latent_length"]) * 128
            assert c[cs : cs + cn].size == cn
            assert v[vs : vs + vn].size == vn
            assert np.isfinite(v[vs : vs + vn].astype(np.float32)).all()
        except Exception as exc:
            sample_errors.append({"id": row.get("id"), "error": repr(exc)})

    result = {
        "manifest": str(args.manifest),
        "rows": sum(rows.values()),
        "hours": sum(seconds.values()) / 3600,
        "rates": {
            str(rate): {
                "rows": rows[rate],
                "hours": seconds[rate] / 3600,
                "dataset_hours": {k: v / 3600 for k, v in sorted(dataset_seconds[rate].items())},
            }
            for rate in sorted(rows)
        },
        "speaker_groups": len(groups),
        "bad_pairable_groups": bad_groups,
        "valid_overlap_rows": dict(overlap),
        "missing_paths": dict(missing),
        "out_of_bounds": dict(out_of_bounds),
        "sampled_loads": len(reservoir),
        "sample_errors": sample_errors,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if bad_groups or overlap or missing or out_of_bounds or sample_errors:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
