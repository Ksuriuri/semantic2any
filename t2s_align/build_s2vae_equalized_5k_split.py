#!/usr/bin/env python3
"""Build the local five-rate S2VAE set from complete code+latent shards.

Policy:
* 24/32/44.1/48 kHz are equalized to the complete pairable 48 kHz pool.
* all pairable 36 kHz rows are retained.
* 44.1 kHz keeps every non-podcast group first; the remaining duration is
  sampled evenly across podcast language datasets.

Selection is at speaker-group granularity so every retained group remains
pairable (at least two rows, with at least one >= 3 seconds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SKIP_DATASETS = {"_extra500_state", "manifests_merged", "splits"}
MAIN_RATES = (24000, 32000, 44100, 48000)
ALL_RATES = set(MAIN_RATES) | {36000}
LOOKUP_PATH = "/mnt/data_sdd/hhy/artifacts/semantic_codec/indextts25_decoder.pt"
LOOKUP_SHA256 = "dbbd497182fdd5acfa48701a17d5dcd495496d18184fab91a6f28735a8ad47c8"


def complete_stems(root: Path, suffix: str) -> set[str]:
    if not root.is_dir():
        return set()
    jsonl = {path.stem for path in root.glob("*.jsonl")}
    binary = {path.name[: -len(suffix)] for path in root.glob(f"*{suffix}")}
    return jsonl & binary


def stable_unit(seed: int, value: str) -> float:
    digest = hashlib.blake2b(f"{seed}\0{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def holdout_identity(row: dict, dataset: str) -> tuple[str, str, str]:
    language = str(row.get("language") or "")
    record_id = str(row.get("id") or "")
    parts = record_id.split("/")
    # ID path is more stable than historical speaker_id variants such as a
    # language suffix appearing only in the code manifest.
    speaker = parts[-2] if len(parts) >= 3 else str(row.get("speaker_id") or "")
    return dataset, language, speaker


def group_key(
    row: dict,
    dataset: str,
    holdout: set[tuple[str, str, str]],
    *,
    all_rates: bool = False,
) -> tuple[int, str, str, str] | None:
    try:
        rate = int(row.get("sample_rate") or 0)
        duration = float(row.get("duration") or 0)
    except (TypeError, ValueError):
        return None
    speaker = row.get("speaker_id")
    if (
        rate <= 0
        or (not all_rates and rate not in ALL_RATES)
        or not (0.5 <= duration <= 30.0)
        or not speaker
        or holdout_identity(row, dataset) in holdout
    ):
        return None
    return rate, dataset, str(row.get("language") or ""), str(speaker)


def shard_inventory(root: Path):
    code_root = root / "indextts25-codes"
    latent_root = root / "dotstts-latents"
    result = []
    for code_dir in sorted(p for p in code_root.iterdir() if p.is_dir() and p.name not in SKIP_DATASETS):
        dataset = code_dir.name
        latent_dir = latent_root / dataset
        stems = sorted(
            complete_stems(code_dir, ".u2.bin")
            & complete_stems(latent_dir, ".f16.bin")
        )
        result.extend((dataset, code_dir, latent_dir, stem) for stem in stems)
    return result


def latent_index(path: Path) -> dict[str, tuple[int, int]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.get("id")
            offset = row.get("vae_latent_offset")
            length = row.get("vae_latent_length")
            if row.get("status") == "encoded" and isinstance(rid, str) and offset is not None and length:
                result[rid] = (int(offset), int(length))
    return result


def choose_until(groups, target_seconds, seed, namespace):
    ordered = sorted(
        groups,
        key=lambda item: stable_unit(seed, f"{namespace}\0{item[0]}")
    )
    chosen = set()
    seconds = 0.0
    for key, stats in ordered:
        if seconds >= target_seconds:
            break
        chosen.add(key)
        seconds += stats[2]
    return chosen, seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("/mnt/data_3t_2/datasets/preprocess/s2vae-train-data"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--all-pairable",
        action="store_true",
        help="Keep every complete pairable group at every positive sample rate.",
    )
    parser.add_argument(
        "--exclude-valid", type=Path,
        default=Path("/mnt/data_3t_2/datasets/preprocess/s2vae-train-data/indextts25-codes/splits/seed1234_valid2000_spk_holdout_20260828/valid.jsonl"),
    )
    args = parser.parse_args()

    holdout: set[tuple[str, str, str]] = set()
    if args.exclude_valid.is_file():
        with args.exclude_valid.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                holdout.add(holdout_identity(row, str(row.get("dataset") or "")))
    print(f"[Holdout] {len(holdout):,} dataset/language/speaker identities", flush=True)

    inventory = shard_inventory(args.root)
    print(f"[Inventory] {len(inventory):,} complete shards", flush=True)
    # stats = [row count, max duration, total seconds]
    stats: dict[tuple[int, str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for index, (dataset, code_dir, latent_dir, stem) in enumerate(inventory, 1):
        ids = set(latent_index(latent_dir / f"{stem}.jsonl"))
        with (code_dir / f"{stem}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "encoded" or row.get("id") not in ids:
                    continue
                key = group_key(row, dataset, holdout, all_rates=args.all_pairable)
                if key is None:
                    continue
                duration = float(row["duration"])
                item = stats[key]
                item[0] += 1
                item[1] = max(item[1], duration)
                item[2] += duration
        if index % 1000 == 0 or index == len(inventory):
            print(f"[Pass1] {index:,}/{len(inventory):,} shards; {len(stats):,} groups", flush=True)

    pairable = {key: value for key, value in stats.items() if value[0] >= 2 and value[1] >= 3.0}
    available = Counter()
    for key, value in pairable.items():
        available[key[0]] += value[2]
    selected: set[tuple[int, str, str, str]] = set()
    selected_seconds = Counter()
    target_seconds = None
    if args.all_pairable:
        selected = set(pairable)
        for key, value in pairable.items():
            selected_seconds[key[0]] += value[2]
        print(
            "[Select] all pairable; "
            + " ".join(f"{rate}={available[rate]/3600:.3f}h" for rate in sorted(available)),
            flush=True,
        )
    else:
        target_seconds = available[48000]
        print(
            f"[Target] equalized={target_seconds / 3600:.6f}h; "
            + " ".join(f"{rate}={available[rate]/3600:.3f}h" for rate in sorted(available)),
            flush=True,
        )
        # 24/32 are deterministic speaker samples; 48 and scarce 36 are all-in.
        for rate in (24000, 32000):
            pool = [(key, value) for key, value in pairable.items() if key[0] == rate]
            chosen, seconds = choose_until(pool, target_seconds, args.seed, str(rate))
            selected.update(chosen)
            selected_seconds[rate] = seconds
        for rate in (36000, 48000):
            chosen = {key for key in pairable if key[0] == rate}
            selected.update(chosen)
            selected_seconds[rate] = sum(pairable[key][2] for key in chosen)

        nonpod = {key for key in pairable if key[0] == 44100 and not key[1].startswith("podcast_")}
        selected.update(nonpod)
        nonpod_seconds = sum(pairable[key][2] for key in nonpod)
        remaining = max(0.0, target_seconds - nonpod_seconds)
        podcast_datasets = sorted({key[1] for key in pairable if key[0] == 44100 and key[1].startswith("podcast_")})
        quota = remaining / len(podcast_datasets)
        podcast_selected_seconds = Counter()
        for dataset in podcast_datasets:
            pool = [
                (key, value) for key, value in pairable.items()
                if key[0] == 44100 and key[1] == dataset
            ]
            chosen, seconds = choose_until(pool, quota, args.seed + 1, dataset)
            selected.update(chosen)
            podcast_selected_seconds[dataset] = seconds
        selected_seconds[44100] = nonpod_seconds + sum(podcast_selected_seconds.values())

        print(
            f"[Select] 44100 nonpod={nonpod_seconds/3600:.3f}h podcast="
            f"{sum(podcast_selected_seconds.values())/3600:.3f}h; "
            + " ".join(f"{name.rsplit('_',1)[-1]}={sec/3600:.2f}h" for name, sec in podcast_selected_seconds.items()),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    actual_seconds = Counter()
    actual_rows = Counter()
    actual_speakers: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    actual_datasets: dict[int, Counter] = defaultdict(Counter)
    with temp.open("w", encoding="utf-8") as sink:
        for index, (dataset, code_dir, latent_dir, stem) in enumerate(inventory, 1):
            latents = latent_index(latent_dir / f"{stem}.jsonl")
            code_bin = (code_dir / f"{stem}.u2.bin").resolve()
            with (code_dir / f"{stem}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rid = row.get("id")
                    latent = latents.get(rid)
                    if row.get("status") != "encoded" or latent is None:
                        continue
                    key = group_key(row, dataset, holdout, all_rates=args.all_pairable)
                    if key not in selected:
                        continue
                    out = {
                        "id": rid,
                        "dataset": dataset,
                        "audio_path": row.get("audio_path", ""),
                        "speaker_id": row.get("speaker_id"),
                        "language": row.get("language", ""),
                        "duration": float(row["duration"]),
                        "sample_rate": int(row["sample_rate"]),
                        "semantic_code_path": str(code_bin),
                        "semantic_code_offset": int(row["semantic_code_offset"]),
                        "semantic_code_length": int(row["semantic_code_length"]),
                        "semantic_frame_rate": float(row.get("semantic_frame_rate", 25.0)),
                        "codebook_size": int(row.get("codebook_size", 8192)),
                        "semantic_codec": "indextts2.5",
                        "semantic_lookup_path": LOOKUP_PATH,
                        "semantic_lookup_sha256": LOOKUP_SHA256,
                        "vae_latent_offset": latent[0],
                        "vae_latent_length": latent[1],
                    }
                    sink.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
                    rate = out["sample_rate"]
                    actual_rows[rate] += 1
                    actual_seconds[rate] += out["duration"]
                    actual_speakers[rate].add((dataset, str(out["language"]), str(out["speaker_id"])))
                    actual_datasets[rate][dataset] += out["duration"]
            if index % 1000 == 0 or index == len(inventory):
                print(f"[Pass2] {index:,}/{len(inventory):,} shards; rows={sum(actual_rows.values()):,}", flush=True)
    temp.replace(args.output)

    summary = {
        "root": str(args.root),
        "output": str(args.output),
        "seed": args.seed,
        "policy": {
            "mode": "all_pairable" if args.all_pairable else "equalized_5k",
            "equalize_to_rate": None if args.all_pairable else 48000,
            "target_hours": None if target_seconds is None else target_seconds / 3600.0,
            "include_all_rate": None if args.all_pairable else 36000,
            "rate_44100": "all pairable" if args.all_pairable else "all pairable non-podcast, then equal-duration sample across podcast datasets",
            "excluded_valid": str(args.exclude_valid),
            "excluded_valid_identities": len(holdout),
        },
        "rates": {
            str(rate): {
                "rows": actual_rows[rate],
                "hours": actual_seconds[rate] / 3600.0,
                "speakers": len(actual_speakers[rate]),
                "dataset_hours": {
                    name: seconds / 3600.0
                    for name, seconds in sorted(actual_datasets[rate].items())
                },
            }
            for rate in sorted(actual_rows)
        },
        "total_rows": sum(actual_rows.values()),
        "total_hours": sum(actual_seconds.values()) / 3600.0,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"[Done] {summary['total_rows']:,} rows / {summary['total_hours']:.3f}h -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
