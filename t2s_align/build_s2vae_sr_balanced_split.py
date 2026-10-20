#!/usr/bin/env python3
"""Build an approximately duration-balanced, speaker-pairable SR subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


RATE_ALIASES = {
    "16": 16000,
    "22": 22050,
    "24": 24000,
    "32": 32000,
    "36": 36000,
    "44": 44100,
    "48": 48000,
}


def parse_rates(value: str) -> list[int]:
    rates = []
    for token in value.split(","):
        token = token.strip().lower().removesuffix("k")
        if not token:
            continue
        rate = RATE_ALIASES.get(token, int(token))
        rates.append(int(rate))
    if len(rates) != len(set(rates)) or not rates:
        raise argparse.ArgumentTypeError("rates must be a non-empty unique list")
    return rates


def parse_rate_hours(value: str) -> dict[int, float]:
    targets = {}
    for item in value.split(","):
        rate_text, separator, hours_text = item.strip().partition(":")
        if not separator:
            raise argparse.ArgumentTypeError("rate-hours entries must be RATE:HOURS")
        rate = parse_rates(rate_text)[0]
        hours = float(hours_text)
        if rate in targets or hours <= 0:
            raise argparse.ArgumentTypeError("rate-hours rates must be unique and hours positive")
        targets[rate] = hours
    if not targets:
        raise argparse.ArgumentTypeError("rate-hours must not be empty")
    return targets


def stable_unit(seed: int, value: str) -> float:
    digest = hashlib.blake2b(
        f"{seed}\0{value}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def speaker_key(row: dict) -> tuple[str, str] | None:
    speaker = row.get("speaker_id")
    if not speaker:
        return None
    return str(row.get("language") or ""), str(speaker)


def usable(row: dict, rates: set[int]) -> bool:
    try:
        duration = float(row.get("duration") or 0.0)
        sample_rate = int(row.get("sample_rate") or 0)
    except (TypeError, ValueError):
        return False
    return (
        0.5 <= duration <= 30.0
        and sample_rate in rates
        and speaker_key(row) is not None
        and bool(row.get("semantic_code_path"))
        and row.get("semantic_code_offset") is not None
        and bool(row.get("semantic_code_length"))
        and row.get("vae_latent_offset") is not None
        and bool(row.get("vae_latent_length"))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", type=parse_rates, default=parse_rates("24,32,44,48"))
    parser.add_argument("--total-hours", type=float, default=1000.0)
    parser.add_argument(
        "--rate-hours",
        type=parse_rate_hours,
        default=None,
        help="Optional unequal targets, e.g. 24:250,32:250,36:3,44:250,48:250.",
    )
    parser.add_argument(
        "--include-all-rates",
        type=parse_rates,
        default=[],
        help="Also include every pairable row at these scarce native rates.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--candidate-factor",
        type=float,
        default=2.0,
        help="Expected candidate duration / target duration before group selection.",
    )
    args = parser.parse_args()
    if args.total_hours <= 0 or args.candidate_factor < 1.0:
        parser.error("total-hours must be positive and candidate-factor >= 1")

    target_by_rate = (
        args.rate_hours
        if args.rate_hours is not None
        else {rate: args.total_hours / len(args.rates) for rate in args.rates}
    )
    overlap = set(target_by_rate) & set(args.include_all_rates)
    if overlap:
        parser.error(f"rates cannot appear in both targets and include-all: {sorted(overlap)}")
    target_by_rate.update({rate: float("inf") for rate in args.include_all_rates})
    ordered_rates = list(target_by_rate)
    rates = set(ordered_rates)
    available_seconds = Counter()
    available_rows = Counter()
    print(f"[Pass1] scanning {args.input}", flush=True)
    with args.input.open() as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not usable(row, rates):
                continue
            rate = int(row["sample_rate"])
            available_seconds[rate] += float(row["duration"])
            available_rows[rate] += 1
            if line_number % 2_000_000 == 0:
                print(f"[Pass1] {line_number:,} rows", flush=True)

    probabilities = {}
    for rate in ordered_rates:
        target_hours = target_by_rate[rate]
        hours = available_seconds[rate] / 3600.0
        include_all = math.isinf(target_hours)
        if not include_all and hours < target_hours:
            raise RuntimeError(
                f"sample rate {rate} has only {hours:.2f} usable hours; "
                f"need {target_hours:.2f}"
            )
        probabilities[rate] = (
            1.0
            if include_all
            else min(1.0, target_hours * args.candidate_factor / hours)
        )
        print(
            f"[Available] {rate}: {available_rows[rate]:,} rows / {hours:.2f}h; "
            f"speaker candidate probability={probabilities[rate]:.6f}",
            flush=True,
        )

    candidates: dict[int, dict[tuple[str, str], list[dict]]] = {
        rate: defaultdict(list) for rate in ordered_rates
    }
    print("[Pass2] collecting deterministic speaker candidates", flush=True)
    with args.input.open() as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not usable(row, rates):
                continue
            rate = int(row["sample_rate"])
            key = speaker_key(row)
            assert key is not None
            priority_key = f"{rate}\0{key[0]}\0{key[1]}"
            if stable_unit(args.seed, priority_key) < probabilities[rate]:
                candidates[rate][key].append(row)
            if line_number % 2_000_000 == 0:
                print(f"[Pass2] {line_number:,} rows", flush=True)

    selected: list[dict] = []
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "target_total_hours": sum(v for v in target_by_rate.values() if math.isfinite(v)),
        "target_hours_by_rate": {
            str(k): (v if math.isfinite(v) else "all") for k, v in target_by_rate.items()
        },
        "rates": {},
    }
    for rate in ordered_rates:
        target_hours = target_by_rate[rate]
        include_all = math.isinf(target_hours)
        groups = []
        for key, rows in candidates[rate].items():
            # Pairing needs a second utterance and at least one >=3 s prompt.
            if len(rows) < 2 or max(float(row["duration"]) for row in rows) < 3.0:
                continue
            priority = stable_unit(args.seed + 1, f"{rate}\0{key[0]}\0{key[1]}")
            groups.append((priority, key, rows))
        groups.sort(key=lambda item: item[0])
        rate_rows = []
        seconds = 0.0
        speakers = 0
        for _priority, _key, rows in groups:
            if not include_all and seconds >= target_hours * 3600:
                break
            rate_rows.extend(rows)
            seconds += sum(float(row["duration"]) for row in rows)
            speakers += 1
        hours = seconds / 3600.0
        if not include_all and hours < target_hours * 0.98:
            raise RuntimeError(
                f"candidate pool for {rate} yielded only {hours:.2f}h after "
                "speaker-pairability filtering; raise --candidate-factor"
            )
        selected.extend(rate_rows)
        summary["rates"][str(rate)] = {
            "rows": len(rate_rows),
            "hours": hours,
            "speakers": speakers,
            "datasets": dict(Counter(str(row.get("dataset")) for row in rate_rows)),
            "languages": dict(Counter(str(row.get("language")) for row in rate_rows)),
        }
        print(
            f"[Selected] {rate}: {len(rate_rows):,} rows / {hours:.2f}h / "
            f"{speakers:,} speakers",
            flush=True,
        )

    selected.sort(key=lambda row: stable_unit(args.seed + 2, str(row.get("id"))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as sink:
        for row in selected:
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary["actual_total_rows"] = len(selected)
    summary["actual_total_hours"] = sum(
        item["hours"] for item in summary["rates"].values()
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"[Done] {len(selected):,} rows / {summary['actual_total_hours']:.2f}h "
        f"-> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
