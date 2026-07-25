#!/usr/bin/env python3
"""Aggregate per-run metric JSONs from a directory of run subdirs into a single TSV/JSON.

Column order matches the standard evaluation sheet:
  run | FAD | FD | LSD | Low_LSD | High_LSD | SI-SDR | Speaker_similarity | MOS | IS | IS_std | pairs | KL_sigmoid | KL_softmax
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def pick(d: dict, *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# Column definitions: (output_name, display_name, source, json_path...)
COLUMNS = [
    ("run",                 "run",               None),
    ("fad",                 "FAD↓",              ("audioldm", "metrics", "frechet_audio_distance")),
    ("fd",                  "FD↓",               ("audioldm", "metrics", "frechet_distance")),
    ("lsd_mean",            "LSD↓",              ("paired",   "lsd", "mean")),
    ("lsd_low_mean",        "Low_LSD↓",          ("paired",   "lsd_low", "mean")),
    ("lsd_high_mean",       "High_LSD↓",         ("paired",   "lsd_high", "mean")),
    ("si_sdr_mean",         "SI-SDR↑",           ("paired",   "si_sdr", "mean")),
    ("speaker_similarity",  "Speaker_sim↑",      ("speaker",  "mean")),
    ("mos",                 "MOS",               None),
    ("is_mean",             "IS↑",               ("audioldm", "metrics", "inception_score_mean")),
    ("is_std",              "IS_std",            ("audioldm", "metrics", "inception_score_std")),
    ("pairs",               "pairs",             ("paired",   "pairs")),
    ("kl_sigmoid",          "KL_sigmoid↓",       ("audioldm", "metrics", "kullback_leibler_divergence_sigmoid")),
    ("kl_softmax",          "KL_softmax↓",       ("audioldm", "metrics", "kullback_leibler_divergence_softmax")),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-root", required=True,
                        help="directory whose subdirectories each contain metric JSONs for one run")
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    metric_root = Path(args.metric_root)
    rows = []
    for run_dir in sorted(p for p in metric_root.iterdir() if p.is_dir()):
        paired   = load_json_optional(run_dir / "paired_metrics.json")
        audioldm = load_json_optional(run_dir / "audioldm_metrics.json")
        speaker  = load_json_optional(run_dir / "speaker_similarity.json")
        sources  = {"paired": paired, "audioldm": audioldm, "speaker": speaker}

        row: dict = {}
        for col_key, _, path in COLUMNS:
            if col_key == "run":
                row["run"] = run_dir.name
            elif col_key == "mos":
                row["mos"] = None
            elif path is not None:
                src, *keys = path
                row[col_key] = pick(sources[src], *keys)
            else:
                row[col_key] = None
        rows.append(row)

    col_keys    = [c[0] for c in COLUMNS]
    col_headers = [c[1] for c in COLUMNS]

    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w") as f:
        f.write("\t".join(col_headers) + "\n")
        for row in rows:
            f.write("\t".join("" if row.get(k) is None else str(row[k]) for k in col_keys) + "\n")
    Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out_tsv}  rows={len(rows)}")


if __name__ == "__main__":
    main()
