#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def pick(d: dict, *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-root", required=True)
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    metric_root = Path(args.metric_root)
    rows = []
    for model_dir in sorted(p for p in metric_root.iterdir() if p.is_dir()):
        paired = load_json(model_dir / "paired_metrics.json")
        audioldm = load_json(model_dir / "audioldm_metrics.json")
        speaker = load_json(model_dir / "speaker_similarity.json")
        row = {
            "model": model_dir.name,
            "pairs": pick(paired, "pairs"),
            "si_sdr_mean": pick(paired, "si_sdr", "mean"),
            "lsd_mean": pick(paired, "lsd", "mean"),
            "fad": pick(audioldm, "metrics", "frechet_audio_distance"),
            "fd": pick(audioldm, "metrics", "frechet_distance"),
            "is_mean": pick(audioldm, "metrics", "inception_score_mean"),
            "is_std": pick(audioldm, "metrics", "inception_score_std"),
            "kl_sigmoid": pick(audioldm, "metrics", "kullback_leibler_divergence_sigmoid"),
            "kl_softmax": pick(audioldm, "metrics", "kullback_leibler_divergence_softmax"),
            "speaker_similarity_mean": pick(speaker, "mean"),
            "speaker_similarity_count": pick(speaker, "count"),
        }
        rows.append(row)

    columns = [
        "model",
        "pairs",
        "si_sdr_mean",
        "lsd_mean",
        "fad",
        "fd",
        "is_mean",
        "is_std",
        "kl_sigmoid",
        "kl_softmax",
        "speaker_similarity_mean",
        "speaker_similarity_count",
    ]
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join("" if row.get(c) is None else str(row.get(c)) for c in columns) + "\n")
    Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out_tsv} rows={len(rows)}")


if __name__ == "__main__":
    main()
