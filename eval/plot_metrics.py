#!/usr/bin/env python3
"""Generate per-metric line plots from a summarize_eval.py JSON output.

Run name format: <experiment>_step<N>[_<suffix>]
  e.g. exp06_step34000_prompt3_cfg0_steps25
       my_run_step20000

Each experiment becomes a line; training step is the x-axis.
One PNG is written per metric to --out-dir.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


METRICS = [
    ("fad",               "FAD ↓",              True),
    ("fd",                "FD ↓",               True),
    ("lsd_mean",          "LSD ↓",              True),
    ("lsd_low_mean",      "Low LSD ↓ (20 Hz–4 kHz)",  True),
    ("lsd_high_mean",     "High LSD ↓ (4 kHz–Nyquist)", True),
    ("si_sdr_mean",       "SI-SDR ↑",           False),
    ("speaker_similarity","Speaker Similarity ↑", False),
    ("is_mean",           "IS ↑",               False),
    ("is_std",            "IS std",              None),
    ("kl_sigmoid",        "KL sigmoid ↓",        True),
    ("kl_softmax",        "KL softmax ↓",        True),
]

_STEP_RE = re.compile(r"_step(\d+)")
_EXP_RE  = re.compile(r"^(.+?)_step\d+")


def parse_run(name: str):
    """Return (experiment_label, step) or (name, None) if no step found."""
    m_step = _STEP_RE.search(name)
    if not m_step:
        return name, None
    step = int(m_step.group(1))
    m_exp = _EXP_RE.match(name)
    exp = m_exp.group(1) if m_exp else name
    return exp, step


def load_rows(summary_json: Path) -> list[dict]:
    with summary_json.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True, help="output of summarize_eval.py --out-json")
    ap.add_argument("--out-dir", required=True, help="directory to write PNG files")
    ap.add_argument("--figsize", default="8x5", help="WxH in inches (default: 8x5)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--experiments", nargs="*",
                    help="only plot these experiment labels (default: all)")
    args = ap.parse_args()

    rows = load_rows(Path(args.summary_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fw, fh = (float(x) for x in args.figsize.split("x"))

    # Group by experiment → list of (step, value)
    by_exp: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    unparsed = []
    for row in rows:
        exp, step = parse_run(row["run"])
        if step is None:
            unparsed.append(row["run"])
            continue
        for metric_key, _, _ in METRICS:
            val = row.get(metric_key)
            if val is not None:
                try:
                    by_exp[exp][metric_key].append((step, float(val)))
                except (TypeError, ValueError):
                    pass

    if unparsed:
        print(f"warning: could not parse step from {len(unparsed)} run(s): {unparsed[:5]}")

    filter_exps = set(args.experiments) if args.experiments else None
    experiments = sorted(by_exp.keys()) if filter_exps is None else sorted(filter_exps & by_exp.keys())

    for metric_key, metric_label, lower_is_better in METRICS:
        series = {exp: sorted(by_exp[exp].get(metric_key, [])) for exp in experiments}
        series = {exp: pts for exp, pts in series.items() if pts}
        if not series:
            continue

        fig, ax = plt.subplots(figsize=(fw, fh))
        for exp, pts in series.items():
            steps, vals = zip(*pts)
            ax.plot(steps, vals, marker="o", markersize=4, label=exp)

        ax.set_xlabel("Training step")
        ax.set_ylabel(metric_label)
        direction = " (lower = better)" if lower_is_better else (" (higher = better)" if lower_is_better is False else "")
        ax.set_title(f"{metric_label}{direction}")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_key = metric_key.replace("_", "-")
        out_path = out_dir / f"{safe_key}_by_step.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"wrote {out_path}")

    print(f"done — {len(list(out_dir.glob('*.png')))} plots in {out_dir}")


if __name__ == "__main__":
    main()
