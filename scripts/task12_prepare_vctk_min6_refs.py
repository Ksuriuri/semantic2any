#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import soundfile as sf


def ref_name(row: dict) -> str:
    source = row.get("source", "")
    if "#" in source:
        source = source.split("#", 1)[1]
    if source:
        return "vctk__" + source.replace("/", "_").rsplit(".", 1)[0] + ".wav"
    return row["id"].replace("/", "_") + ".wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full-ref-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--min-duration", type=float, default=6.0)
    parser.add_argument("--prompt-seconds", type=float, default=3.01)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--sample-rate-mel", type=int, default=22050)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    full_ref_dir = Path(args.full_ref_dir)
    out_root = Path(args.out_root)
    input_dir = out_root / "input_full_wav_min6"
    ref_reference = out_root / "reference_tail_style-reference"
    ref_none = out_root / "reference_tail_style-none"
    subset_manifest = out_root / "manifest_min6_prompt3p01.jsonl"

    input_dir.mkdir(parents=True, exist_ok=True)
    ref_reference.mkdir(parents=True, exist_ok=True)
    ref_none.mkdir(parents=True, exist_ok=True)

    prompt_frames = int(args.prompt_seconds * args.sample_rate_mel / args.hop_length)
    cut_seconds = prompt_frames * args.hop_length / args.sample_rate_mel

    selected: list[dict] = []
    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row["duration"]) < args.min_duration:
                continue
            name = ref_name(row)
            src = full_ref_dir / name
            if not src.exists():
                raise FileNotFoundError(src)
            selected.append({**row, "reference_wav": str(src), "file_name": name})

    for row in selected:
        src = Path(row["reference_wav"])
        link = input_dir / src.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src, link)

        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        cut = int(round(cut_seconds * sr))
        if len(wav) - cut <= int(round(3.0 * sr)):
            raise ValueError(f"target shorter than 3s after cut: {src} sr={sr} len={len(wav)} cut={cut}")
        stem = src.stem
        targets = [
            ref_reference / f"{stem}_s2mel_style-reference.wav",
            ref_none / f"{stem}_s2mel_style-none.wav",
        ]
        for dst in targets:
            sf.write(dst, wav[cut:], sr)

    with subset_manifest.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "manifest": str(manifest),
        "full_ref_dir": str(full_ref_dir),
        "out_root": str(out_root),
        "input_dir": str(input_dir),
        "reference_tail_style_reference": str(ref_reference),
        "reference_tail_style_none": str(ref_none),
        "subset_manifest": str(subset_manifest),
        "min_duration": args.min_duration,
        "prompt_seconds_arg": args.prompt_seconds,
        "prompt_frames": prompt_frames,
        "cut_seconds": cut_seconds,
        "count": len(selected),
        "speakers": len({r["speaker_id"] for r in selected}),
        "min_source_duration": min(float(r["duration"]) for r in selected),
        "max_source_duration": max(float(r["duration"]) for r in selected),
    }
    (out_root / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
