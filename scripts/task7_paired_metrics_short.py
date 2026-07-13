#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import stft
from tqdm import tqdm


def load_mono(path, sr):
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav.astype(np.float32)


def si_sdr(est, ref, eps=1e-8):
    n = min(len(est), len(ref))
    est = est[:n] - np.mean(est[:n])
    ref = ref[:n] - np.mean(ref[:n])
    target = np.dot(est, ref) * ref / (np.dot(ref, ref) + eps)
    noise = est - target
    return float(10 * np.log10((np.sum(target**2) + eps) / (np.sum(noise**2) + eps)))


def lsd(est, ref, sr, n_fft=2048, hop_length=512, eps=1e-7):
    n = min(len(est), len(ref))
    est = est[:n]
    ref = ref[:n]
    if n < 2:
        return float("nan")
    nperseg = min(n_fft, n)
    noverlap = min(max(0, nperseg - 1), n_fft - hop_length)
    _, _, ze = stft(est, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    _, _, zr = stft(ref, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    me = np.maximum(np.abs(ze), eps)
    mr = np.maximum(np.abs(zr), eps)
    frames = min(me.shape[1], mr.shape[1])
    bins = min(me.shape[0], mr.shape[0])
    if frames == 0 or bins == 0:
        return float("nan")
    diff = 20 * (np.log10(me[:bins, :frames]) - np.log10(mr[:bins, :frames]))
    return float(np.mean(np.sqrt(np.mean(diff**2, axis=0))))


def summarize(vals):
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else math.nan,
        "median": float(np.median(arr)) if arr.size else math.nan,
        "std": float(np.std(arr)) if arr.size else math.nan,
        "min": float(np.min(arr)) if arr.size else math.nan,
        "max": float(np.max(arr)) if arr.size else math.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--generated-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    gen_dir = Path(args.generated_dir)
    names = sorted(set(p.name for p in ref_dir.glob("*.wav")) & set(p.name for p in gen_dir.glob("*.wav")))
    if args.limit:
        names = names[: args.limit]
    rows = []
    sis = []
    lsds = []
    for name in tqdm(names, desc=gen_dir.name):
        ref = load_mono(ref_dir / name, args.sample_rate)
        gen = load_mono(gen_dir / name, args.sample_rate)
        s = si_sdr(gen, ref)
        l = lsd(gen, ref, args.sample_rate)
        rows.append({"file": name, "si_sdr": s, "lsd": l})
        sis.append(s)
        lsds.append(l)
    out = {
        "reference_dir": str(ref_dir),
        "generated_dir": str(gen_dir),
        "sample_rate": args.sample_rate,
        "pairs": len(names),
        "si_sdr": summarize(sis),
        "lsd": summarize(lsds),
        "per_file": rows,
        "note": "task7 short-audio compatible LSD; same as paired_metrics.py except adaptive nperseg/noverlap for very short clips",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ["generated_dir", "pairs", "si_sdr", "lsd"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
