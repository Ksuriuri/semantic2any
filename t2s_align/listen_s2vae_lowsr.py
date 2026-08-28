#!/usr/bin/env python3
"""s2vae listen pack: 3 Paimon pairs + a few native-rate <44 kHz pairs.

Frozen dots.tts AudioVAE decode at 48 kHz. No BigVGAN.
Runs on ONE GPU (PACK_GPU) while training is live.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tarfile

import soundfile as sf
import torchaudio

ROOT = pathlib.Path("/mnt/data_sdd/hhy/noiz-tts/semantic2any")
SRC = ROOT / "outputs/laion7_v2_step80000_listen"
VALID = pathlib.Path(
    "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-v2/indextts25-codes/"
    "splits/seed1234_valid2000_spk_holdout_20260827/valid.jsonl"
)
EXP = ROOT / "exp/s2vae_dit_indextts25"
CFG = ROOT / "configs/s2vae_dit_indextts25.yaml"
GPU = os.environ.get("PACK_GPU", "6")
DTYPE = os.environ.get("PACK_DTYPE", "float32")
CFG_RATE = 0.7
# speaker_id, prefer_sr, label
LOW_SR_SPEAKERS = (
    ("laion_emolia__EN_B00055_S04603", 24000, "en24k"),
    ("StarRail__女性的声音", 32000, "en32k"),
    ("laion_emolia__ZH_B00039_S02831", 32000, "zh32k"),
    ("laion_emolia__JA_B00000_S02570", 24000, "ja24k"),
)


def latest_step() -> str:
    steps = []
    for p in EXP.glob("s2mel_step*.pth"):
        n = p.name.removeprefix("s2mel_step").removesuffix(".pth")
        if n.isdigit():
            steps.append(int(n))
    if not steps:
        raise SystemExit(f"no s2mel_step*.pth in {EXP}")
    return str(max(steps))


STEP = os.environ.get("PACK_STEP") or latest_step()
CKPT = EXP / f"s2mel_step{STEP}.pth"
OUT = ROOT / f"outputs/s2vae_listen{STEP}_{DTYPE}_lowsr"
PACK = OUT / "pack"
TAR = OUT / f"s2vae_step{STEP}_{DTYPE}_listen.tar.gz"
PREP = OUT / "prep"


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mono(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, int(sr)


def write_concat(prompt_path: str, target_path: str, dest: pathlib.Path) -> tuple[float, float, int]:
    p, psr = load_mono(prompt_path)
    t, tsr = load_mono(target_path)
    if psr != tsr:
        t = torchaudio.functional.resample(t, tsr, psr)
    concat = torch.cat([p, t], dim=-1).clamp(-1, 1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dest), concat, psr)
    return p.size(-1) / psr, t.size(-1) / psr, psr


def infer_one(concat: pathlib.Path, out_dir: pathlib.Path, prompt_seconds: float, log_path: pathlib.Path) -> int:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/infer_s2vae.py"),
        "--config", str(CFG),
        "--checkpoint", str(CKPT),
        "--input", str(concat),
        "--output-dir", str(out_dir),
        "--model-dir", str(ROOT / "checkpoints/feature-extractors"),
        "--dots-tts-dir", str(ROOT / "checkpoints/dots-tts"),
        "--prompt-seconds", str(prompt_seconds),
        "--inference-steps", "25",
        "--inference-cfg-rate", str(CFG_RATE),
        "--temperature", "0.7",
        "--style-mode", "none",
        "--seed", "1234",
        "--dtype", DTYPE,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def load_low_sr_pairs() -> list[dict]:
    wanted = {spk: (sr, lab) for spk, sr, lab in LOW_SR_SPEAKERS}
    by_spk: dict[str, list[dict]] = {spk: [] for spk in wanted}
    with VALID.open() as f:
        for line in f:
            rec = json.loads(line)
            spk = rec.get("speaker_id")
            if spk not in wanted:
                continue
            sr_w, _ = wanted[spk]
            if rec.get("sample_rate") != sr_w or not rec.get("audio_path"):
                continue
            by_spk[spk].append(rec)
    pairs = []
    for idx, (spk, (sr_w, lab)) in enumerate(wanted.items(), start=3):
        xs = sorted(by_spk[spk], key=lambda r: float(r["duration"]), reverse=True)
        prompt = next((r for r in xs if float(r["duration"]) >= 3.0), None)
        target = next((r for r in xs if r is not prompt and 0.8 <= float(r["duration"]) <= 18), None)
        if prompt is None or target is None:
            raise SystemExit(f"cannot pair {spk} sr={sr_w}: n={len(xs)}")
        if not pathlib.Path(prompt["audio_path"]).is_file() or not pathlib.Path(target["audio_path"]).is_file():
            raise SystemExit(f"missing audio for {spk}")
        concat = PREP / f"pair{idx}_{lab}_concat.wav"
        psec, tsec, sr = write_concat(prompt["audio_path"], target["audio_path"], concat)
        pairs.append(
            {
                "idx": idx,
                "lang": lab,
                "dataset": prompt.get("dataset"),
                "speaker": spk,
                "native_sr": sr,
                "prompt_seconds": psec,
                "target_seconds": tsec,
                "prompt_audio": prompt["audio_path"],
                "target_audio": target["audio_path"],
                "concat": str(concat),
            }
        )
    return pairs


def main() -> int:
    if not CKPT.is_file():
        raise SystemExit(f"missing ckpt {CKPT}")
    OUT.mkdir(parents=True, exist_ok=True)
    PREP.mkdir(parents=True, exist_ok=True)
    paimon = json.loads((SRC / "pairs.json").read_text())
    low = load_low_sr_pairs()
    all_pairs = []
    for m in paimon:
        all_pairs.append(
            {
                "idx": m["idx"],
                "lang": {0: "en", 1: "ja", 2: "ko"}[m["idx"]],
                "dataset": "Genshin",
                "speaker": m["speaker"],
                "native_sr": 48000,
                "prompt_seconds": float(m["prompt_seconds"]),
                "target_seconds": float(m["target_seconds"]),
                "prompt_audio": m["prompt_audio"],
                "target_audio": m["target_audio"],
                "concat": m["concat"],
            }
        )
    all_pairs.extend(low)
    print(f"s2vae step={STEP} gpu={GPU} dtype={DTYPE} ckpt={CKPT.name} n={len(all_pairs)}", flush=True)

    gen_dir = OUT / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for m in all_pairs:
        rc = infer_one(
            pathlib.Path(m["concat"]),
            gen_dir,
            float(m["prompt_seconds"]),
            OUT / f"infer_pair{m['idx']}.log",
        )
        print(f"  pair{m['idx']} {m['lang']} sr={m['native_sr']}: exit={rc}", flush=True)
        if rc != 0:
            failures += 1
            print((OUT / f"infer_pair{m['idx']}.log").read_text()[-2500:], flush=True)
    if failures:
        return 1

    if PACK.exists():
        shutil.rmtree(PACK)
    (PACK / "gen_cfg0.7").mkdir(parents=True)
    (PACK / "ref_original").mkdir(parents=True)

    rows = []
    for m in all_pairs:
        gt_wav, gt_sr = load_mono(m["target_audio"])
        gt_path = PACK / "ref_original" / f"pair{m['idx']}_{m['lang']}_target.wav"
        torchaudio.save(str(gt_path), gt_wav.clamp(-1, 1), gt_sr)
        stem = pathlib.Path(m["concat"]).stem
        gen = gen_dir / f"{stem}_s2vae_style-none.wav"
        if not gen.is_file():
            raise SystemExit(f"missing gen {gen}")
        info = sf.info(str(gen))
        if info.samplerate != 48000:
            raise SystemExit(f"{gen} is {info.samplerate} Hz, expected 48000")
        dest = PACK / "gen_cfg0.7" / f"pair{m['idx']}_{m['lang']}_gen.wav"
        shutil.copy2(gen, dest)
        rows.append(
            f"pair{m['idx']} {m['lang']} {m['dataset']} native_sr={m['native_sr']} "
            f"prompt={m['prompt_seconds']:.1f}s target={m['target_seconds']:.1f}s "
            f"gen_sr={info.samplerate}"
        )

    table = "\n".join(rows)
    print(table, flush=True)
    (PACK / "README.txt").write_text(
        f"s2vae semantic->AudioVAE latent, step {STEP}, 48 kHz official decoder\n"
        "config: configs/s2vae_dit_indextts25.yaml\n"
        f"DiT: {CKPT} md5 {md5(CKPT)}\n"
        "decode: frozen dots.tts AudioVAE (not BigVGAN)\n"
        f"W&B hf4zm6sy. infer: steps=25 cfg=0.7 temp=0.7 seed=1234 style=none {DTYPE}\n"
        "pair0-2: same 3 Paimon (en/ja/ko), native 48 kHz\n"
        "pair3-6: holdout clips native <44 kHz (24 kHz / 32 kHz)\n"
        "\n"
        "gen_cfg0.7/     s2vae, listen to this (always 48 kHz decode)\n"
        "ref_original/   GT at native rate\n"
        "\n"
        f"{table}\n"
    )
    if TAR.exists():
        TAR.unlink()
    with tarfile.open(TAR, "w:gz") as tf:
        tf.add(PACK, arcname=f"s2vae_step{STEP}_listen")
    print(f"packed {TAR} ({TAR.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
