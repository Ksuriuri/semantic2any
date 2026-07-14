from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic2any.models import Semantic2MelModel
from semantic2any.utils.checkpoint import load_compatible_checkpoint
from semantic2any.utils.indextts_adapters import IndexTTSFeatureAdapter
from scripts.infer_s2mel_zipformer import resolve_dtype


def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ground-truth and generated target mel comparisons for paired inference."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32"), default="auto")
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--style-mode",
        choices=("reference", "none"),
        default="reference",
        help="Use the prompt reference style embedding or mask it after projection.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No pairs found in {path}")
    return rows


def robust_color_limits(target: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([target.reshape(-1), generated.reshape(-1)])
    vmin, vmax = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return float(values.min()), float(values.max())
    return float(vmin), float(vmax)


def render_comparison(
    *,
    target: torch.Tensor,
    generated: torch.Tensor,
    output_path: Path,
    sample_rate: int,
    hop_length: int,
    title: str,
) -> None:
    target_np = target.float().cpu().numpy()
    generated_np = generated.float().cpu().numpy()
    vmin, vmax = robust_color_limits(target_np, generated_np)
    duration = target_np.shape[-1] * hop_length / sample_rate

    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    images = []
    for axis, mel, label in zip(
        axes,
        (target_np, generated_np),
        ("Ground-truth target mel", "Generated target mel"),
        strict=True,
    ):
        image = axis.imshow(
            mel,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0.0, duration, 0, mel.shape[0]),
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        axis.set_title(label)
        axis.set_ylabel("Mel bin")
    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle(title)
    figure.colorbar(images[0], ax=axes, label="Log-mel value", shrink=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    manifest_path = Path(args.pair_manifest).expanduser().resolve()
    rows = load_rows(manifest_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else manifest_path.parent / "mel_comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(device, args.dtype)

    feature_adapter = IndexTTSFeatureAdapter(cfg).to(device=device)
    feature_adapter.eval()
    model = Semantic2MelModel(cfg.s2mel)
    epoch, step = load_compatible_checkpoint(model, args.checkpoint, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    preprocess = _get(cfg, "preprocess_params")
    spect = _get(preprocess, "spect_params")
    sample_rate = int(_get(preprocess, "sr"))
    hop_length = int(_get(spect, "hop_length"))
    inference_steps = args.inference_steps or int(_get(cfg.s2mel, "inference_steps", 25))
    model.models["cfm"].setup_estimator_caches(
        max_batch_size=2 if args.inference_cfg_rate > 0 else 1,
        max_seq_length=int(_get(_get(cfg.s2mel, "DiT"), "block_size", 1)),
    )
    output_rows = []

    for row in rows:
        prompt_path = str(row["prompt_audio_path"])
        target_path = str(row["target_audio_path"])
        batch = feature_adapter.extract_paired_from_audio_paths([prompt_path], [target_path])
        mel = batch["mel"].to(device=device, dtype=dtype)
        mel_lens = batch["mel_lens"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)
        semantic = batch["semantic"].to(device=device, dtype=dtype)
        semantic_lens = batch["semantic_lens"].to(device)
        prompt_semantic_lens = batch["prompt_semantic_lens"].to(device)
        style = batch["style"].to(device=device, dtype=dtype)

        mu = model.build_paired_condition(
            semantic,
            mel_lens,
            prompt_lens,
            semantic_lens,
            prompt_semantic_lens,
        )
        prompt_len = int(prompt_lens[0].item())
        mel_len = int(mel_lens[0].item())
        prompt = mel[:, :, :prompt_len]
        sampled = model.models["cfm"].inference(
            mu=mu,
            x_lens=mel_lens,
            prompt=prompt,
            style=style,
            f0=None,
            n_timesteps=inference_steps,
            temperature=args.temperature,
            inference_cfg_rate=args.inference_cfg_rate,
            show_progress=False,
            drop_style=args.style_mode == "none",
        )
        target_mel = mel[0, :, prompt_len:mel_len]
        generated_mel = sampled[0, :, prompt_len:mel_len]

        output_name = f"{Path(row['output_path']).stem}_style-{args.style_mode}_mel.png"
        output_path = output_dir / output_name
        render_comparison(
            target=target_mel,
            generated=generated_mel,
            output_path=output_path,
            sample_rate=sample_rate,
            hop_length=hop_length,
            title=(
                f"Pair {int(row['pair_index']):02d} | {row['speaker_id']} | "
                f"checkpoint step {step}"
            ),
        )
        output_row = {
            "pair_index": row["pair_index"],
            "speaker_id": row["speaker_id"],
            "prompt_id": row["prompt_id"],
            "target_id": row["target_id"],
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_epoch": epoch,
            "checkpoint_step": step,
            "style_mode": args.style_mode,
            "inference_cfg_rate": args.inference_cfg_rate,
            "temperature": args.temperature,
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "mel_bins": int(target_mel.size(0)),
            "target_frames": int(target_mel.size(1)),
            "image_path": str(output_path),
        }
        output_rows.append(output_row)
        print(f">> wrote {output_path}")

    visualization_manifest = output_dir / "mel_visualization_manifest.jsonl"
    visualization_manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {visualization_manifest}")


if __name__ == "__main__":
    main()
