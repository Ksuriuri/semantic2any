from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic2any.models import Semantic2MelModel
from semantic2any.utils.checkpoint import load_compatible_checkpoint
from semantic2any.utils.dots_audiovae import is_vae_latent_target
from semantic2any.utils.indextts_adapters import (
    S2MelFeatureAdapter,
    build_feature_adapter,
)
from semantic2any.utils.semantic_codecs import resolve_semantic_codec_config
from scripts.infer_s2mel_zipformer import (
    iter_audio_paths,
    resolve_dtype,
    save_wav,
)


def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run semantic-to-AudioVAE-latent inference. CFM samples a 128-d / 25 Hz "
            "latent; frozen dots.tts AudioVAE decodes 48 kHz audio. BigVGAN is not used."
        )
    )
    parser.add_argument("--config", default="configs/s2vae_dit_indextts25.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default="assets/test", help="Audio file or directory.")
    parser.add_argument("--output-dir", default="outputs/s2vae")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--dots-tts-dir", default=None)
    parser.add_argument("--semantic-codec", choices=("maskgct", "sac"), default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float32",
        help=(
            "Full-network dtype for DiT + Euler. float16 collapses VAE latents "
            "(silent output). auto also resolves to float32. bfloat16 is audible "
            "but worse than float32."
        ),
    )
    parser.add_argument("--prompt-seconds", type=float, default=3.0)
    parser.add_argument("--min-generate-frames", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--inference-cfg-rate", type=float, default=None)
    parser.add_argument(
        "--style-mode",
        choices=("reference", "none"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def prompt_frames_from_seconds(
    adapter: S2MelFeatureAdapter,
    n_frames: int,
    prompt_seconds: float,
    min_generate_frames: int,
) -> int:
    if prompt_seconds < 3.0:
        raise ValueError(f"--prompt-seconds must be at least 3.0, got {prompt_seconds}")
    sample_rate = int(adapter.sample_rate_acoustic)
    hop_length = int(adapter.acoustic_hop_size)
    requested = max(1, int(prompt_seconds * sample_rate / hop_length))
    max_prompt = max(1, n_frames - min_generate_frames)
    if requested > max_prompt:
        minimum_audio_seconds = (requested + min_generate_frames) * hop_length / sample_rate
        raise ValueError(
            "Audio is too short for the requested prompt and generated target: "
            f"need at least {minimum_audio_seconds:.3f}s for a {prompt_seconds:.3f}s prompt."
        )
    return requested


@torch.inference_mode()
def infer_one(
    *,
    audio_path: Path,
    output_path: Path,
    feature_adapter: S2MelFeatureAdapter,
    model: Semantic2MelModel,
    device: torch.device,
    dtype: torch.dtype,
    prompt_seconds: float,
    min_generate_frames: int,
    inference_steps: int,
    inference_cfg_rate: float,
    temperature: float,
    show_progress: bool,
    style_mode: str,
) -> None:
    audio_vae = feature_adapter.audio_vae
    if audio_vae is None:
        raise RuntimeError("s2vae inference requires target.type=vae_latent")

    batch = feature_adapter.extract_from_audio_paths([str(audio_path)])
    latent = batch["mel"].to(device=device, dtype=dtype)
    latent_lens = batch["mel_lens"].to(device=device)
    semantic = batch["semantic"].to(device=device, dtype=dtype)
    semantic_lens = batch["semantic_lens"].to(device=device)
    style = batch["style"].to(device=device, dtype=dtype)

    latent_len = int(latent_lens[0].item())
    prompt_len = prompt_frames_from_seconds(
        feature_adapter, latent_len, prompt_seconds, min_generate_frames
    )
    if latent_len <= prompt_len:
        raise ValueError(f"Audio is too short for prompt-only split: {audio_path}")

    mu = model.build_condition(semantic, latent_lens, semantic_lens=semantic_lens)
    prompt = latent[:, :, :prompt_len]
    generated = model.models["cfm"].inference(
        mu=mu,
        x_lens=latent_lens,
        prompt=prompt,
        style=style,
        f0=None,
        n_timesteps=inference_steps,
        temperature=temperature,
        inference_cfg_rate=inference_cfg_rate,
        show_progress=show_progress,
        drop_style=style_mode == "none",
    )
    generated = generated[:, :, prompt_len:latent_len]
    wav = audio_vae.decode(generated.float(), normalized=True)[0]
    save_wav(output_path, wav, int(audio_vae.sample_rate))
    print(
        f">> wrote {output_path} "
        f"(style_mode={style_mode}, frames={latent_len}, prompt_frames={prompt_len}, "
        f"generated_frames={latent_len - prompt_len}, sample_rate={audio_vae.sample_rate})"
    )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.model_dir is not None:
        cfg.paths.model_dir = args.model_dir
    if args.dots_tts_dir is not None:
        cfg.paths.dots_tts_dir = args.dots_tts_dir
    if not is_vae_latent_target(cfg):
        raise ValueError("scripts/infer_s2vae.py requires target.type=vae_latent")
    codec = resolve_semantic_codec_config(cfg, args.semantic_codec)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    requested = "float32" if args.dtype == "auto" else args.dtype
    if requested == "float16":
        print(">> warning: --dtype float16 collapses s2vae Euler; prefer float32")
    dtype = resolve_dtype(device, requested)
    input_paths = iter_audio_paths(Path(args.input).expanduser())
    if not input_paths:
        raise ValueError(f"No supported audio files found under {args.input}")

    print(f">> Loading s2vae checkpoint: {args.checkpoint}")
    model = Semantic2MelModel(cfg.s2mel)
    epoch, step = load_compatible_checkpoint(model, args.checkpoint, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f">> s2vae restored at epoch={epoch}, step={step}, dtype={dtype}")

    print(
        f">> Loading {codec.name} feature adapter + AudioVAE on {device} "
        f"({codec.semantic_fps:g} Hz, {codec.semantic_dim} dims)"
    )
    feature_adapter = build_feature_adapter(cfg).to(device=device)
    feature_adapter.eval()
    if feature_adapter.audio_vae is None:
        raise RuntimeError("Feature adapter did not load AudioVAE")

    inference_steps = (
        int(args.inference_steps)
        if args.inference_steps is not None
        else int(_get(cfg.s2mel, "inference_steps", 25))
    )
    inference_cfg_rate = (
        args.inference_cfg_rate
        if args.inference_cfg_rate is not None
        else float(_get(cfg.s2mel, "inference_cfg_rate", 0.7))
    )
    model.models["cfm"].setup_estimator_caches(
        max_batch_size=2 if inference_cfg_rate > 0 else 1,
        max_seq_length=int(_get(_get(cfg.s2mel, "DiT"), "block_size", 1)),
    )
    output_dir = Path(args.output_dir)
    for audio_path in input_paths:
        output_path = output_dir / f"{audio_path.stem}_s2vae_style-{args.style_mode}.wav"
        infer_one(
            audio_path=audio_path,
            output_path=output_path,
            feature_adapter=feature_adapter,
            model=model,
            device=device,
            dtype=dtype,
            prompt_seconds=args.prompt_seconds,
            min_generate_frames=args.min_generate_frames,
            inference_steps=inference_steps,
            inference_cfg_rate=inference_cfg_rate,
            temperature=args.temperature,
            show_progress=args.show_progress,
            style_mode=args.style_mode,
        )


if __name__ == "__main__":
    main()
