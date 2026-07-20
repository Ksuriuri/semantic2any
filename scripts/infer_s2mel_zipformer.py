from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic2any.models import Semantic2MelModel
from semantic2any.utils.checkpoint import load_compatible_checkpoint
from semantic2any.utils.indextts_adapters import (
    S2MelFeatureAdapter,
    add_indextts_to_path,
    build_feature_adapter,
)
from semantic2any.utils.semantic_codecs import (
    resolve_semantic_codec_config,
)


AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run semantic2any ZipFormer s2mel inference on audio files. "
            "The output contains only the generated target segment."
        )
    )
    parser.add_argument("--config", default="exp/s2mel_zipformer-vctk-8gpu/config.resolved.yaml")
    parser.add_argument("--checkpoint", default="exp/s2mel_zipformer-vctk-8gpu/s2mel_final.pth")
    parser.add_argument("--input", default="assets/test", help="Audio file or directory to process.")
    parser.add_argument("--output-dir", default="outputs/s2mel_zipformer-vctk-8gpu")
    parser.add_argument("--indextts-root", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--semantic-codec", choices=("maskgct", "sac"), default=None)
    parser.add_argument("--vocoder-model", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Dtype for the s2mel model and BigVGAN. Feature extraction stays in float32.",
    )
    parser.add_argument(
        "--prompt-seconds",
        type=float,
        default=3.0,
        help="Reference prompt duration in seconds; must be at least 3 seconds.",
    )
    parser.add_argument("--min-generate-frames", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=50,
        help="Number of Euler sampling steps (default: 50).",
    )
    parser.add_argument("--inference-cfg-rate", type=float, default=None)
    parser.add_argument(
        "--style-mode",
        choices=("reference", "none"),
        default="reference",
        help="Use the reference style embedding or mask it after projection.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def resolve_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested == "float32" or device.type == "cpu":
        return torch.float32
    if requested == "float16":
        return torch.float16
    return torch.float16 if device.type == "cuda" else torch.float32


def iter_audio_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def load_vocoder(cfg, device: torch.device, dtype: torch.dtype):
    paths_cfg = _get(cfg, "paths")
    indextts_root = add_indextts_to_path(_get(paths_cfg, "indextts_root"))
    model_dir = Path(_get(paths_cfg, "model_dir")).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"IndexTTS model_dir does not exist: {model_dir}")

    from indextts.s2mel.modules.bigvgan import bigvgan

    vocoder_cfg = _get(cfg, "vocoder", None)
    model_id = str(_get(vocoder_cfg, "model_id", "") or "")
    source = model_id or str(model_dir / "bigvgan")
    cache_dir = str(_get(vocoder_cfg, "cache_dir", "") or "")
    load_kwargs = {}
    if model_id:
        load_kwargs["cache_dir"] = cache_dir or None
        load_kwargs["local_files_only"] = bool(
            _get(vocoder_cfg, "local_files_only", False)
        )
    vocoder = bigvgan.BigVGAN.from_pretrained(source, **load_kwargs)

    preprocess = _get(cfg, "preprocess_params")
    spect = _get(preprocess, "spect_params")
    expected = {
        "sampling_rate": int(_get(preprocess, "sr", 22050)),
        "num_mels": int(_get(spect, "n_mels", 80)),
        "n_fft": int(_get(spect, "n_fft", 1024)),
        "hop_size": int(_get(spect, "hop_length", 256)),
        "win_size": int(_get(spect, "win_length", 1024)),
    }
    mismatches = {
        key: (value, int(_get(vocoder.h, key)))
        for key, value in expected.items()
        if int(_get(vocoder.h, key)) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: config={config_value}, vocoder={vocoder_value}"
            for key, (config_value, vocoder_value) in mismatches.items()
        )
        raise ValueError(f"BigVGAN mel configuration mismatch: {details}")

    vocoder = vocoder.to(device=device, dtype=dtype)
    vocoder.remove_weight_norm()
    vocoder.eval()
    print(f">> IndexTTS root: {indextts_root}")
    print(f">> BigVGAN restored from: {source}")
    return vocoder


def prompt_frames_from_seconds(cfg, mel_len: int, prompt_seconds: float, min_generate_frames: int) -> int:
    if prompt_seconds < 3.0:
        raise ValueError(f"--prompt-seconds must be at least 3.0, got {prompt_seconds}")
    preprocess = _get(cfg, "preprocess_params")
    spect = _get(preprocess, "spect_params")
    sample_rate = int(_get(preprocess, "sr", 22050))
    hop_length = int(_get(spect, "hop_length", 256))
    requested = max(1, int(prompt_seconds * sample_rate / hop_length))
    max_prompt = max(1, mel_len - min_generate_frames)
    if requested > max_prompt:
        minimum_audio_seconds = (requested + min_generate_frames) * hop_length / sample_rate
        raise ValueError(
            "Audio is too short for the requested prompt and generated target: "
            f"need at least {minimum_audio_seconds:.3f}s for a {prompt_seconds:.3f}s prompt."
        )
    return requested


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = torch.clamp(wav, -1.0, 1.0).detach().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate, encoding="PCM_S", bits_per_sample=16)


@torch.inference_mode()
def infer_one(
    *,
    audio_path: Path,
    output_path: Path,
    cfg,
    feature_adapter: S2MelFeatureAdapter,
    model: Semantic2MelModel,
    vocoder,
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
    batch = feature_adapter.extract_from_audio_paths([str(audio_path)])
    mel = batch["mel"].to(device=device, dtype=dtype)
    mel_lens = batch["mel_lens"].to(device=device)
    semantic = batch["semantic"].to(device=device, dtype=dtype)
    semantic_lens = batch["semantic_lens"].to(device=device)
    style = batch["style"].to(device=device, dtype=dtype)

    mel_len = int(mel_lens[0].item())
    prompt_len = prompt_frames_from_seconds(cfg, mel_len, prompt_seconds, min_generate_frames)
    if mel_len <= prompt_len:
        raise ValueError(f"Audio is too short for prompt-only split: {audio_path}")

    mu = model.build_condition(semantic, mel_lens, semantic_lens=semantic_lens)
    prompt = mel[:, :, :prompt_len]
    generated = model.models["cfm"].inference(
        mu=mu,
        x_lens=mel_lens,
        prompt=prompt,
        style=style,
        f0=None,
        n_timesteps=inference_steps,
        temperature=temperature,
        inference_cfg_rate=inference_cfg_rate,
        show_progress=show_progress,
        drop_style=style_mode == "none",
    )
    generated = generated[:, :, prompt_len:mel_len]

    wav = vocoder(generated.to(device=device, dtype=dtype))[0]
    sample_rate = int(_get(_get(cfg, "preprocess_params"), "sr", 22050))
    save_wav(output_path, wav, sample_rate)
    print(
        f">> wrote {output_path} "
        f"(style_mode={style_mode}, frames={mel_len}, prompt_frames={prompt_len}, "
        f"generated_frames={mel_len - prompt_len})"
    )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.indextts_root is not None:
        cfg.paths.indextts_root = args.indextts_root
    if args.model_dir is not None:
        cfg.paths.model_dir = args.model_dir
    if args.vocoder_model is not None:
        if _get(cfg, "vocoder", None) is None:
            cfg.vocoder = {}
        cfg.vocoder.model_id = args.vocoder_model
    codec = resolve_semantic_codec_config(cfg, args.semantic_codec)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    input_paths = iter_audio_paths(Path(args.input).expanduser())
    if not input_paths:
        raise ValueError(f"No supported audio files found under {args.input}")

    print(f">> Loading s2mel checkpoint: {args.checkpoint}")
    model = Semantic2MelModel(cfg.s2mel)
    epoch, step = load_compatible_checkpoint(model, args.checkpoint, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f">> s2mel restored at epoch={epoch}, step={step}, dtype={dtype}")

    print(
        f">> Loading {codec.name} feature adapter on {device} "
        f"({codec.semantic_fps:g} Hz, {codec.semantic_dim} dims)"
    )
    feature_adapter = build_feature_adapter(cfg).to(device=device)
    feature_adapter.eval()

    vocoder = load_vocoder(cfg, device, dtype)

    inference_steps = int(args.inference_steps)
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
        output_path = output_dir / f"{audio_path.stem}_s2mel_style-{args.style_mode}.wav"
        infer_one(
            audio_path=audio_path,
            output_path=output_path,
            cfg=cfg,
            feature_adapter=feature_adapter,
            model=model,
            vocoder=vocoder,
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
