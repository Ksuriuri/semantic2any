"""Inference with train-consistent feature extraction (separate prompt/target)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
import torchaudio
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from semantic2any.models import Semantic2MelModel
from semantic2any.utils.checkpoint import load_compatible_checkpoint
from semantic2any.utils.indextts_adapters import build_feature_adapter


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = torch.clamp(wav, -1.0, 1.0).detach().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav, sample_rate, encoding="PCM_S", bits_per_sample=16)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--target-audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="checkpoints/feature-extractors")
    parser.add_argument("--inference-steps", type=int, default=200)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_dir:
        cfg.paths.model_dir = args.model_dir

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Load model
    model = Semantic2MelModel(cfg.s2mel)
    epoch, step = load_compatible_checkpoint(model, args.checkpoint, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f">> Model restored at epoch={epoch}, step={step}")

    # Load feature adapter
    adapter = build_feature_adapter(cfg).to(device=device)
    adapter.eval()

    # KEY FIX: Extract features from prompt and target SEPARATELY (like training)
    prompt_batch = adapter.extract_from_audio_paths([args.prompt_audio])
    target_batch = adapter.extract_from_audio_paths([args.target_audio])

    prompt_mel = prompt_batch["mel"].to(device=device, dtype=dtype)
    target_mel = target_batch["mel"].to(device=device, dtype=dtype)
    prompt_semantic = prompt_batch["semantic"].to(device=device, dtype=dtype)
    target_semantic = target_batch["semantic"].to(device=device, dtype=dtype)
    style = prompt_batch["style"].to(device=device, dtype=dtype)

    # Concatenate (same as training collator)
    full_mel = torch.cat([prompt_mel, target_mel], dim=-1)  # [1, C, T_prompt+T_target]
    full_semantic = torch.cat([prompt_semantic, target_semantic], dim=1)  # [1, S, D]
    mel_lens = torch.tensor([full_mel.shape[2]], device=device)
    semantic_lens = torch.tensor([full_semantic.shape[1]], device=device)
    prompt_frames = prompt_mel.shape[2]

    print(f">> Prompt mel frames: {prompt_frames}, Target mel frames: {target_mel.shape[2]}")
    print(f">> Total mel: {full_mel.shape[2]}, Total semantic: {full_semantic.shape[1]}")

    # Load vocoder
    from semantic2any.third_party.indextts.bigvgan import BigVGAN
    from semantic2any.defaults import DEFAULT_BIGVGAN_MODEL_ID

    def _get(obj, name, default=None):
        return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)

    vocoder_cfg = _get(cfg, "vocoder", None)
    model_id = (
        DEFAULT_BIGVGAN_MODEL_ID
        if vocoder_cfg is None
        else str(_get(vocoder_cfg, "model_id", "") or DEFAULT_BIGVGAN_MODEL_ID)
    )
    vocoder = BigVGAN.from_pretrained(model_id)
    vocoder = vocoder.to(device=device, dtype=dtype)
    vocoder.remove_weight_norm()
    vocoder.eval()
    print(f">> BigVGAN loaded: {model_id}")

    # Setup caches
    block_size = int(_get(cfg.s2mel.DiT, "block_size", full_mel.shape[2]))
    inference_cfg_rate = args.inference_cfg_rate
    model.models["cfm"].setup_estimator_caches(
        max_batch_size=2 if inference_cfg_rate > 0 else 1,
        max_seq_length=block_size,
    )

    # Generate
    mu = model.build_condition(full_semantic, mel_lens, semantic_lens=semantic_lens)
    prompt = full_mel[:, :, :prompt_frames]
    generated = model.models["cfm"].inference(
        mu=mu,
        x_lens=mel_lens,
        prompt=prompt,
        style=style,
        f0=None,
        n_timesteps=args.inference_steps,
        temperature=args.temperature,
        inference_cfg_rate=inference_cfg_rate,
        show_progress=False,
        drop_style=True,
    )
    generated_target = generated[:, :, prompt_frames:full_mel.shape[2]]

    # Vocoder
    wav = vocoder(generated_target.to(device=device, dtype=dtype))[0]
    sr = int(_get(_get(cfg, "preprocess_params"), "sr", 44100))
    save_wav(Path(args.output), wav, sr)
    print(f">> Saved: {args.output} (frames={generated_target.shape[2]})")


if __name__ == "__main__":
    main()
