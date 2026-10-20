#!/usr/bin/env python3
"""Reconstruct target audio with the official IndexTTS2.5 s2mel stack.

The manifest contains explicit prompt/target WAV pairs.  Target audio is first
converted to IndexTTS2.5 EnhancedCodec embeddings, then the official s2mel and
22.05 kHz BigVGAN decode it.  This bypasses text/GPT generation so the script
measures only the semantic-to-acoustic path.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import librosa
import torch
import torchaudio
from omegaconf import OmegaConf
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo", default="/mnt/data_sdd/hhy/indextts-2.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    repo = pathlib.Path(args.repo)
    model_dir = repo / "checkpoints"
    sys.path.insert(0, str(repo))
    from indextts.codec.models import EnhancedCodec
    from indextts.s2mel.modules.audio import mel_spectrogram
    from indextts.s2mel.modules.bigvgan import bigvgan
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2

    device = torch.device(args.device)
    cfg = OmegaConf.load(model_dir / "config.yaml")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">> load IndexTTS2.5 semantic frontend", flush=True)
    w2v_dir = model_dir / "hf_cache/w2v-bert-2.0"
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(w2v_dir, local_files_only=True)
    sem_model = Wav2Vec2BertModel.from_pretrained(w2v_dir, local_files_only=True)
    sem_model = sem_model.to(device).eval().requires_grad_(False)
    stats = torch.load(model_dir / cfg.w2v_stat, map_location="cpu", weights_only=True)
    sem_mean = stats["mean"].to(device, dtype=torch.float32)
    sem_std = stats["var"].sqrt().to(device, dtype=torch.float32)

    codec = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
    codec.load_checkpoint(model_dir / "codec.pth")
    codec = codec.to(device).eval().requires_grad_(False)

    print(">> load official IndexTTS2.5 s2mel", flush=True)
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=False)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel,
        None,
        model_dir / cfg.s2mel_checkpoint,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    s2mel = s2mel.to(device).eval().requires_grad_(False)
    s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(model_dir / "hf_cache/campplus_cn_common.bin", map_location="cpu"))
    campplus = campplus.to(device).eval().requires_grad_(False)

    vocoder = bigvgan.BigVGAN.from_pretrained(model_dir / "hf_cache/bigvgan", use_cuda_kernel=False)
    vocoder = vocoder.to(device).eval().requires_grad_(False)
    vocoder.remove_weight_norm()

    spect = cfg.s2mel.preprocess_params.spect_params
    sr = int(cfg.s2mel.preprocess_params.sr)
    hop = int(spect.hop_length)
    mel_fn = lambda x: mel_spectrogram(
        x,
        n_fft=int(spect.n_fft),
        win_size=int(spect.win_length),
        hop_size=hop,
        num_mels=int(spect.n_mels),
        sampling_rate=sr,
        fmin=float(spect.get("fmin", 0)),
        fmax=None if str(spect.get("fmax", "None")) == "None" else float(spect.fmax),
        center=False,
    )
    # EnhancedCodec codes are 25 Hz, but official s2mel does not consume the
    # quantizer's internal 25 Hz latent.  codec.decode() restores the 50 Hz,
    # 1024-d semantic sequence used by infer_v2_5.py, whose mel expansion is
    # 86.13 / 50 ~= 1.72.
    semantic_fps = 50.0
    semantic_to_mel = 1.72  # exact constant used by IndexTTS2.5 infer_v2_5.py
    print(f">> decoded_semantic_fps={semantic_fps:g}, semantic_to_mel={semantic_to_mel:.6f}", flush=True)

    @torch.inference_mode()
    def semantic_feature(wav16):
        inputs = extractor(wav16, sampling_rate=16000, return_tensors="pt")
        features = inputs["input_features"].to(device)
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        hidden = sem_model(
            input_features=features,
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states[17].float()
        return (hidden - sem_mean) / sem_std

    cases = json.loads(pathlib.Path(args.manifest).read_text())
    report = []
    for pos, case in enumerate(cases, 1):
        case_id = case["case_id"]
        print(f">> [{pos}/{len(cases)}] {case_id}", flush=True)
        prompt22, _ = librosa.load(case["prompt_wav"], sr=sr, mono=True)
        target16, _ = librosa.load(case["target_wav"], sr=16000, mono=True)
        prompt16, _ = librosa.load(case["prompt_wav"], sr=16000, mono=True)
        prompt22_t = torch.from_numpy(prompt22).unsqueeze(0).to(device)

        with torch.inference_mode():
            prompt_feat = semantic_feature(prompt16)
            target_feat = semantic_feature(target16)
            target_codes, _quantizer_latent = codec.quantize(target_feat)
            target_emb = codec.decode(target_codes)
            ref_mel = mel_fn(prompt22_t.float())
            ref_lengths = torch.tensor([ref_mel.shape[-1]], device=device, dtype=torch.long)
            prompt_cond = s2mel.models["length_regulator"](
                prompt_feat, ylens=ref_lengths, n_quantizers=3, f0=None
            )[0]

            feat = torchaudio.compliance.kaldi.fbank(
                torch.from_numpy(prompt16).unsqueeze(0).to(device),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = campplus(feat.unsqueeze(0))

            target_frames = max(1, round(int(target_emb.shape[1]) * semantic_to_mel))
            target_lengths = torch.tensor([target_frames], device=device, dtype=torch.long)
            target_cond = s2mel.models["length_regulator"](
                target_emb, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]
            cond = torch.cat([prompt_cond, target_cond], dim=1)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            mel = s2mel.models["cfm"].inference(
                cond,
                torch.tensor([cond.shape[1]], device=device, dtype=torch.long),
                ref_mel,
                style,
                None,
                25,
                inference_cfg_rate=0.7,
            )[:, :, ref_mel.shape[-1] :]
            wav = vocoder(mel.float())
            if isinstance(wav, (tuple, list)):
                wav = wav[0]
            wav = wav.squeeze().clamp(-1, 1).cpu()

        dest = out_dir / f"{case_id}_indextts25_s2mel.wav"
        torchaudio.save(str(dest), wav.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16)
        report.append(
            {
                "case_id": case_id,
                "codes": int(target_codes.shape[-1]),
                "mel_frames": int(mel.shape[-1]),
                "seconds": float(wav.numel() / sr),
                "output": str(dest),
            }
        )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(">> ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
