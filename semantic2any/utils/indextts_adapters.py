from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from semantic2any.data.s2mel_dataset import choose_prompt_len


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def add_indextts_to_path(indextts_root: str | Path) -> Path:
    root = Path(indextts_root).expanduser().resolve()
    if not (root / "indextts").exists():
        raise FileNotFoundError(f"IndexTTS package not found under {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _resolve_model_path(model_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = model_dir / path
    return path


def _find_semantic_codec_ckpt(model_dir: Path, configured: str | None) -> Path:
    if configured:
        path = _resolve_model_path(model_dir, configured)
        if path.exists():
            return path
        raise FileNotFoundError(f"semantic codec checkpoint not found: {path}")

    candidates = [
        "semantic_codec.safetensors",
        "semantic_codec/model.safetensors",
        "semantic_codec.pth",
        "semantic_codec.pt",
    ]
    for candidate in candidates:
        path = model_dir / candidate
        if path.exists():
            return path
    matches = sorted(model_dir.glob("**/*semantic*codec*.safetensors"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find semantic codec checkpoint. Set paths.semantic_codec_ckpt in the config."
    )


def _load_audio(path: str | Path, max_audio_seconds: float | None = None) -> tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(str(path))
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if max_audio_seconds is not None:
        max_samples = int(max_audio_seconds * sr)
        audio = audio[:, :max_samples]
    return audio, sr


def _resample(audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return audio
    return torchaudio.functional.resample(audio, src_sr, dst_sr)


class IndexTTSFeatureAdapter(nn.Module):
    """Frozen IndexTTS feature stack for online semantic2mel training batches."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        from omegaconf import OmegaConf
        import safetensors.torch
        from transformers import SeamlessM4TFeatureExtractor

        paths_cfg = _get(cfg, "paths")
        self.indextts_root = add_indextts_to_path(_get(paths_cfg, "indextts_root"))
        self.model_dir = Path(_get(paths_cfg, "model_dir")).expanduser().resolve()
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model_dir does not exist: {self.model_dir}")

        from indextts.s2mel.modules.audio import mel_spectrogram
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model

        index_cfg_path = self.model_dir / "config.yaml"
        index_cfg = OmegaConf.load(index_cfg_path) if index_cfg_path.exists() else cfg
        semantic_codec_cfg = _get(index_cfg, "semantic_codec", _get(cfg, "semantic_codec", None))
        if semantic_codec_cfg is None:
            raise ValueError("semantic_codec config not found in current config or IndexTTS config.yaml")

        w2v_stat = _resolve_model_path(self.model_dir, _get(paths_cfg, "w2v_stat", "wav2vec2bert_stats.pt"))
        w2v_bert_dir = _resolve_model_path(self.model_dir, _get(paths_cfg, "w2v_bert_dir", "w2v-bert-2.0"))
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            str(w2v_bert_dir), local_files_only=True
        )
        semantic_model, semantic_mean, semantic_std = build_semantic_model(
            str(w2v_stat), model_path=str(w2v_bert_dir)
        )
        self.semantic_model = semantic_model.eval()
        self.register_buffer("semantic_mean", semantic_mean.float())
        self.register_buffer("semantic_std", semantic_std.float())

        self.semantic_codec = build_semantic_codec(semantic_codec_cfg).eval()
        semantic_ckpt = _find_semantic_codec_ckpt(
            self.model_dir, _get(paths_cfg, "semantic_codec_ckpt", "")
        )
        if semantic_ckpt.suffix == ".safetensors":
            safetensors.torch.load_model(self.semantic_codec, str(semantic_ckpt))
        else:
            state = torch.load(semantic_ckpt, map_location="cpu")
            self.semantic_codec.load_state_dict(state.get("state_dict", state), strict=False)

        campplus_ckpt = _resolve_model_path(
            self.model_dir, _get(paths_cfg, "campplus_ckpt", "campplus/campplus_cn_common.bin")
        )
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        self.campplus_model.eval()

        preprocess = _get(cfg, "preprocess_params", _get(_get(cfg, "s2mel"), "preprocess_params", None))
        spect = _get(preprocess, "spect_params")
        fmax = _get(spect, "fmax", "None")
        fmax = None if fmax in (None, "None") else float(fmax)
        self.mel_args = {
            "n_fft": int(_get(spect, "n_fft", 1024)),
            "num_mels": int(_get(spect, "n_mels", 80)),
            "sampling_rate": int(_get(preprocess, "sr", 22050)),
            "hop_size": int(_get(spect, "hop_length", 256)),
            "win_size": int(_get(spect, "win_length", 1024)),
            "fmin": float(_get(spect, "fmin", 0)),
            "fmax": fmax,
            "center": False,
        }
        self.mel_spectrogram = mel_spectrogram

        data_cfg = _get(cfg, "data")
        self.max_audio_seconds = float(_get(data_cfg, "max_audio_seconds", 20.0))
        self.min_prompt_seconds = float(_get(data_cfg, "min_prompt_seconds", 1.0))
        self.max_prompt_seconds = float(_get(data_cfg, "max_prompt_seconds", 5.0))
        self.min_generated_frames = int(_get(data_cfg, "min_generated_frames", 8))
        self.sample_rate_16k = int(_get(data_cfg, "sample_rate_16k", 16000))
        self.sample_rate_mel = int(_get(data_cfg, "sample_rate_mel", self.mel_args["sampling_rate"]))

        for module in (self.semantic_model, self.semantic_codec, self.campplus_model):
            module.requires_grad_(False)

    @torch.no_grad()
    def _semantic_from_audio(self, audio_16k: torch.Tensor) -> torch.Tensor:
        device = self.semantic_mean.device
        waveform = audio_16k.squeeze(0).detach().cpu().numpy()
        inputs = self.extract_features(waveform, sampling_rate=self.sample_rate_16k, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        out = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        _, semantic = self.semantic_codec.quantize(feat)
        if not torch.is_floating_point(semantic):
            semantic = self.semantic_codec.quantizer.vq2emb(semantic.unsqueeze(1))
            if semantic.ndim == 3 and semantic.size(1) != feat.size(1):
                semantic = semantic.transpose(1, 2)
        elif semantic.ndim == 3 and semantic.size(1) != feat.size(1) and semantic.size(2) == feat.size(1):
            semantic = semantic.transpose(1, 2)
        return semantic.squeeze(0).float()

    @torch.no_grad()
    def _style_from_audio(self, audio_16k: torch.Tensor) -> torch.Tensor:
        device = self.semantic_mean.device
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=self.sample_rate_16k,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus_model(feat.unsqueeze(0)).squeeze(0).float()

    @torch.no_grad()
    def extract_from_audio_paths(self, audio_paths: list[str]) -> dict[str, torch.Tensor]:
        device = self.semantic_mean.device
        mels = []
        semantics = []
        styles = []
        prompt_lens = []

        for path in audio_paths:
            audio, sr = _load_audio(path, self.max_audio_seconds)
            audio_22k = _resample(audio, sr, self.sample_rate_mel).to(device)
            audio_16k = _resample(audio, sr, self.sample_rate_16k)

            mel = self.mel_spectrogram(audio_22k.float(), **self.mel_args).squeeze(0)
            semantic = self._semantic_from_audio(audio_16k)
            style = self._style_from_audio(audio_16k)
            prompt_len = choose_prompt_len(
                mel.size(-1),
                hop_length=self.mel_args["hop_size"],
                sample_rate=self.sample_rate_mel,
                min_prompt_seconds=self.min_prompt_seconds,
                max_prompt_seconds=self.max_prompt_seconds,
                min_generated_frames=self.min_generated_frames,
            )

            mels.append(mel.transpose(0, 1))
            semantics.append(semantic)
            styles.append(style)
            prompt_lens.append(prompt_len)

        mel_lens = torch.tensor([x.size(0) for x in mels], dtype=torch.long, device=device)
        semantic_lens = torch.tensor([x.size(0) for x in semantics], dtype=torch.long, device=device)
        mel = pad_sequence(mels, batch_first=True, padding_value=0.0).transpose(1, 2).to(device)
        semantic = pad_sequence(semantics, batch_first=True, padding_value=0.0).to(device)
        style = torch.stack(styles).to(device)
        prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        return {
            "mel": mel,
            "mel_lens": mel_lens,
            "semantic": semantic,
            "semantic_lens": semantic_lens,
            "style": style,
            "prompt_lens": prompt_lens_tensor,
        }


def move_feature_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("mel", "mel_lens", "semantic", "semantic_lens", "style", "prompt_lens"):
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device)
    return out
