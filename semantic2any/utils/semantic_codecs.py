from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


@dataclass(frozen=True)
class SemanticCodecInfo:
    name: str
    semantic_dim: int
    semantic_fps: float
    sample_rate: int
    is_discrete: bool
    source_model: str
    tokenizer_model: str = ""
    revision: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fingerprint": self.fingerprint()}


SEMANTIC_CODEC_SPECS = {
    "maskgct": SemanticCodecInfo(
        name="maskgct",
        semantic_dim=1024,
        semantic_fps=50.0,
        sample_rate=16000,
        is_discrete=False,
        source_model="IndexTTS/MaskGCT-RepCodec",
    ),
    "sac": SemanticCodecInfo(
        name="sac",
        semantic_dim=1280,
        semantic_fps=12.5,
        sample_rate=16000,
        is_discrete=False,
        source_model="Soul-AILab/SAC-16k-62_5Hz",
        tokenizer_model="zai-org/glm-4-voice-tokenizer",
        revision="a5f2404e63c84e92f5238908e1706316324ebafa",
    ),
}


def semantic_codec_type(cfg: Any) -> str:
    codec_cfg = _get(cfg, "semantic_codec", None)
    name = str(_get(codec_cfg, "type", "maskgct")).lower()
    if name not in SEMANTIC_CODEC_SPECS:
        choices = ", ".join(sorted(SEMANTIC_CODEC_SPECS))
        raise ValueError(f"Unsupported semantic codec {name!r}; choose one of: {choices}")
    return name


def resolve_semantic_codec_config(cfg: Any, codec_type: str | None = None) -> SemanticCodecInfo:
    """Resolve dimensions/rate from one selector so ablations cannot drift."""

    from omegaconf import OmegaConf

    if _get(cfg, "semantic_codec", None) is None:
        cfg.semantic_codec = OmegaConf.create({})
    if codec_type is not None:
        cfg.semantic_codec.type = str(codec_type).lower()
    elif _get(cfg.semantic_codec, "type", None) is None:
        cfg.semantic_codec.type = "maskgct"

    name = semantic_codec_type(cfg)
    info = semantic_codec_info(cfg)

    if _get(cfg, "data", None) is None:
        cfg.data = OmegaConf.create({})
    if _get(_get(cfg, "s2mel", None), "length_regulator", None) is None:
        cfg.s2mel.length_regulator = OmegaConf.create({})
    cfg.data.semantic_fps = info.semantic_fps
    cfg.data.sample_rate_16k = info.sample_rate
    cfg.s2mel.length_regulator.is_discrete = info.is_discrete
    cfg.s2mel.length_regulator.in_channels = info.semantic_dim
    return info


def semantic_codec_info(cfg: Any) -> SemanticCodecInfo:
    name = semantic_codec_type(cfg)
    base = SEMANTIC_CODEC_SPECS[name]
    codec_cfg = _get(cfg, "semantic_codec", None)
    if name == "maskgct":
        paths_cfg = _get(cfg, "paths", None)
        model_dir = str(_get(paths_cfg, "model_dir", "") or "")
        checkpoint = str(_get(paths_cfg, "semantic_codec_ckpt", "") or "auto")
        if not model_dir:
            return base
        return SemanticCodecInfo(
            name=name,
            semantic_dim=base.semantic_dim,
            semantic_fps=base.semantic_fps,
            sample_rate=base.sample_rate,
            is_discrete=False,
            source_model=model_dir,
            revision=checkpoint,
        )
    tokenizer_path = str(_get(codec_cfg, "tokenizer_path", "") or "")
    tokenizer_model = (
        f"local:{Path(tokenizer_path).expanduser().resolve()}"
        if tokenizer_path
        else str(_get(codec_cfg, "tokenizer_model", base.tokenizer_model))
    )
    return SemanticCodecInfo(
        name=name,
        semantic_dim=base.semantic_dim,
        semantic_fps=base.semantic_fps,
        sample_rate=base.sample_rate,
        is_discrete=False,
        source_model=str(_get(codec_cfg, "source_model", base.source_model)),
        tokenizer_model=tokenizer_model,
        revision=str(_get(codec_cfg, "revision", base.revision)),
    )


def prepare_feature_metadata(
    output_dir: str | Path,
    cfg: Any,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    expected = semantic_codec_info(cfg).to_dict()
    metadata_path = output_dir / "feature_metadata.json"
    existing_feature_files = (output_dir / "feats").exists()
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != expected["fingerprint"] and not overwrite:
            raise ValueError(
                "Precomputed semantic codec mismatch: "
                f"existing={existing.get('name')}@{existing.get('fingerprint')}, "
                f"requested={expected['name']}@{expected['fingerprint']}. "
                "Use a different --output-dir or pass --overwrite."
            )
    elif existing_feature_files and not overwrite:
        raise ValueError(
            f"{output_dir} contains legacy features without codec metadata. "
            "Use a different --output-dir or pass --overwrite."
        )
    metadata_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return expected


class MaskGCTSemanticCodec(nn.Module):
    def __init__(self, cfg: Any, model_dir: Path) -> None:
        super().__init__()

        import safetensors.torch
        from omegaconf import OmegaConf
        from transformers import SeamlessM4TFeatureExtractor

        from semantic2any.third_party.indextts.maskgct import (
            build_semantic_codec,
            build_semantic_model,
        )

        paths_cfg = _get(cfg, "paths")
        index_cfg_path = model_dir / "config.yaml"
        semantic_codec_cfg = None
        if index_cfg_path.exists():
            semantic_codec_cfg = _get(
                OmegaConf.load(index_cfg_path), "semantic_codec", None
            )

        def resolve_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else model_dir / path

        w2v_stat = resolve_path(_get(paths_cfg, "w2v_stat", "wav2vec2bert_stats.pt"))
        w2v_bert_dir = resolve_path(_get(paths_cfg, "w2v_bert_dir", "w2v-bert-2.0"))
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            str(w2v_bert_dir), local_files_only=True
        )
        model, mean, std = build_semantic_model(
            str(w2v_stat), model_path=str(w2v_bert_dir)
        )
        self.semantic_model = model.eval()
        self.register_buffer("semantic_mean", mean.float())
        self.register_buffer("semantic_std", std.float())
        # RepCodec's constructor defaults are the published MaskGCT dimensions.
        # A legacy IndexTTS config can still override them, but the minimal
        # asset bundle intentionally does not require the unrelated config.
        self.codec = build_semantic_codec(semantic_codec_cfg).eval()

        configured = str(_get(paths_cfg, "semantic_codec_ckpt", "") or "")
        if configured:
            checkpoint = resolve_path(configured)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"semantic codec checkpoint not found: {checkpoint}")
        else:
            candidates = [
                model_dir / "semantic_codec.safetensors",
                model_dir / "semantic_codec/model.safetensors",
                model_dir / "semantic_codec.pth",
                model_dir / "semantic_codec.pt",
            ]
            checkpoint = next((path for path in candidates if path.is_file()), None)
            if checkpoint is None:
                matches = sorted(model_dir.glob("**/*semantic*codec*.safetensors"))
                checkpoint = matches[0] if matches else None
            if checkpoint is None:
                raise FileNotFoundError(
                    "Could not find semantic codec checkpoint. "
                    "Set paths.semantic_codec_ckpt in the config."
                )
        if checkpoint.suffix == ".safetensors":
            safetensors.torch.load_model(
                self.codec, str(checkpoint), strict=True, device="cpu"
            )
        else:
            state = torch.load(checkpoint, map_location="cpu")
            self.codec.load_state_dict(state.get("state_dict", state), strict=False)
        self.requires_grad_(False)
        self.eval()

    @property
    def info(self) -> SemanticCodecInfo:
        return SEMANTIC_CODEC_SPECS["maskgct"]

    def _quantize(self, feature: torch.Tensor) -> torch.Tensor:
        _, semantic = self.codec.quantize(feature)
        if not torch.is_floating_point(semantic):
            semantic = self.codec.quantizer.vq2emb(semantic.unsqueeze(1))
            if semantic.ndim == 3 and semantic.size(1) != feature.size(1):
                semantic = semantic.transpose(1, 2)
        elif (
            semantic.ndim == 3
            and semantic.size(1) != feature.size(1)
            and semantic.size(2) == feature.size(1)
        ):
            semantic = semantic.transpose(1, 2)
        return semantic.squeeze(0).float()

    @torch.no_grad()
    def extract(self, waveforms: list[np.ndarray]) -> list[torch.Tensor]:
        inputs = self.feature_extractor(
            waveforms, sampling_rate=16000, return_tensors="pt", padding=True
        )
        device = self.semantic_mean.device
        input_features = inputs["input_features"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        output = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feature = (output.hidden_states[17] - self.semantic_mean) / self.semantic_std
        lengths = attention_mask.sum(dim=1).long()
        return [
            self._quantize(feature[index : index + 1, : int(lengths[index])])
            for index in range(feature.size(0))
        ]


class SACSemanticCodec(nn.Module):
    """SAC raw semantic embedding without loading any acoustic modules."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        from huggingface_hub import snapshot_download
        from transformers import WhisperFeatureExtractor

        from semantic2any.third_party.sac_whisper import (
            load_whisper_vq_semantic_encoder,
        )

        codec_cfg = _get(cfg, "semantic_codec")
        info = semantic_codec_info(cfg)
        cache_dir = str(_get(codec_cfg, "cache_dir", "") or "") or None
        local_files_only = bool(_get(codec_cfg, "local_files_only", False))
        local_path = str(_get(codec_cfg, "tokenizer_path", "") or "")
        if local_path:
            model_dir = Path(local_path).expanduser().resolve()
            if not model_dir.is_dir():
                raise FileNotFoundError(f"SAC tokenizer_path does not exist: {model_dir}")
        else:
            model_dir = Path(
                snapshot_download(
                    repo_id=info.tokenizer_model,
                    revision=info.revision or None,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    allow_patterns=(
                        "config.json",
                        "preprocessor_config.json",
                        "model*.safetensors",
                    ),
                )
            )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.encoder = load_whisper_vq_semantic_encoder(model_dir)
        if self.encoder.semantic_dim != info.semantic_dim:
            raise ValueError(
                f"SAC semantic dimension mismatch: expected {info.semantic_dim}, "
                f"loaded {self.encoder.semantic_dim}"
            )
        if self.encoder.codebook_size != 16384:
            raise ValueError(
                f"SAC semantic codebook mismatch: expected 16384, "
                f"loaded {self.encoder.codebook_size}"
            )
        self._info = info
        self.max_chunk_samples = int(
            float(_get(codec_cfg, "max_chunk_seconds", 30.0)) * info.sample_rate
        )
        self.chunk_batch_size = max(
            1, int(_get(codec_cfg, "chunk_batch_size", _get(_get(cfg, "data"), "feature_batch_size", 16)))
        )
        self.requires_grad_(False)
        self.eval()

    @property
    def info(self) -> SemanticCodecInfo:
        return self._info

    @torch.no_grad()
    def extract(self, waveforms: list[np.ndarray]) -> list[torch.Tensor]:
        chunks: list[np.ndarray] = []
        owners: list[int] = []
        for owner, waveform in enumerate(waveforms):
            waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if waveform.size == 0:
                raise ValueError("Cannot extract SAC semantic features from empty audio")
            for start in range(0, waveform.size, self.max_chunk_samples):
                chunks.append(waveform[start : start + self.max_chunk_samples])
                owners.append(owner)

        outputs: list[list[torch.Tensor]] = [[] for _ in waveforms]
        device = next(self.encoder.parameters()).device
        stride = (
            int(self.encoder.conv1.stride[0])
            * int(self.encoder.conv2.stride[0])
            * int(self.encoder.pooling_kernel_size)
            * int(self.feature_extractor.hop_length)
        )
        for start in range(0, len(chunks), self.chunk_batch_size):
            chunk_batch = chunks[start : start + self.chunk_batch_size]
            features = self.feature_extractor(
                chunk_batch,
                sampling_rate=self.info.sample_rate,
                return_attention_mask=True,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=stride,
            )
            _, embeddings, mask = self.encoder(
                features.input_features.to(device),
                features.attention_mask.to(device),
            )
            for offset in range(len(chunk_batch)):
                owner = owners[start + offset]
                outputs[owner].append(embeddings[offset][mask[offset]].float())
        return [torch.cat(parts, dim=0) for parts in outputs]


def build_semantic_codec(cfg: Any, *, model_dir: Path) -> nn.Module:
    name = semantic_codec_type(cfg)
    if name == "maskgct":
        return MaskGCTSemanticCodec(cfg, model_dir)
    return SACSemanticCodec(cfg)
