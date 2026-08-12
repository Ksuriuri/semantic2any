from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


# The new default.  Set ``semantic_codec.type: maskgct`` to get the old
# MaskGCT RepCodec back.
DEFAULT_SEMANTIC_CODEC = "indextts25"


SEMANTIC_CODEC_SPECS = {
    "maskgct": SemanticCodecInfo(
        name="maskgct",
        semantic_dim=1024,
        semantic_fps=50.0,
        sample_rate=16000,
        is_discrete=False,
        source_model="IndexTTS/MaskGCT-RepCodec",
    ),
    "indextts25": SemanticCodecInfo(
        name="indextts25",
        semantic_dim=1024,
        # Code rate.  The decoder upsamples 2x, so the features s2mel consumes
        # stay at 50 Hz -- see SEMANTIC_CODEC_FRAMES_PER_CODE.
        semantic_fps=25.0,
        sample_rate=16000,
        is_discrete=False,
        source_model="IndexTTS-2.5/EnhancedCodec",
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

# Decoded feature frames per stored code.  MaskGCT decodes per frame;
# IndexTTS-2.5 downsamples by 2 before quantizing and upsamples again when
# decoding, so its codes run at 25 Hz while the features stay at 50 Hz.
#
# Deliberately kept out of SemanticCodecInfo: its fields feed fingerprint(),
# and every manifest already written carries the fingerprint of the current
# field set.
SEMANTIC_CODEC_FRAMES_PER_CODE = {"maskgct": 1, "indextts25": 2, "sac": 1}

# Codecs sharing MaskGCT's w2v-bert layer-17 front end and asset layout.
W2V_BERT_CODECS = ("maskgct", "indextts25")

# Manifest spellings that mean the same codec.  The code-generation workers
# stamp "indextts2.5"; the config selector stays a plain identifier because it
# also names files.
SEMANTIC_CODEC_ALIASES = {
    "indextts2.5": "indextts25",
    "indextts-2.5": "indextts25",
    "indextts_2.5": "indextts25",
    "enhancedcodec": "indextts25",
}


def canonical_semantic_codec(name: Any) -> str:
    key = str(name).strip().lower()
    return SEMANTIC_CODEC_ALIASES.get(key, key)


def semantic_frames_per_code(name: str) -> int:
    return SEMANTIC_CODEC_FRAMES_PER_CODE[str(name).lower()]


def semantic_feature_fps(info: SemanticCodecInfo) -> float:
    """Rate of the features s2mel consumes; codes may be slower."""
    return info.semantic_fps * semantic_frames_per_code(info.name)


def semantic_codec_type(cfg: Any) -> str:
    codec_cfg = _get(cfg, "semantic_codec", None)
    name = canonical_semantic_codec(_get(codec_cfg, "type", DEFAULT_SEMANTIC_CODEC))
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
        cfg.semantic_codec.type = DEFAULT_SEMANTIC_CODEC

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
    if name in W2V_BERT_CODECS:
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
    CODEC_NAME = "maskgct"
    DOWNSAMPLE_SCALE = 1
    STRICT_LOAD = False
    CKPT_CANDIDATES = (
        "semantic_codec.safetensors",
        "semantic_codec/model.safetensors",
        "semantic_codec.pth",
        "semantic_codec.pt",
    )
    CKPT_GLOB = "**/*semantic*codec*.safetensors"

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
        self.codec = build_semantic_codec(
            semantic_codec_cfg, downsample_scale=self.DOWNSAMPLE_SCALE
        ).eval()

        configured = str(_get(paths_cfg, "semantic_codec_ckpt", "") or "")
        if configured:
            checkpoint = resolve_path(configured)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"semantic codec checkpoint not found: {checkpoint}")
        else:
            candidates = [model_dir / name for name in self.CKPT_CANDIDATES]
            checkpoint = next((path for path in candidates if path.is_file()), None)
            if checkpoint is None:
                matches = sorted(model_dir.glob(self.CKPT_GLOB))
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
            for key in ("model", "state_dict"):
                if isinstance(state, dict) and isinstance(state.get(key), dict):
                    state = state[key]
                    break
            self.codec.load_state_dict(state, strict=self.STRICT_LOAD)
        self.checkpoint_path = checkpoint.resolve()
        self.requires_grad_(False)
        self.eval()

    @property
    def info(self) -> SemanticCodecInfo:
        return SEMANTIC_CODEC_SPECS[self.CODEC_NAME]

    def _quantize(
        self, feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes, semantic = self.codec.quantize(feature)
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
        if codes.ndim == 2 and codes.size(0) == 1:
            codes = codes.squeeze(0)
        if codes.ndim != 1:
            raise ValueError(
                "MaskGCT extraction expects one codebook and one utterance, "
                f"got codes with shape {tuple(codes.shape)}"
            )
        return codes.long(), semantic.squeeze(0).float()

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode MaskGCT indices to the continuous features used by s2mel."""
        squeeze = False
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
            squeeze = True
        elif codes.ndim == 3 and codes.size(1) == 1:
            codes = codes.squeeze(1)
        if codes.ndim != 2:
            raise ValueError(
                f"MaskGCT codes must be [T], [B,T], or [B,1,T], got {tuple(codes.shape)}"
            )
        codebook_size = int(self.codec.quantizer.codebook_size)
        if codes.numel() and (
            int(codes.min().item()) < 0 or int(codes.max().item()) >= codebook_size
        ):
            raise ValueError(f"MaskGCT codes must be in [0, {codebook_size})")
        decoded = self.codec.quantizer.vq2emb(codes.long().unsqueeze(0))
        decoded = decoded.transpose(1, 2).contiguous().float()
        return decoded.squeeze(0) if squeeze else decoded

    @torch.no_grad()
    def codebook_lookup(self) -> torch.Tensor:
        """Materialize the frozen 8192x1024 post-projection lookup table."""
        codebook_size = int(self.codec.quantizer.codebook_size)
        device = next(self.codec.parameters()).device
        codes = torch.arange(codebook_size, device=device, dtype=torch.long)
        return self.decode_codes(codes).detach().cpu().float().contiguous()

    def codebook_metadata(self) -> dict[str, Any]:
        lookup = self.codebook_lookup()
        return {
            **self.info.to_dict(),
            "representation": f"{self.CODEC_NAME}_codes",
            "codebook_size": int(lookup.size(0)),
            "codebook_dim": int(lookup.size(1)),
            "code_dtype": "uint16",
            "lookup_dtype": "float32",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
        }

    def _encode_semantic_features(
        self, waveforms: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return feature, attention_mask.sum(dim=1).long()

    @torch.no_grad()
    def extract(self, waveforms: list[np.ndarray]) -> list[torch.Tensor]:
        feature, lengths = self._encode_semantic_features(waveforms)
        return [
            self._quantize(feature[index : index + 1, : int(lengths[index])])[1]
            for index in range(feature.size(0))
        ]

    def _codes_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return self._quantize(feature)[0]

    @torch.no_grad()
    def extract_codes(self, waveforms: list[np.ndarray]) -> list[torch.Tensor]:
        feature, lengths = self._encode_semantic_features(waveforms)
        return [
            self._codes_from_feature(feature[index : index + 1, : int(lengths[index])])
            for index in range(feature.size(0))
        ]


class IndexTTS25SemanticCodec(MaskGCTSemanticCodec):
    """IndexTTS-2.5's EnhancedCodec: same w2v-bert front end, 25 Hz codes.

    Architecturally this is the vendored ``RepCodec`` with
    ``downsample_scale=2`` -- the same module hierarchy upstream calls
    ``EnhancedCodec`` -- so ``codec.pth`` loads without any key translation.

    Its decode side is **context dependent** (ConvNeXt stack plus a 2x
    upsample), which has two consequences the MaskGCT path never had to worry
    about: there is no per-code embedding table to precompute, and code
    sequences must be decoded one unpadded utterance at a time.
    """

    CODEC_NAME = "indextts25"
    DOWNSAMPLE_SCALE = 2
    STRICT_LOAD = True
    CKPT_CANDIDATES = ("codec.pth", "codec.pt", "codec.safetensors")
    CKPT_GLOB = "**/codec.pth"

    DECODE_PREFIXES = ("quantizer.", "decoder.", "up.")

    def _codes_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        codes, _ = self.codec.quantize(feature)
        if codes.ndim == 2 and codes.size(0) == 1:
            codes = codes.squeeze(0)
        if codes.ndim != 1:
            raise ValueError(
                "IndexTTS-2.5 extraction expects one codebook and one utterance, "
                f"got codes with shape {tuple(codes.shape)}"
            )
        return codes.long()

    def _quantize(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # quantize() returns the 25 Hz quantizer output; what s2mel consumes is
        # decode()'s 50 Hz reconstruction, so ignore the former.
        codes = self._codes_from_feature(feature)
        return codes, self.decode_codes(codes).float()

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode code ids to the 50 Hz features s2mel consumes."""
        squeeze = False
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
            squeeze = True
        elif codes.ndim == 3 and codes.size(1) == 1:
            codes = codes.squeeze(1)
        if codes.ndim != 2:
            raise ValueError(
                "IndexTTS-2.5 codes must be [T], [B,T], or [B,1,T], "
                f"got {tuple(codes.shape)}"
            )
        codebook_size = int(self.codec.quantizer.codebook_size)
        if codes.numel() and (
            int(codes.min().item()) < 0 or int(codes.max().item()) >= codebook_size
        ):
            raise ValueError(f"IndexTTS-2.5 codes must be in [0, {codebook_size})")
        decoded = self.codec.decode(codes.long()).float()
        return decoded.squeeze(0) if squeeze else decoded

    def codebook_lookup(self) -> torch.Tensor:
        raise TypeError(
            "IndexTTS-2.5's decode is context dependent, so no "
            "[codebook_size, dim] lookup table exists: decoding "
            "arange(codebook_size) as one sequence would return a table in "
            "which every code has been mixed through its neighbours. Use "
            "decoder_bundle() instead."
        )

    def decoder_bundle(self) -> dict[str, Any]:
        """Decode-side weights, to be stored beside the codes they belong to."""
        state = {
            key: value.detach().cpu().clone()
            for key, value in self.codec.state_dict().items()
            if key.startswith(self.DECODE_PREFIXES)
        }
        if not state:
            raise ValueError("Refusing to write an empty decoder bundle")
        return {
            "codec_type": self.CODEC_NAME,
            "frames_per_code": semantic_frames_per_code(self.CODEC_NAME),
            "semantic_dim": int(self.info.semantic_dim),
            "arch": {
                "codebook_size": int(self.codec.codebook_size),
                "hidden_size": int(self.codec.hidden_size),
                "codebook_dim": int(self.codec.codebook_dim),
                "vocos_dim": int(self.codec.vocos_dim),
                "vocos_intermediate_dim": int(self.codec.vocos_intermediate_dim),
                "vocos_num_layers": int(self.codec.vocos_num_layers),
                "num_quantizers": int(self.codec.num_quantizers),
                "downsample_scale": int(self.codec.downsample_scale),
            },
            "state_dict": state,
            "source_checkpoint": str(self.checkpoint_path),
            "source_checkpoint_sha256": sha256_file(self.checkpoint_path),
        }

    def codebook_metadata(self) -> dict[str, Any]:
        return {
            **self.info.to_dict(),
            "representation": f"{self.CODEC_NAME}_codes",
            "codebook_size": int(self.codec.quantizer.codebook_size),
            "codebook_dim": int(self.info.semantic_dim),
            "frames_per_code": semantic_frames_per_code(self.CODEC_NAME),
            "code_dtype": "uint16",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
        }


class SemanticCodeDecoder(nn.Module):
    """Frozen decode side of a semantic codec, paired with stored code ids.

    ``decode_sequences`` deliberately takes a list of already-sliced, unpadded
    1-D code tensors rather than a padded ``[B, T]`` batch: a context-dependent
    decoder has no length mask, so a padded batch leaks padding into every
    sample's tail (measured 2-3 orders of magnitude above the decoder's own
    noise floor) while inference never pads.
    """

    codec_name = "maskgct"
    frames_per_code = 1

    @torch.no_grad()
    def decode_sequences(self, sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode one sequence at a time; always correct, never batched.

        Subclasses override this only to batch it where batching is provably
        equivalent.
        """
        return [self(item.reshape(1, -1))[0] for item in sequences]


class MaskGCTCodebookDecoder(SemanticCodeDecoder):
    """Lightweight frozen decoder for precomputed MaskGCT indices."""

    def __init__(
        self,
        lookup_path: str | Path,
        *,
        expected_sha256: str | None = None,
        payload: Any | None = None,
    ) -> None:
        super().__init__()
        lookup_path = Path(lookup_path).expanduser()
        if not lookup_path.is_file():
            raise FileNotFoundError(f"MaskGCT lookup table not found: {lookup_path}")
        self.lookup_sha256 = sha256_file(lookup_path)
        if expected_sha256:
            if self.lookup_sha256 != expected_sha256:
                raise ValueError(
                    "MaskGCT lookup table checksum mismatch: "
                    f"expected={expected_sha256}, actual={self.lookup_sha256}"
                )
        if payload is None:
            payload = torch.load(lookup_path, map_location="cpu")
        lookup = payload.get("lookup") if isinstance(payload, dict) else payload
        if not isinstance(lookup, torch.Tensor):
            raise TypeError(f"Invalid MaskGCT lookup payload in {lookup_path}")
        lookup = lookup.float().contiguous()
        expected_dim = SEMANTIC_CODEC_SPECS["maskgct"].semantic_dim
        if lookup.ndim != 2 or lookup.size(0) != 8192 or lookup.size(1) != expected_dim:
            raise ValueError(
                "MaskGCT lookup must be [8192, 1024], "
                f"got {tuple(lookup.shape)}"
            )
        self.lookup_path = lookup_path.resolve()
        self.register_buffer("lookup", lookup, persistent=False)

    @torch.no_grad()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 3:
            if codes.size(1) != 1:
                raise ValueError(
                    f"MaskGCT uses one codebook, got shape {tuple(codes.shape)}"
                )
            codes = codes[:, 0]
        if codes.ndim not in (1, 2):
            raise ValueError(f"MaskGCT codes must be [T] or [B,T], got {tuple(codes.shape)}")
        codes = codes.long()
        if codes.numel() and (
            int(codes.min().item()) < 0 or int(codes.max().item()) >= self.lookup.size(0)
        ):
            raise ValueError(f"MaskGCT codes must be in [0, {self.lookup.size(0)})")
        return F.embedding(codes, self.lookup)

    @torch.no_grad()
    def decode_sequences(self, sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        if not sequences:
            return []
        # A per-code gather over the concatenation is exactly equal to decoding
        # each sequence on its own, so this keeps the single-kernel fast path.
        lengths = [int(item.numel()) for item in sequences]
        flat = self(torch.cat([item.reshape(-1) for item in sequences], dim=0))
        return list(torch.split(flat, lengths, dim=0))


class IndexTTS25CodeDecoder(SemanticCodeDecoder):
    """Frozen IndexTTS-2.5 decode side for precomputed 25 Hz code ids."""

    codec_name = "indextts25"
    frames_per_code = 2

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        expected_sha256: str | None = None,
        payload: Any | None = None,
    ) -> None:
        super().__init__()
        from semantic2any.third_party.indextts.maskgct import RepCodec

        bundle_path = Path(bundle_path).expanduser()
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"IndexTTS-2.5 decoder bundle not found: {bundle_path}"
            )
        self.lookup_sha256 = sha256_file(bundle_path)
        if expected_sha256 and self.lookup_sha256 != expected_sha256:
            raise ValueError(
                "IndexTTS-2.5 decoder bundle checksum mismatch: "
                f"expected={expected_sha256}, actual={self.lookup_sha256}"
            )
        if payload is None:
            payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
        arch, state = _indextts25_bundle_contents(payload, bundle_path)
        frames_per_code = int(payload.get("frames_per_code", self.frames_per_code))
        if frames_per_code != int(arch.get("downsample_scale", 2)):
            raise ValueError(
                "IndexTTS-2.5 bundle is inconsistent: frames_per_code="
                f"{frames_per_code} but downsample_scale={arch.get('downsample_scale')}"
            )
        self.frames_per_code = frames_per_code
        codec = RepCodec(**arch)
        missing, unexpected = codec.load_state_dict(state, strict=False)
        stray = [key for key in missing if not key.startswith(("encoder.", "down."))]
        if unexpected or stray:
            raise ValueError(
                "IndexTTS-2.5 decoder bundle does not match the vendored "
                f"RepCodec: missing={stray[:5]}, unexpected={list(unexpected)[:5]}"
            )
        # The encode side is dead weight once the codes are stored.
        codec.encoder = None
        if getattr(codec, "down", None) is not None:
            codec.down = None
        codec.eval()
        codec.requires_grad_(False)
        self.codec = codec
        self.codebook_size = int(codec.quantizer.codebook_size)
        self.semantic_dim = int(arch["hidden_size"])
        self.bundle_path = bundle_path.resolve()
        self.source_checkpoint = str(payload.get("source_checkpoint", ""))
        self.source_checkpoint_sha256 = str(payload.get("source_checkpoint_sha256", ""))

    @torch.no_grad()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        elif codes.ndim == 3:
            if codes.size(1) != 1:
                raise ValueError(
                    f"IndexTTS-2.5 uses one codebook, got shape {tuple(codes.shape)}"
                )
            codes = codes[:, 0]
        if codes.ndim != 2:
            raise ValueError(
                f"IndexTTS-2.5 codes must be [T] or [B,T], got {tuple(codes.shape)}"
            )
        if codes.numel() and (
            int(codes.min().item()) < 0 or int(codes.max().item()) >= self.codebook_size
        ):
            raise ValueError(f"IndexTTS-2.5 codes must be in [0, {self.codebook_size})")
        # Frozen fp32 feature extractor: keep it out of the training autocast so
        # the same codes always decode to the same features.
        with torch.autocast(device_type=codes.device.type, enabled=False):
            return self.codec.decode(codes.long()).float()

    @torch.no_grad()
    def decode_sequences(self, sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        # One decode per sequence.  Batching would need equal lengths, and
        # padding is not equivalent here (see SemanticCodeDecoder).
        return [self(item.reshape(1, -1))[0] for item in sequences]


def _indextts25_bundle_contents(
    payload: Any, bundle_path: Path
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid IndexTTS-2.5 decoder payload in {bundle_path}")
    if isinstance(payload.get("state_dict"), dict) and isinstance(
        payload.get("arch"), dict
    ):
        return dict(payload["arch"]), dict(payload["state_dict"])
    # A raw upstream codec.pth is accepted too, so a decoder can be built
    # straight from IndexTTS-2.5's release.
    state = payload.get("model") if isinstance(payload.get("model"), dict) else payload
    if not isinstance(state, dict) or "quantizer.quantizers.0.codebook.weight" not in state:
        raise TypeError(f"Invalid IndexTTS-2.5 decoder payload in {bundle_path}")
    codebook = state["quantizer.quantizers.0.codebook.weight"]
    hidden = int(state["decoder.1.weight"].shape[0])
    arch = {
        "codebook_size": int(codebook.shape[0]),
        "codebook_dim": int(codebook.shape[1]),
        "hidden_size": hidden,
        "vocos_dim": int(state["decoder.0.embed.weight"].shape[0]),
        "vocos_intermediate_dim": int(state["decoder.0.convnext.0.pwconv1.weight"].shape[0]),
        "vocos_num_layers": 1
        + max(
            int(key.split(".")[3])
            for key in state
            if key.startswith("decoder.0.convnext.")
        ),
        "num_quantizers": 1
        + max(
            int(key.split(".")[2])
            for key in state
            if key.startswith("quantizer.quantizers.")
        ),
        "downsample_scale": 2 if any(key.startswith("up.") for key in state) else 1,
    }
    return arch, {
        key: value
        for key, value in state.items()
        if key.startswith(IndexTTS25SemanticCodec.DECODE_PREFIXES)
    }


def build_semantic_code_decoder(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_codec: str | None = None,
) -> SemanticCodeDecoder:
    """Build the decode side from the artifact the codes were stored with.

    The artifact, not the config, decides how codes decode, so stored codes and
    their decoder cannot drift apart.  ``expected_codec`` only cross-checks the
    configured selector against what the artifact actually is.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Semantic code decoder artifact not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("codec_type"):
        codec_name = canonical_semantic_codec(payload["codec_type"])
    elif isinstance(payload, dict) and (
        isinstance(payload.get("model"), dict)
        or "quantizer.quantizers.0.codebook.weight" in payload
    ):
        codec_name = "indextts25"
    else:
        codec_name = "maskgct"
    if expected_codec is not None and codec_name != canonical_semantic_codec(
        expected_codec
    ):
        raise ValueError(
            "Precomputed semantic codes were stored with codec "
            f"{codec_name!r} but semantic_codec.type is "
            f"{canonical_semantic_codec(expected_codec)!r}: {path}"
        )
    if codec_name == "maskgct":
        return MaskGCTCodebookDecoder(
            path, expected_sha256=expected_sha256, payload=payload
        )
    if codec_name == "indextts25":
        return IndexTTS25CodeDecoder(
            path, expected_sha256=expected_sha256, payload=payload
        )
    raise ValueError(f"Unsupported semantic code decoder artifact: {codec_name}")


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
    if name == "indextts25":
        return IndexTTS25SemanticCodec(cfg, model_dir)
    return SACSemanticCodec(cfg)
