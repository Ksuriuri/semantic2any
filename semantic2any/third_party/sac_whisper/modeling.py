# Copyright 2025 Soul-AILab
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0. This is a minimal,
# inference-only adaptation of SAC's WhisperVQ encoder. See NOTICE.md.

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer


class CausalConv1d(nn.Conv1d):
    """The causal convolution used by GLM-4-Voice's WhisperVQ encoder."""

    def __init__(self, *args, **kwargs) -> None:
        kernel_size = kwargs.get("kernel_size", args[2] if len(args) > 2 else None)
        dilation = kwargs.get("dilation", 1)
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        kernel = kernel_size[0] if isinstance(kernel_size, tuple) else int(kernel_size)
        dil = dilation[0] if isinstance(dilation, tuple) else int(dilation)
        self.left_padding = dil * (kernel - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(inputs, (self.left_padding, 0)))


def _block_attention_mask(
    attention_mask: torch.Tensor,
    *,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allow all history plus bidirectional attention within each block."""

    batch_size, length = attention_mask.shape
    causal = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=attention_mask.device)
    )
    block = torch.zeros_like(causal)
    for start in range(0, length, block_size):
        stop = min(start + block_size, length)
        block[start:stop, start:stop] = True
    allowed = (causal | block).unsqueeze(0)
    allowed = allowed & attention_mask[:, None, :].bool()
    additive = torch.zeros(
        batch_size, length, length, dtype=dtype, device=attention_mask.device
    )
    additive.masked_fill_(~allowed, torch.finfo(dtype).min)
    return additive.unsqueeze(1)


class WhisperVQSemanticEncoder(nn.Module):
    """Inference-only GLM-4-Voice quantizer used by SAC's semantic stream."""

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"
        self.config = config
        embed_dim = int(config.d_model)
        conv_class = CausalConv1d if bool(config.encoder_causal_convolution) else nn.Conv1d
        self.conv1 = conv_class(
            int(config.num_mel_bins), embed_dim, kernel_size=3, padding=1
        )
        self.conv2 = conv_class(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(int(config.max_source_positions), embed_dim)
        self.layers = nn.ModuleList(
            WhisperEncoderLayer(config) for _ in range(int(config.quantize_position))
        )
        if int(config.pooling_position) != int(config.quantize_position):
            raise ValueError(
                "The minimal SAC semantic encoder requires pooling_position == "
                "quantize_position"
            )
        if str(config.pooling_type) != "avg":
            raise ValueError("The SAC 62.5 Hz semantic encoder requires average pooling")
        self.pooling_kernel_size = int(config.pooling_kernel_size)
        self.pooling = nn.AvgPool1d(kernel_size=self.pooling_kernel_size)
        self.codebook = nn.Embedding(int(config.quantize_vocab_size), embed_dim)

    @property
    def semantic_dim(self) -> int:
        return int(self.codebook.embedding_dim)

    @property
    def codebook_size(self) -> int:
        return int(self.codebook.num_embeddings)

    def embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dtype != torch.long:
            raise TypeError(f"semantic IDs must be torch.long, got {ids.dtype}")
        return self.codebook(ids)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.conv1(input_features))
        hidden = F.gelu(self.conv2(hidden)).transpose(1, 2)
        attention_mask = attention_mask[:, ::2]
        length = hidden.size(1)
        attention_mask = attention_mask[:, :length]
        if length > self.embed_positions.num_embeddings:
            raise ValueError(
                f"WhisperVQ chunk has {length} frames, maximum is "
                f"{self.embed_positions.num_embeddings}"
            )
        hidden = hidden + self.embed_positions.weight[:length]

        block_size = int(getattr(self.config, "quantize_causal_block_size", 0) or 0)
        if block_size > 0:
            extended_mask = _block_attention_mask(
                attention_mask, block_size=block_size, dtype=hidden.dtype
            )
        else:
            allowed = attention_mask[:, None, None, :].bool()
            extended_mask = torch.zeros(
                hidden.size(0), 1, length, length, dtype=hidden.dtype, device=hidden.device
            )
            extended_mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)

        for layer in self.layers:
            hidden = layer(
                hidden,
                extended_mask,
                layer_head_mask=None,
                output_attentions=False,
            )[0]

        hidden = hidden.transpose(1, 2)
        remainder = hidden.size(-1) % self.pooling_kernel_size
        if remainder:
            hidden = F.pad(hidden, (0, self.pooling_kernel_size - remainder))
        hidden = self.pooling(hidden).transpose(1, 2)
        semantic_mask = attention_mask[:, :: self.pooling_kernel_size]
        semantic_mask = semantic_mask[:, : hidden.size(1)].bool()

        flat = hidden.reshape(-1, hidden.size(-1)).float()
        codebook = self.codebook.weight.float()
        distances = torch.addmm(
            codebook.square().sum(dim=1) + flat.square().sum(dim=1, keepdim=True),
            flat,
            codebook.transpose(0, 1),
            beta=1.0,
            alpha=-2.0,
        )
        ids = distances.argmin(dim=1).view(hidden.size(0), hidden.size(1))
        embeddings = self.embed_ids(ids)
        return ids, embeddings, semantic_mask


def load_whisper_vq_semantic_encoder(model_dir: str | Path) -> WhisperVQSemanticEncoder:
    """Load only the encoder prefix and codebook from GLM-4-Voice weights."""

    model_dir = Path(model_dir)
    config = WhisperConfig.from_pretrained(str(model_dir), local_files_only=True)
    model = WhisperVQSemanticEncoder(config)
    target_keys = set(model.state_dict())
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("model*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                prefix = "model.encoder."
                local_key = key[len(prefix) :] if key.startswith(prefix) else key
                if local_key in target_keys:
                    state[local_key] = handle.get_tensor(key)
    if not state:
        raise FileNotFoundError(
            f"No model.encoder weights found in {model_dir}/model*.safetensors"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "GLM-4-Voice semantic encoder state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.codebook.weight.requires_grad_(False)
    model.requires_grad_(False)
    model.eval()
    return model
