# Derived from IndexTTS2's indextts/s2mel/modules/diffusion_transformer.py.
# Licensed under the Apache License, Version 2.0. See models/indextts_dit/LICENSE.

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from semantic2any.defaults import DEFAULT_MEL_CHANNELS
from semantic2any.models.common import sequence_mask
from semantic2any.models.indextts_dit.gpt_fast.model import ModelArgs, Transformer
from semantic2any.models.indextts_dit.wavenet import WN


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """IndexTTS2 sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, timesteps: Tensor) -> Tensor:
        args = 1000 * timesteps[:, None] * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(hidden_size, out_channels, bias=True))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class DiTEstimator(nn.Module):
    """IndexTTS2 DiT plus WaveNet CFM velocity estimator."""

    def __init__(self, args) -> None:
        super().__init__()
        dit_cfg = _get(args, "DiT")
        wavenet_cfg = _get(args, "wavenet")
        style_cfg = _get(args, "style_encoder")
        if dit_cfg is None or wavenet_cfg is None or style_cfg is None:
            raise ValueError("DiT estimator requires DiT, wavenet, and style_encoder config blocks")

        self.in_channels = int(_get(dit_cfg, "in_channels", DEFAULT_MEL_CHANNELS))
        self.content_dim = int(_get(dit_cfg, "content_dim", 512))
        self.style_dim = int(_get(style_cfg, "dim", 192))
        self.hidden_dim = int(_get(dit_cfg, "hidden_dim", 512))
        self.block_size = int(_get(dit_cfg, "block_size", 16384))
        self.time_as_token = bool(_get(dit_cfg, "time_as_token", False))
        self.style_as_token = bool(_get(dit_cfg, "style_as_token", False))
        self.style_condition = bool(_get(dit_cfg, "style_condition", False))
        self.final_layer_type = str(_get(dit_cfg, "final_layer_type", "wavenet"))
        self.class_dropout_prob = float(_get(dit_cfg, "class_dropout_prob", 0.0))
        self.is_causal = bool(_get(dit_cfg, "is_causal", False))
        self.long_skip_connection = bool(_get(dit_cfg, "long_skip_connection", True))
        if self.block_size <= 0:
            raise ValueError("DiT.block_size must be positive")
        if self.style_as_token and not self.style_condition:
            raise ValueError("DiT.style_as_token requires DiT.style_condition=true")

        num_heads = int(_get(dit_cfg, "num_heads", 8))
        if self.hidden_dim % num_heads:
            raise ValueError("DiT.hidden_dim must be divisible by DiT.num_heads")
        self.transformer = Transformer(
            ModelArgs(
                block_size=self.block_size,
                n_layer=int(_get(dit_cfg, "depth", 13)),
                n_head=num_heads,
                dim=self.hidden_dim,
                head_dim=self.hidden_dim // num_heads,
                vocab_size=1024,
                uvit_skip_connection=bool(_get(dit_cfg, "uvit_skip_connection", True)),
                time_as_token=self.time_as_token,
            )
        )
        self.x_embedder = weight_norm(nn.Linear(self.in_channels, self.hidden_dim, bias=True))
        self.content_type = str(_get(dit_cfg, "content_type", "continuous"))
        self.content_codebook_size = int(_get(dit_cfg, "content_codebook_size", 1024))
        self.cond_embedder = nn.Embedding(self.content_codebook_size, self.hidden_dim)
        self.cond_projection = nn.Linear(self.content_dim, self.hidden_dim, bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_dim)
        self.register_buffer("input_pos", torch.arange(self.block_size))

        if self.final_layer_type == "wavenet":
            wavenet_style_condition = bool(_get(wavenet_cfg, "style_condition", False))
            if wavenet_style_condition != self.style_condition:
                raise ValueError("DiT.style_condition and wavenet.style_condition must match")
            wavenet_hidden = int(_get(wavenet_cfg, "hidden_dim", self.hidden_dim))
            self.t_embedder2 = TimestepEmbedder(wavenet_hidden)
            self.conv1 = nn.Linear(self.hidden_dim, wavenet_hidden)
            self.conv2 = nn.Conv1d(wavenet_hidden, self.in_channels, 1)
            self.wavenet = WN(
                hidden_channels=wavenet_hidden,
                kernel_size=int(_get(wavenet_cfg, "kernel_size", 5)),
                dilation_rate=int(_get(wavenet_cfg, "dilation_rate", 1)),
                n_layers=int(_get(wavenet_cfg, "num_layers", 8)),
                gin_channels=wavenet_hidden,
                p_dropout=float(_get(wavenet_cfg, "p_dropout", 0.0)),
                causal=False,
            )
            self.final_layer = FinalLayer(wavenet_hidden, wavenet_hidden)
            self.res_projection = nn.Linear(self.hidden_dim, wavenet_hidden)
        elif self.final_layer_type == "mlp":
            self.final_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.in_channels),
            )
        else:
            raise ValueError(f"Unsupported DiT.final_layer_type={self.final_layer_type!r}")

        style_width = self.style_dim if self.style_condition and not self.style_as_token else 0
        self.cond_x_merge_linear = nn.Linear(
            self.hidden_dim + self.in_channels * 2 + style_width,
            self.hidden_dim,
        )
        self.content_mask_embedder = nn.Embedding(1, self.hidden_dim)
        self.skip_linear = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        if self.style_as_token:
            self.style_in = nn.Linear(self.style_dim, self.hidden_dim)

    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        if max_seq_length > self.block_size:
            raise ValueError(
                f"DiT cache length {max_seq_length} exceeds DiT.block_size={self.block_size}; "
                "increase block_size or shorten the audio."
            )
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)

    def forward(
        self,
        x: Tensor,
        prompt_x: Tensor,
        x_lens: Tensor,
        t: Tensor,
        style: Tensor,
        cond: Tensor,
        prompt_lens: Tensor | None = None,
        drop_style: bool = False,
    ) -> Tensor:
        del prompt_lens
        batch, _, frames = x.shape
        if frames > self.block_size:
            raise ValueError(
                f"DiT received {frames} frames, exceeding DiT.block_size={self.block_size}; "
                "shorten the sample or increase DiT.block_size."
            )
        if cond.size(1) != frames:
            raise ValueError(f"DiT conditioning has {cond.size(1)} frames but x has {frames}")
        if torch.any(x_lens > frames):
            raise ValueError("x_lens cannot exceed the mel frame dimension")

        class_dropout = self.training and bool(torch.rand((), device=x.device) < self.class_dropout_prob)
        if drop_style and self.style_condition:
            # The DiT merges raw style directly; zeroing here removes all
            # style-dependent weights before the shared merge-layer bias.
            style = torch.zeros_like(style)

        time_condition = self.t_embedder(t)
        cond = self.cond_projection(cond)
        x_frames = x.transpose(1, 2)
        prompt_frames = prompt_x.transpose(1, 2)
        parts = [x_frames, prompt_frames, cond]
        if self.style_condition and not self.style_as_token:
            parts.append(style.unsqueeze(1).expand(-1, frames, -1))
        x_in = torch.cat(parts, dim=-1)
        if class_dropout:
            x_in[..., self.in_channels :] = 0
        x_in = self.cond_x_merge_linear(x_in)

        token_count = frames
        if self.style_as_token:
            style_token = self.style_in(style)
            if class_dropout:
                style_token = torch.zeros_like(style_token)
            x_in = torch.cat([style_token.unsqueeze(1), x_in], dim=1)
            token_count += 1
        if self.time_as_token:
            x_in = torch.cat([time_condition.unsqueeze(1), x_in], dim=1)
            token_count += 1
        if token_count > self.block_size:
            raise ValueError(f"DiT token count {token_count} exceeds DiT.block_size={self.block_size}")

        x_mask = sequence_mask(x_lens + int(self.style_as_token) + int(self.time_as_token)).unsqueeze(1)
        input_pos = self.input_pos[:token_count]
        attention_mask = None
        if not self.is_causal:
            attention_mask = x_mask[:, None, :].expand(-1, 1, token_count, -1)
        x_res = self.transformer(x_in, time_condition.unsqueeze(1), input_pos, attention_mask)
        if self.time_as_token:
            x_res = x_res[:, 1:]
        if self.style_as_token:
            x_res = x_res[:, 1:]
        if self.long_skip_connection:
            x_res = self.skip_linear(torch.cat([x_res, x_frames], dim=-1))

        if self.final_layer_type == "wavenet":
            output = self.conv1(x_res).transpose(1, 2)
            wavenet_condition = self.t_embedder2(t).unsqueeze(2)
            output = self.wavenet(output, x_mask, g=wavenet_condition).transpose(1, 2)
            output = output + self.res_projection(x_res)
            output = self.final_layer(output, time_condition).transpose(1, 2)
            return self.conv2(output)
        return self.final_mlp(x_res).transpose(1, 2)
