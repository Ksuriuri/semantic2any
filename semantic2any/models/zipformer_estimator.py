from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from semantic2any.defaults import DEFAULT_MEL_CHANNELS
from semantic2any.models.common import lengths_to_padding_mask
from semantic2any.models.zipvoice_modules import TTSZipformer


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def _fit_time(x: Tensor, frames: int, value: float = 0.0) -> Tensor:
    if x.size(1) == frames:
        return x
    if x.size(1) > frames:
        return x[:, :frames]
    return F.pad(x, (0, 0, 0, frames - x.size(1)), value=value)


class ZipFormerEstimator(nn.Module):
    """CFM velocity estimator using a ZipVoice-inspired ZipFormer decoder."""

    def __init__(self, args) -> None:
        super().__init__()
        dit_cfg = _get(args, "DiT")
        zip_cfg = _get(args, "ZipFormer")
        style_cfg = _get(args, "style_encoder")

        self.in_channels = int(_get(dit_cfg, "in_channels", DEFAULT_MEL_CHANNELS))
        self.content_dim = int(_get(dit_cfg, "content_dim", 512))
        self.style_dim = int(_get(style_cfg, "dim", 192))
        self.style_condition = bool(_get(zip_cfg, "style_condition", False))
        self.class_dropout_prob = float(_get(zip_cfg, "class_dropout_prob", 0.0))
        self.condition_dropout_prob = float(_get(zip_cfg, "condition_dropout_prob", self.class_dropout_prob))

        input_dim = self.in_channels * 2 + self.content_dim
        if self.style_condition:
            input_dim += self.style_dim

        self.cond_projection = nn.Linear(self.content_dim, self.content_dim)
        self.style_projection = nn.Linear(self.style_dim, self.style_dim)
        self.decoder = TTSZipformer(
            in_dim=input_dim,
            out_dim=self.in_channels,
            downsampling_factor=tuple(_get(zip_cfg, "downsampling_factor", (1, 2, 4, 2, 1))),
            num_encoder_layers=tuple(_get(zip_cfg, "num_layers", (2, 2, 4, 4, 4))),
            cnn_module_kernel=tuple(_get(zip_cfg, "cnn_module_kernel", (31, 15, 7, 15, 31))),
            encoder_dim=int(_get(zip_cfg, "hidden_dim", 512)),
            feedforward_dim=int(_get(zip_cfg, "feedforward_dim", 1536)),
            num_heads=int(_get(zip_cfg, "num_heads", 4)),
            query_head_dim=int(_get(zip_cfg, "query_head_dim", 32)),
            value_head_dim=int(_get(zip_cfg, "value_head_dim", 12)),
            pos_head_dim=int(_get(zip_cfg, "pos_head_dim", 4)),
            pos_dim=int(_get(zip_cfg, "pos_dim", 48)),
            time_embed_dim=int(_get(zip_cfg, "time_embed_dim", 192)),
            use_time_embed=True,
            use_conv=bool(_get(zip_cfg, "use_conv", True)),
        )

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
        x_in = x.transpose(1, 2)
        prompt_in = prompt_x.transpose(1, 2)
        cond = _fit_time(cond, frames)
        cond = self.cond_projection(cond)

        if self.training and self.condition_dropout_prob > 0:
            drop = torch.rand(batch, 1, 1, device=x.device) < self.condition_dropout_prob
            cond = torch.where(drop, torch.zeros_like(cond), cond)
            prompt_in = torch.where(drop, torch.zeros_like(prompt_in), prompt_in)

        parts = [x_in, prompt_in, cond]
        if self.style_condition:
            style = self.style_projection(style)
            if drop_style:
                style = torch.zeros_like(style)
            elif self.training and self.class_dropout_prob > 0:
                style_dropout_mask = torch.rand(batch, 1, device=x.device) < self.class_dropout_prob
                style = torch.where(style_dropout_mask, torch.zeros_like(style), style)
            parts.append(style.unsqueeze(1).expand(-1, frames, -1))

        decoder_in = torch.cat(parts, dim=-1)
        padding_mask = lengths_to_padding_mask(x_lens, max_length=frames)
        out = self.decoder(decoder_in, t=t, padding_mask=padding_mask)
        return out.transpose(1, 2)
