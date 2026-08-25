from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from semantic2any.models.common import sequence_mask


_TIME_ALIGN_MODES = ("nearest", "pack2")


class InterpolateRegulator(nn.Module):
    """IndexTTS-compatible semantic-to-frame length regulator.

    It accepts either continuous semantic embeddings ``[B, T, C]`` or discrete
    codebooks ``[B, Q, T]`` and returns target-rate conditioning ``[B, T_y, D]``.

    ``time_align="nearest"`` (default) gathers one source frame per output frame.
    ``time_align="pack2"`` keeps a 2× semantic grid: linear-resample to
    ``2 * ylens``, then concatenate each pair so both 50 Hz frames reach the
    25 Hz CFM condition instead of dropping every other frame.
    """

    def __init__(
        self,
        channels: int,
        sampling_ratios: Sequence[int],
        is_discrete: bool = False,
        in_channels: int | None = None,
        codebook_size: int = 8192,
        out_channels: int | None = None,
        groups: int = 1,
        n_codebooks: int = 1,
        quantizer_dropout: float = 0.0,
        f0_condition: bool = False,
        n_f0_bins: int = 512,
        time_align: str = "nearest",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.is_discrete = is_discrete
        self.n_codebooks = n_codebooks
        self.quantizer_dropout = quantizer_dropout
        self.f0_condition = f0_condition
        if time_align not in _TIME_ALIGN_MODES:
            raise ValueError(
                f"time_align must be one of {_TIME_ALIGN_MODES}, got {time_align!r}"
            )
        if time_align == "pack2" and is_discrete:
            raise ValueError("time_align='pack2' requires continuous semantic features")
        self.time_align = time_align
        out_channels = out_channels or channels

        layers: list[nn.Module] = []
        self.interpolate = len(tuple(sampling_ratios)) > 0
        for _ in sampling_ratios:
            layers.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(groups, channels),
                    nn.Mish(),
                ]
            )
        layers.append(nn.Conv1d(channels, out_channels, kernel_size=1))
        self.model = nn.Sequential(*layers)

        if is_discrete:
            self.embedding = nn.Embedding(codebook_size, channels)
            if n_codebooks > 1:
                self.extra_codebooks = nn.ModuleList(
                    nn.Embedding(codebook_size, channels) for _ in range(n_codebooks - 1)
                )
        else:
            if in_channels is None:
                raise ValueError("in_channels must be set for continuous semantic inputs")
            proj_in = 2 * int(in_channels) if time_align == "pack2" else int(in_channels)
            self.content_in_proj = nn.Linear(proj_in, channels)

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
            self.n_f0_bins = n_f0_bins

    def _embed_discrete(self, x: torch.Tensor, n_quantizers: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected discrete semantic codes with 3 dims, got {tuple(x.shape)}")
        x = x.long()
        out = self.embedding(x[:, 0])
        for idx, emb in enumerate(getattr(self, "extra_codebooks", []), start=1):
            active = (n_quantizers > idx).to(out.dtype).view(-1, 1, 1)
            out = out + active * emb(x[:, idx])
        return out

    def _resolve_xlens(self, x: torch.Tensor, xlens: torch.Tensor | None) -> torch.Tensor:
        if xlens is None:
            return torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        if xlens.ndim != 1 or xlens.size(0) != x.size(0):
            raise ValueError("xlens must be a 1-D length tensor matching the batch size")
        return xlens

    def _align_nearest(
        self,
        x: torch.Tensor,
        ylens: torch.Tensor,
        xlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Map ``[B, T_x, C]`` to ``[B, C, max_y]`` by per-sample nearest gather."""
        max_y = int(ylens.max().item())
        if not self.interpolate:
            xt = x.transpose(1, 2).contiguous()
            return xt[..., :max_y]
        if xlens is None:
            return F.interpolate(x.transpose(1, 2).contiguous(), size=max_y, mode="nearest")
        xlens = self._resolve_xlens(x, xlens)
        positions = torch.arange(max_y, device=x.device)
        source_indices = torch.div(
            positions.unsqueeze(0) * xlens.unsqueeze(1),
            ylens.clamp_min(1).unsqueeze(1),
            rounding_mode="floor",
        )
        source_indices = source_indices.clamp(min=0, max=x.size(1) - 1)
        gathered = x.gather(1, source_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        valid = positions.unsqueeze(0) < ylens.unsqueeze(1)
        return gathered.masked_fill(~valid.unsqueeze(-1), 0).transpose(1, 2)

    def _linear_resample_time(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        out_lens: torch.Tensor,
        out_max: int,
    ) -> torch.Tensor:
        """Per-sample 1-D linear resample matching ``F.interpolate(..., align_corners=False)``."""
        channels = x.size(-1)
        positions = torch.arange(out_max, device=x.device, dtype=x.dtype)
        in_len = xlens.clamp_min(1).unsqueeze(1).to(dtype=x.dtype)
        out_len = out_lens.clamp_min(1).unsqueeze(1).to(dtype=x.dtype)
        src = (positions.unsqueeze(0) + 0.5) * in_len / out_len - 0.5
        src_max = (xlens - 1).clamp_min(0).unsqueeze(1).to(dtype=x.dtype)
        src = src.clamp(min=torch.zeros_like(src_max), max=src_max)
        src0 = src.floor().long().clamp(min=0, max=x.size(1) - 1)
        src1 = (src0 + 1).clamp_max(x.size(1) - 1)
        weight = (src - src0.to(dtype=src.dtype)).unsqueeze(-1)
        left = x.gather(1, src0.unsqueeze(-1).expand(-1, -1, channels))
        right = x.gather(1, src1.unsqueeze(-1).expand(-1, -1, channels))
        resampled = left * (1.0 - weight) + right * weight
        valid = positions.unsqueeze(0) < out_lens.unsqueeze(1)
        return resampled.masked_fill(~valid.unsqueeze(-1), 0)

    def _pack2_align(
        self,
        x: torch.Tensor,
        ylens: torch.Tensor,
        xlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Resample to ``2 * ylens`` then concat pairs → ``[B, max_y, 2C]``."""
        xlens = self._resolve_xlens(x, xlens)
        max_y = int(ylens.max().item())
        fine_lens = ylens * 2
        fine = self._linear_resample_time(x, xlens, fine_lens, 2 * max_y)
        packed = fine.view(x.size(0), max_y, 2, x.size(-1)).reshape(
            x.size(0), max_y, 2 * x.size(-1)
        )
        valid = torch.arange(max_y, device=x.device).unsqueeze(0) < ylens.unsqueeze(1)
        return packed.masked_fill(~valid.unsqueeze(-1), 0)

    def forward(
        self,
        x: torch.Tensor,
        ylens: torch.Tensor,
        n_quantizers: int | torch.Tensor | None = None,
        f0: torch.Tensor | None = None,
        xlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        if ylens.ndim != 1:
            raise ValueError("ylens must be a 1-D length tensor")

        if self.training and self.is_discrete:
            n_quantizers_tensor = torch.full(
                (x.shape[0],), self.n_codebooks, dtype=torch.long, device=x.device
            )
            if self.quantizer_dropout > 0 and self.n_codebooks > 1:
                count = int(x.shape[0] * self.quantizer_dropout)
                if count > 0:
                    n_quantizers_tensor[:count] = torch.randint(
                        1, self.n_codebooks + 1, (count,), device=x.device
                    )
        elif isinstance(n_quantizers, torch.Tensor):
            n_quantizers_tensor = n_quantizers.to(device=x.device, dtype=torch.long)
        else:
            n = self.n_codebooks if n_quantizers is None else int(n_quantizers)
            n_quantizers_tensor = torch.full((x.shape[0],), n, dtype=torch.long, device=x.device)

        if self.is_discrete:
            x = self._embed_discrete(x, n_quantizers_tensor)
            x = self._align_nearest(x, ylens, xlens)
        else:
            if not torch.is_floating_point(x):
                raise TypeError("Continuous length regulator expects floating point semantic features")
            if self.time_align == "pack2":
                packed = self._pack2_align(x, ylens, xlens)
                x = self.content_in_proj(packed).transpose(1, 2).contiguous()
            else:
                x = self._align_nearest(self.content_in_proj(x), ylens, xlens)

        max_y = int(ylens.max().item())
        mask = sequence_mask(ylens, max_y).unsqueeze(-1).to(x.dtype)

        if self.f0_condition:
            if f0 is None:
                x = x + self.f0_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            else:
                f0 = f0.clamp(0, self.n_f0_bins - 1).long()
                f0_emb = self.f0_embedding(f0)
                f0_emb = F.interpolate(f0_emb.transpose(1, 2).contiguous(), size=max_y, mode="nearest")
                x = x + f0_emb

        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, ylens, None, None, None
