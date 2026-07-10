from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from semantic2any.models.common import sequence_mask


class InterpolateRegulator(nn.Module):
    """IndexTTS-compatible semantic-to-frame length regulator.

    It accepts either continuous semantic embeddings ``[B, T, C]`` or discrete
    codebooks ``[B, Q, T]`` and returns mel-rate conditioning ``[B, T_mel, D]``.
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
    ) -> None:
        super().__init__()
        self.channels = channels
        self.is_discrete = is_discrete
        self.n_codebooks = n_codebooks
        self.quantizer_dropout = quantizer_dropout
        self.f0_condition = f0_condition
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
            self.content_in_proj = nn.Linear(in_channels, channels)

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
        else:
            if not torch.is_floating_point(x):
                raise TypeError("Continuous length regulator expects floating point semantic features")
            x = self.content_in_proj(x)

        max_y = int(ylens.max().item())
        mask = sequence_mask(ylens, max_y).unsqueeze(-1).to(x.dtype)
        if self.interpolate:
            xt = x.transpose(1, 2).contiguous()
            if xlens is not None:
                # Stretch each sample by its own semantic/mel length pair; a single
                # batch-wide interpolation to max_y misaligns shorter samples whose
                # semantic:mel ratio differs from the longest one.
                if xlens.ndim != 1 or xlens.size(0) != x.size(0):
                    raise ValueError("xlens must be a 1-D length tensor matching the batch size")
                out = xt.new_zeros(xt.size(0), xt.size(1), max_y)
                for idx in range(xt.size(0)):
                    xlen = int(xlens[idx].item())
                    ylen = int(ylens[idx].item())
                    if xlen <= 0 or ylen <= 0:
                        continue
                    out[idx, :, :ylen] = F.interpolate(
                        xt[idx : idx + 1, :, :xlen], size=ylen, mode="nearest"
                    )[0]
                x = out
            else:
                x = F.interpolate(xt, size=max_y, mode="nearest")
        else:
            x = x.transpose(1, 2).contiguous()
            x = x[..., :max_y]

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
