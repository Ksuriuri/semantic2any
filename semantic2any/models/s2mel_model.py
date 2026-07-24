from __future__ import annotations

import torch
from torch import nn

from semantic2any.models.flow_matching import CFM
from semantic2any.models.length_regulator import InterpolateRegulator


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


class Semantic2MelModel(nn.Module):
    """IndexTTS-style s2mel wrapper.

    The public ``models`` ModuleDict mirrors IndexTTS checkpoint keys.
    """

    def __init__(self, args) -> None:
        super().__init__()
        lr_cfg = _get(args, "length_regulator")
        use_gpt_latent = bool(_get(args, "use_gpt_latent", False))

        length_regulator = InterpolateRegulator(
            channels=int(_get(lr_cfg, "channels")),
            sampling_ratios=tuple(_get(lr_cfg, "sampling_ratios", (1, 1, 1, 1))),
            is_discrete=bool(_get(lr_cfg, "is_discrete", False)),
            in_channels=_get(lr_cfg, "in_channels", None),
            codebook_size=int(_get(lr_cfg, "content_codebook_size", 8192)),
            n_codebooks=int(_get(lr_cfg, "n_codebooks", 1)),
            quantizer_dropout=float(_get(lr_cfg, "quantizer_dropout", 0.0)),
            f0_condition=bool(_get(lr_cfg, "f0_condition", False)),
            n_f0_bins=int(_get(lr_cfg, "n_f0_bins", 512)),
        )

        modules = {
            "cfm": CFM(args),
            "length_regulator": length_regulator,
        }
        if use_gpt_latent:
            modules["gpt_layer"] = nn.Sequential(
                nn.Linear(1280, 256),
                nn.Linear(256, 128),
                nn.Linear(128, int(_get(lr_cfg, "in_channels", 1024))),
            )
        self.models = nn.ModuleDict(modules)

    def build_condition(
        self,
        semantic: torch.Tensor,
        target_lengths,
        n_quantizers: int | None = None,
        f0=None,
        semantic_lens: torch.Tensor | None = None,
    ):
        return self.models["length_regulator"](
            semantic,
            ylens=target_lengths,
            n_quantizers=n_quantizers,
            f0=f0,
            xlens=semantic_lens,
        )[0]

    @staticmethod
    def _slice_semantic_segments(
        semantic: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_length = int(lengths.max().item())
        positions = torch.arange(max_length, device=semantic.device)
        source_indices = starts.unsqueeze(1) + positions.unsqueeze(0)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        if torch.is_floating_point(semantic):
            if semantic.ndim != 3:
                raise ValueError("Continuous semantic batch must be [B, T, C]")
            indices = source_indices.clamp_max(semantic.size(1) - 1)
            segments = semantic.gather(
                1,
                indices.unsqueeze(-1).expand(-1, -1, semantic.size(-1)),
            )
            return segments.masked_fill(~valid.unsqueeze(-1), 0)

        if semantic.ndim != 3:
            raise ValueError("Discrete semantic batch must be [B, Q, T]")
        indices = source_indices.clamp_max(semantic.size(-1) - 1)
        segments = semantic.gather(
            2,
            indices.unsqueeze(1).expand(-1, semantic.size(1), -1),
        )
        return segments.masked_fill(~valid.unsqueeze(1), 0)

    def build_paired_condition(
        self,
        semantic: torch.Tensor,
        mel_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        semantic_lens: torch.Tensor,
        prompt_semantic_lens: torch.Tensor,
        max_mel_frames: int | None = None,
    ) -> torch.Tensor:
        """Length-regulate prompt and target independently across the utterance boundary."""
        target_lens = mel_lens - prompt_lens
        target_semantic_lens = semantic_lens - prompt_semantic_lens
        if (
            torch.any(prompt_lens <= 0)
            or torch.any(target_lens <= 0)
            or torch.any(prompt_semantic_lens <= 0)
            or torch.any(target_semantic_lens <= 0)
        ):
            raise ValueError("Paired prompt and target lengths must all be positive")

        zeros = torch.zeros_like(prompt_semantic_lens)
        prompt_semantic = self._slice_semantic_segments(
            semantic, zeros, prompt_semantic_lens
        )
        target_semantic = self._slice_semantic_segments(
            semantic, prompt_semantic_lens, target_semantic_lens
        )
        prompt_mu = self.build_condition(
            prompt_semantic,
            prompt_lens,
            semantic_lens=prompt_semantic_lens,
        )
        target_mu = self.build_condition(
            target_semantic,
            target_lens,
            semantic_lens=target_semantic_lens,
        )

        total_frames = (
            int(max_mel_frames)
            if max_mel_frames is not None
            else int(mel_lens.max().item())
        )
        positions = torch.arange(total_frames, device=prompt_mu.device)
        prompt_positions = positions.unsqueeze(0).expand(prompt_mu.size(0), -1)
        target_positions = prompt_positions - prompt_lens.unsqueeze(1)
        prompt_values = prompt_mu.gather(
            1,
            prompt_positions.clamp_max(prompt_mu.size(1) - 1)
            .unsqueeze(-1)
            .expand(-1, -1, prompt_mu.size(-1)),
        )
        target_values = target_mu.gather(
            1,
            target_positions.clamp(min=0, max=target_mu.size(1) - 1)
            .unsqueeze(-1)
            .expand(-1, -1, target_mu.size(-1)),
        )
        prompt_mask = prompt_positions < prompt_lens.unsqueeze(1)
        valid_mask = prompt_positions < mel_lens.unsqueeze(1)
        mu = torch.where(prompt_mask.unsqueeze(-1), prompt_values, target_values)
        return mu.masked_fill(~valid_mask.unsqueeze(-1), 0)

    def forward(
        self,
        mel,
        mel_lens,
        prompt_lens,
        semantic_or_mu,
        style,
        semantic_is_mu: bool = False,
        semantic_lens: torch.Tensor | None = None,
        prompt_semantic_lens: torch.Tensor | None = None,
    ):
        if semantic_is_mu:
            mu = semantic_or_mu
        elif prompt_semantic_lens is not None:
            if semantic_lens is None:
                raise ValueError("semantic_lens is required for paired semantic conditioning")
            mu = self.build_paired_condition(
                semantic_or_mu,
                mel_lens,
                prompt_lens,
                semantic_lens,
                prompt_semantic_lens,
                max_mel_frames=mel.size(-1),
            )
        else:
            mu = self.build_condition(
                semantic_or_mu, mel_lens, semantic_lens=semantic_lens
            )
        return self.models["cfm"](mel, mel_lens, prompt_lens, mu, style)

    def enable_torch_compile(self) -> None:
        self.models["cfm"].enable_torch_compile()
