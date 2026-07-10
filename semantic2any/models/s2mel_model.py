from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

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
        if torch.is_floating_point(semantic):
            segments = [
                semantic[index, int(start.item()) : int((start + length).item())]
                for index, (start, length) in enumerate(zip(starts, lengths, strict=True))
            ]
            return pad_sequence(segments, batch_first=True, padding_value=0.0)

        if semantic.ndim != 3:
            raise ValueError("Discrete semantic batch must be [B, Q, T]")
        max_length = int(lengths.max().item())
        segments = semantic.new_zeros(semantic.size(0), semantic.size(1), max_length)
        for index, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            start_value = int(start.item())
            length_value = int(length.item())
            segments[index, :, :length_value] = semantic[
                index, :, start_value : start_value + length_value
            ]
        return segments

    def build_paired_condition(
        self,
        semantic: torch.Tensor,
        mel_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        semantic_lens: torch.Tensor,
        prompt_semantic_lens: torch.Tensor,
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

        mu = prompt_mu.new_zeros(
            prompt_mu.size(0), int(mel_lens.max().item()), prompt_mu.size(-1)
        )
        for index, (prompt_len, target_len) in enumerate(
            zip(prompt_lens, target_lens, strict=True)
        ):
            prompt_length = int(prompt_len.item())
            target_length = int(target_len.item())
            mu[index, :prompt_length] = prompt_mu[index, :prompt_length]
            mu[index, prompt_length : prompt_length + target_length] = target_mu[
                index, :target_length
            ]
        return mu

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
            )
        else:
            mu = self.build_condition(
                semantic_or_mu, mel_lens, semantic_lens=semantic_lens
            )
        return self.models["cfm"](mel, mel_lens, prompt_lens, mu, style)

    def enable_torch_compile(self) -> None:
        self.models["cfm"].enable_torch_compile()
