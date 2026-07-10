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

    def forward(
        self,
        mel,
        mel_lens,
        prompt_lens,
        semantic_or_mu,
        style,
        semantic_is_mu: bool = False,
        semantic_lens: torch.Tensor | None = None,
    ):
        mu = (
            semantic_or_mu
            if semantic_is_mu
            else self.build_condition(semantic_or_mu, mel_lens, semantic_lens=semantic_lens)
        )
        return self.models["cfm"](mel, mel_lens, prompt_lens, mu, style)

    def enable_torch_compile(self) -> None:
        self.models["cfm"].enable_torch_compile()
