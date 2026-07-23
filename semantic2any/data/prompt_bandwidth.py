"""Prompt-only bandwidth simulation for low sample-rate reference mel."""

from __future__ import annotations

import torch
import torchaudio
from torch.nn import functional as F


def simulate_lower_sample_rate(
    waveform: torch.Tensor,
    sample_rate: int,
    simulate_sample_rate: int,
) -> torch.Tensor:
    """Band-limit ``waveform`` as if it were captured at ``simulate_sample_rate``.

    Downs ample to the simulated rate and upsample back so anti-aliasing matches
    a real low-rate capture, then pad/trim to the original length.
    """
    sample_rate = int(sample_rate)
    simulate_sample_rate = int(simulate_sample_rate)
    if simulate_sample_rate <= 0:
        raise ValueError(f"simulate_sample_rate must be positive, got {simulate_sample_rate}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if simulate_sample_rate >= sample_rate:
        return waveform

    squeezed = False
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
        squeezed = True
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be [channels, samples], got {tuple(waveform.shape)}")

    original_length = waveform.size(-1)
    limited = torchaudio.functional.resample(
        torchaudio.functional.resample(waveform, sample_rate, simulate_sample_rate),
        simulate_sample_rate,
        sample_rate,
    )
    if limited.size(-1) > original_length:
        limited = limited[..., :original_length]
    elif limited.size(-1) < original_length:
        limited = F.pad(limited, (0, original_length - limited.size(-1)))

    if squeezed:
        limited = limited.squeeze(0)
    return limited
