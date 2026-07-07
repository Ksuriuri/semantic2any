from __future__ import annotations

import torch


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Return a bool mask with shape ``[batch, max_length]``."""
    if max_length is None:
        max_length = int(length.max().item())
    positions = torch.arange(max_length, device=length.device)
    return positions.unsqueeze(0) < length.unsqueeze(1)


def lengths_to_padding_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Return a key padding mask where True means padded."""
    return ~sequence_mask(length, max_length=max_length)
