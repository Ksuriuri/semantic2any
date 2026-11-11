"""Chunk windows and block masks for streaming S2VAE.

The full-sequence CFM path stays the default. These helpers are used only when
the streaming switch is on.
"""

from __future__ import annotations

import torch
from torch import Tensor


def draw_chunk_starts(max_start: Tensor) -> Tensor:
    """Uniform inclusive start in ``0 .. max_start`` for each batch item."""
    unit = torch.rand(max_start.shape, device=max_start.device, dtype=torch.float32)
    starts = torch.floor(unit * (max_start.to(dtype=torch.float32) + 1)).to(dtype=max_start.dtype)
    return torch.minimum(starts, max_start)


def plan_stream_windows(
    prompt_lens: Tensor,
    target_lens: Tensor,
    chunk_frames: int,
    history_frames: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pick a target chunk and the ground-truth history immediately on its left.

    Returns chunk starts within the target, history lengths, and chunk lengths.
    A target shorter than the chunk uses the whole target and no history.
    """
    if chunk_frames < 1:
        raise ValueError("stream_chunk_frames must be positive")
    if history_frames < 0:
        raise ValueError("stream_history_frames cannot be negative")
    if torch.any(target_lens < 0):
        raise ValueError("target length cannot be negative")
    chunk_lim = torch.full_like(target_lens, int(chunk_frames))
    chunk_lens = torch.minimum(target_lens.clamp_min(0), chunk_lim)
    max_start = (target_lens - chunk_lens).clamp_min(0)
    starts = draw_chunk_starts(max_start)
    history_lim = torch.full_like(starts, int(history_frames))
    history_lens = torch.minimum(starts, history_lim)
    return starts, history_lens, chunk_lens


def stream_block_mask(
    prompt_lens: Tensor,
    history_lens: Tensor,
    chunk_lens: Tensor,
    total: int,
    prefix: int = 0,
) -> Tensor:
    """Bool mask ``[B, 1, total, total]``. True means the query may attend to the key.

    Content order is reference, history, then chunk. Reference sees only itself,
    history sees reference and history, and the chunk sees all three. Prefix
    tokens see only prefix tokens. Content queries may see prefix tokens, so a
    time or style token cannot carry future chunk frames back to the history.
    """
    if prefix < 0 or total < prefix:
        raise ValueError(f"invalid streaming mask shape prefix={prefix} total={total}")
    batch = int(prompt_lens.shape[0])
    pos = torch.arange(total, device=prompt_lens.device)
    query = pos.view(1, total, 1)
    key = pos.view(1, 1, total)
    frame_q = query - prefix
    frame_k = key - prefix
    prompt = prompt_lens.view(batch, 1, 1)
    history_end = (prompt_lens + history_lens).view(batch, 1, 1)
    chunk_end = (prompt_lens + history_lens + chunk_lens).view(batch, 1, 1)
    query_prefix = query < prefix
    key_prefix = key < prefix
    query_ref = (frame_q >= 0) & (frame_q < prompt)
    key_ref = (frame_k >= 0) & (frame_k < prompt)
    query_hist = (frame_q >= prompt) & (frame_q < history_end)
    key_hist = (frame_k >= prompt) & (frame_k < history_end)
    query_chunk = (frame_q >= history_end) & (frame_q < chunk_end)
    key_chunk = (frame_k >= history_end) & (frame_k < chunk_end)
    allowed = (
        (query_ref & key_ref)
        | (query_hist & (key_ref | key_hist))
        | (query_chunk & (key_ref | key_hist | key_chunk))
        | (query_prefix & key_prefix)
        | ((query_ref | query_hist | query_chunk) & key_prefix)
    )
    return allowed.unsqueeze(1)


def assemble_stream_batch(
    x1: Tensor,
    mu: Tensor,
    prompt_lens: Tensor,
    starts: Tensor,
    history_lens: Tensor,
    chunk_lens: Tensor,
    *,
    sigma_min: float,
    time: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build ``[reference | clean history | noised chunk]`` for one training step.

    History stays in the flow input and is not copied into ``prompt``. Only the
    chunk is noised. ``velocity`` is zero outside the chunk.
    """
    batch, channels, _frames = x1.shape
    assembled = prompt_lens + history_lens + chunk_lens
    total = int(assembled.max().item()) if batch else 0
    if total <= 0:
        raise ValueError("streaming batch has no frames")
    flow = x1.new_zeros(batch, channels, total)
    prompt = x1.new_zeros(batch, channels, total)
    cond = mu.new_zeros(batch, total, mu.size(-1))
    velocity = x1.new_zeros(batch, channels, total)
    noise_full = x1.new_zeros(batch, channels, total)
    step = time.view(batch, 1, 1)
    for index in range(batch):
        ref = int(prompt_lens[index])
        hist = int(history_lens[index])
        chunk = int(chunk_lens[index])
        start = int(starts[index])
        hist_src = ref + start - hist
        chunk_src = ref + start
        if ref:
            prompt[index, :, :ref] = x1[index, :, :ref]
            cond[index, :ref] = mu[index, :ref]
        if hist:
            flow[index, :, ref : ref + hist] = x1[index, :, hist_src : hist_src + hist]
            cond[index, ref : ref + hist] = mu[index, hist_src : hist_src + hist]
        if chunk:
            target = x1[index, :, chunk_src : chunk_src + chunk]
            noise = torch.randn_like(target)
            mixed = (1 - (1 - sigma_min) * step[index]) * noise + step[index] * target
            dest = slice(ref + hist, ref + hist + chunk)
            flow[index, :, dest] = mixed
            noise_full[index, :, dest] = noise
            velocity[index, :, dest] = target - (1 - sigma_min) * noise
            cond[index, ref + hist : ref + hist + chunk] = mu[index, chunk_src : chunk_src + chunk]
    return flow, prompt, cond, velocity, noise_full, assembled
