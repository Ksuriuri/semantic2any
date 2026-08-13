"""Regression tests for the flow loss mask being a select, not a multiply.

v18 (`uqgfkvaw`) skipped ~0.3-10% of steps on `[SkipStep]` with `valid=nan`, and
kusuriuri's wandb read on 2026-08-12 narrowed it to the flow loss alone.  Cause:
`masked_loss = element_loss * loss_mask.unsqueeze(1)` under `mixed_precision:
fp16`, where an overflow to inf in a region the mask *excludes* -- the prompt
segment, whose `y` is zeroed and where the model is unconstrained, or the
padding tail -- became `inf * 0 = nan` and poisoned the whole batch loss.

These tests pin the intended semantics: excluded positions and excluded samples
contribute nothing, whatever they contain.  A non-finite value *inside* the mask
must still propagate, because that is a genuinely bad step and the trainer's
SkipStep guard is what should catch it.

`torch.randn_like` is stubbed to zeros so that with `x1 = 0` both the noise and
the target velocity vanish, which makes `element_loss` exactly the injected
spike squared instead of something that depends on the sampled time and noise.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from semantic2any.models.flow_matching import CFM


class _InjectingEstimator(nn.Module):
    """Returns a fixed tensor, so `element_loss` is controllable per position."""

    def __init__(self, spike: torch.Tensor) -> None:
        super().__init__()
        self.spike = spike

    def forward(
        self,
        x: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,
        drop_style: bool = False,
    ) -> torch.Tensor:
        del prompt_x, x_lens, t, style, cond, prompt_lens, drop_style
        return self.spike.to(dtype=x.dtype, device=x.device)


def _cfm(spike: torch.Tensor) -> CFM:
    cfm = CFM.__new__(CFM)
    nn.Module.__init__(cfm)
    cfm.in_channels = spike.size(1)
    cfm.sigma_min = 1e-6
    cfm.criterion = nn.MSELoss()
    cfm.zero_prompt_speech_token = False
    cfm.estimator = _InjectingEstimator(spike)
    return cfm


def _batch(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    batch, channels, frames = 2, 3, 8
    return {
        "x1": torch.zeros(batch, channels, frames, dtype=dtype),
        "x_lens": torch.tensor([6, 8]),
        "prompt_lens": torch.tensor([2, 2]),
        "mu": torch.zeros(batch, frames, 4, dtype=dtype),
        "style": torch.zeros(batch, 4, dtype=dtype),
    }


def _loss(spike: torch.Tensor, args: dict[str, torch.Tensor]) -> torch.Tensor:
    torch.manual_seed(1234)
    with patch("torch.randn_like", side_effect=torch.zeros_like):
        loss, _ = _cfm(spike)(**args)
    return loss


class FlowLossMaskTest(unittest.TestCase):
    def test_inf_in_prompt_region_does_not_poison_the_batch(self) -> None:
        args = _batch(torch.float32)
        spike = torch.zeros_like(args["x1"])
        spike[0, :, 0] = float("inf")  # inside prompt_lens=2, excluded
        loss = _loss(spike, args)
        self.assertTrue(torch.isfinite(loss), f"loss={loss}")

    def test_inf_in_padding_tail_does_not_poison_the_batch(self) -> None:
        args = _batch(torch.float32)
        spike = torch.zeros_like(args["x1"])
        spike[0, :, 7] = float("inf")  # beyond x_lens=6, excluded
        loss = _loss(spike, args)
        self.assertTrue(torch.isfinite(loss), f"loss={loss}")

    def test_fp16_overflow_in_prompt_region_stays_finite(self) -> None:
        """The real mechanism: square() overflows fp16 above |difference| ~ 256."""
        args = _batch(torch.float16)
        spike = torch.zeros_like(args["x1"])
        spike[0, :, 0] = 3.0e4  # 9e8 >> 65504, so element_loss is inf here
        loss = _loss(spike, args)
        self.assertTrue(torch.isfinite(loss), f"loss={loss}")

    def test_degenerate_sample_is_dropped_not_multiplied_by_zero(self) -> None:
        args = _batch(torch.float32)
        args["x_lens"] = torch.tensor([2, 8])  # sample 0: x_lens == prompt_lens
        spike = torch.zeros_like(args["x1"])
        spike[0, :, 1] = float("inf")
        loss = _loss(spike, args)
        self.assertTrue(torch.isfinite(loss), f"loss={loss}")

    def test_inf_inside_the_mask_still_propagates(self) -> None:
        """A genuinely bad step must stay visible for the SkipStep guard."""
        args = _batch(torch.float32)
        spike = torch.zeros_like(args["x1"])
        spike[0, :, 3] = float("inf")  # prompt_lens=2 <= 3 < x_lens=6, included
        loss = _loss(spike, args)
        self.assertFalse(torch.isfinite(loss), f"loss={loss}")

    def test_masking_is_unchanged_when_everything_is_finite(self) -> None:
        """The fix must not move the loss value on healthy batches."""
        args = _batch(torch.float32)
        spike = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 7.0
        loss = _loss(spike, args)
        keep = torch.tensor(
            [[[0, 0, 1, 1, 1, 1, 0, 0]], [[0, 0, 1, 1, 1, 1, 1, 1]]],
            dtype=torch.float32,
        )
        per_sample = (spike.square() * keep).sum(dim=(1, 2)) / (
            torch.tensor([4.0, 6.0]) * 3
        )
        self.assertTrue(torch.allclose(loss, per_sample.mean(), atol=1e-6),
                        f"loss={loss} expected={per_sample.mean()}")


if __name__ == "__main__":
    unittest.main()
