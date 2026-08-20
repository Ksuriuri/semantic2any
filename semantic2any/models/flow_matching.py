from __future__ import annotations

from abc import ABC
import os as _fm_os

import torch
from torch import nn
from tqdm.auto import tqdm

from semantic2any.defaults import DEFAULT_MEL_CHANNELS


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


_X1HAT_MODE = _fm_os.environ.get("S2MEL_X1HAT_MODE", "onestep").strip().lower()
if _X1HAT_MODE not in ("onestep", "velocity"):
    raise ValueError(
        f"S2MEL_X1HAT_MODE must be 'onestep' or 'velocity', got {_X1HAT_MODE!r}"
    )


class BASECFM(nn.Module, ABC):
    """Conditional flow matching wrapper compatible with IndexTTS s2mel."""

    def __init__(self, args) -> None:
        super().__init__()
        self.sigma_min = 1e-6
        self.estimator: nn.Module | None = None
        dit_cfg = _get(args, "DiT")
        self.in_channels = int(_get(dit_cfg, "in_channels", DEFAULT_MEL_CHANNELS))
        reg_loss_type = _get(args, "reg_loss_type", "l1")
        self.criterion = nn.MSELoss() if reg_loss_type == "l2" else nn.L1Loss()
        self.zero_prompt_speech_token = bool(_get(dit_cfg, "zero_prompt_speech_token", False))
        # ZipVoice-style: scale log-mel into the same ballpark as N(0,1).
        # 1.0 = current recipe. Inference divides the ODE output back.
        self.feat_scale = float(_get(args, "feat_scale", 1.0) or 1.0)
        if self.feat_scale <= 0.0:
            raise ValueError(f"feat_scale must be positive, got {self.feat_scale}")
        # Extra weight on the highest mel band (lowest stays 1). 0 = off.
        self.high_band_mse_extra = float(_get(args, "high_band_mse_extra", 0.0) or 0.0)

    @torch.inference_mode()
    def inference(
        self,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor | None,
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
        show_progress: bool = False,
        drop_style: bool = False,
    ) -> torch.Tensor:
        del f0
        batch, total_frames = mu.shape[:2]
        z = torch.randn(batch, self.in_channels, total_frames, device=mu.device, dtype=mu.dtype)
        z = z * temperature
        if self.feat_scale != 1.0:
            prompt = prompt * self.feat_scale
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        x = self.solve_euler(
            z,
            x_lens,
            prompt,
            mu,
            style,
            t_span,
            inference_cfg_rate,
            show_progress,
            drop_style=drop_style,
        )
        if self.feat_scale != 1.0:
            x = x / self.feat_scale
        return x

    def setup_estimator_caches(self, max_batch_size: int, max_seq_length: int) -> None:
        """Initialize estimator caches when the selected backbone requires them."""
        if self.estimator is None:
            raise RuntimeError("CFM estimator has not been initialized")
        setup_caches = getattr(self.estimator, "setup_caches", None)
        if setup_caches is not None:
            setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

    def solve_euler(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: float = 0.5,
        show_progress: bool = False,
        drop_style: bool = False,
    ) -> torch.Tensor:
        if self.estimator is None:
            raise RuntimeError("CFM estimator has not been initialized")

        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x = x.clone()
        mu = mu.clone()
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        t = t_span[0]
        iterator = range(1, len(t_span))
        if show_progress:
            iterator = tqdm(iterator, desc="CFM sampling")

        for step in iterator:
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_lens = torch.cat([x_lens, x_lens], dim=0)
                stacked_t = t.reshape(1).expand(stacked_x.size(0))
                dphi_dt, cfg_dphi_dt = self.estimator(
                    stacked_x,
                    stacked_prompt_x,
                    stacked_lens,
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                    drop_style=drop_style,
                ).chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(
                    x,
                    prompt_x,
                    x_lens,
                    t.reshape(1).expand(x.size(0)),
                    style,
                    mu,
                    drop_style=drop_style,
                )

            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0

        return x

    def forward(
        self,
        x1: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.estimator is None:
            raise RuntimeError("CFM estimator has not been initialized")

        if self.feat_scale != 1.0:
            x1 = x1 * self.feat_scale
        batch = x1.shape[0]
        time = torch.rand(batch, 1, 1, device=x1.device, dtype=x1.dtype)
        # Stashed for the auxiliary loss.  x1_hat's error is
        # (1-t)*(v_pred - velocity), so at small t it is noise dominated and the
        # vocoder-space target is unreachable; AUX_T_MIN / AUX_T_POW gate and
        # weight on this.  Kept on the module rather than widening the return
        # tuple, which every caller unpacks as (loss, x1_hat).
        self.last_time = time.detach().reshape(-1)
        noise = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * time) * noise + time * x1
        velocity = x1 - (1 - self.sigma_min) * noise

        prompt = torch.zeros_like(x1)
        y = y.clone()
        mu = mu.clone()
        positions = torch.arange(x1.size(-1), device=x1.device)
        prompt_mask = positions.unsqueeze(0) < prompt_lens.unsqueeze(1)
        prompt_mask_channels = prompt_mask.unsqueeze(1)
        prompt = torch.where(prompt_mask_channels, x1, prompt)
        # Keep the unmasked interpolant for the one-step x1 estimate below.
        y_unmasked = y
        y = y.masked_fill(prompt_mask_channels, 0)
        if self.zero_prompt_speech_token:
            mu = mu.masked_fill(prompt_mask.unsqueeze(-1), 0)

        estimator_out = self.estimator(
            y,
            prompt,
            x_lens,
            time.squeeze(2).squeeze(1),
            style,
            mu,
            prompt_lens=prompt_lens,
        )

        loss_mask = (
            (positions.unsqueeze(0) >= prompt_lens.unsqueeze(1))
            & (positions.unsqueeze(0) < x_lens.unsqueeze(1))
        )
        # fp32 for the residual and everything downstream: under fp16 autocast
        # |d| > 256 already overflows on squaring (max 65504), and one inf
        # inside the loss mask is enough to lose the whole step.
        difference = (estimator_out - velocity).float()
        element_loss = (
            difference.square()
            if isinstance(self.criterion, nn.MSELoss)
            else difference.abs()
        )
        if self.high_band_mse_extra != 0.0:
            n_mels = element_loss.size(1)
            ramp = torch.linspace(
                0.0, 1.0, n_mels, device=element_loss.device, dtype=element_loss.dtype
            )
            band_w = 1.0 + self.high_band_mse_extra * ramp.square()
            element_loss = element_loss * band_w.view(1, n_mels, 1)
        # Select, do not multiply: under fp16 `element_loss` can be inf in a
        # region this mask excludes (the unconstrained prompt segment, or the
        # padding tail), and `inf * 0 = nan` would poison the whole batch.
        masked_loss = torch.where(
            loss_mask.unsqueeze(1), element_loss, element_loss.new_zeros(())
        )
        denominators = (
            loss_mask.sum(dim=1).clamp_min(1).to(element_loss.dtype)
            * estimator_out.size(1)
        )
        per_sample_loss = masked_loss.sum(dim=(1, 2)) / denominators
        valid_samples = x_lens > prompt_lens
        # Same reason as above: a degenerate sample (x_lens <= prompt_lens) is
        # excluded, so its loss must be dropped rather than multiplied by zero.
        loss = torch.where(
            valid_samples, per_sample_loss, per_sample_loss.new_zeros(())
        ).sum() / valid_samples.sum().clamp_min(1)

        # One-step estimate of x1, consumed only by the auxiliary loss.
        #   y + (1-t) * v_pred  == x1 + sigma_min * noise  when v_pred is exact,
        # and its error is (1-t) * (v_pred - velocity), so it becomes accurate as
        # t -> 1.  The older `estimator_out + (1-sigma_min)*noise` is also exact
        # for a perfect v_pred but carries the full velocity error at every t,
        # which pinned the aux's phase term at the random-phase bound.
        if _X1HAT_MODE == "onestep":
            x1_hat = y_unmasked + (1 - time) * estimator_out
        else:
            x1_hat = estimator_out + (1 - self.sigma_min) * noise
        if self.feat_scale != 1.0:
            x1_hat = x1_hat / self.feat_scale
        return loss, x1_hat


class CFM(BASECFM):
    def __init__(self, args) -> None:
        super().__init__(args)
        dit_type = _get(args, "dit_type", "ZipFormer")
        if dit_type == "ZipFormer":
            from semantic2any.models.zipformer_estimator import ZipFormerEstimator

            self.estimator = ZipFormerEstimator(args)
        elif dit_type == "DiT":
            from semantic2any.models.dit_estimator import DiTEstimator

            self.estimator = DiTEstimator(args)
        else:
            raise NotImplementedError(f"Unknown diffusion estimator type: {dit_type}")

    def enable_torch_compile(self) -> None:
        if self.estimator is None:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.estimator = torch.compile(self.estimator, fullgraph=True, dynamic=True)
