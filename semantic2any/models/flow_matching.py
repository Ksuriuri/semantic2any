from __future__ import annotations

from abc import ABC
import os as _fm_os

import torch
from torch import nn
from tqdm.auto import tqdm

from semantic2any.defaults import DEFAULT_MEL_CHANNELS
from semantic2any.models.stream_layout import assemble_stream_batch, plan_stream_windows


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
        self.streaming = bool(_get(args, "streaming", False))
        self.stream_chunk_frames = int(_get(args, "stream_chunk_frames", 25))
        self.stream_history_frames = int(_get(args, "stream_history_frames", 150))
        if self.stream_chunk_frames < 1 or self.stream_history_frames < 0:
            raise ValueError(
                "stream_chunk_frames must be positive and stream_history_frames cannot be negative, "
                f"got {self.stream_chunk_frames} and {self.stream_history_frames}"
            )

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
        solver = self.solve_euler_stream if getattr(self, "streaming", False) else self.solve_euler
        x = solver(
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

    def solve_euler_stream(
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
        """Denoise the target in fixed chunks. Euler updates only the current chunk."""
        if self.estimator is None:
            raise RuntimeError("CFM estimator has not been initialized")
        prompt_len = int(prompt.size(-1))
        batch, channels, total = x.shape
        if mu.size(1) != total:
            raise ValueError(f"streaming mu has {mu.size(1)} frames but x has {total}")
        target_lens = (x_lens - prompt_len).clamp_min(0)
        max_target = int(target_lens.max().item()) if batch else 0
        generated = x.new_zeros(batch, channels, max_target)
        source = x.clone()
        mu = mu.clone()
        if self.zero_prompt_speech_token and prompt_len:
            mu[:, :prompt_len] = 0
        cursor = 0
        while cursor < max_target:
            remain = (target_lens - cursor).clamp_min(0)
            chunk_lens = torch.minimum(
                remain, torch.full_like(remain, self.stream_chunk_frames)
            )
            if int(chunk_lens.max().item()) == 0:
                break
            history_lens = torch.where(
                chunk_lens > 0,
                torch.full_like(chunk_lens, min(self.stream_history_frames, cursor)),
                torch.zeros_like(chunk_lens),
            )
            prompt_lens = torch.full_like(chunk_lens, prompt_len)
            assembled = prompt_lens + history_lens + chunk_lens
            width = int(assembled.max().item())
            flow = x.new_zeros(batch, channels, width)
            prompt_x = x.new_zeros(batch, channels, width)
            cond = mu.new_zeros(batch, width, mu.size(-1))
            if prompt_len:
                prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]
                cond[:, :prompt_len] = mu[:, :prompt_len]
            for index in range(batch):
                hist = int(history_lens[index])
                chunk = int(chunk_lens[index])
                if hist:
                    hist_src = cursor - hist
                    flow[index, :, prompt_len : prompt_len + hist] = generated[
                        index, :, hist_src:cursor
                    ]
                    cond[index, prompt_len : prompt_len + hist] = mu[
                        index, prompt_len + hist_src : prompt_len + cursor
                    ]
                if chunk:
                    chunk_src = prompt_len + cursor
                    flow[index, :, prompt_len + hist : prompt_len + hist + chunk] = source[
                        index, :, chunk_src : chunk_src + chunk
                    ]
                    cond[index, prompt_len + hist : prompt_len + hist + chunk] = mu[
                        index, chunk_src : chunk_src + chunk
                    ]
            t = t_span[0]
            iterator = range(1, len(t_span))
            if show_progress:
                iterator = tqdm(iterator, desc="CFM stream sampling")
            positions = torch.arange(width, device=x.device)
            chunk_begin = (prompt_lens + history_lens).unsqueeze(1)
            chunk_end = chunk_begin + chunk_lens.unsqueeze(1)
            region = (positions.unsqueeze(0) >= chunk_begin) & (positions.unsqueeze(0) < chunk_end)
            region = region.unsqueeze(1)
            for step in iterator:
                dt = t_span[step] - t_span[step - 1]
                dphi = self._stream_velocity(
                    flow,
                    prompt_x,
                    assembled,
                    t,
                    style,
                    cond,
                    prompt_lens,
                    history_lens,
                    chunk_lens,
                    inference_cfg_rate,
                    drop_style,
                )
                flow = flow + dt * dphi * region.to(dphi.dtype)
                t = t + dt
            for index in range(batch):
                chunk = int(chunk_lens[index])
                hist = int(history_lens[index])
                if chunk:
                    begin = prompt_len + hist
                    generated[index, :, cursor : cursor + chunk] = flow[
                        index, :, begin : begin + chunk
                    ]
            cursor += int(chunk_lens.max().item())
        output = x.new_zeros(batch, channels, total)
        if max_target:
            output[:, :, prompt_len : prompt_len + max_target] = generated
        return output

    def _stream_velocity(
        self,
        flow: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        mu: torch.Tensor,
        prompt_lens: torch.Tensor,
        history_lens: torch.Tensor,
        chunk_lens: torch.Tensor,
        inference_cfg_rate: float,
        drop_style: bool,
    ) -> torch.Tensor:
        step = t.reshape(1).expand(flow.size(0))
        if inference_cfg_rate > 0:
            # Training dropout clears only the semantic condition, so the
            # unconditional branch keeps the reference prompt and the style.
            zeros_mu = torch.zeros_like(mu)
            dphi, uncond = self.estimator(
                torch.cat([flow, flow], dim=0),
                torch.cat([prompt_x, prompt_x], dim=0),
                torch.cat([x_lens, x_lens], dim=0),
                step.repeat(2),
                torch.cat([style, style], dim=0),
                torch.cat([mu, zeros_mu], dim=0),
                prompt_lens=torch.cat([prompt_lens, prompt_lens], dim=0),
                stream_history_lens=torch.cat([history_lens, history_lens], dim=0),
                stream_chunk_lens=torch.cat([chunk_lens, chunk_lens], dim=0),
                drop_style=drop_style,
            ).chunk(2, dim=0)
            return (1.0 + inference_cfg_rate) * dphi - inference_cfg_rate * uncond
        return self.estimator(
            flow,
            prompt_x,
            x_lens,
            step,
            style,
            mu,
            prompt_lens=prompt_lens,
            stream_history_lens=history_lens,
            stream_chunk_lens=chunk_lens,
            drop_style=drop_style,
        )

    def _reduce_flow_loss(
        self,
        estimator_out: torch.Tensor,
        velocity: torch.Tensor,
        loss_mask: torch.Tensor,
        valid_samples: torch.Tensor,
    ) -> torch.Tensor:
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
        masked_loss = torch.where(
            loss_mask.unsqueeze(1), element_loss, element_loss.new_zeros(())
        )
        denominators = (
            loss_mask.sum(dim=1).clamp_min(1).to(element_loss.dtype)
            * estimator_out.size(1)
        )
        per_sample_loss = masked_loss.sum(dim=(1, 2)) / denominators
        return torch.where(
            valid_samples, per_sample_loss, per_sample_loss.new_zeros(())
        ).sum() / valid_samples.sum().clamp_min(1)

    def _overlay_chunk_estimate(
        self,
        base: torch.Tensor,
        estimate: torch.Tensor,
        prompt_lens: torch.Tensor,
        starts: torch.Tensor,
        history_lens: torch.Tensor,
        chunk_lens: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        for index in range(base.size(0)):
            row = base[index].detach()
            chunk = int(chunk_lens[index])
            if chunk:
                src = int(prompt_lens[index]) + int(starts[index])
                assembled_at = int(prompt_lens[index]) + int(history_lens[index])
                value = estimate[index, :, assembled_at : assembled_at + chunk]
                row = torch.cat((row[:, :src], value, row[:, src + chunk :]), dim=-1)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def _forward_streaming(
        self,
        x1: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.estimator is None:
            raise RuntimeError("CFM estimator has not been initialized")
        x1_raw = x1
        if self.feat_scale != 1.0:
            x1 = x1 * self.feat_scale
        batch = x1.shape[0]
        time = torch.rand(batch, 1, 1, device=x1.device, dtype=x1.dtype)
        self.last_time = time.detach().reshape(-1)
        target_lens = x_lens - prompt_lens
        starts, history_lens, chunk_lens = plan_stream_windows(
            prompt_lens,
            target_lens,
            self.stream_chunk_frames,
            self.stream_history_frames,
        )
        flow, prompt, cond, velocity, noise, assembled = assemble_stream_batch(
            x1,
            mu,
            prompt_lens,
            starts,
            history_lens,
            chunk_lens,
            sigma_min=self.sigma_min,
            time=time,
        )
        if self.zero_prompt_speech_token:
            ref_positions = torch.arange(cond.size(1), device=cond.device)
            ref_mask = ref_positions.unsqueeze(0) < prompt_lens.unsqueeze(1)
            cond = cond.masked_fill(ref_mask.unsqueeze(-1), 0)
        estimator_out = self.estimator(
            flow,
            prompt,
            assembled,
            time.squeeze(2).squeeze(1),
            style,
            cond,
            prompt_lens=prompt_lens,
            stream_history_lens=history_lens,
            stream_chunk_lens=chunk_lens,
        )
        positions = torch.arange(flow.size(-1), device=flow.device)
        chunk_begin = prompt_lens + history_lens
        loss_mask = (positions.unsqueeze(0) >= chunk_begin.unsqueeze(1)) & (
            positions.unsqueeze(0) < (chunk_begin + chunk_lens).unsqueeze(1)
        )
        loss = self._reduce_flow_loss(estimator_out, velocity, loss_mask, chunk_lens > 0)
        if _X1HAT_MODE == "onestep":
            estimate = flow + (1 - time) * estimator_out
        else:
            estimate = estimator_out + (1 - self.sigma_min) * noise
        if self.feat_scale != 1.0:
            estimate = estimate / self.feat_scale
        # Only the chunk is a real x1 estimate. Other frames stay the detached
        # target so an auxiliary loss does not train the reference or history.
        x1_hat = self._overlay_chunk_estimate(
            x1_raw,
            estimate,
            prompt_lens,
            starts,
            history_lens,
            chunk_lens,
        )
        return loss, x1_hat

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
        if getattr(self, "streaming", False):
            return self._forward_streaming(x1, x_lens, prompt_lens, mu, style)

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
        # fp32 select, not a multiply: under fp16 an excluded prompt or padding
        # frame can overflow, and `inf * 0 = nan` would poison the batch.
        valid_samples = x_lens > prompt_lens
        loss = self._reduce_flow_loss(estimator_out, velocity, loss_mask, valid_samples)

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
            if self.streaming:
                raise NotImplementedError("streaming is implemented for the DiT estimator only")
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
