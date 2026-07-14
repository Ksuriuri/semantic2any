from __future__ import annotations

from abc import ABC

import torch
from torch import nn
from tqdm.auto import tqdm


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


class BASECFM(nn.Module, ABC):
    """Conditional flow matching wrapper compatible with IndexTTS s2mel."""

    def __init__(self, args) -> None:
        super().__init__()
        self.sigma_min = 1e-6
        self.estimator: nn.Module | None = None
        dit_cfg = _get(args, "DiT")
        self.in_channels = int(_get(dit_cfg, "in_channels", 80))
        reg_loss_type = _get(args, "reg_loss_type", "l1")
        self.criterion = nn.MSELoss() if reg_loss_type == "l2" else nn.L1Loss()
        self.zero_prompt_speech_token = bool(_get(dit_cfg, "zero_prompt_speech_token", False))

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
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        return self.solve_euler(
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

        batch = x1.shape[0]
        time = torch.rand(batch, 1, 1, device=x1.device, dtype=x1.dtype)
        noise = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * time) * noise + time * x1
        velocity = x1 - (1 - self.sigma_min) * noise

        prompt = torch.zeros_like(x1)
        y = y.clone()
        mu = mu.clone()
        for idx in range(batch):
            prompt_len = int(prompt_lens[idx].item())
            prompt[idx, :, :prompt_len] = x1[idx, :, :prompt_len]
            y[idx, :, :prompt_len] = 0
            if self.zero_prompt_speech_token:
                mu[idx, :prompt_len] = 0

        estimator_out = self.estimator(
            y,
            prompt,
            x_lens,
            time.squeeze(2).squeeze(1),
            style,
            mu,
            prompt_lens=prompt_lens,
        )

        losses = []
        for idx in range(batch):
            start = int(prompt_lens[idx].item())
            end = int(x_lens[idx].item())
            if end <= start:
                continue
            losses.append(self.criterion(estimator_out[idx, :, start:end], velocity[idx, :, start:end]))
        if not losses:
            loss = estimator_out.sum() * 0
        else:
            loss = torch.stack(losses).mean()

        return loss, estimator_out + (1 - self.sigma_min) * noise


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
