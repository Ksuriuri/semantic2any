"""Joint BigVGAN fine-tuning inside the s2mel loop: losses + optimizer plumbing.

Turned off by default.  ``VOCODER_TRAIN=1`` unfreezes the vocoder; see
``VocoderGANTrainer.from_env``.

Why this is not just ``requires_grad_(True)``
--------------------------------------------
* **The MR-STFT aux target is the vocoder's own output.**  ``BigVGANMRSTFTLoss``
  builds its "ground truth" as ``vocoder(gt_mel_chunk)`` under ``no_grad``.  With
  a frozen vocoder that is a fixed target; unfreeze it and the target moves with
  the thing being trained, so the pair can reduce the loss by agreeing on
  something quieter and smoother rather than by getting better.  A trainable
  vocoder therefore needs **real audio** as the target, which is why
  ``VOCODER_TRAIN`` also switches the dataset to return the target waveform.
* **DDP does not see it.**  The vocoder lives inside a loss module built after
  ``accelerator.prepare``, and it is called more than once per step (predicted
  mel branch + real mel branch) with gradient checkpointing on top -- all three
  are things ``DistributedDataParallel`` does not support.  So the gradients are
  averaged explicitly in :meth:`VocoderGANTrainer.clip_and_step`.
* **The discriminators start from scratch.**  The HF checkpoint contains
  ``bigvgan_generator.pt`` only, so for the first few thousand steps D is noise.
  The adversarial and feature-matching weights ramp in linearly over
  ``VOCODER_GAN_WARMUP_STEPS``.

Loss layout (weights are the upstream BigVGAN v2 values where one exists):

    real-mel branch   vocoder(real mel)  vs real audio   -> adv + fm + mel(15)
    predicted branch  vocoder(x1_hat)    vs real audio   -> the existing aux loss
                                                            (+ adv/fm only when
                                                             VOCODER_GAN_ON
                                                             includes 'pred')

The real-mel branch is the anchor: it is exactly what BigVGAN's own training
optimises, so it keeps the vocoder's true-mel mapping intact while the flow model
drags the predicted-mel branch around.  Without it the vocoder adapts to blurry
mels, stops being reusable, and copy-synthesis stops being a valid reference.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import torch
import torch.distributed as dist
from torch import nn


# --- losses -------------------------------------------------------------------


def discriminator_loss(
    real_scores: Sequence[torch.Tensor], fake_scores: Sequence[torch.Tensor]
) -> torch.Tensor:
    """LSGAN discriminator loss, as in HiFi-GAN and BigVGAN."""
    loss = None
    for score_r, score_g in zip(real_scores, fake_scores):
        term = (1.0 - score_r).pow(2).mean() + score_g.pow(2).mean()
        loss = term if loss is None else loss + term
    if loss is None:
        raise ValueError("discriminator_loss called with no sub-discriminators")
    return loss


def generator_adversarial_loss(fake_scores: Sequence[torch.Tensor]) -> torch.Tensor:
    loss = None
    for score_g in fake_scores:
        term = (1.0 - score_g).pow(2).mean()
        loss = term if loss is None else loss + term
    if loss is None:
        raise ValueError("generator_adversarial_loss called with no sub-discriminators")
    return loss


def feature_matching_loss(
    real_maps: Sequence[Sequence[torch.Tensor]],
    fake_maps: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    """L1 between D's intermediate activations, x2 as upstream weights it."""
    loss = None
    for maps_r, maps_g in zip(real_maps, fake_maps):
        for layer_r, layer_g in zip(maps_r, maps_g):
            term = (layer_r.detach() - layer_g).abs().mean()
            loss = term if loss is None else loss + term
    if loss is None:
        raise ValueError("feature_matching_loss called with no feature maps")
    return loss * 2.0


class MelReconstructionLoss(nn.Module):
    """L1 between the log-mel of a generated waveform and a target log-mel.

    Single resolution, deliberately: ``mel_spectrogram``'s filterbank cache is
    keyed on ``(sampling_rate, fmax, device, dtype)`` and omits ``n_fft`` and
    ``num_mels``, so asking it for a second resolution at the same sample rate
    silently reuses the first one's basis.  Upstream's ``use_multiscale_melloss``
    would need its own filterbanks to be safe.
    """

    def __init__(self, mel_args: dict):
        super().__init__()
        self.mel_args = dict(mel_args)

    def forward(self, wav: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        from semantic2any.third_party.indextts import mel_spectrogram

        with torch.amp.autocast(device_type="cuda", enabled=False):
            mel = mel_spectrogram(wav.float(), **self.mel_args)
        frames = min(mel.size(-1), target_mel.size(-1))
        return (mel[..., :frames] - target_mel[..., :frames].float()).abs().mean()


# --- distributed helpers ------------------------------------------------------


def _all_reduce_grads(params: Iterable[nn.Parameter], world_size: int) -> None:
    """Average grads across ranks, materialising the ones that stayed None.

    A rank whose whole micro-batch was dropped by the t gate produces no vocoder
    gradient at all.  Leaving its ``.grad`` as None would either crash the
    all-reduce or, worse, let that rank step on stale values -- so it
    contributes an explicit zero and every rank reduces the same tensor list.
    """
    if world_size <= 1:
        return
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _broadcast_params(module: nn.Module, src: int = 0) -> None:
    """Make every rank agree on a randomly initialised module."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


# --- the trainer --------------------------------------------------------------


class VocoderGANTrainer(nn.Module):
    """Owns the trainable vocoder, its discriminators, and their optimizers."""

    def __init__(
        self,
        vocoder: nn.Module,
        *,
        mel_args: dict,
        world_size: int = 1,
        learning_rate: float = 1.0e-05,
        discriminator_lr: float = 1.0e-04,
        betas: tuple[float, float] = (0.8, 0.99),
        lr_decay: float = 0.9999996,
        adv_weight: float = 1.0,
        fm_weight: float = 2.0,
        mel_weight: float = 15.0,
        real_mel_weight: float = 15.0,
        real_wave_l1_weight: float = 0.0,
        grad_clip: float = 500.0,
        warmup_steps: int = 2000,
        gan_on: str = "real",
        mpd_periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        mrd_resolutions: tuple[tuple[int, int, int], ...] = (
            (1024, 120, 600),
            (2048, 240, 1200),
            (512, 50, 240),
        ),
        channel_mult: int = 1,
    ):
        super().__init__()
        from semantic2any.third_party.indextts.discriminators import (
            MultiPeriodDiscriminator,
            MultiResolutionDiscriminator,
        )

        if gan_on not in ("real", "pred", "both", "off"):
            raise ValueError(
                f"VOCODER_GAN_ON must be one of real/pred/both/off, got {gan_on!r}"
            )
        self.vocoder = vocoder
        for param in self.vocoder.parameters():
            param.requires_grad_(True)
        self.mpd = MultiPeriodDiscriminator(mpd_periods, channel_mult=channel_mult)
        self.mrd = MultiResolutionDiscriminator(mrd_resolutions, channel_mult=channel_mult)
        self.mel_loss = MelReconstructionLoss(mel_args)

        self.world_size = int(world_size)
        self.adv_weight = float(adv_weight)
        self.fm_weight = float(fm_weight)
        self.mel_weight = float(mel_weight)
        self.real_mel_weight = float(real_mel_weight)
        self.real_wave_l1_weight = float(real_wave_l1_weight)
        self.grad_clip = float(grad_clip)
        self.warmup_steps = int(warmup_steps)
        self.gan_on = gan_on

        self.optimizer_g = torch.optim.AdamW(
            self.vocoder.parameters(), lr=learning_rate, betas=betas
        )
        self.optimizer_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.mrd.parameters()),
            lr=discriminator_lr,
            betas=betas,
        )
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_g, gamma=lr_decay
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_d, gamma=lr_decay
        )
        # Waveforms stashed by generator_loss for the discriminator's own step.
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.last_components: dict[str, torch.Tensor] = {}
        self.component_keys = (
            "d_loss", "adv", "fm", "mel", "real_mel", "real_wave_l1",
        )

    # -- construction ---------------------------------------------------------

    @staticmethod
    def enabled() -> bool:
        return os.environ.get("VOCODER_TRAIN", "0") == "1"

    @classmethod
    def from_env(
        cls, vocoder: nn.Module, *, mel_args: dict, world_size: int
    ) -> "VocoderGANTrainer":
        env = os.environ.get
        return cls(
            vocoder,
            mel_args=mel_args,
            world_size=world_size,
            learning_rate=float(env("VOCODER_LR", "1.0e-05")),
            discriminator_lr=float(env("VOCODER_D_LR", "1.0e-04")),
            lr_decay=float(env("VOCODER_LR_DECAY", "0.9999996")),
            adv_weight=float(env("VOCODER_ADV_WEIGHT", "1.0")),
            fm_weight=float(env("VOCODER_FM_WEIGHT", "2.0")),
            mel_weight=float(env("VOCODER_MEL_WEIGHT", "15.0")),
            real_mel_weight=float(env("VOCODER_REAL_MEL_WEIGHT", "15.0")),
            real_wave_l1_weight=float(env("VOCODER_REAL_WAVE_L1_WEIGHT", "0.0")),
            grad_clip=float(env("VOCODER_GRAD_CLIP", "500.0")),
            warmup_steps=int(env("VOCODER_GAN_WARMUP_STEPS", "2000")),
            gan_on=env("VOCODER_GAN_ON", "real"),
        )

    def describe(self) -> str:
        return (
            f"[Vocoder] joint training ON: lr={self.optimizer_g.param_groups[0]['lr']:.2e} "
            f"d_lr={self.optimizer_d.param_groups[0]['lr']:.2e} gan_on={self.gan_on} "
            f"adv={self.adv_weight} fm={self.fm_weight} mel={self.mel_weight} "
            f"real_mel={self.real_mel_weight} warmup={self.warmup_steps} "
            f"grad_clip={self.grad_clip} world_size={self.world_size}"
        )

    def sync_initial_weights(self) -> None:
        """Broadcast the randomly initialised discriminators from rank 0."""
        _broadcast_params(self.mpd)
        _broadcast_params(self.mrd)

    # -- generator side (inside the main graph) -------------------------------

    def gan_ramp(self, global_step: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(global_step) / float(self.warmup_steps)))

    def generator_loss(
        self,
        *,
        wav_real: torch.Tensor,
        wav_from_real_mel: torch.Tensor | None = None,
        wav_from_pred_mel: torch.Tensor | None = None,
        real_mel_chunk: torch.Tensor | None = None,
        global_step: int = 0,
    ) -> torch.Tensor:
        """Vocoder-side losses; also queues the D pass for :meth:`discriminator_backward`."""
        components: dict[str, torch.Tensor] = {}
        zero = wav_real.new_zeros(())
        total = zero
        ramp = self.gan_ramp(global_step)

        branches: list[tuple[str, torch.Tensor]] = []
        if wav_from_real_mel is not None and self.gan_on in ("real", "both"):
            branches.append(("real", wav_from_real_mel))
        if wav_from_pred_mel is not None and self.gan_on in ("pred", "both"):
            branches.append(("pred", wav_from_pred_mel))

        adv_total, fm_total = zero, zero
        for _name, wav_fake in branches:
            length = min(wav_fake.size(-1), wav_real.size(-1))
            fake = wav_fake[..., :length].unsqueeze(1)
            real = wav_real[..., :length].unsqueeze(1).detach()
            for discriminator in (self.mpd, self.mrd):
                _, fake_scores, real_maps, fake_maps = discriminator(real, fake)
                adv_total = adv_total + generator_adversarial_loss(fake_scores)
                fm_total = fm_total + feature_matching_loss(real_maps, fake_maps)
            # The D update needs the same pair, detached from the vocoder graph.
            self._pending.append((real.detach(), fake.detach()))
        if branches:
            total = total + ramp * (self.adv_weight * adv_total + self.fm_weight * fm_total)
        components["adv"] = adv_total.detach()
        components["fm"] = fm_total.detach()

        # Mel reconstruction.  For the predicted branch the target is still the
        # real mel, so this term also pulls the flow model; for the real branch it
        # is analysis-by-synthesis and pulls the vocoder only.
        mel_term, real_mel_term = zero, zero
        if real_mel_chunk is not None:
            if wav_from_pred_mel is not None and self.mel_weight > 0:
                mel_term = self.mel_loss(wav_from_pred_mel, real_mel_chunk)
                total = total + self.mel_weight * mel_term
            if wav_from_real_mel is not None and self.real_mel_weight > 0:
                real_mel_term = self.mel_loss(wav_from_real_mel, real_mel_chunk)
                total = total + self.real_mel_weight * real_mel_term
        components["mel"] = mel_term.detach()
        components["real_mel"] = real_mel_term.detach()

        wave_term = zero
        if wav_from_real_mel is not None and self.real_wave_l1_weight > 0:
            length = min(wav_from_real_mel.size(-1), wav_real.size(-1))
            wave_term = (
                wav_from_real_mel[..., :length] - wav_real[..., :length].detach()
            ).abs().mean()
            total = total + self.real_wave_l1_weight * wave_term
        components["real_wave_l1"] = wave_term.detach()

        components["d_loss"] = self.last_components.get("d_loss", zero)
        self.last_components = components
        return total

    # -- discriminator side (its own graph) -----------------------------------

    def discriminator_backward(self, scale: float = 1.0) -> torch.Tensor | None:
        """Backward the D loss on the pairs queued by :meth:`generator_loss`.

        ``scale`` should be ``1 / grad_accumulation``: accelerate scales the main
        loss for us, this graph is ours to scale.
        """
        if not self._pending:
            return None
        total = None
        for real, fake in self._pending:
            for discriminator in (self.mpd, self.mrd):
                real_scores, fake_scores, _, _ = discriminator(real, fake)
                term = discriminator_loss(real_scores, fake_scores)
                total = term if total is None else total + term
        self._pending.clear()
        if total is None:
            return None
        (total * float(scale)).backward()
        self.last_components["d_loss"] = total.detach()
        return total.detach()

    def discard_pending(self) -> None:
        """Drop queued pairs without a D update (skipped step)."""
        self._pending.clear()

    # -- optimizer plumbing ---------------------------------------------------

    def _trainable(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def clip_and_step(self) -> tuple[float, float]:
        """Average grads across ranks, clip, and step both optimizers."""
        vocoder_params = list(self.vocoder.parameters())
        d_params = list(self.mpd.parameters()) + list(self.mrd.parameters())
        _all_reduce_grads(vocoder_params, self.world_size)
        _all_reduce_grads(d_params, self.world_size)
        g_norm = float(
            torch.nn.utils.clip_grad_norm_(vocoder_params, self.grad_clip)
        )
        d_norm = float(torch.nn.utils.clip_grad_norm_(d_params, self.grad_clip))
        self.optimizer_g.step()
        self.optimizer_d.step()
        self.scheduler_g.step()
        self.scheduler_d.step()
        self.zero_grad_all()
        return g_norm, d_norm

    def zero_grad_all(self) -> None:
        self.optimizer_g.zero_grad(set_to_none=True)
        self.optimizer_d.zero_grad(set_to_none=True)
        self._pending.clear()

    # -- checkpointing --------------------------------------------------------

    def training_state(self) -> dict:
        return {
            "vocoder": self.vocoder.state_dict(),
            "mpd": self.mpd.state_dict(),
            "mrd": self.mrd.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
        }

    def load_training_state(self, state: dict, *, strict: bool = True) -> None:
        self.vocoder.load_state_dict(state["vocoder"], strict=strict)
        self.mpd.load_state_dict(state["mpd"], strict=strict)
        self.mrd.load_state_dict(state["mrd"], strict=strict)
        self.optimizer_g.load_state_dict(state["optimizer_g"])
        self.optimizer_d.load_state_dict(state["optimizer_d"])
        self.scheduler_g.load_state_dict(state["scheduler_g"])
        self.scheduler_d.load_state_dict(state["scheduler_d"])
