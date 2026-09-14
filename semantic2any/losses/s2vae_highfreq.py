"""High-frequency supervision through the frozen dots.tts AudioVAE decoder."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class S2VAEHighFrequencyLoss(nn.Module):
    """Compare decoded prediction/target spectra above a configurable cutoff.

    The flow model predicts normalized AudioVAE latents.  AudioVAE parameters stay
    frozen, but its decoder is intentionally evaluated with autograd enabled so
    the spectral error reaches the latent prediction and then the DiT.
    """

    component_keys = ("total", "log_mag", "linear_mag", "samples", "frames")

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        cutoff_hz: float = 12_000.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        chunk_frames: int = 32,
        min_chunk_frames: int = 16,
        max_samples: int = 4,
        t_min: float = 0.5,
        t_power: float = 2.0,
        linear_weight: float = 0.1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= cutoff_hz < sample_rate / 2:
            raise ValueError("cutoff_hz must be in [0, Nyquist)")
        if n_fft <= 0 or hop_length <= 0 or chunk_frames <= 0:
            raise ValueError("n_fft, hop_length, and chunk_frames must be positive")
        if min_chunk_frames <= 0 or min_chunk_frames > chunk_frames:
            raise ValueError("min_chunk_frames must be in [1, chunk_frames]")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if not 0.0 <= t_min < 1.0:
            raise ValueError("t_min must be in [0, 1)")
        if t_power < 0.0:
            raise ValueError("t_power must be non-negative")
        self.sample_rate = int(sample_rate)
        self.cutoff_hz = float(cutoff_hz)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.chunk_frames = int(chunk_frames)
        self.min_chunk_frames = int(min_chunk_frames)
        self.max_samples = int(max_samples)
        self.t_min = float(t_min)
        self.t_power = float(t_power)
        self.linear_weight = float(linear_weight)
        self.use_checkpoint = bool(use_checkpoint)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.last_components: dict[str, torch.Tensor] = {}

    def describe(self) -> str:
        return (
            "[S2VAE-HF] frozen decoder spectral loss: "
            f">={self.cutoff_hz:g} Hz, n_fft={self.n_fft}, "
            f"chunk<={self.chunk_frames} latent frames, samples<={self.max_samples}, "
            f"t>{self.t_min:g}, t^{self.t_power:g}, checkpoint={self.use_checkpoint}"
        )

    def _zero(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.sum() * 0.0
        self.last_components = {
            "total": zero.detach(),
            "log_mag": zero.detach(),
            "linear_mag": zero.detach(),
            "samples": zero.detach(),
            "frames": zero.detach(),
        }
        return zero

    def _decode(
        self,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        latent: torch.Tensor,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        if not with_grad:
            with torch.no_grad():
                return decode_fn(latent)
        if self.use_checkpoint:
            return checkpoint(decode_fn, latent, use_reentrant=False)
        return decode_fn(latent)

    def _magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.float().reshape(waveform.size(0), -1)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=True,
            return_complex=True,
        )
        first_bin = int(self.cutoff_hz * self.n_fft / self.sample_rate + 0.999999)
        return spectrum[:, first_bin:, :].abs()

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        prompt_lengths: torch.Tensor,
        times: torch.Tensor,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if predicted.shape != target.shape or predicted.ndim != 3:
            raise ValueError(
                f"predicted/target must have matching [B,C,T] shapes, got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}"
            )
        times = times.detach().reshape(-1).to(predicted.device, torch.float32)
        target_lengths = (lengths - prompt_lengths).to(torch.long)
        eligible = (times > self.t_min) & (target_lengths >= self.min_chunk_frames)
        indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            return self._zero(predicted)
        if indices.numel() > self.max_samples:
            # The most trustworthy one-step x1 estimates are those closest to t=1.
            top = torch.topk(times[indices], self.max_samples, sorted=False).indices
            indices = indices[top]

        chunk = min(
            self.chunk_frames,
            min(int(target_lengths[index].item()) for index in indices),
        )
        predicted_chunks = []
        target_chunks = []
        for index in indices.tolist():
            first = int(prompt_lengths[index].item())
            available = int(target_lengths[index].item())
            offset = 0
            if available > chunk:
                offset = int(
                    torch.randint(
                        available - chunk + 1,
                        (),
                        device=predicted.device,
                    ).item()
                )
            start = first + offset
            predicted_chunks.append(predicted[index, :, start : start + chunk])
            target_chunks.append(target[index, :, start : start + chunk])
        predicted_latent = torch.stack(predicted_chunks).contiguous()
        target_latent = torch.stack(target_chunks).contiguous()

        predicted_wav = self._decode(
            decode_fn, predicted_latent, with_grad=True
        )
        target_wav = self._decode(decode_fn, target_latent, with_grad=False)
        predicted_mag = self._magnitude(predicted_wav)
        target_mag = self._magnitude(target_wav)

        eps = 1.0e-5
        per_sample_log = (
            predicted_mag.clamp_min(eps).log()
            - target_mag.clamp_min(eps).log()
        ).abs().mean(dim=(1, 2))
        target_scale = target_mag.mean(dim=(1, 2)).clamp_min(1.0e-4)
        per_sample_linear = (
            (predicted_mag - target_mag).abs().mean(dim=(1, 2)) / target_scale
        )
        weights = times[indices].clamp_min(0.0).pow(self.t_power)
        weights = weights / weights.sum().clamp_min(1.0e-8)
        log_mag = (per_sample_log * weights).sum()
        linear_mag = (per_sample_linear * weights).sum()
        total = log_mag + self.linear_weight * linear_mag
        self.last_components = {
            "total": total.detach(),
            "log_mag": log_mag.detach(),
            "linear_mag": linear_mag.detach(),
            "samples": total.detach().new_tensor(float(indices.numel())),
            "frames": total.detach().new_tensor(float(chunk)),
        }
        return total


class S2VAEBreathAwareLoss(S2VAEHighFrequencyLoss):
    """Decoded-audio loss for aperiodic/breathy high-frequency texture.

    Unlike :class:`S2VAEHighFrequencyLoss`, this objective does not only match
    aggregate magnitude above one cutoff.  It combines multi-resolution
    4--18 kHz magnitude matching with target-derived breath-frame weighting,
    periodicity and spectral-flatness matching, plus a one-sided penalty for
    energy synthesized beyond the target's effective bandwidth.

    Target statistics and frame weights are detached.  Exact waveform phase is
    deliberately not supervised because breath noise has stochastic phase.
    """

    component_keys = (
        "total",
        "mr_log_mag",
        "mr_linear_mag",
        "periodicity",
        "flatness",
        "excess_band",
        "breath_weight",
        "samples",
        "frames",
    )

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        band_low_hz: float = 4_000.0,
        band_high_hz: float = 18_000.0,
        n_ffts: tuple[int, ...] = (512, 1024, 2048),
        hop_ratio: int = 4,
        chunk_frames: int = 64,
        min_chunk_frames: int = 32,
        max_samples: int = 4,
        t_min: float = 0.5,
        t_power: float = 2.0,
        breath_boost: float = 4.0,
        linear_weight: float = 0.1,
        periodicity_weight: float = 0.25,
        flatness_weight: float = 0.25,
        excess_band_weight: float = 0.5,
        excess_start_hz: float = 12_000.0,
        bandwidth_floor_db: float = -35.0,
        excess_margin_db: float = 3.0,
        use_checkpoint: bool = True,
    ) -> None:
        if not n_ffts or any(value <= 0 for value in n_ffts):
            raise ValueError("n_ffts must contain positive values")
        if hop_ratio <= 0:
            raise ValueError("hop_ratio must be positive")
        if not 0.0 <= band_low_hz < band_high_hz <= sample_rate / 2:
            raise ValueError("breath band must lie inside [0, Nyquist]")
        if not band_low_hz < excess_start_hz < sample_rate / 2:
            raise ValueError("excess_start_hz must lie above band_low_hz and below Nyquist")
        super().__init__(
            sample_rate=sample_rate,
            cutoff_hz=band_low_hz,
            n_fft=max(n_ffts),
            hop_length=max(n_ffts) // hop_ratio,
            chunk_frames=chunk_frames,
            min_chunk_frames=min_chunk_frames,
            max_samples=max_samples,
            t_min=t_min,
            t_power=t_power,
            linear_weight=linear_weight,
            use_checkpoint=use_checkpoint,
        )
        self.band_low_hz = float(band_low_hz)
        self.band_high_hz = float(band_high_hz)
        self.n_ffts = tuple(int(value) for value in n_ffts)
        self.hop_ratio = int(hop_ratio)
        self.breath_boost = float(breath_boost)
        self.periodicity_weight = float(periodicity_weight)
        self.flatness_weight = float(flatness_weight)
        self.excess_band_weight = float(excess_band_weight)
        self.excess_start_hz = float(excess_start_hz)
        self.bandwidth_floor_db = float(bandwidth_floor_db)
        self.excess_margin_db = float(excess_margin_db)
        # Delegate each STFT to the already proven single-resolution helper.
        # On the E0904 RTX 4090/PyTorch stack, an equivalent inline STFT graph
        # fails during cuFFT backward, while this registered-window path is
        # stable.  Start at 200 Hz so breath weighting can still compare the
        # high band with the voiced-band reference.
        self.mr_spectral_helpers = nn.ModuleDict(
            {
                str(n_fft): S2VAEHighFrequencyLoss(
                    sample_rate=self.sample_rate,
                    cutoff_hz=200.0,
                    n_fft=n_fft,
                    hop_length=max(1, n_fft // self.hop_ratio),
                    chunk_frames=self.chunk_frames,
                    min_chunk_frames=self.min_chunk_frames,
                    max_samples=self.max_samples,
                    t_min=self.t_min,
                    t_power=self.t_power,
                    linear_weight=self.linear_weight,
                    use_checkpoint=False,
                )
                for n_fft in self.n_ffts
            }
        )

    def describe(self) -> str:
        resolutions = "/".join(str(value) for value in self.n_ffts)
        return (
            "[S2VAE-Breath] frozen decoder breath-aware loss: "
            f"{self.band_low_hz:g}-{self.band_high_hz:g} Hz, "
            f"n_fft={resolutions}, breath_boost={self.breath_boost:g}, "
            f"excess>{self.excess_start_hz:g} Hz, chunk<={self.chunk_frames} "
            f"latent frames, samples<={self.max_samples}, t>{self.t_min:g}"
        )

    def _zero(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.sum() * 0.0
        self.last_components = {
            key: zero.detach() for key in self.component_keys
        }
        return zero

    @staticmethod
    def _safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1.0e-8)

    def _stft_magnitude(
        self,
        waveform: torch.Tensor,
        n_fft: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        helper = self.mr_spectral_helpers[str(n_fft)]
        magnitude = helper._magnitude(waveform)
        all_frequencies = torch.fft.rfftfreq(
            n_fft,
            d=1.0 / self.sample_rate,
            device=waveform.device,
        )
        first_bin = int(200.0 * n_fft / self.sample_rate + 0.999999)
        return magnitude, all_frequencies[first_bin:]

    def _breath_weights(
        self,
        target_mag: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1.0e-6
        n_fft = 2 * (frequencies.numel() - 1)
        full_first = max(0, int(200.0 * n_fft / self.sample_rate + 0.999999))
        full_last = min(
            frequencies.numel(),
            int(self.band_high_hz * n_fft / self.sample_rate + 0.999999),
        )
        band_first = max(
            0, int(self.band_low_hz * n_fft / self.sample_rate + 0.999999)
        )
        band_last = min(
            frequencies.numel(), int(12_000.0 * n_fft / self.sample_rate + 0.999999)
        )
        band_mag = target_mag[:, band_first:band_last, :].clamp_min(eps)
        full_energy = (
            target_mag[:, full_first:full_last, :]
            .square()
            .sum(dim=1)
            .clamp_min(eps)
        )
        band_energy = band_mag.square().sum(dim=1)
        band_ratio_db = 10.0 * torch.log10((band_energy / full_energy).clamp_min(eps))
        flatness = torch.exp(band_mag.log().mean(dim=1)) / band_mag.mean(dim=1)
        energy_score = torch.sigmoid((band_ratio_db + 18.0) / 3.0)
        flatness_score = torch.sigmoid((flatness.clamp_min(eps).log() + 3.5) / 0.7)
        breath_score = (energy_score * flatness_score).detach()
        return 1.0 + self.breath_boost * breath_score, breath_score

    def _periodicity(self, waveform: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
        waveform = waveform.float().reshape(waveform.size(0), -1)
        if waveform.size(-1) < n_fft:
            waveform = torch.nn.functional.pad(waveform, (0, n_fft - waveform.size(-1)))
        frames = waveform.unfold(-1, n_fft, hop)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        frames = frames * torch.hann_window(
            n_fft, device=waveform.device, dtype=waveform.dtype
        )
        fft_size = 2 * n_fft
        spectrum = torch.fft.rfft(frames, n=fft_size, dim=-1)
        autocorrelation = torch.fft.irfft(
            spectrum * spectrum.conj(), n=fft_size, dim=-1
        )[..., :n_fft]
        lag_low = max(1, int(self.sample_rate / 500.0))
        lag_high = min(n_fft - 1, int(self.sample_rate / 80.0))
        if lag_high <= lag_low:
            return autocorrelation[..., 0] * 0.0
        peak = autocorrelation[..., lag_low : lag_high + 1].amax(dim=-1)
        return peak / autocorrelation[..., 0].clamp_min(1.0e-6)

    def _excess_band_loss(
        self,
        predicted_mag: torch.Tensor,
        target_mag: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1.0e-6
        sample_mean = target_mag.detach().mean(dim=-1).clamp_min(eps)
        reference = (frequencies >= self.band_low_hz) & (frequencies < 12_000.0)
        eligible = frequencies >= self.excess_start_hz
        reference_db = 20.0 * torch.log10(
            sample_mean[:, reference].amax(dim=1).clamp_min(eps)
        )
        mean_db = 20.0 * torch.log10(sample_mean)
        active = eligible.unsqueeze(0) & (
            mean_db >= reference_db.unsqueeze(1) + self.bandwidth_floor_db
        )
        indices = torch.arange(frequencies.numel(), device=frequencies.device).unsqueeze(0)
        last_active = torch.where(active, indices, torch.zeros_like(indices)).amax(dim=1)
        start_bin = int(self.excess_start_hz * (frequencies.numel() - 1) / (self.sample_rate / 2))
        last_active = last_active.clamp_min(start_bin)
        tail = indices.unsqueeze(-1) > last_active[:, None, None]
        margin = self.excess_margin_db * (torch.log(torch.tensor(10.0, device=target_mag.device)) / 20.0)
        overshoot = (
            predicted_mag.clamp_min(eps).log()
            - target_mag.detach().clamp_min(eps).log()
            - margin
        ).relu()
        tail_count = tail.sum().clamp_min(1)
        return (overshoot * tail).sum() / (tail_count * overshoot.size(-1))

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
        prompt_lengths: torch.Tensor,
        times: torch.Tensor,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if predicted.shape != target.shape or predicted.ndim != 3:
            raise ValueError(
                f"predicted/target must have matching [B,C,T] shapes, got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}"
            )
        times = times.detach().reshape(-1).to(predicted.device, torch.float32)
        target_lengths = (lengths - prompt_lengths).to(torch.long)
        indices = torch.nonzero(
            (times > self.t_min) & (target_lengths >= self.min_chunk_frames),
            as_tuple=False,
        ).reshape(-1)
        if indices.numel() == 0:
            return self._zero(predicted)
        if indices.numel() > self.max_samples:
            indices = indices[
                torch.topk(times[indices], self.max_samples, sorted=False).indices
            ]
        chunk = min(
            self.chunk_frames,
            min(int(target_lengths[index].item()) for index in indices),
        )
        predicted_chunks = []
        target_chunks = []
        for index in indices.tolist():
            first = int(prompt_lengths[index].item())
            available = int(target_lengths[index].item())
            offset = 0
            if available > chunk:
                offset = int(
                    torch.randint(
                        available - chunk + 1,
                        (),
                        device=predicted.device,
                    ).item()
                )
            start = first + offset
            predicted_chunks.append(predicted[index, :, start : start + chunk])
            target_chunks.append(target[index, :, start : start + chunk])
        predicted_latent = torch.stack(predicted_chunks).contiguous()
        target_latent = torch.stack(target_chunks).contiguous()
        predicted_wav = self._decode(decode_fn, predicted_latent, with_grad=True)
        target_wav = self._decode(decode_fn, target_latent, with_grad=False)

        eps = 1.0e-6
        log_losses = []
        linear_losses = []
        reference = None
        for n_fft in self.n_ffts:
            predicted_mag, frequencies = self._stft_magnitude(predicted_wav, n_fft)
            target_mag, _ = self._stft_magnitude(target_wav, n_fft)
            band_first = max(
                0,
                int(self.band_low_hz * n_fft / self.sample_rate + 0.999999),
            )
            band_last = min(
                frequencies.numel(),
                int(self.band_high_hz * n_fft / self.sample_rate + 0.999999),
            )
            predicted_band = predicted_mag[:, band_first:band_last, :].clamp_min(eps)
            target_band = target_mag[:, band_first:band_last, :].clamp_min(eps)
            frame_weights, _ = self._breath_weights(target_mag, frequencies)
            per_frame_log = (predicted_band.log() - target_band.log()).abs().mean(dim=1)
            log_losses.append(self._safe_weighted_mean(per_frame_log, frame_weights))
            if self.linear_weight > 0.0:
                target_scale = target_band.mean(dim=1).clamp_min(1.0e-4)
                per_frame_linear = (
                    (predicted_band - target_band).abs().mean(dim=1) / target_scale
                )
                linear_losses.append(
                    self._safe_weighted_mean(per_frame_linear, frame_weights)
                )
            if n_fft == max(self.n_ffts):
                reference = (predicted_mag, target_mag, frequencies, frame_weights)
        mr_log_mag = torch.stack(log_losses).mean()
        mr_linear_mag = (
            torch.stack(linear_losses).mean()
            if linear_losses
            else mr_log_mag.detach().new_zeros(())
        )
        assert reference is not None
        predicted_mag, target_mag, frequencies, frame_weights = reference
        ref_n_fft = max(self.n_ffts)
        band_first = max(
            0,
            int(self.band_low_hz * ref_n_fft / self.sample_rate + 0.999999),
        )
        band_last = min(
            frequencies.numel(),
            int(self.band_high_hz * ref_n_fft / self.sample_rate + 0.999999),
        )
        predicted_band = predicted_mag[:, band_first:band_last, :].clamp_min(eps)
        target_band = target_mag[:, band_first:band_last, :].clamp_min(eps)
        flatness = mr_log_mag.detach().new_zeros(())
        if self.flatness_weight > 0.0:
            predicted_flatness = (
                torch.exp(predicted_band.log().mean(dim=1))
                / predicted_band.mean(dim=1)
            )
            target_flatness = (
                torch.exp(target_band.log().mean(dim=1))
                / target_band.mean(dim=1)
            )
            flatness = self._safe_weighted_mean(
                (
                    predicted_flatness.clamp_min(eps).log()
                    - target_flatness.clamp_min(eps).log()
                ).abs(),
                frame_weights,
            )
        periodicity = mr_log_mag.detach().new_zeros(())
        if self.periodicity_weight > 0.0:
            ref_hop = max(1, ref_n_fft // self.hop_ratio)
            predicted_periodicity = self._periodicity(
                predicted_wav, ref_n_fft, ref_hop
            )
            target_periodicity = self._periodicity(
                target_wav, ref_n_fft, ref_hop
            ).detach()
            periodic_frames = min(
                predicted_periodicity.size(-1),
                target_periodicity.size(-1),
                frame_weights.size(-1),
            )
            periodicity = self._safe_weighted_mean(
                (
                    predicted_periodicity[..., :periodic_frames]
                    - target_periodicity[..., :periodic_frames]
                ).abs(),
                frame_weights[..., :periodic_frames],
            )
        excess_band = mr_log_mag.detach().new_zeros(())
        if self.excess_band_weight > 0.0:
            excess_band = self._excess_band_loss(
                predicted_mag, target_mag, frequencies
            )
        total = (
            mr_log_mag
            + self.linear_weight * mr_linear_mag
            + self.periodicity_weight * periodicity
            + self.flatness_weight * flatness
            + self.excess_band_weight * excess_band
        )
        self.last_components = {
            "total": total.detach(),
            "mr_log_mag": mr_log_mag.detach(),
            "mr_linear_mag": mr_linear_mag.detach(),
            "periodicity": periodicity.detach(),
            "flatness": flatness.detach(),
            "excess_band": excess_band.detach(),
            "breath_weight": frame_weights.detach().mean(),
            "samples": total.detach().new_tensor(float(indices.numel())),
            "frames": total.detach().new_tensor(float(chunk)),
        }
        return total


__all__ = ["S2VAEHighFrequencyLoss", "S2VAEBreathAwareLoss"]
