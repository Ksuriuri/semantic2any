"""Gated low-band harmonic-buzz loss after frozen AudioVAE decode."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class Conv1DSTFT(nn.Module):
    """Magnitude STFT via a fixed DFT conv1d kernel.

    E0904 (CUDA 13.0 / driver 580.x) breaks ``torch.stft`` backward with
    ``CUDA driver error: invalid argument``. This path matches the WaveFM
    aux loss that already trains on this machine.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int | None = None) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        n_freq = self.n_fft // 2 + 1
        freqs = torch.arange(n_freq, dtype=torch.float32)
        t = torch.arange(self.n_fft, dtype=torch.float32)
        phase = 2.0 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0) / self.n_fft
        window = torch.hann_window(self.win_length)
        if self.win_length < self.n_fft:
            pad = (self.n_fft - self.win_length) // 2
            window = F.pad(window, (pad, self.n_fft - self.win_length - pad))
        cos_basis = torch.cos(phase) * window.unsqueeze(0)
        sin_basis = torch.sin(phase) * window.unsqueeze(0)
        self.register_buffer("kernel", torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1))
        self.n_freq = n_freq

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        pad = self.n_fft // 2
        framed = F.pad(waveform, (pad, pad), mode="reflect")
        spectrum = F.conv1d(framed.unsqueeze(1), self.kernel, stride=self.hop_length)
        real = spectrum[:, : self.n_freq]
        imag = spectrum[:, self.n_freq :]
        return torch.sqrt(real.square() + imag.square() + 1.0e-8)


class S2VAEGatedBuzzLoss(nn.Module):
    """Penalize over-sharp 0.2-1.5 kHz harmonic teeth on breathy GT frames.

    GT / predicted latents are decoded to waveform. The gate and F0 come from
    the GT wav only (no gradient). The hinge compares peakiness so a prediction
    that is *more* vowel-like than GT is penalized; matching or breathier is not.
    """

    component_keys = (
        "total",
        "excess",
        "gated",
        "samples",
        "frames",
        "gt_peakiness",
        "pred_peakiness",
    )

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        band_low_hz: float = 200.0,
        band_high_hz: float = 1500.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        chunk_frames: int = 32,
        min_chunk_frames: int = 16,
        max_samples: int = 16,
        max_scan_windows: int = 8,
        t_min: float = 0.0,
        t_power: float = 0.0,
        gate_peakiness: float = 0.70,
        margin: float = 0.03,
        f0_min_hz: float = 80.0,
        f0_max_hz: float = 400.0,
        n_harmonics: int = 6,
        relative_bandwidth: float = 0.12,
        energy_floor: float = 1.0e-4,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= band_low_hz < band_high_hz <= sample_rate / 2:
            raise ValueError("buzz band must lie inside [0, Nyquist]")
        if n_fft <= 0 or hop_length <= 0 or chunk_frames <= 0:
            raise ValueError("n_fft, hop_length, and chunk_frames must be positive")
        if min_chunk_frames <= 0 or min_chunk_frames > chunk_frames:
            raise ValueError("min_chunk_frames must be in [1, chunk_frames]")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if max_scan_windows <= 0:
            raise ValueError("max_scan_windows must be positive")
        if not 0.0 <= t_min < 1.0:
            raise ValueError("t_min must be in [0, 1)")
        if t_power < 0.0:
            raise ValueError("t_power must be non-negative")
        if not 0.0 < gate_peakiness < 1.0:
            raise ValueError("gate_peakiness must be in (0, 1)")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        self.sample_rate = int(sample_rate)
        self.band_low_hz = float(band_low_hz)
        self.band_high_hz = float(band_high_hz)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.chunk_frames = int(chunk_frames)
        self.min_chunk_frames = int(min_chunk_frames)
        self.max_samples = int(max_samples)
        self.max_scan_windows = int(max_scan_windows)
        self.t_min = float(t_min)
        self.t_power = float(t_power)
        self.gate_peakiness = float(gate_peakiness)
        self.margin = float(margin)
        self.f0_min_hz = float(f0_min_hz)
        self.f0_max_hz = float(f0_max_hz)
        self.n_harmonics = int(n_harmonics)
        self.relative_bandwidth = float(relative_bandwidth)
        self.energy_floor = float(energy_floor)
        self.use_checkpoint = bool(use_checkpoint)
        self.stft = Conv1DSTFT(self.n_fft, self.hop_length, self.n_fft)
        freqs = torch.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1)
        self.register_buffer("fft_freqs", freqs, persistent=False)
        self.last_components: dict[str, torch.Tensor] = {}
        self.last_flow: torch.Tensor | None = None
        self.last_starts: torch.Tensor | None = None

    def describe(self) -> str:
        return (
            "[S2VAE-buzz] gated low-band peakiness hinge: "
            f"{self.band_low_hz:g}-{self.band_high_hz:g} Hz, "
            f"gate<{self.gate_peakiness:g}, margin={self.margin:g}, "
            f"chunk={self.chunk_frames}, flattest of <={self.max_scan_windows} windows, "
            f"samples<={self.max_samples}, t>={self.t_min:g} t^{self.t_power:g}, "
            f"sequential decode, checkpoint={self.use_checkpoint}, stft=conv1d"
        )

    def _zero(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.sum() * 0.0
        self.last_components = {
            "total": zero.detach(),
            "excess": zero.detach(),
            "gated": zero.detach(),
            "samples": zero.detach(),
            "frames": zero.detach(),
            "gt_peakiness": zero.detach(),
            "pred_peakiness": zero.detach(),
        }
        self.last_starts = None
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
        return self.stft(waveform)

    def _estimate_f0(self, mag: torch.Tensor) -> torch.Tensor:
        """Harmonic product spectrum on time-averaged GT magnitude. Stopgrad."""
        mean = mag.mean(dim=-1)
        freqs = self.fft_freqs.to(device=mean.device, dtype=mean.dtype)
        band = (freqs >= self.f0_min_hz) & (freqs <= self.f0_max_hz)
        idx = torch.nonzero(band, as_tuple=False).reshape(-1)
        hps = mean.index_select(1, idx).clamp_min(1.0e-8)
        for harmonic in (2, 3, 4):
            dest = (idx * harmonic).clamp(max=mean.size(1) - 1)
            hps = hps * mean.index_select(1, dest).clamp_min(1.0e-8)
        peak = torch.argmax(hps, dim=1)
        return freqs.index_select(0, idx).index_select(0, peak).detach()

    def _peakiness(self, mag: torch.Tensor, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        power = mag.square().mean(dim=-1)
        freqs = self.fft_freqs.to(device=power.device, dtype=power.dtype).unsqueeze(0)
        f0 = f0.reshape(-1, 1).to(device=power.device, dtype=power.dtype)
        band = (freqs >= self.band_low_hz) & (freqs <= self.band_high_hz)
        harm_mask = torch.zeros_like(power)
        for harmonic in range(1, self.n_harmonics + 1):
            center = f0 * float(harmonic)
            bandwidth = torch.clamp(self.relative_bandwidth * center, min=20.0)
            harm_mask = harm_mask + torch.exp(-0.5 * ((freqs - center) / bandwidth).square())
        harm_mask = harm_mask.clamp(max=1.0) * band.to(power.dtype)
        between = band.to(power.dtype) * (1.0 - harm_mask)
        harm_e = (power * harm_mask).sum(dim=1)
        between_e = (power * between).sum(dim=1)
        energy = (power * band.to(power.dtype)).sum(dim=1)
        peakiness = harm_e / (harm_e + between_e).clamp_min(1.0e-8)
        return peakiness, energy

    def _scan_offsets(self, available: int, chunk: int) -> list[int]:
        last = available - chunk
        if last <= 0:
            return [0]
        if last <= (self.max_scan_windows - 1) * chunk:
            offsets = list(range(0, last + 1, chunk))
            if offsets[-1] != last:
                offsets.append(last)
            return offsets
        return [
            int(round(i * last / (self.max_scan_windows - 1)))
            for i in range(self.max_scan_windows)
        ]

    def _decode_nograd_batched(
        self,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        latent: torch.Tensor,
        batch_size: int = 8,
    ) -> torch.Tensor:
        pieces = []
        for start in range(0, latent.size(0), batch_size):
            pieces.append(
                self._decode(decode_fn, latent[start : start + batch_size], with_grad=False)
            )
        return torch.cat(pieces, dim=0)

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
                "predicted/target must have matching [B,C,T] shapes, got "
                f"{tuple(predicted.shape)} and {tuple(target.shape)}"
            )
        times = times.detach().reshape(-1).to(predicted.device, torch.float32)
        target_lengths = (lengths - prompt_lengths).to(torch.long)
        eligible = (times >= self.t_min) & (target_lengths >= self.min_chunk_frames)
        indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            return self._zero(predicted)
        if indices.numel() > self.max_samples:
            perm = torch.randperm(indices.numel(), device=indices.device)[: self.max_samples]
            indices = indices[perm]

        groups: dict[int, list[tuple[int, int, int]]] = {}
        for local_index, index in enumerate(indices.tolist()):
            first = int(prompt_lengths[index].item())
            available = int(target_lengths[index].item())
            chunk = min(self.chunk_frames, available)
            for offset in self._scan_offsets(available, chunk):
                groups.setdefault(chunk, []).append((local_index, first + offset, index))

        n_local = int(indices.numel())
        best_peak = torch.full((n_local,), 1.0e9, device=predicted.device, dtype=torch.float32)
        best_energy = torch.zeros(n_local, device=predicted.device, dtype=torch.float32)
        best_f0 = torch.zeros(n_local, device=predicted.device, dtype=torch.float32)
        best_start = torch.zeros(n_local, device=predicted.device, dtype=torch.long)
        best_chunk = torch.zeros(n_local, device=predicted.device, dtype=torch.long)

        for chunk, items in groups.items():
            crops = torch.stack(
                [target[index, :, start : start + chunk] for _local, start, index in items]
            ).contiguous()
            wav = self._decode_nograd_batched(decode_fn, crops)
            mag = self._magnitude(wav)
            f0 = self._estimate_f0(mag)
            peakiness, energy = self._peakiness(mag, f0)
            for row, (local_index, start, _index) in enumerate(items):
                if float(peakiness[row]) < float(best_peak[local_index]):
                    best_peak[local_index] = peakiness[row].detach()
                    best_energy[local_index] = energy[row].detach()
                    best_f0[local_index] = f0[row]
                    best_start[local_index] = start
                    best_chunk[local_index] = chunk

        gt_peakiness = best_peak.detach()
        self.last_starts = best_start.detach()
        gate = (best_energy > self.energy_floor) & (gt_peakiness < self.gate_peakiness)
        log_frames = float(best_chunk.to(torch.float32).mean().clamp_min(1.0))
        if not bool(gate.any()):
            total = self._zero(predicted)
            self.last_components["gt_peakiness"] = gt_peakiness.mean().detach()
            self.last_components["samples"] = predicted.new_tensor(float(n_local))
            self.last_components["frames"] = predicted.new_tensor(log_frames)
            return total

        sample_times = times[indices].clamp_min(0.0)
        weights = sample_times.pow(self.t_power) * gate.to(sample_times.dtype)
        weights = weights / weights.sum().clamp_min(1.0e-8)

        total = predicted.sum() * 0.0
        pred_peaks = []
        excesses = []
        for local_index in range(n_local):
            if not bool(gate[local_index]):
                continue
            index = int(indices[local_index].item())
            start = int(best_start[local_index].item())
            chunk = int(best_chunk[local_index].item())
            predicted_wav = self._decode(
                decode_fn,
                predicted[index : index + 1, :, start : start + chunk].contiguous(),
                with_grad=True,
            )
            predicted_mag = self._magnitude(predicted_wav)
            pred_peakiness, _ = self._peakiness(
                predicted_mag, best_f0[local_index : local_index + 1]
            )
            excess = torch.relu(pred_peakiness - gt_peakiness[local_index] - self.margin)
            total = total + weights[local_index] * excess.squeeze()
            pred_peaks.append(pred_peakiness.detach())
            excesses.append(excess.detach())

        pred_peakiness = torch.cat(pred_peaks).mean()
        excess = torch.cat(excesses).mean()
        self.last_components = {
            "total": total.detach(),
            "excess": excess,
            "gated": gate.to(total.dtype).mean().detach(),
            "samples": total.new_tensor(float(n_local)),
            "frames": total.new_tensor(log_frames),
            "gt_peakiness": gt_peakiness.mean().detach(),
            "pred_peakiness": pred_peakiness,
        }
        return total
