"""Auxiliary losses for s2mel flow-matching training."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


class MultiResolutionMelLoss(nn.Module):
    """Multi-resolution spectral loss applied directly in mel domain.

    Computes L1 + spectral convergence at multiple temporal resolutions
    (original + downsampled by avg pooling).
    """

    def __init__(self, resolutions=(1, 2, 4, 8), sc_weight=1.0, mag_weight=1.0):
        super().__init__()
        self.resolutions = resolutions
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

    def forward(
        self,
        x1_hat: torch.Tensor,
        x1: torch.Tensor,
        mel_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)
        T = x1.size(-1)
        positions = torch.arange(T, device=x1.device)
        base_mask = (
            (positions.unsqueeze(0) >= prompt_lens.unsqueeze(1))
            & (positions.unsqueeze(0) < mel_lens.unsqueeze(1))
        )

        for pool_size in self.resolutions:
            if pool_size > 1:
                pred = F.avg_pool1d(x1_hat, pool_size, pool_size)
                target = F.avg_pool1d(x1, pool_size, pool_size)
                mask = base_mask[:, ::pool_size][:, : pred.shape[-1]]
            else:
                pred, target = x1_hat, x1
                mask = base_mask

            mask_expanded = mask.unsqueeze(1)
            diff = (pred - target) * mask_expanded

            target_masked = target * mask_expanded
            sc = diff.norm(dim=(1, 2)) / target_masked.norm(dim=(1, 2)).clamp_min(1e-7)
            sc_loss = sc.mean()

            denom = mask.sum(dim=1).clamp_min(1).to(diff.dtype) * pred.size(1)
            mag_loss = diff.abs().sum(dim=(1, 2)) / denom
            mag_loss = mag_loss.mean()

            total_loss = total_loss + self.sc_weight * sc_loss + self.mag_weight * mag_loss

        return total_loss / len(self.resolutions)


class Conv1DSTFT(nn.Module):
    """STFT implemented via conv1d with fixed DFT basis.

    Avoids torch.stft which has broken backward on CUDA 13.0 (driver 580.x).
    The DFT basis is registered as a non-trainable buffer.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        n_freq = n_fft // 2 + 1

        # DFT basis
        freqs = torch.arange(n_freq, dtype=torch.float32)
        t = torch.arange(n_fft, dtype=torch.float32)
        phase = 2.0 * math.pi * freqs.unsqueeze(1) * t.unsqueeze(0) / n_fft
        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        # Window
        window = torch.hann_window(self.win_length)
        if self.win_length < n_fft:
            pad = (n_fft - self.win_length) // 2
            window = F.pad(window, (pad, n_fft - self.win_length - pad))
        cos_basis = cos_basis * window.unsqueeze(0)
        sin_basis = sin_basis * window.unsqueeze(0)

        # (2*n_freq, 1, n_fft) for conv1d
        kernel = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel)
        self.n_freq = n_freq

    def forward(
        self, wav: torch.Tensor, return_complex: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """STFT of ``wav``.

        Default returns the magnitude spectrogram (B, n_freq, T).  With
        ``return_complex=True`` returns ``(real, imag)`` so callers can derive
        phases or complex products.
        """
        pad_len = self.n_fft // 2
        x = F.pad(wav, (pad_len, pad_len), mode="reflect")
        out = F.conv1d(x.unsqueeze(1), self.kernel, stride=self.hop_length)
        real = out[:, : self.n_freq, :]
        imag = out[:, self.n_freq :, :]
        if return_complex:
            return real, imag
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)


class MultiResolutionSTFTLoss(nn.Module):
    """WaveFM-style refined multi-resolution STFT loss (NAACL 2025, App. C).

    For each STFT resolution the loss combines:
      * anti-wrapped phase-angle L1, masked where either squared magnitude is
        below ``mag_min`` (phases are meaningless there),
      * log-magnitude L1 over all bins,
      * MSE on the frequency/time gradients and the Laplacian of the
        magnitude spectrogram (edge/structure supervision).

    Resolution defaults follow the WaveFM appendix (fft/hop/win), which are
    hop = fft/8 and win = fft/2.  The conv1d-based STFT is used instead of
    ``torch.stft`` to stay compatible with the CUDA 13.0 backward bug.

    Every forward stores each term (per resolution and averaged over
    resolutions) in ``last_components`` so the trainer can log the individual
    loss values.  ``component_keys`` is fixed at construction time, so all
    ranks always report the same keys even on the early-return paths.
    """

    TERM_NAMES = ("mag_l1", "phase_l1", "grad_freq", "grad_time", "laplacian", "total")

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (1024, 2048, 512),
        hop_sizes: tuple[int, ...] = (128, 256, 64),
        win_lengths: tuple[int, ...] = (512, 1024, 256),
        mag_min: float = 1e-6,
        phase_weight: float = 1.0,
        mag_weight: float = 1.0,
        grad_weight_freq: float = 4.0,
        grad_weight_time: float = 4.0,
        grad_weight_lap: float = 2.0,
    ):
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft/hop/win resolution lists must have equal length")
        self.mag_min = mag_min
        self.phase_weight = phase_weight
        self.mag_weight = mag_weight
        self.grad_weight_freq = grad_weight_freq
        self.grad_weight_time = grad_weight_time
        self.grad_weight_lap = grad_weight_lap
        self.stft_modules = nn.ModuleList(
            Conv1DSTFT(n_fft, hop, win)
            for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        )
        self.resolution_names = tuple(f"res{n_fft}" for n_fft in fft_sizes)
        self.component_keys = tuple(
            list(self.TERM_NAMES)
            + [f"{res}/{term}" for res in self.resolution_names for term in self.TERM_NAMES]
        )
        self.last_components: dict[str, torch.Tensor] = {}
        # Structure kernels on (F, T) magnitude spectrograms (WaveFM App. C).
        freq_kernel = torch.tensor(
            [[-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 2, 3) / 4.0
        time_kernel = torch.tensor(
            [[-1.0, 1.0], [-2.0, 2.0], [-1.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 2) / 4.0
        lap_kernel = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 8.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("freq_kernel", freq_kernel)
        self.register_buffer("time_kernel", time_kernel)
        self.register_buffer("lap_kernel", lap_kernel)

    def zero_components(self, device, dtype) -> None:
        """Fill ``last_components`` with zeros (early-return paths)."""
        self.last_components = {
            key: torch.zeros((), device=device, dtype=dtype)
            for key in self.component_keys
        }

    @staticmethod
    def _filter2d(
        x: torch.Tensor,
        kernel: torch.Tensor,
        pad: tuple[int, int, int, int],
    ) -> torch.Tensor:
        x = F.pad(x.unsqueeze(1), pad, mode="constant")
        return F.conv2d(x, kernel).squeeze(1)

    def forward(
        self,
        wav_pred: torch.Tensor,
        wav_gt: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """WaveFM MR-STFT loss between predicted and GT waveforms (B, T).

        `sample_weights` (B,) turns every elementwise term into the per-sample
        weighted mean `sum_i w_i * mean_bins(e_i) / sum_i w_i`.  The phase term
        is excluded: its significant-bin masking flattens the batch axis, so it
        has no per-sample axis to weight.
        """
        if sample_weights is not None and self.phase_weight != 0:
            raise ValueError(
                "sample_weights cannot be combined with phase_weight != 0: the "
                "phase term's masked indexing has no per-sample axis"
            )
        if wav_pred.shape != wav_gt.shape:
            min_len = min(wav_pred.size(-1), wav_gt.size(-1))
            if min_len < 256:
                self.zero_components(wav_pred.device, wav_pred.dtype)
                return torch.zeros((), device=wav_pred.device, dtype=wav_pred.dtype)
            wav_pred = wav_pred[..., :min_len]
            wav_gt = wav_gt[..., :min_len]

        device, dtype = wav_pred.device, wav_pred.dtype
        _w = (
            None
            if sample_weights is None
            else sample_weights.to(device=device, dtype=dtype).reshape(-1)
        )
        _w_sum = None if _w is None else _w.sum().clamp_min(1e-8)

        def _reduce(err: torch.Tensor) -> torch.Tensor:
            """Plain mean, or the per-sample weighted mean over dim 0."""
            if _w is None:
                return err.mean()
            return (err.flatten(1).mean(dim=1) * _w).sum() / _w_sum

        components: dict[str, torch.Tensor] = {}
        term_sums = {
            name: torch.zeros((), device=device, dtype=dtype)
            for name in self.TERM_NAMES
        }
        total = torch.zeros((), device=device, dtype=dtype)
        for res_name, stft_mod in zip(self.resolution_names, self.stft_modules):
            real_pred, imag_pred = stft_mod(wav_pred, return_complex=True)
            real_gt, imag_gt = stft_mod(wav_gt, return_complex=True)
            sq_pred = real_pred ** 2 + imag_pred ** 2
            sq_gt = real_gt ** 2 + imag_gt ** 2
            mask = (sq_gt > self.mag_min) & (sq_pred > self.mag_min)
            mag_pred = torch.sqrt(sq_pred + self.mag_min)
            mag_gt = torch.sqrt(sq_gt + self.mag_min)

            # Log-magnitude L1 over all bins.
            mag_loss = _reduce((mag_gt.log() - mag_pred.log()).abs())

            # Anti-wrapped phase-angle L1 on significant bins.
            if mask.any():
                phase_pred = torch.atan2(imag_pred[mask], real_pred[mask])
                phase_gt = torch.atan2(imag_gt[mask], real_gt[mask])
                delta = phase_gt - phase_pred
                phase_loss = torch.atan2(
                    torch.sin(delta), torch.cos(delta)
                ).abs().mean()
            else:
                phase_loss = torch.zeros((), device=wav_pred.device, dtype=wav_pred.dtype)

            # Edge/structure terms on magnitude spectrograms.
            df_gt = self._filter2d(mag_gt, self.freq_kernel, (1, 1, 1, 0))
            df_pred = self._filter2d(mag_pred, self.freq_kernel, (1, 1, 1, 0))
            dt_gt = self._filter2d(mag_gt, self.time_kernel, (1, 0, 1, 1))
            dt_pred = self._filter2d(mag_pred, self.time_kernel, (1, 0, 1, 1))
            lap_gt = self._filter2d(mag_gt, self.lap_kernel, (1, 1, 1, 1))
            lap_pred = self._filter2d(mag_pred, self.lap_kernel, (1, 1, 1, 1))
            df_loss = _reduce((df_gt - df_pred).pow(2))
            dt_loss = _reduce((dt_gt - dt_pred).pow(2))
            lap_loss = _reduce((lap_gt - lap_pred).pow(2))

            res_total = (
                self.phase_weight * phase_loss
                + self.mag_weight * mag_loss
                + self.grad_weight_freq * df_loss
                + self.grad_weight_time * dt_loss
                + self.grad_weight_lap * lap_loss
            )
            total = total + res_total

            res_terms = {
                "mag_l1": mag_loss,
                "phase_l1": phase_loss,
                "grad_freq": df_loss,
                "grad_time": dt_loss,
                "laplacian": lap_loss,
                "total": res_total,
            }
            for name, value in res_terms.items():
                components[f"{res_name}/{name}"] = value.detach()
                term_sums[name] = term_sums[name] + value.detach()

        n_res = len(self.stft_modules)
        for name, value in term_sums.items():
            components[name] = value / n_res
        self.last_components = components
        return total / n_res

class BigVGANLoopLoss(nn.Module):
    """BigVGAN-in-the-loop loss: mel reconstruction + STFT reconstruction.

    pred mel -> BigVGAN -> waveform -> {conv1d STFT, conv1d mel} -> loss vs GT.
    Uses conv1d-based spectral analysis to avoid torch.stft backward bug on CUDA 13.0.
    Final loss = mel_recon_loss + 0.5 * stft_recon_loss.
    """

    def __init__(
        self,
        vocoder: nn.Module,
        sr: int = 44100,
        n_fft_list: tuple[int, ...] = (2048, 1024, 512),
        hop_list: tuple[int, ...] = (512, 256, 128),
        win_list: tuple[int, ...] = (2048, 1024, 512),
        n_mels: int = 128,
        max_chunk_frames: int = 128,
        stft_weight: float = 0.5,
    ):
        super().__init__()
        self.vocoder = vocoder
        for p in self.vocoder.parameters():
            p.requires_grad_(False)
        self.sr = sr
        self.n_mels = n_mels
        self.max_chunk_frames = max_chunk_frames
        self.stft_weight = stft_weight
        self._gt_wav_cache = None

        # Conv1D STFT modules for multi-resolution
        self.stft_modules = nn.ModuleList([
            Conv1DSTFT(n_fft, hop, win)
            for n_fft, hop, win in zip(n_fft_list, hop_list, win_list)
        ])

        # Mel filterbank for each resolution
        try:
            import torchaudio
            mel_fbs = []
            for n_fft in n_fft_list:
                fb = torchaudio.functional.melscale_fbanks(
                    n_freqs=n_fft // 2 + 1,
                    f_min=0.0,
                    f_max=sr / 2.0,
                    n_mels=n_mels,
                    sample_rate=sr,
                )
                mel_fbs.append(fb)
        except ImportError:
            mel_fbs = [self._make_mel_fb(n_fft // 2 + 1, n_mels, sr) for n_fft in n_fft_list]

        for i, fb in enumerate(mel_fbs):
            self.register_buffer(f"mel_fb_{i}", fb)

    @staticmethod
    def _make_mel_fb(n_freqs: int, n_mels: int, sr: int) -> torch.Tensor:
        """Fallback mel filterbank if torchaudio unavailable."""
        f_max = sr / 2.0
        mel_low = 2595.0 * math.log10(1.0 + 0.0 / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + f_max / 700.0)
        mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = (hz_points / f_max * (n_freqs - 1)).long()
        fb = torch.zeros(n_freqs, n_mels)
        for m in range(n_mels):
            left, center, right = bins[m], bins[m + 1], bins[m + 2]
            for k in range(left, center):
                fb[k, m] = (k - left).float() / (center - left).float()
            for k in range(center, right):
                fb[k, m] = (right - k).float() / (right - center).float()
        return fb

    def _vocoder_forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.vocoder(mel).squeeze(1)

    def _mel_recon_loss(self, wav_pred: torch.Tensor, wav_gt: torch.Tensor) -> torch.Tensor:
        """Multi-resolution mel reconstruction loss."""
        loss = torch.zeros((), device=wav_pred.device, dtype=wav_pred.dtype)
        for i, stft_mod in enumerate(self.stft_modules):
            mag_pred = stft_mod(wav_pred)
            mag_gt = stft_mod(wav_gt)
            mel_fb = getattr(self, f"mel_fb_{i}")
            # (B, n_freq, T) -> (B, T, n_freq) @ (n_freq, n_mels) -> (B, T, n_mels) -> (B, n_mels, T)
            mel_pred = torch.matmul(mag_pred.transpose(1, 2), mel_fb).transpose(1, 2)
            mel_gt = torch.matmul(mag_gt.transpose(1, 2), mel_fb).transpose(1, 2)
            log_mel_pred = torch.log(mel_pred.clamp_min(1e-5))
            log_mel_gt = torch.log(mel_gt.clamp_min(1e-5))
            # L1 + spectral convergence in mel domain
            diff = log_mel_pred - log_mel_gt
            loss = loss + diff.abs().mean() + diff.norm(dim=(1, 2)).mean() / log_mel_gt.norm(dim=(1, 2)).clamp_min(1e-7).mean()
        return loss / len(self.stft_modules)

    def _stft_recon_loss(self, wav_pred: torch.Tensor, wav_gt: torch.Tensor) -> torch.Tensor:
        """Multi-resolution STFT magnitude loss."""
        loss = torch.zeros((), device=wav_pred.device, dtype=wav_pred.dtype)
        for stft_mod in self.stft_modules:
            mag_pred = stft_mod(wav_pred)
            mag_gt = stft_mod(wav_gt)
            sc = (mag_pred - mag_gt).norm(dim=(1, 2)) / mag_gt.norm(dim=(1, 2)).clamp_min(1e-7)
            log_pred = torch.log(mag_pred.clamp_min(1e-7))
            log_gt = torch.log(mag_gt.clamp_min(1e-7))
            mag_l1 = (log_pred - log_gt).abs().mean(dim=(1, 2))
            loss = loss + sc.mean() + mag_l1.mean()
        return loss / len(self.stft_modules)

    def forward(
        self,
        x1_hat: torch.Tensor,
        x1: torch.Tensor,
        mel_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        gt_wav: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x1_hat.size(0)
        prompt_frames = prompt_lens[0].item() if B == 1 else int(prompt_lens.max().item())
        mel_end = mel_lens[0].item() if B == 1 else int(mel_lens.max().item())

        gen_start = prompt_frames
        gen_end = min(mel_end, gen_start + self.max_chunk_frames)
        if gen_end <= gen_start:
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        x1_hat_chunk = x1_hat[:, :, gen_start:gen_end].float()
        if gt_wav is None:
            # Vocode only the generated chunk of the ground-truth mel, per
            # batch.  Caching the first batch's full waveform was both wrong
            # (compared every batch against batch #1) and fragile (a NaN batch
            # poisoned the cache forever).
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                gt_wav = self.vocoder(x1[:, :, gen_start:gen_end].float()).squeeze(1)
            gt_wav = gt_wav.detach()

        # Never let a single corrupt (NaN) sample poison the loss: return a
        # zero aux loss so the flow loss still trains on healthy samples.
        if not torch.isfinite(x1_hat).all() or not torch.isfinite(gt_wav).all():
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            wav_pred = self._vocoder_forward(x1_hat_chunk)

        # gt_wav is the vocoded chunk for frames [gen_start, gen_end), so its
        # sample axis is already chunk-relative; no absolute-frame offset.
        gt_wav_chunk = gt_wav

        # Trim to matching length
        min_len = min(wav_pred.size(-1), gt_wav_chunk.size(-1))
        if min_len < 512:
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)
        wav_pred = wav_pred[..., :min_len]
        gt_wav_chunk = gt_wav_chunk[..., :min_len]

        mel_loss = self._mel_recon_loss(wav_pred, gt_wav_chunk)
        stft_loss = self._stft_recon_loss(wav_pred, gt_wav_chunk)
        return mel_loss + self.stft_weight * stft_loss


class BigVGANWaveformLoss(nn.Module):
    """BigVGAN-in-the-loop waveform L1 loss.

    pred mel -> BigVGAN -> waveform -> multi-scale waveform L1 vs GT waveform.
    Direct time-domain supervision that effectively suppresses artifacts.
    """

    def __init__(
        self,
        vocoder: nn.Module,
        sr: int = 44100,
        max_chunk_frames: int = 128,
        pool_sizes: tuple[int, ...] = (2, 4, 8),
    ):
        super().__init__()
        self.vocoder = vocoder
        for p in self.vocoder.parameters():
            p.requires_grad_(False)
        self.sr = sr
        self.max_chunk_frames = max_chunk_frames
        self.pool_sizes = pool_sizes
        self._gt_wav_cache = None

    def _vocoder_forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.vocoder(mel).squeeze(1)

    def forward(
        self,
        x1_hat: torch.Tensor,
        x1: torch.Tensor,
        mel_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        gt_wav: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x1_hat.size(0)
        prompt_frames = prompt_lens[0].item() if B == 1 else int(prompt_lens.max().item())
        mel_end = mel_lens[0].item() if B == 1 else int(mel_lens.max().item())

        gen_start = prompt_frames
        gen_end = min(mel_end, gen_start + self.max_chunk_frames)
        if gen_end <= gen_start:
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        x1_hat_chunk = x1_hat[:, :, gen_start:gen_end].float()
        if gt_wav is None:
            # Vocode only the generated chunk of the ground-truth mel, per
            # batch.  Caching the first batch's full waveform was both wrong
            # (compared every batch against batch #1) and fragile (a NaN batch
            # poisoned the cache forever).
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                gt_wav = self.vocoder(x1[:, :, gen_start:gen_end].float()).squeeze(1)
            gt_wav = gt_wav.detach()

        # Never let a single corrupt (NaN) sample poison the loss: return a
        # zero aux loss so the flow loss still trains on healthy samples.
        if not torch.isfinite(x1_hat).all() or not torch.isfinite(gt_wav).all():
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            wav_pred = self._vocoder_forward(x1_hat_chunk)

        # gt_wav is the vocoded chunk for frames [gen_start, gen_end), so its
        # sample axis is already chunk-relative; no absolute-frame offset.
        gt_wav_chunk = gt_wav

        min_len = min(wav_pred.size(-1), gt_wav_chunk.size(-1))
        if min_len < 512:
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)
        wav_pred = wav_pred[..., :min_len]
        gt_wav_chunk = gt_wav_chunk[..., :min_len]

        # Multi-scale waveform L1
        loss = (wav_pred - gt_wav_chunk).abs().mean()
        for pool_size in self.pool_sizes:
            pred_ds = F.avg_pool1d(wav_pred.unsqueeze(1), pool_size, pool_size).squeeze(1)
            gt_ds = F.avg_pool1d(gt_wav_chunk.unsqueeze(1), pool_size, pool_size).squeeze(1)
            loss = loss + (pred_ds - gt_ds).abs().mean()
        return loss / (1.0 + len(self.pool_sizes))


def slice_target_waveform(
    target_wav: torch.Tensor,
    starts: Sequence[int],
    prompt_lens: torch.Tensor,
    chunk_len: int,
    hop: int,
    target_wav_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cut the real target audio for the mel frames the aux loss picked.

    ``mel_spectrogram`` reflect-pads by ``(n_fft - hop) / 2`` before the STFT, so
    mel frame ``f`` covers samples ``[f*hop, (f+1)*hop)`` -- the same frame/sample
    mapping BigVGAN upsamples with, which is what makes this slice sample-aligned
    with the vocoder output.  ``starts`` are absolute frames in the
    ``[prompt, target]`` timeline while the waveform covers the target segment
    only, hence the ``prompt_lens`` offset.
    """
    want = int(chunk_len) * int(hop)
    rows = []
    for index, start in enumerate(starts):
        offset = (int(start) - int(prompt_lens[index])) * int(hop)
        if offset < 0:
            raise RuntimeError(
                f"aux chunk for sample {index} starts at frame {int(start)}, "
                f"before its prompt ends ({int(prompt_lens[index])})"
            )
        if target_wav_lens is None:
            row = target_wav[index, offset : offset + want]
        else:
            # Cut at the row's own content, not at the batch's padded width:
            # `target_wav` is pad_sequence output, so slicing to `want` on a
            # short row silently returns the padding of a longer neighbour and
            # nothing looks wrong.  The collator trims the waveform to
            # target_frames * hop, so the only legitimate shortfall is the tail
            # of the very last frame; anything larger means the frame/sample
            # mapping has drifted, and zero-padding it would train the vocoder
            # to emit silence where the mel has content.
            available = max(int(target_wav_lens[index]) - offset, 0)
            if want - available > hop:
                raise RuntimeError(
                    f"target_wav row {index} has {available} samples for "
                    f"{chunk_len} mel frames ({want} samples) at frame "
                    f"{int(start)}: the waveform and mel are misaligned"
                )
            row = target_wav[index, offset : offset + min(want, available)]
        if row.numel() < want:
            row = F.pad(row, (0, want - row.numel()))
        rows.append(row)
    return torch.stack(rows).float().detach()


class BigVGANMRSTFTLoss(nn.Module):
    """WaveFM MR-STFT in the BigVGAN loop.

    pred mel chunk -> frozen BigVGAN -> waveform -> WaveFM-style
    multi-resolution STFT loss vs the GT waveform chunk.  Optionally mixes a
    small waveform L1 term (``wave_l1_weight``) so the time-domain signal stays
    aligned while the spectral terms push out broadband hiss.

    ``last_components`` carries every individual term of the last forward (the
    MR-STFT sub-terms, the waveform L1 and the combined aux total) so the
    trainer can log them separately.
    """

    def __init__(
        self,
        vocoder: nn.Module,
        sr: int = 44100,
        max_chunk_frames: int = 128,
        wave_l1_weight: float = 0.0,
        stft_kwargs: dict | None = None,
        random_chunk_offset: bool = False,
        checkpoint_vocoder: bool = False,
        trainable_vocoder: bool = False,
        vocode_real_mel: bool = False,
        hop_size: int = 512,
        wavlm_weight: float = 0.0,
        wavlm_model_id: str = "microsoft/wavlm-large",
        wavlm_cache_dir: str = "",
        wavlm_layers: tuple[int, ...] = (6, 8, 10, 12),
        wavlm_local_files_only: bool = False,
    ):
        super().__init__()
        self.vocoder = vocoder
        # With a trainable vocoder the MR-STFT target may no longer be
        # `vocoder(gt_mel)` -- that target would move with the thing being
        # trained.  `forward` then requires real audio instead.
        self.trainable_vocoder = bool(trainable_vocoder)
        # Also vocode the *real* mel chunk with gradients, so the caller can add
        # BigVGAN's own objectives (mel reconstruction / GAN) on that branch and
        # stop the vocoder from drifting towards blurry predicted mels.
        self.vocode_real_mel = bool(vocode_real_mel)
        self.hop_size = int(hop_size)
        if not self.trainable_vocoder:
            for p in self.vocoder.parameters():
                p.requires_grad_(False)
        self.sr = sr
        self.max_chunk_frames = max_chunk_frames
        self.wave_l1_weight = wave_l1_weight
        # A fixed start supervises only the first max_chunk_frames of every
        # target for the whole run; a random start covers the segment in
        # expectation at identical cost.
        self.random_chunk_offset = bool(random_chunk_offset)
        # Trade vocoder recompute for its retained activations (5.3x less peak).
        self.checkpoint_vocoder = bool(checkpoint_vocoder)
        self.stft_loss = MultiResolutionSTFTLoss(**(stft_kwargs or {}))
        self.wavlm_weight = float(wavlm_weight)
        self.wavlm = None
        if self.wavlm_weight > 0.0:
            from semantic2any.losses.wavlm_perceptual import WavLMPerceptualLoss
            self.wavlm = WavLMPerceptualLoss(
                model_id=wavlm_model_id,
                cache_dir=wavlm_cache_dir,
                layers=tuple(wavlm_layers),
                input_sr=sr,
                local_files_only=wavlm_local_files_only,
            )
        keys = ["total", "mrstft", "wave_l1", "wavlm"]
        self.component_keys = tuple(
            keys + [f"mrstft/{key}" for key in self.stft_loss.component_keys]
        )
        self.last_components: dict[str, torch.Tensor] = {}
        # What the last forward produced, for a caller that wants to add its own
        # losses on the same chunk (see semantic2any/losses/vocoder_gan.py).
        # Keys: pred_wav, real_wav, real_mel_wav, real_mel_chunk.  Cleared at the
        # top of every forward so a skipped step cannot serve stale audio.
        self.last_waveforms: dict[str, torch.Tensor] = {}

    def _sample_chunk_starts(
        self, mel_lens: torch.Tensor, prompt_lens: torch.Tensor
    ) -> tuple[list[int], int]:
        """Draw one chunk start per sample, each bounded by its own lengths.

        A single batch-wide window is clamped to
        ``[prompt_lens.max(), mel_lens.min() - chunk)``, which leaves part of
        every shorter-prompt / longer-target sample permanently unsupervised.
        Here sample i draws from ``[prompt_lens[i], mel_lens[i] - chunk_len]``,
        so its whole target segment is reachable.  ``chunk_len`` is shared
        because the slices are stacked into one tensor.
        """
        mel = [int(v) for v in mel_lens.tolist()]
        prompt = [int(v) for v in prompt_lens.tolist()]
        chunk_len = min([self.max_chunk_frames] + [m - p for m, p in zip(mel, prompt)])
        if chunk_len <= 0:
            return [], 0
        starts: list[int] = []
        for m, p in zip(mel, prompt):
            span = m - chunk_len - p
            offset = int(torch.randint(0, span + 1, (1,)).item()) if span > 0 else 0
            starts.append(p + offset)
        return starts, chunk_len

    def zero_components(self, device, dtype) -> None:
        """Fill ``last_components`` with zeros (early-return paths)."""
        self.stft_loss.zero_components(device, dtype)
        self.last_components = {
            key: torch.zeros((), device=device, dtype=dtype)
            for key in self.component_keys
        }
        # The trainer calls this instead of forward() when the t gate drops the
        # whole batch; leaving the previous step's audio here would let the
        # vocoder/GAN losses train on a stale graph.
        self.last_waveforms = {}

    def _record_components(
        self,
        mrstft: torch.Tensor,
        wave_l1: torch.Tensor,
        total: torch.Tensor,
        wavlm: torch.Tensor | None = None,
    ) -> None:
        components = {
            f"mrstft/{key}": value
            for key, value in self.stft_loss.last_components.items()
        }
        components["mrstft"] = mrstft.detach()
        components["wave_l1"] = wave_l1.detach()
        components["wavlm"] = (
            wavlm.detach() if wavlm is not None
            else mrstft.detach().new_zeros(())
        )
        components["total"] = total.detach()
        self.last_components = components

    def _vocoder_stage(self, index: int, x: torch.Tensor) -> torch.Tensor:
        """One upsample stage of BigVGAN: ups[index] then its resblocks, averaged."""
        voc = self.vocoder
        for upsampler in voc.ups[index]:
            x = upsampler(x)
        summed = None
        for kernel_index in range(voc.num_kernels):
            value = voc.resblocks[index * voc.num_kernels + kernel_index](x)
            summed = value if summed is None else summed + value
        return summed / voc.num_kernels

    def _vocoder_post(self, x: torch.Tensor) -> torch.Tensor:
        voc = self.vocoder
        x = voc.conv_post(voc.activation_post(x))
        return torch.tanh(x) if voc.use_tanh_at_final else torch.clamp(x, -1, 1)

    def _vocoder_forward(self, mel: torch.Tensor) -> torch.Tensor:
        # A trainable vocoder needs its activations kept even when the *input* mel
        # is a detached real mel, so checkpointing must not key on mel alone.
        needs_backward = mel.requires_grad or self.trainable_vocoder
        if not (self.checkpoint_vocoder and torch.is_grad_enabled() and needs_backward):
            return self.vocoder(mel).squeeze(1)
        # Mirror of BigVGAN.forward with one checkpoint segment per upsample
        # stage, plus the post block (which runs at the full 44.1 kHz rate).
        # Freezing the vocoder saves only its 0.46 GiB of weights; what costs
        # 19-26 GiB is the activations kept for the backward into the predicted
        # mel, and those are exactly what recompute drops.
        x = self.vocoder.conv_pre(mel)
        for index in range(self.vocoder.num_upsamples):
            x = _checkpoint(self._vocoder_stage, index, x, use_reentrant=False)
        return _checkpoint(self._vocoder_post, x, use_reentrant=False).squeeze(1)

    def forward(
        self,
        x1_hat: torch.Tensor,
        x1: torch.Tensor,
        mel_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        gt_wav: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
        target_wav: torch.Tensor | None = None,
        target_wav_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_waveforms = {}
        B = x1_hat.size(0)
        prompt_frames = prompt_lens[0].item() if B == 1 else int(prompt_lens.max().item())
        mel_end = mel_lens[0].item() if B == 1 else int(mel_lens.max().item())

        if self.random_chunk_offset:
            # One start per sample: the batch-wide window can never cover the
            # part of a target that lies before prompt_lens.max() or after
            # mel_lens.min().
            starts, chunk_len = self._sample_chunk_starts(mel_lens, prompt_lens)
            if chunk_len <= 0:
                self.zero_components(x1_hat.device, x1_hat.dtype)
                return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)
            x1_hat_chunk = torch.stack(
                [x1_hat[i, :, s : s + chunk_len] for i, s in enumerate(starts)]
            ).float()
            gt_chunk = torch.stack(
                [x1[i, :, s : s + chunk_len] for i, s in enumerate(starts)]
            ).float()
        else:
            gen_start = prompt_frames
            gen_end = min(mel_end, gen_start + self.max_chunk_frames)
            if gen_end <= gen_start:
                self.zero_components(x1_hat.device, x1_hat.dtype)
                return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)
            x1_hat_chunk = x1_hat[:, :, gen_start:gen_end].float()
            gt_chunk = x1[:, :, gen_start:gen_end].float()
            starts = [gen_start] * B
            chunk_len = gen_end - gen_start

        if target_wav is not None:
            gt_wav = slice_target_waveform(
                target_wav,
                starts,
                prompt_lens,
                chunk_len,
                self.hop_size,
                target_wav_lens=target_wav_lens,
            )
        elif self.trainable_vocoder:
            raise RuntimeError(
                "trainable_vocoder=True needs real audio: the default target is "
                "vocoder(gt_mel), which moves with the vocoder being trained, so "
                "the pair can lower the loss without sounding better. Pass "
                "target_wav (set VOCODER_TRAIN=1 so the dataset returns it)."
            )

        if gt_wav is None:
            # Vocode only the generated chunk of the ground-truth mel, per
            # batch.  Caching the first batch's full waveform was both wrong
            # (compared every batch against batch #1) and fragile (a NaN batch
            # poisoned the cache forever).  pred and GT use the same slices, so
            # they cannot drift apart.
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                gt_wav = self.vocoder(gt_chunk).squeeze(1)
            gt_wav = gt_wav.detach()

        # Never let a single corrupt (NaN) sample poison the loss: return a
        # zero aux loss so the flow loss still trains on healthy samples.
        if not torch.isfinite(x1_hat).all() or not torch.isfinite(gt_wav).all():
            self.zero_components(x1_hat.device, x1_hat.dtype)
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            wav_pred = self._vocoder_forward(x1_hat_chunk)
            # Second vocoder pass on the *real* mel, kept in the graph.  This is
            # the branch BigVGAN itself trains on, and the only one whose target
            # (real audio) is independent of the vocoder's current weights.
            real_mel_wav = (
                self._vocoder_forward(gt_chunk) if self.vocode_real_mel else None
            )

        # gt_wav is the vocoded chunk for frames [gen_start, gen_end), so its
        # sample axis is already chunk-relative; no absolute-frame offset.
        min_len = min(wav_pred.size(-1), gt_wav.size(-1))
        if min_len < 512:
            self.zero_components(x1_hat.device, x1_hat.dtype)
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)
        wav_pred = wav_pred[..., :min_len]
        gt_wav_chunk = gt_wav[..., :min_len]
        self.last_waveforms = {
            "pred_wav": wav_pred,
            "real_wav": gt_wav_chunk,
            "real_mel_chunk": gt_chunk,
        }
        if real_mel_wav is not None:
            self.last_waveforms["real_mel_wav"] = real_mel_wav[..., :min_len]

        mrstft = self.stft_loss(
            wav_pred, gt_wav_chunk, sample_weights=sample_weights
        )

        def _wave_l1() -> torch.Tensor:
            err = (wav_pred - gt_wav_chunk).abs()
            if sample_weights is None:
                return err.mean()
            w = sample_weights.to(device=err.device, dtype=err.dtype).reshape(-1)
            return (err.flatten(1).mean(dim=1) * w).sum() / w.sum().clamp_min(1e-8)

        if self.wave_l1_weight > 0:
            wave_l1 = _wave_l1()
            loss = mrstft + self.wave_l1_weight * wave_l1
        else:
            # Still reported for monitoring, but kept out of the graph.
            with torch.no_grad():
                wave_l1 = _wave_l1()
            loss = mrstft
        wavlm = None
        if self.wavlm is not None and self.wavlm_weight > 0:
            wavlm = self.wavlm(
                wav_pred, gt_wav_chunk, sample_weights=sample_weights
            )
            loss = loss + self.wavlm_weight * wavlm
        self._record_components(mrstft, wave_l1, loss, wavlm=wavlm)
        return loss
