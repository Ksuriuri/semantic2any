"""Auxiliary losses for s2mel flow-matching training."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Returns magnitude spectrogram (B, n_freq, T)."""
        pad_len = self.n_fft // 2
        x = F.pad(wav, (pad_len, pad_len), mode="reflect")
        out = F.conv1d(x.unsqueeze(1), self.kernel, stride=self.hop_length)
        real = out[:, : self.n_freq, :]
        imag = out[:, self.n_freq :, :]
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)


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
        if gt_wav is None:
            gt_wav = self._gt_wav_cache
        if gt_wav is None:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                gt_wav = self.vocoder(x1.float()).squeeze(1)
            self._gt_wav_cache = gt_wav.detach()

        B = x1_hat.size(0)
        prompt_frames = prompt_lens[0].item() if B == 1 else int(prompt_lens.max().item())
        mel_end = mel_lens[0].item() if B == 1 else int(mel_lens.max().item())

        gen_start = prompt_frames
        gen_end = min(mel_end, gen_start + self.max_chunk_frames)
        if gen_end <= gen_start:
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        x1_hat_chunk = x1_hat[:, :, gen_start:gen_end].float()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            wav_pred = self._vocoder_forward(x1_hat_chunk)

        # GT waveform for this chunk
        hop_size = 512
        wav_start = gen_start * hop_size
        wav_end = gen_end * hop_size
        gt_wav_chunk = gt_wav[:, wav_start:wav_end]

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
        if gt_wav is None:
            gt_wav = self._gt_wav_cache
        if gt_wav is None:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                gt_wav = self.vocoder(x1.float()).squeeze(1)
            self._gt_wav_cache = gt_wav.detach()

        B = x1_hat.size(0)
        prompt_frames = prompt_lens[0].item() if B == 1 else int(prompt_lens.max().item())
        mel_end = mel_lens[0].item() if B == 1 else int(mel_lens.max().item())

        gen_start = prompt_frames
        gen_end = min(mel_end, gen_start + self.max_chunk_frames)
        if gen_end <= gen_start:
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        x1_hat_chunk = x1_hat[:, :, gen_start:gen_end].float()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            wav_pred = self._vocoder_forward(x1_hat_chunk)

        hop_size = 512
        wav_start = gen_start * hop_size
        wav_end = gen_end * hop_size
        gt_wav_chunk = gt_wav[:, wav_start:wav_end]

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
