"""Auxiliary losses for s2mel flow-matching training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class BigVGANLoopLoss(nn.Module):
    """BigVGAN-in-the-loop mel-GAN loss with gradient checkpointing.

    Passes predicted mel through frozen BigVGAN (with gradient checkpointing
    to manage memory), then computes waveform-domain multi-resolution STFT
    loss. Gradients flow through BigVGAN back to the mel prediction.
    """

    def __init__(
        self,
        vocoder: nn.Module,
        sr: int = 44100,
        n_fft_list=(2048, 1024, 512),
        hop_list=(512, 256, 128),
        win_list=(2048, 1024, 512),
        max_chunk_frames: int = 128,
    ):
        super().__init__()
        self.vocoder = vocoder
        for p in self.vocoder.parameters():
            p.requires_grad_(False)
        self.sr = sr
        self.n_fft_list = n_fft_list
        self.hop_list = hop_list
        self.win_list = win_list
        self.max_chunk_frames = max_chunk_frames
        self._gt_wav_cache = None

    def _vocoder_forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Wrapper for checkpointing — must be a plain function call."""
        return self.vocoder(mel).squeeze(1)

    def _stft_mag(self, wav: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        window = torch.hann_window(win, device=wav.device, dtype=wav.dtype)
        stft = torch.stft(
            wav, n_fft, hop_length=hop, win_length=win,
            window=window, center=True, return_complex=True,
        )
        return stft.abs()

    def _multi_res_stft_loss(
        self, wav_pred: torch.Tensor, wav_gt: torch.Tensor
    ) -> torch.Tensor:
        # Waveform L1 + multi-scale L1 (STFT backward broken on CUDA 13.0)
        min_len = min(wav_pred.size(-1), wav_gt.size(-1))
        wav_pred = wav_pred[..., :min_len]
        wav_gt = wav_gt[..., :min_len]
        loss = (wav_pred - wav_gt).abs().mean()
        for pool_size in (2, 4, 8):
            pred_ds = F.avg_pool1d(wav_pred.unsqueeze(1), pool_size, pool_size).squeeze(1)
            gt_ds = F.avg_pool1d(wav_gt.unsqueeze(1), pool_size, pool_size).squeeze(1)
            loss = loss + (pred_ds - gt_ds).abs().mean()
        return loss / 4.0

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

        # Only process the non-prompt (generated) portion to save memory
        B = x1_hat.size(0)
        prompt_frames = prompt_lens[0].item() if B == 1 else int(prompt_lens.max().item())
        mel_end = mel_lens[0].item() if B == 1 else int(mel_lens.max().item())

        # Take a chunk of max_chunk_frames from the generated region
        gen_start = prompt_frames
        gen_end = min(mel_end, gen_start + self.max_chunk_frames)
        if gen_end <= gen_start:
            return torch.zeros((), device=x1_hat.device, dtype=x1_hat.dtype)

        x1_hat_chunk = x1_hat[:, :, gen_start:gen_end].float()
        x1_chunk = x1[:, :, gen_start:gen_end].float()

        # Direct vocoder forward (gradient checkpoint disabled for CUDA 13.0 compat)
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

        return self._multi_res_stft_loss(wav_pred, gt_wav_chunk)
