"""IndexTTS mel frontend, vendored from indextts/s2mel/modules/audio.py."""

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
):
    """Compute the log-mel representation used by IndexTTS s2mel."""

    cache_key = f"{sampling_rate}_{fmax}_{y.device}_{y.dtype}"
    window_key = f"{sampling_rate}_{win_size}_{y.device}_{y.dtype}"
    if cache_key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis[cache_key] = torch.from_numpy(mel).float().to(y.device)
    if window_key not in hann_window:
        hann_window[window_key] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[window_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis[cache_key], spec)
    return spectral_normalize_torch(spec)


def mel_spectrogram_batch(
    waveforms,
    *,
    batch_size=2,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
):
    """Compute variable-length mels in small GPU batches without changing edges."""
    if not waveforms:
        return []
    if center:
        raise ValueError("Variable-length batched mel currently requires center=False")

    outputs = []
    padding = int((n_fft - hop_size) / 2)
    for start in range(0, len(waveforms), batch_size):
        chunk = [waveform.float() for waveform in waveforms[start : start + batch_size]]
        device = chunk[0].device
        dtype = chunk[0].dtype
        cache_key = f"{sampling_rate}_{fmax}_{device}_{dtype}"
        window_key = f"{sampling_rate}_{win_size}_{device}_{dtype}"
        if cache_key not in mel_basis:
            mel = librosa_mel_fn(
                sr=sampling_rate,
                n_fft=n_fft,
                n_mels=num_mels,
                fmin=fmin,
                fmax=fmax,
            )
            mel_basis[cache_key] = torch.from_numpy(mel).to(device=device, dtype=dtype)
        if window_key not in hann_window:
            hann_window[window_key] = torch.hann_window(
                win_size,
                device=device,
                dtype=dtype,
            )

        frame_lengths = [waveform.size(-1) // hop_size for waveform in chunk]
        edge_padded = [
            F.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect")
            .squeeze(0)
            .squeeze(0)
            for waveform in chunk
        ]
        padded = pad_sequence(edge_padded, batch_first=True)
        spec = torch.view_as_real(
            torch.stft(
                padded,
                n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=hann_window[window_key],
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        mels = spectral_normalize_torch(
            torch.matmul(mel_basis[cache_key].unsqueeze(0), spec)
        )
        outputs.extend(
            mel[..., :frames]
            for mel, frames in zip(mels, frame_lengths, strict=True)
        )
    return outputs
