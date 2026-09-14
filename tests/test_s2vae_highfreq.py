from __future__ import annotations

import torch
import torch.nn.functional as F

from semantic2any.losses.s2vae_highfreq import (
    S2VAEBreathAwareLoss,
    S2VAEHighFrequencyLoss,
)
from semantic2any.utils.dots_audiovae import DotsAudioVAE, tiny_audio_vae_config


def _fake_decode(latent: torch.Tensor) -> torch.Tensor:
    waveform = latent.mean(dim=1, keepdim=True)
    return F.interpolate(waveform, scale_factor=128, mode="linear", align_corners=False)


def test_highfreq_loss_backpropagates_only_through_prediction() -> None:
    torch.manual_seed(7)
    predicted = torch.randn(3, 8, 30, requires_grad=True)
    target = torch.randn(3, 8, 30)
    loss_fn = S2VAEHighFrequencyLoss(
        sample_rate=16000,
        cutoff_hz=3000,
        n_fft=128,
        hop_length=32,
        chunk_frames=8,
        min_chunk_frames=4,
        max_samples=2,
        t_min=0.5,
        use_checkpoint=True,
    )
    loss = loss_fn(
        predicted,
        target,
        torch.tensor([30, 30, 30]),
        torch.tensor([10, 10, 10]),
        torch.tensor([0.2, 0.8, 0.9]),
        _fake_decode,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert predicted.grad is not None
    assert predicted.grad[0].abs().sum() == 0
    assert predicted.grad[1:].abs().sum() > 0
    assert loss_fn.last_components["samples"].item() == 2
    assert loss_fn.last_components["frames"].item() == 8


def test_empty_t_gate_returns_connected_zero() -> None:
    predicted = torch.randn(2, 8, 20, requires_grad=True)
    loss_fn = S2VAEHighFrequencyLoss(
        sample_rate=16000,
        cutoff_hz=3000,
        n_fft=128,
        hop_length=32,
        chunk_frames=8,
        min_chunk_frames=4,
        t_min=0.9,
    )
    loss = loss_fn(
        predicted,
        torch.randn_like(predicted),
        torch.tensor([20, 20]),
        torch.tensor([8, 8]),
        torch.tensor([0.1, 0.2]),
        _fake_decode,
    )
    loss.backward()
    assert loss.item() == 0
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() == 0


def test_dots_decoder_has_latent_gradient_but_frozen_parameters() -> None:
    vae = DotsAudioVAE.from_config(tiny_audio_vae_config(), remove_weight_norm=True)
    latent = torch.randn(1, vae.latent_dim, 5, requires_grad=True)
    waveform = vae.decode_differentiable(latent)
    waveform.square().mean().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in vae.parameters())


def test_breath_aware_loss_is_zero_for_identical_audio_and_backpropagates() -> None:
    torch.manual_seed(11)
    target = torch.randn(2, 8, 24)
    predicted = target.clone().requires_grad_(True)
    loss_fn = S2VAEBreathAwareLoss(
        sample_rate=16000,
        band_low_hz=2000,
        band_high_hz=7000,
        n_ffts=(64, 128),
        chunk_frames=8,
        min_chunk_frames=4,
        max_samples=2,
        excess_start_hz=4000,
        use_checkpoint=False,
    )
    loss = loss_fn(
        predicted,
        target,
        torch.tensor([24, 24]),
        torch.tensor([8, 8]),
        torch.tensor([0.8, 0.9]),
        _fake_decode,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1.0e-6
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert set(loss_fn.last_components) == set(loss_fn.component_keys)


def test_breath_aware_one_sided_tail_penalizes_only_excess() -> None:
    loss_fn = S2VAEBreathAwareLoss(
        sample_rate=16000,
        band_low_hz=2000,
        band_high_hz=7000,
        n_ffts=(128,),
        excess_start_hz=4000,
        bandwidth_floor_db=-25,
        excess_margin_db=1,
    )
    frequencies = torch.fft.rfftfreq(128, d=1.0 / 16000)
    target = torch.ones(1, frequencies.numel(), 5)
    target[:, frequencies >= 5500] = 1.0e-3
    matched = loss_fn._excess_band_loss(target, target, frequencies)
    predicted = target.clone()
    predicted[:, frequencies >= 6000] = 0.1
    excess = loss_fn._excess_band_loss(predicted, target, frequencies)
    assert matched.item() == 0.0
    assert excess.item() > 0.0
