import math

import torch

from semantic2any.losses.s2vae_gated_buzz import S2VAEGatedBuzzLoss


def _harmonic_wav(f0: float, n: int, sr: int, noise: float, n_harm: int = 6) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / sr
    wav = torch.zeros(n)
    for harmonic in range(1, n_harm + 1):
        wav = wav + torch.sin(2 * math.pi * harmonic * f0 * t) / harmonic
    if noise > 0:
        generator = torch.Generator().manual_seed(0)
        wav = wav + noise * torch.randn(n, generator=generator)
    return wav / wav.abs().max().clamp_min(1e-6)


def _decode_from_amp(latent: torch.Tensor) -> torch.Tensor:
    amp = latent.mean(dim=(1, 2)).sigmoid()
    sr = 16_000
    n = 4096
    t = torch.arange(n, device=latent.device, dtype=latent.dtype) / sr
    harm = torch.zeros(latent.size(0), n, device=latent.device, dtype=latent.dtype)
    for harmonic in range(1, 6):
        harm = harm + torch.sin(2 * math.pi * harmonic * 220.0 * t) / harmonic
    noise = 0.35 * torch.sin(2 * math.pi * 173.0 * t)
    return amp.unsqueeze(1) * harm + (1.0 - amp).unsqueeze(1) * noise


def test_peaky_prediction_has_positive_loss() -> None:
    loss = S2VAEGatedBuzzLoss(
        sample_rate=16_000,
        n_fft=1024,
        hop_length=160,
        chunk_frames=8,
        min_chunk_frames=4,
        max_samples=2,
        t_min=0.0,
        gate_peakiness=0.90,
        margin=0.0,
        use_checkpoint=False,
        energy_floor=1e-8,
    )
    predicted = torch.full((1, 4, 8), 4.0, requires_grad=True)
    target = torch.full((1, 4, 8), -4.0)
    value = loss(
        predicted,
        target,
        torch.tensor([8]),
        torch.tensor([0]),
        torch.tensor([0.8]),
        _decode_from_amp,
    )
    assert float(value.detach()) > 0.0
    value.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_vowel_like_gt_is_gated_off() -> None:
    loss = S2VAEGatedBuzzLoss(
        sample_rate=16_000,
        n_fft=1024,
        hop_length=160,
        chunk_frames=8,
        min_chunk_frames=4,
        t_min=0.0,
        gate_peakiness=0.20,
        margin=0.0,
        use_checkpoint=False,
        energy_floor=1e-8,
    )
    peaky = _harmonic_wav(220.0, 4096, 16_000, noise=0.01)

    def decode(latent: torch.Tensor) -> torch.Tensor:
        return peaky.to(device=latent.device, dtype=torch.float32).expand(latent.size(0), -1)

    predicted = torch.ones(1, 4, 8, requires_grad=True)
    target = torch.ones(1, 4, 8)
    value = loss(
        predicted,
        target,
        torch.tensor([8]),
        torch.tensor([0]),
        torch.tensor([0.8]),
        decode,
    )
    assert float(value) == 0.0
    value.backward()
    assert predicted.grad is not None
    assert float(predicted.grad.abs().sum()) == 0.0


def test_latent_gradient_exists_for_buzzier_decode() -> None:
    loss = S2VAEGatedBuzzLoss(
        sample_rate=16_000,
        n_fft=1024,
        hop_length=160,
        chunk_frames=8,
        min_chunk_frames=4,
        t_min=0.0,
        gate_peakiness=0.90,
        margin=0.0,
        use_checkpoint=False,
        energy_floor=1e-8,
    )
    predicted = torch.full((1, 4, 8), 3.0, requires_grad=True)
    target = torch.full((1, 4, 8), -3.0)
    value = loss(
        predicted,
        target,
        torch.tensor([8]),
        torch.tensor([0]),
        torch.tensor([0.9]),
        _decode_from_amp,
    )
    assert torch.isfinite(value)
    value.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_selects_flattest_gt_window() -> None:
    peaky = _harmonic_wav(220.0, 4096, 16_000, noise=0.01)
    generator = torch.Generator().manual_seed(1)
    breathy = torch.randn(4096, generator=generator)
    breathy = breathy / breathy.abs().max().clamp_min(1e-6)

    def decode(latent: torch.Tensor) -> torch.Tensor:
        amp = latent.mean(dim=(1, 2)).sigmoid().unsqueeze(1)
        peaky_b = peaky.to(device=latent.device, dtype=latent.dtype).unsqueeze(0)
        breathy_b = breathy.to(device=latent.device, dtype=latent.dtype).unsqueeze(0)
        return amp * peaky_b + (1.0 - amp) * breathy_b

    loss = S2VAEGatedBuzzLoss(
        sample_rate=16_000,
        n_fft=1024,
        hop_length=160,
        chunk_frames=8,
        min_chunk_frames=8,
        max_samples=1,
        max_scan_windows=4,
        t_min=0.0,
        t_power=0.0,
        gate_peakiness=0.75,
        margin=0.0,
        use_checkpoint=False,
        energy_floor=1e-8,
    )
    predicted = torch.cat(
        [torch.full((1, 4, 8), 4.0), torch.full((1, 4, 8), 4.0)],
        dim=-1,
    ).requires_grad_(True)
    target = torch.cat(
        [torch.full((1, 4, 8), 4.0), torch.full((1, 4, 8), -4.0)],
        dim=-1,
    )
    value = loss(
        predicted,
        target,
        torch.tensor([16]),
        torch.tensor([0]),
        torch.tensor([0.2]),
        decode,
    )
    assert loss.last_starts is not None
    assert int(loss.last_starts.item()) == 8
    assert float(loss.last_components["gt_peakiness"]) < 0.75
