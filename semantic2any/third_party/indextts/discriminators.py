"""BigVGAN / HiFi-GAN discriminators, for fine-tuning the vocoder in the loop.

Re-implemented to match the hyper-parameters recorded in the checkpoint we
actually load, ``nvidia/bigvgan_v2_44khz_128band_512x`` ``config.json``:

    mpd_reshapes = [2, 3, 5, 7, 11]
    discriminator_channel_mult = 1
    use_spectral_norm = False
    use_cqtd_instead_of_mrd = True     <-- see below

Two things to know before using these:

1. **The checkpoint ships the generator only** (``bigvgan_generator.pt``); no
   discriminator weights exist upstream.  Whatever we build here starts from a
   random init, so for the first few thousand steps its output is noise and the
   adversarial term is a random push on the vocoder.  Ramp the adversarial and
   feature-matching weights in from zero (``VOCODER_GAN_WARMUP_STEPS``) instead
   of applying them at full strength from step 0.
2. **Upstream v2 used a multi-scale sub-band CQT discriminator, not the MRD.**
   That variant needs ``nnAudio`` for a differentiable CQT, which this repo does
   not depend on.  MPD + MRD is the BigVGAN v1 / HiFi-GAN pairing and is the
   standard, dependency-free substitute; it is a deliberate deviation from the
   recipe that produced these weights, not an oversight.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(nn.Module):
    """Period discriminator: reshape the waveform to 2D on `period` and conv it."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        channel_mult: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        c = channel_mult
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, int(32 * c), (kernel_size, 1), (stride, 1),
                              padding=(_padding(5), 0))),
                norm_f(Conv2d(int(32 * c), int(128 * c), (kernel_size, 1), (stride, 1),
                              padding=(_padding(5), 0))),
                norm_f(Conv2d(int(128 * c), int(512 * c), (kernel_size, 1), (stride, 1),
                              padding=(_padding(5), 0))),
                norm_f(Conv2d(int(512 * c), int(1024 * c), (kernel_size, 1), (stride, 1),
                              padding=(_padding(5), 0))),
                norm_f(Conv2d(int(1024 * c), int(1024 * c), (kernel_size, 1), 1,
                              padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(int(1024 * c), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feature_maps = []
        b, c, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
            t = x.size(-1)
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class DiscriminatorR(nn.Module):
    """Resolution discriminator: conv over one STFT magnitude resolution."""

    def __init__(
        self,
        resolution: tuple[int, int, int],
        channel_mult: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        c = int(32 * channel_mult)
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, c, (3, 9), padding=(1, 4))),
                norm_f(Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(Conv2d(c, c, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(Conv2d(c, 1, (3, 3), padding=(1, 1)))

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        ).squeeze(1)
        # No window on purpose, matching upstream: torch warns about spectral
        # leakage from the implied rectangular window, but a discriminator only
        # has to see the same transform of both signals, and changing it here
        # would make D's features differ from the recipe these weights and
        # weightings come from.
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )
        return torch.view_as_real(spec).pow(2).sum(-1).add(1e-9).sqrt()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feature_maps = []
        x = self._spectrogram(x).unsqueeze(1)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class _DiscriminatorBank(nn.Module):
    """Run every sub-discriminator on the real and the generated waveform."""

    discriminators: nn.ModuleList

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> tuple[list, list, list, list]:
        real_scores, fake_scores, real_maps, fake_maps = [], [], [], []
        for discriminator in self.discriminators:
            score_r, maps_r = discriminator(y)
            score_g, maps_g = discriminator(y_hat)
            real_scores.append(score_r)
            fake_scores.append(score_g)
            real_maps.append(maps_r)
            fake_maps.append(maps_g)
        return real_scores, fake_scores, real_maps, fake_maps


class MultiPeriodDiscriminator(_DiscriminatorBank):
    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        channel_mult: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(
                    period,
                    channel_mult=channel_mult,
                    use_spectral_norm=use_spectral_norm,
                )
                for period in periods
            ]
        )


class MultiResolutionDiscriminator(_DiscriminatorBank):
    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = (
            (1024, 120, 600),
            (2048, 240, 1200),
            (512, 50, 240),
        ),
        channel_mult: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(
                    resolution,
                    channel_mult=channel_mult,
                    use_spectral_norm=use_spectral_norm,
                )
                for resolution in resolutions
            ]
        )
