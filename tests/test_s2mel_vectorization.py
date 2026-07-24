from __future__ import annotations

import torch
from torch.nn import functional as F

from semantic2any.models.length_regulator import InterpolateRegulator
from semantic2any.models.s2mel_model import Semantic2MelModel


def test_vectorized_semantic_slicing_matches_per_sample_reference() -> None:
    semantic = torch.randn(3, 11, 5)
    starts = torch.tensor([0, 2, 5])
    lengths = torch.tensor([7, 4, 6])

    actual = Semantic2MelModel._slice_semantic_segments(semantic, starts, lengths)
    expected = semantic.new_zeros(3, 7, 5)
    for index in range(semantic.size(0)):
        start = int(starts[index])
        length = int(lengths[index])
        expected[index, :length] = semantic[index, start : start + length]

    torch.testing.assert_close(actual, expected)


def test_vectorized_length_interpolation_matches_nearest_reference() -> None:
    torch.manual_seed(1234)
    regulator = InterpolateRegulator(
        channels=4,
        sampling_ratios=(1,),
        in_channels=3,
    ).eval()
    semantic = torch.randn(3, 8, 3)
    semantic_lens = torch.tensor([8, 5, 3])
    mel_lens = torch.tensor([11, 7, 9])

    actual = regulator(semantic, ylens=mel_lens, xlens=semantic_lens)[0]

    projected = regulator.content_in_proj(semantic)
    max_mel = int(mel_lens.max())
    interpolated = projected.new_zeros(3, 4, max_mel)
    for index in range(semantic.size(0)):
        xlen = int(semantic_lens[index])
        ylen = int(mel_lens[index])
        interpolated[index, :, :ylen] = F.interpolate(
            projected[index : index + 1, :xlen].transpose(1, 2),
            size=ylen,
            mode="nearest",
        )[0]
    expected = regulator.model(interpolated).transpose(1, 2)
    mask = torch.arange(max_mel).unsqueeze(0) < mel_lens.unsqueeze(1)
    expected = expected * mask.unsqueeze(-1)

    torch.testing.assert_close(actual, expected)
