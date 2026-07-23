from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from semantic2any.data.prompt_bandwidth import simulate_lower_sample_rate
from semantic2any.utils.indextts_adapters import IndexTTSFeatureAdapter


class SimulateLowerSampleRateTest(unittest.TestCase):
    def test_noop_when_simulate_rate_not_lower(self) -> None:
        waveform = torch.randn(1, 1024)
        out = simulate_lower_sample_rate(waveform, 44100, 44100)
        self.assertTrue(torch.equal(out, waveform))
        out = simulate_lower_sample_rate(waveform, 22050, 44100)
        self.assertTrue(torch.equal(out, waveform))

    def test_preserves_shape_and_attenuates_high_frequencies(self) -> None:
        sample_rate = 44100
        duration = 0.25
        t = torch.arange(int(sample_rate * duration), dtype=torch.float32) / sample_rate
        # 12 kHz tone is above 16 kHz Nyquist and should be strongly attenuated.
        high = torch.sin(2 * torch.pi * 12000 * t).unsqueeze(0)
        limited = simulate_lower_sample_rate(high, sample_rate, 16000)
        self.assertEqual(tuple(limited.shape), tuple(high.shape))
        self.assertLess(limited.pow(2).mean().sqrt().item(), 0.2)

    def test_rejects_non_positive_rates(self) -> None:
        with self.assertRaises(ValueError):
            simulate_lower_sample_rate(torch.zeros(1, 8), 0, 16000)


class PromptBandwidthAugAdapterTest(unittest.TestCase):
    def _adapter(self) -> IndexTTSFeatureAdapter:
        adapter = IndexTTSFeatureAdapter.__new__(IndexTTSFeatureAdapter)
        torch.nn.Module.__init__(adapter)
        adapter.semantic_mean = torch.zeros(1)
        adapter.prompt_bandwidth_aug_prob = 1.0
        adapter.prompt_bandwidth_aug_rates = (16000, 22050)
        adapter.sample_rate_mel = 44100
        return adapter

    def test_skips_when_disabled_even_if_prob_is_one(self) -> None:
        adapter = self._adapter()
        waveform = torch.randn(1, 2048)
        with patch(
            "semantic2any.utils.indextts_adapters.simulate_lower_sample_rate"
        ) as simulate:
            out = adapter._maybe_limit_prompt_bandwidth(
                waveform, 44100, enabled=False
            )
        simulate.assert_not_called()
        self.assertTrue(torch.equal(out, waveform))

    def test_applies_chosen_rate_when_enabled(self) -> None:
        adapter = self._adapter()
        waveform = torch.randn(1, 2048)
        limited = torch.zeros_like(waveform)
        with (
            patch("semantic2any.utils.indextts_adapters.random.random", return_value=0.0),
            patch(
                "semantic2any.utils.indextts_adapters.random.choice",
                return_value=16000,
            ) as choice,
            patch(
                "semantic2any.utils.indextts_adapters.simulate_lower_sample_rate",
                return_value=limited,
            ) as simulate,
        ):
            out = adapter._maybe_limit_prompt_bandwidth(
                waveform, 44100, enabled=True
            )
        choice.assert_called_once_with([16000, 22050])
        simulate.assert_called_once_with(waveform, 44100, 16000)
        self.assertTrue(torch.equal(out, limited))

    def test_skips_when_probability_misses(self) -> None:
        adapter = self._adapter()
        adapter.prompt_bandwidth_aug_prob = 0.3
        waveform = torch.randn(1, 2048)
        with (
            patch("semantic2any.utils.indextts_adapters.random.random", return_value=0.99),
            patch(
                "semantic2any.utils.indextts_adapters.simulate_lower_sample_rate"
            ) as simulate,
        ):
            out = adapter._maybe_limit_prompt_bandwidth(
                waveform, 44100, enabled=True
            )
        simulate.assert_not_called()
        self.assertTrue(torch.equal(out, waveform))


if __name__ == "__main__":
    unittest.main()
