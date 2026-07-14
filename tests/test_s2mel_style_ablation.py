from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from scripts.infer_s2mel_zipformer import parse_args as parse_inference_args
from scripts.render_paired_mel_comparisons import parse_args as parse_paired_inference_args
from semantic2any.models.flow_matching import CFM
from semantic2any.models.zipformer_estimator import ZipFormerEstimator
from trainers.train_s2mel_zipformer import apply_overrides, parse_args as parse_training_args


class _CapturingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        del t, padding_mask
        self.inputs.append(x.detach().clone())
        return x[..., :1]


class _RecordingEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[bool, int]] = []

    def forward(
        self,
        x: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: torch.Tensor,
        *,
        drop_style: bool = False,
    ) -> torch.Tensor:
        del prompt_x, x_lens, t, style, cond
        self.calls.append((drop_style, x.size(0)))
        return torch.zeros_like(x)


def _minimal_estimator() -> tuple[ZipFormerEstimator, _CapturingDecoder]:
    estimator = ZipFormerEstimator.__new__(ZipFormerEstimator)
    nn.Module.__init__(estimator)
    estimator.in_channels = 1
    estimator.content_dim = 1
    estimator.style_dim = 2
    estimator.style_condition = True
    estimator.class_dropout_prob = 0.0
    estimator.condition_dropout_prob = 0.0
    estimator.cond_projection = nn.Identity()
    estimator.style_projection = nn.Linear(2, 2)
    with torch.no_grad():
        estimator.style_projection.weight.copy_(torch.eye(2))
        estimator.style_projection.bias.copy_(torch.tensor([3.0, -4.0]))
    decoder = _CapturingDecoder()
    estimator.decoder = decoder
    estimator.eval()
    return estimator, decoder


def _minimal_cfm() -> tuple[CFM, _RecordingEstimator]:
    cfm = CFM.__new__(CFM)
    nn.Module.__init__(cfm)
    cfm.in_channels = 1
    cfm.zero_prompt_speech_token = False
    estimator = _RecordingEstimator()
    cfm.estimator = estimator
    return cfm, estimator


class TrainingStyleConditionCliTest(unittest.TestCase):
    def test_training_cli_overrides_style_condition(self) -> None:
        for option, expected in (("--style-condition", True), ("--no-style-condition", False)):
            with self.subTest(option=option), patch.object(sys, "argv", ["train", option]):
                args = parse_training_args()
                cfg = OmegaConf.create(
                    {
                        "s2mel": {
                            "dit_type": "ZipFormer",
                            "ZipFormer": {"style_condition": not expected},
                        }
                    }
                )
                resolved = apply_overrides(cfg, args)

            self.assertIs(resolved.s2mel.ZipFormer.style_condition, expected)

    def test_training_cli_overrides_dit_and_wavenet_style_condition(self) -> None:
        for option, expected in (("--style-condition", True), ("--no-style-condition", False)):
            with self.subTest(option=option), patch.object(sys, "argv", ["train", option]):
                args = parse_training_args()
                cfg = OmegaConf.create(
                    {
                        "s2mel": {
                            "dit_type": "DiT",
                            "DiT": {"style_condition": not expected},
                            "wavenet": {"style_condition": not expected},
                        }
                    }
                )
                resolved = apply_overrides(cfg, args)

            self.assertIs(resolved.s2mel.DiT.style_condition, expected)
            self.assertIs(resolved.s2mel.wavenet.style_condition, expected)


class StyleMaskingTest(unittest.TestCase):
    def test_drop_style_masks_projected_embedding_including_bias(self) -> None:
        estimator, decoder = _minimal_estimator()
        kwargs = {
            "x": torch.zeros(1, 1, 3),
            "prompt_x": torch.zeros(1, 1, 3),
            "x_lens": torch.tensor([3]),
            "t": torch.zeros(1),
            "style": torch.tensor([[4.0, 5.0]]),
            "cond": torch.zeros(1, 3, 1),
        }

        estimator(**kwargs)
        reference_style = decoder.inputs[-1][..., -2:]
        torch.testing.assert_close(reference_style, torch.tensor([[[7.0, 1.0]]]).expand(1, 3, 2))

        estimator(**kwargs, drop_style=True)
        masked_style = decoder.inputs[-1][..., -2:]
        torch.testing.assert_close(masked_style, torch.zeros_like(masked_style))

    def test_cfm_forwards_style_mask_for_cfg_and_non_cfg(self) -> None:
        for cfg_rate, expected_batch in ((0.0, 1), (0.7, 2)):
            with self.subTest(inference_cfg_rate=cfg_rate):
                cfm, estimator = _minimal_cfm()
                cfm.solve_euler(
                    x=torch.zeros(1, 1, 3),
                    x_lens=torch.tensor([3]),
                    prompt=torch.zeros(1, 1, 1),
                    mu=torch.zeros(1, 3, 1),
                    style=torch.zeros(1, 2),
                    t_span=torch.tensor([0.0, 1.0]),
                    inference_cfg_rate=cfg_rate,
                    drop_style=True,
                )
                self.assertEqual(estimator.calls, [(True, expected_batch)])

    def test_cfm_default_keeps_style_enabled(self) -> None:
        cfm, estimator = _minimal_cfm()
        cfm.solve_euler(
            x=torch.zeros(1, 1, 3),
            x_lens=torch.tensor([3]),
            prompt=torch.zeros(1, 1, 1),
            mu=torch.zeros(1, 3, 1),
            style=torch.zeros(1, 2),
            t_span=torch.tensor([0.0, 1.0]),
        )
        self.assertEqual(estimator.calls, [(False, 2)])


class InferenceStyleModeCliTest(unittest.TestCase):
    def test_style_mode_defaults_and_overrides(self) -> None:
        with patch.object(sys, "argv", ["infer"]):
            self.assertEqual(parse_inference_args().style_mode, "reference")
        with patch.object(sys, "argv", ["infer", "--style-mode", "none"]):
            self.assertEqual(parse_inference_args().style_mode, "none")
        with patch.object(
            sys,
            "argv",
            [
                "render",
                "--config",
                "config.yaml",
                "--checkpoint",
                "checkpoint.pth",
                "--pair-manifest",
                "pairs.jsonl",
                "--style-mode",
                "none",
            ],
        ):
            self.assertEqual(parse_paired_inference_args().style_mode, "none")


if __name__ == "__main__":
    unittest.main()
