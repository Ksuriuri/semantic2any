from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from semantic2any.models import Semantic2MelModel
from semantic2any.models.dit_estimator import DiTEstimator
from semantic2any.utils.checkpoint import load_compatible_checkpoint, save_compatible_checkpoint
from trainers.train_s2mel_zipformer import model_parameter_metadata, validate_resume_backbone


def _tiny_dit_config(*, style_condition: bool = True):
    return OmegaConf.create(
        {
            "dit_type": "DiT",
            "reg_loss_type": "l1",
            "style_encoder": {"dim": 3},
            "length_regulator": {
                "channels": 8,
                "in_channels": 4,
                "sampling_ratios": [1],
                "is_discrete": False,
                "content_codebook_size": 8,
                "n_codebooks": 1,
                "quantizer_dropout": 0.0,
                "f0_condition": False,
                "n_f0_bins": 4,
            },
            "DiT": {
                "hidden_dim": 16,
                "num_heads": 4,
                "depth": 2,
                "class_dropout_prob": 0.0,
                "block_size": 16,
                "in_channels": 4,
                "style_condition": style_condition,
                "final_layer_type": "wavenet",
                "content_dim": 8,
                "content_codebook_size": 8,
                "content_type": "continuous",
                "is_causal": False,
                "long_skip_connection": True,
                "zero_prompt_speech_token": False,
                "time_as_token": False,
                "style_as_token": False,
                "uvit_skip_connection": True,
            },
            "wavenet": {
                "hidden_dim": 16,
                "num_layers": 2,
                "kernel_size": 3,
                "dilation_rate": 1,
                "p_dropout": 0.0,
                "style_condition": style_condition,
            },
        }
    )


def _estimator_inputs():
    return {
        "x": torch.zeros(1, 4, 6),
        "prompt_x": torch.zeros(1, 4, 6),
        "x_lens": torch.tensor([6]),
        "t": torch.tensor([0.5]),
        "style": torch.tensor([[2.0, -1.0, 0.5]]),
        "cond": torch.zeros(1, 6, 8),
    }


class DiTEstimatorTest(unittest.TestCase):
    def test_factory_requires_cache_and_returns_expected_shape(self) -> None:
        model = Semantic2MelModel(_tiny_dit_config())
        cfm = model.models["cfm"]
        self.assertIsInstance(cfm.estimator, DiTEstimator)

        with self.assertRaisesRegex(RuntimeError, "caches are not initialized"):
            cfm.estimator(**_estimator_inputs())

        cfm.setup_estimator_caches(max_batch_size=2, max_seq_length=16)
        output = cfm.estimator(**_estimator_inputs())
        self.assertEqual(tuple(output.shape), (1, 4, 6))

    def test_style_condition_changes_merge_input_width(self) -> None:
        with_style = Semantic2MelModel(_tiny_dit_config(style_condition=True)).models["cfm"].estimator
        without_style = Semantic2MelModel(_tiny_dit_config(style_condition=False)).models["cfm"].estimator
        self.assertEqual(with_style.cond_x_merge_linear.in_features, 16 + 2 * 4 + 3)
        self.assertEqual(without_style.cond_x_merge_linear.in_features, 16 + 2 * 4)
        without_style.setup_caches(max_batch_size=1, max_seq_length=16)
        output = without_style(**_estimator_inputs(), drop_style=True)
        self.assertEqual(tuple(output.shape), (1, 4, 6))

    def test_drop_style_zeros_raw_style_before_merge(self) -> None:
        model = Semantic2MelModel(_tiny_dit_config())
        estimator = model.models["cfm"].estimator
        estimator.eval()
        model.models["cfm"].setup_estimator_caches(max_batch_size=2, max_seq_length=16)
        merged_inputs: list[torch.Tensor] = []
        hook = estimator.cond_x_merge_linear.register_forward_pre_hook(
            lambda _module, inputs: merged_inputs.append(inputs[0].detach().clone())
        )
        try:
            estimator(**_estimator_inputs())
            reference_style = merged_inputs[-1][..., -3:]
            torch.testing.assert_close(
                reference_style,
                torch.tensor([[[2.0, -1.0, 0.5]]]).expand(1, 6, 3),
            )
            estimator(**_estimator_inputs(), drop_style=True)
            torch.testing.assert_close(merged_inputs[-1][..., -3:], torch.zeros(1, 6, 3))
        finally:
            hook.remove()

    def test_cfg_passes_style_mask_to_dit(self) -> None:
        model = Semantic2MelModel(_tiny_dit_config())
        model.eval()
        cfm = model.models["cfm"]
        cfm.setup_estimator_caches(max_batch_size=2, max_seq_length=16)
        estimator = cfm.estimator
        merged_inputs: list[torch.Tensor] = []
        hook = estimator.cond_x_merge_linear.register_forward_pre_hook(
            lambda _module, inputs: merged_inputs.append(inputs[0].detach().clone())
        )
        try:
            cfm.inference(
                mu=torch.zeros(1, 6, 8),
                x_lens=torch.tensor([6]),
                prompt=torch.zeros(1, 4, 2),
                style=torch.tensor([[2.0, -1.0, 0.5]]),
                f0=None,
                n_timesteps=1,
                inference_cfg_rate=0.5,
                drop_style=True,
            )
            self.assertEqual(merged_inputs[-1].shape[0], 2)
            torch.testing.assert_close(merged_inputs[-1][..., -3:], torch.zeros(2, 6, 3))
        finally:
            hook.remove()

    def test_strict_checkpoint_roundtrip(self) -> None:
        config = _tiny_dit_config()
        model = Semantic2MelModel(config)
        metadata = model_parameter_metadata(model, OmegaConf.create({"s2mel": config}))
        self.assertEqual(metadata["dit_type"], "DiT")
        self.assertGreater(metadata["estimator_parameters"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "s2mel_dit.pth"
            save_compatible_checkpoint(checkpoint, model, config=OmegaConf.to_container(config, resolve=True))
            restored = Semantic2MelModel(config)
            load_compatible_checkpoint(restored, checkpoint, strict=True)
            restored.eval()
            restored.models["cfm"].setup_estimator_caches(max_batch_size=1, max_seq_length=16)
            sampled = restored.models["cfm"].inference(
                mu=torch.zeros(1, 6, 8),
                x_lens=torch.tensor([6]),
                prompt=torch.zeros(1, 4, 2),
                style=torch.zeros(1, 3),
                f0=None,
                n_timesteps=1,
                inference_cfg_rate=0.0,
            )
            self.assertEqual(tuple(sampled.shape), (1, 4, 6))

    def test_resume_rejects_another_backbone(self) -> None:
        target_config = OmegaConf.create({"s2mel": {"dit_type": "DiT"}})
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "zipformer.pth"
            torch.save({"config": {"s2mel": {"dit_type": "ZipFormer"}}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "not compatible"):
                validate_resume_backbone(target_config, checkpoint)


if __name__ == "__main__":
    unittest.main()
