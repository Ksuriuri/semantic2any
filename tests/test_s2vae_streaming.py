from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf
from torch import nn

from semantic2any.models import Semantic2MelModel
from semantic2any.models.stream_layout import (
    assemble_stream_batch,
    plan_stream_windows,
    stream_block_mask,
)
from semantic2any.utils.dots_audiovae import DotsAudioVAE, LatentStreamDecoder, tiny_audio_vae_config


def _config(*, streaming: bool, class_dropout_prob: float = 0.0, dit_type: str = "DiT"):
    return OmegaConf.create(
        {
            "dit_type": dit_type,
            "streaming": streaming,
            "stream_chunk_frames": 2,
            "stream_history_frames": 4,
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
                "class_dropout_prob": class_dropout_prob,
                "block_size": 32,
                "in_channels": 4,
                "style_condition": False,
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
                "style_condition": False,
            },
        }
    )


class StreamLayoutTest(unittest.TestCase):
    def test_block_mask_matches_the_three_regions(self) -> None:
        mask = stream_block_mask(
            torch.tensor([2]),
            torch.tensor([3]),
            torch.tensor([2]),
            7,
        )[0, 0]
        self.assertEqual(tuple(mask.shape), (7, 7))
        reference = mask[:2]
        history = mask[2:5]
        chunk = mask[5:]
        self.assertTrue(torch.equal(reference[:, :2], torch.ones(2, 2, dtype=torch.bool)))
        self.assertFalse(reference[:, 2:].any())
        self.assertTrue(torch.equal(history[:, :5], torch.ones(3, 5, dtype=torch.bool)))
        self.assertFalse(history[:, 5:].any())
        self.assertTrue(torch.equal(chunk, torch.ones(2, 7, dtype=torch.bool)))

    def test_prefix_tokens_cannot_carry_the_chunk_backwards(self) -> None:
        mask = stream_block_mask(
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([1]),
            4,
            prefix=1,
        )[0, 0]
        self.assertTrue(bool(mask[0, 0]))
        self.assertFalse(mask[0, 1:].any())
        self.assertTrue(bool(mask[1, 0]))
        self.assertFalse(mask[1, 2:].any())
        self.assertFalse(bool(mask[2, 3]))

    def test_short_target_has_no_history(self) -> None:
        starts, history, chunk = plan_stream_windows(
            torch.tensor([3, 4]),
            torch.tensor([2, 10]),
            chunk_frames=25,
            history_frames=150,
        )
        torch.testing.assert_close(starts, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(history, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(chunk, torch.tensor([2, 10]))

    def test_history_is_clean_and_only_the_chunk_is_noised(self) -> None:
        torch.manual_seed(0)
        x1 = torch.arange(2 * 4 * 9, dtype=torch.float32).reshape(2, 4, 9)
        mu = torch.arange(2 * 9 * 3, dtype=torch.float32).reshape(2, 9, 3)
        prompt_lens = torch.tensor([3, 2])
        starts = torch.tensor([4, 0])
        history = torch.tensor([4, 0])
        chunk = torch.tensor([2, 4])
        flow, prompt, cond, velocity, _noise, assembled = assemble_stream_batch(
            x1,
            mu,
            prompt_lens,
            starts,
            history,
            chunk,
            sigma_min=1e-6,
            time=torch.tensor([0.4, 0.7]),
        )
        self.assertEqual(int(assembled.max()), 9)
        torch.testing.assert_close(flow[0, :, :3], torch.zeros(4, 3))
        torch.testing.assert_close(flow[0, :, 3:7], x1[0, :, 3:7])
        self.assertGreater((flow[0, :, 7:9] - x1[0, :, 7:9]).abs().sum().item(), 0)
        torch.testing.assert_close(prompt[0, :, :3], x1[0, :, :3])
        torch.testing.assert_close(prompt[0, :, 3:], torch.zeros(4, 6))
        torch.testing.assert_close(velocity[0, :, :7], torch.zeros(4, 7))
        self.assertGreater(velocity[0, :, 7:].abs().sum().item(), 0)
        torch.testing.assert_close(cond[0, 3:7], mu[0, 3:7])
        torch.testing.assert_close(cond[1, :2], mu[1, :2])
        torch.testing.assert_close(flow[1, :, :2], torch.zeros(4, 2))


class StreamingModelTest(unittest.TestCase):
    def test_offline_wavenet_stays_noncausal_and_loads_into_streaming(self) -> None:
        offline = Semantic2MelModel(_config(streaming=False)).models["cfm"].estimator
        streaming = Semantic2MelModel(_config(streaming=True)).models["cfm"].estimator
        self.assertFalse(offline.wavenet.in_layers[0].causal)
        self.assertTrue(streaming.wavenet.in_layers[0].causal)
        streaming.load_state_dict(offline.state_dict(), strict=True)

    def test_dropout_clears_only_semantic_condition_when_streaming(self) -> None:
        offline = Semantic2MelModel(_config(streaming=False, class_dropout_prob=1.0))
        streaming = Semantic2MelModel(_config(streaming=True, class_dropout_prob=1.0))
        for model in (offline, streaming):
            model.train()
            model.models["cfm"].setup_estimator_caches(max_batch_size=1, max_seq_length=32)
        seen: list[torch.Tensor] = []

        def capture(module, inputs):
            seen.append(inputs[0].detach().clone())

        hooks = [
            model.models["cfm"].estimator.cond_x_merge_linear.register_forward_pre_hook(capture)
            for model in (offline, streaming)
        ]
        try:
            common = dict(
                x=torch.zeros(1, 4, 6),
                prompt_x=torch.ones(1, 4, 6),
                x_lens=torch.tensor([6]),
                t=torch.tensor([0.5]),
                style=torch.zeros(1, 3),
                cond=torch.ones(1, 6, 8),
            )
            offline.models["cfm"].estimator(**common)
            streaming.models["cfm"].estimator(
                **common,
                prompt_lens=torch.tensor([2]),
                stream_history_lens=torch.tensor([2]),
                stream_chunk_lens=torch.tensor([2]),
            )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(seen[0][..., 4:].abs().sum().item(), 0)
        torch.testing.assert_close(seen[1][..., 4:8], torch.ones(1, 6, 4))
        self.assertEqual(seen[1][..., 8:].abs().sum().item(), 0)

    def test_causal_wavenet_ignores_future_frames(self) -> None:
        streaming = Semantic2MelModel(_config(streaming=True)).models["cfm"].estimator
        offline = Semantic2MelModel(_config(streaming=False)).models["cfm"].estimator
        offline.wavenet.load_state_dict(streaming.wavenet.state_dict())
        current = torch.randn(1, 16, 8)
        changed = current.clone()
        changed[:, :, -1] += 4
        mask = torch.ones(1, 1, 8)
        condition = torch.randn(1, 16, 1)
        causal_now = streaming.wavenet(current, mask, condition)
        causal_future = streaming.wavenet(changed, mask, condition)
        torch.testing.assert_close(causal_now[..., :-1], causal_future[..., :-1])
        offline_now = offline.wavenet(current, mask, condition)
        offline_future = offline.wavenet(changed, mask, condition)
        self.assertGreater((offline_now[..., :-1] - offline_future[..., :-1]).abs().max().item(), 0)

    def test_forward_keeps_reference_and_puts_loss_on_the_chunk(self) -> None:
        import semantic2any.models.stream_layout as layout

        original = layout.draw_chunk_starts
        layout.draw_chunk_starts = lambda max_start: max_start.clone()
        try:
            model = Semantic2MelModel(_config(streaming=True))
            cfm = model.models["cfm"]
            cfm.setup_estimator_caches(max_batch_size=1, max_seq_length=32)
            seen: list[tuple[torch.Tensor, torch.Tensor]] = []

            def capture(_module, args, kwargs):
                seen.append((args[0].detach().clone(), args[1].detach().clone()))

            hook = cfm.estimator.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                mel = torch.randn(1, 4, 9)
                loss, hat = model(
                    mel,
                    torch.tensor([9]),
                    torch.tensor([3]),
                    torch.randn(1, 9, 8),
                    torch.zeros(1, 3),
                    semantic_is_mu=True,
                )
            finally:
                hook.remove()
            loss.backward()
        finally:
            layout.draw_chunk_starts = original
        flow, prompt = seen[0]
        self.assertEqual(tuple(flow.shape), (1, 4, 9))
        torch.testing.assert_close(flow[:, :, :3], torch.zeros(1, 4, 3))
        torch.testing.assert_close(flow[:, :, 3:7], mel[:, :, 3:7])
        torch.testing.assert_close(prompt[:, :, :3], mel[:, :, :3])
        torch.testing.assert_close(prompt[:, :, 3:], torch.zeros(1, 4, 6))
        torch.testing.assert_close(hat[:, :, :7], mel[:, :, :7])
        self.assertGreater((hat[:, :, 7:] - mel[:, :, 7:]).abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(cfm.estimator.cond_projection.weight.grad)

    def test_euler_updates_only_the_current_chunk(self) -> None:
        model = Semantic2MelModel(_config(streaming=True))
        cfm = model.models["cfm"]
        cfm.stream_chunk_frames = 2
        cfm.stream_history_frames = 4
        seen: list[torch.Tensor] = []

        class Stub(nn.Module):
            def forward(self, x, *_args, **_kwargs):
                seen.append(x.detach().clone())
                return torch.ones_like(x)

        cfm.estimator = Stub()
        source = torch.randn(1, 4, 6)
        output = cfm.solve_euler_stream(
            source,
            torch.tensor([6]),
            torch.zeros(1, 4, 2),
            torch.zeros(1, 6, 8),
            torch.zeros(1, 3),
            torch.linspace(0, 1, 2),
            inference_cfg_rate=0.0,
        )
        torch.testing.assert_close(output[:, :, :2], torch.zeros(1, 4, 2))
        torch.testing.assert_close(output[:, :, 2:], source[:, :, 2:] + 1)
        torch.testing.assert_close(seen[1][:, :, 2:4], source[:, :, 2:4] + 1)

    def test_zipformer_streaming_is_rejected(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "DiT"):
            Semantic2MelModel(_config(streaming=True, dit_type="ZipFormer"))


class LatentStreamDecoderTest(unittest.TestCase):
    def test_chunked_decode_matches_the_full_utterance(self) -> None:
        torch.manual_seed(1)
        vae = DotsAudioVAE.from_config(
            tiny_audio_vae_config(),
            enforce_released_shapes=False,
        )
        self.assertEqual(vae.vocoder.decoder_lookahead, 2)
        latent = torch.randn(1, vae.latent_dim, 7)
        full = vae.decode(latent)
        decoder = LatentStreamDecoder(vae)
        pieces = [
            decoder.push(latent[:, :, index : index + 3], flush=index + 3 >= latent.size(-1))
            for index in range(0, latent.size(-1), 3)
        ]
        audio = torch.cat(pieces, dim=-1)
        torch.testing.assert_close(audio, full)


if __name__ == "__main__":
    unittest.main()
