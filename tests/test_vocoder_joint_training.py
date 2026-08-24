"""CPU tests for joint BigVGAN training (VOCODER_TRAIN=1).

The real vocoder is 122 M parameters and needs a downloaded snapshot, so these
tests stand in a tiny weight-normed conv for it.  What they pin down is the
plumbing that has no other way of failing loudly: the frame/sample alignment of
the real audio, the manual cross-rank gradient average that replaces DDP, the
checkpoint round-trip, and the fold-after-load ordering that a weight-normed
state dict depends on.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from semantic2any.losses.auxiliary_losses import slice_target_waveform
from semantic2any.losses.vocoder_gan import VocoderGANTrainer, _all_reduce_grads
from trainers.train_s2mel_zipformer import rotate_checkpoints

HOP = 512
MEL_ARGS = {
    "n_fft": 2048,
    "num_mels": 128,
    "sampling_rate": 44100,
    "hop_size": HOP,
    "win_size": 2048,
    "fmin": 0,
    "fmax": None,
}


class TinyVocoder(nn.Module):
    """Stands in for BigVGAN: 128 mel bands in, `hop` samples per frame out."""

    def __init__(self, num_mels: int = 128, hop: int = HOP):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(num_mels, hop, 1))
        self.hop = hop

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # [B, num_mels, T] -> [B, 1, T * hop], the same layout BigVGAN returns.
        out = self.conv(mel)
        return out.transpose(1, 2).reshape(mel.size(0), 1, -1)

    def remove_weight_norm(self) -> None:
        nn.utils.remove_weight_norm(self.conv)


def make_trainer(world_size: int = 1, **kwargs) -> VocoderGANTrainer:
    return VocoderGANTrainer(
        TinyVocoder(),
        mel_args=MEL_ARGS,
        world_size=world_size,
        # One period / one resolution keeps the test fast; the code paths under
        # test iterate over the bank either way.
        mpd_periods=(2,),
        mrd_resolutions=((256, 64, 256),),
        **kwargs,
    )


class TargetWaveformAlignmentTest(unittest.TestCase):
    def test_slice_starts_at_the_frames_hop_boundary(self) -> None:
        # A ramp whose value at sample i is i, so the slice's first value proves
        # which sample it started at.
        target_wav = torch.arange(40 * HOP, dtype=torch.float32).reshape(2, 20 * HOP)
        prompt_lens = torch.tensor([5, 7])
        starts = [8, 11]  # absolute pair frames

        out = slice_target_waveform(target_wav, starts, prompt_lens, 4, HOP)

        self.assertEqual(tuple(out.shape), (2, 4 * HOP))
        # Sample index within the target segment, i.e. (start - prompt) * hop.
        self.assertEqual(out[0, 0].item(), (8 - 5) * HOP)
        self.assertEqual(out[1, 0].item(), 20 * HOP + (11 - 7) * HOP)

    def test_a_short_tail_is_zero_padded_but_a_real_gap_raises(self) -> None:
        target_wav = torch.ones(1, 10 * HOP)
        prompt_lens = torch.tensor([0])
        # 9.5 frames of audio in a 10-frame row: the last frame is a partial one,
        # which is what the collator's trim legitimately leaves behind.
        lens = torch.tensor([9 * HOP + HOP // 2])
        out = slice_target_waveform(
            target_wav, [0], prompt_lens, 10, HOP, target_wav_lens=lens
        )
        self.assertEqual(out.shape[-1], 10 * HOP)

        # Two whole frames missing is a misalignment, not a partial tail.  Note
        # the row itself is full width -- target_wav is pad_sequence output, so
        # only target_wav_lens can tell a short row from a long one, and a guard
        # keyed on the slice's length would never fire here.
        with self.assertRaisesRegex(RuntimeError, "misaligned"):
            slice_target_waveform(
                target_wav,
                [0],
                prompt_lens,
                10,
                HOP,
                target_wav_lens=torch.tensor([8 * HOP]),
            )

    def test_a_short_row_in_a_padded_batch_is_caught(self) -> None:
        # Row 1 holds 4 frames of audio inside a batch padded to 10 frames.
        target_wav = torch.ones(2, 10 * HOP)
        lens = torch.tensor([10 * HOP, 4 * HOP])
        with self.assertRaisesRegex(RuntimeError, "row 1"):
            slice_target_waveform(
                target_wav,
                [0, 0],
                torch.tensor([0, 0]),
                10,
                HOP,
                target_wav_lens=lens,
            )

    def test_a_chunk_starting_before_the_prompt_ends_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "before its prompt ends"):
            slice_target_waveform(
                torch.zeros(1, 4 * HOP), [1], torch.tensor([3]), 1, HOP
            )


class CollatedTargetWaveformTest(unittest.TestCase):
    def test_collate_trims_the_waveform_to_the_frames_it_keeps(self) -> None:
        from semantic2any.data.s2mel_dataset import collate_paired_features

        def feature(frames: int, with_wav: bool) -> dict:
            item = {
                "mel": torch.randn(128, frames),
                "semantic": torch.randn(frames, 1024),
                "style": torch.randn(192),
            }
            if with_wav:
                # Longer than frames * hop on purpose: the collator must cut it
                # back, or sample i * hop would stop belonging to frame i.
                item["wav"] = torch.arange(frames * HOP + 777, dtype=torch.float32)
            return item

        prompts = [feature(400, False), feature(400, False)]
        targets = [feature(600, True), feature(500, True)]
        batch = collate_paired_features(
            prompts,
            targets,
            hop_length=HOP,
            sample_rate=44100,
            max_pair_seconds=60.0,
            min_prompt_seconds=3.0,
            min_generated_frames=8,
        )

        self.assertIn("target_wav", batch)
        target_frames = batch["mel_lens"] - batch["prompt_lens"]
        expected = target_frames * HOP
        self.assertTrue(torch.equal(batch["target_wav_lens"], expected))
        # The ramp starts at 0, so row[k] == k proves nothing was dropped in front.
        self.assertEqual(batch["target_wav"][0, 0].item(), 0.0)
        self.assertEqual(batch["target_wav"][1, 123].item(), 123.0)


class MainProcessTargetWaveformTest(unittest.TestCase):
    """The main-process extraction path must hand over the real target audio.

    This is the path v24 and v25 actually use: `data.extract_mel_in_worker`
    defaults to False, so the worker branch that used to be the only producer of
    `target_wav` never runs.  Everything expensive here is stubbed -- the point is
    which fields come out, not what they contain.
    """

    HOP = 512
    RATE = 44100

    def _adapter(self):
        from semantic2any.utils import indextts_adapters as ia

        hop = self.HOP
        rate = self.RATE

        class _Stub(ia.S2MelFeatureAdapter):
            def __init__(self) -> None:
                nn.Module.__init__(self)
                self.max_audio_seconds = 30.0
                self.max_prompt_seconds = 30.0
                self.min_pair_prompt_seconds = 3.0
                self.min_target_seconds = 3.0
                self.max_target_seconds = 30.0
                self.min_generated_frames = 8
                self.sample_rate_mel = rate
                self.sample_rate_16k = 16000
                self.feature_batch_size = 16
                self.use_style_condition = False
                self.style_dim = 192
                self.mel_args = {"hop_size": hop}
                # Not `mel_spectrogram` itself, so the batched fast path (which
                # is an identity check on that function) is skipped.
                self.mel_spectrogram = self._fake_mel
                self.semantic_decoder = self

            @staticmethod
            def _fake_mel(waveform, **kwargs):
                frames = waveform.size(-1) // hop
                return torch.zeros(1, 128, frames)

            def _module_device(self):
                return torch.device("cpu")

            def _prepare_audio_batch(self, audio_paths, waveforms, sample_rates, **kw):
                return list(waveforms), [int(item) for item in sample_rates]

            def _resample_waveform_batch(self, waveforms, sample_rates, target_rates):
                # Already at the mel rate in this test, so resampling is identity.
                return {target: list(waveforms) for target in target_rates}

            def _code_rows(self, codes, lengths):
                return [codes[i, : int(lengths[i])] for i in range(codes.size(0))]

            def decode_sequences(self, sequences):
                return [torch.zeros(item.numel(), 1024) for item in sequences]

        return _Stub()

    def _run(self, *, prompt_secs: float, target_secs: float):
        adapter = self._adapter()
        prompt_samples = int(prompt_secs * self.RATE) // self.HOP * self.HOP
        target_samples = int(target_secs * self.RATE) // self.HOP * self.HOP
        # A ramp, so a shifted or truncated row is visible in the values.
        prompt = torch.arange(prompt_samples, dtype=torch.float32).reshape(1, -1)
        target = torch.arange(target_samples, dtype=torch.float32).reshape(1, -1)
        return (
            adapter.extract_paired_from_audio_paths(
                ["prompt.wav"],
                ["target.wav"],
                prompt_waveforms=[prompt],
                prompt_sample_rates=[self.RATE],
                target_waveforms=[target],
                target_sample_rates=[self.RATE],
                prompt_semantic_codes=torch.zeros(1, prompt_samples // self.HOP, dtype=torch.long),
                prompt_semantic_code_lens=torch.tensor([prompt_samples // self.HOP]),
                target_semantic_codes=torch.zeros(1, target_samples // self.HOP, dtype=torch.long),
                target_semantic_code_lens=torch.tensor([target_samples // self.HOP]),
            ),
            target_samples,
        )

    def _gate(self, enabled: bool):
        """Patch the gate in place rather than reloading the module.

        `_RETURN_TARGET_WAVEFORM` is read from the environment at import time, so
        the obvious way to flip it is `importlib.reload` -- but a reload rebinds
        the module's classes for the whole process and would quietly break every
        test that runs after this one.
        """
        from semantic2any.utils import indextts_adapters as ia

        return mock.patch.object(ia, "_RETURN_TARGET_WAVEFORM", enabled)

    def test_target_wav_is_absent_unless_vocoder_train_is_set(self) -> None:
        with self._gate(False):
            batch, _ = self._run(prompt_secs=5.0, target_secs=6.0)
        self.assertNotIn("target_wav", batch)

    def test_target_wav_is_present_and_mel_aligned_with_vocoder_train(self) -> None:
        with self._gate(True):
            batch, target_samples = self._run(prompt_secs=5.0, target_secs=6.0)

        self.assertIn("target_wav", batch)
        self.assertIn("target_wav_lens", batch)
        target_frames = batch["mel_lens"] - batch["prompt_lens"]
        # One hop of audio per generated mel frame, or the aux slice and the mel
        # stop describing the same span.
        self.assertTrue(torch.equal(batch["target_wav_lens"], target_frames * self.HOP))
        self.assertLessEqual(int(batch["target_wav_lens"][0]), target_samples)
        # The ramp starts at 0: this is the target's own audio, not the prompt's
        # and not shifted.
        self.assertEqual(batch["target_wav"][0, 0].item(), 0.0)
        self.assertEqual(batch["target_wav"][0, 123].item(), 123.0)


def _grad_average_worker(rank: int, world_size: int, init_file: str, out: dict) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    param = nn.Parameter(torch.zeros(4))
    # Rank 0 contributes 2.0; rank 1 contributes nothing at all, which is what a
    # rank whose whole micro-batch was dropped by the t gate looks like.
    if rank == 0:
        param.grad = torch.full((4,), 2.0)
    _all_reduce_grads([param], world_size)
    out[rank] = param.grad.clone()
    dist.destroy_process_group()


class CrossRankGradientTest(unittest.TestCase):
    def test_a_rank_with_no_gradient_still_reduces_and_averages(self) -> None:
        world_size = 2
        manager = mp.Manager()
        out = manager.dict()
        with tempfile.TemporaryDirectory() as tmp:
            init_file = str(Path(tmp) / "init")
            mp.spawn(
                _grad_average_worker,
                args=(world_size, init_file, out),
                nprocs=world_size,
                join=True,
            )
        self.assertEqual(sorted(out.keys()), [0, 1])
        for rank in (0, 1):
            # (2.0 + 0.0) / 2 on every rank, including the one that had None.
            self.assertTrue(torch.allclose(out[rank], torch.full((4,), 1.0)))


class VocoderGANTrainerTest(unittest.TestCase):
    def _batch(self, frames: int = 8, batch: int = 2):
        real_mel = torch.randn(batch, 128, frames)
        wav_real = torch.randn(batch, frames * HOP)
        trainer = make_trainer(warmup_steps=0)
        wav_from_real_mel = trainer.vocoder(real_mel).squeeze(1)
        return trainer, real_mel, wav_real, wav_from_real_mel

    def test_a_step_updates_the_vocoder_and_the_discriminators(self) -> None:
        trainer, real_mel, wav_real, wav_fake = self._batch()
        before = trainer.vocoder.conv.weight_v.detach().clone()
        d_before = next(trainer.mpd.parameters()).detach().clone()

        loss = trainer.generator_loss(
            wav_real=wav_real,
            wav_from_real_mel=wav_fake,
            real_mel_chunk=real_mel,
            global_step=10,
        )
        loss.backward()
        self.assertIsNotNone(trainer.discriminator_backward(scale=1.0))
        g_norm, d_norm = trainer.clip_and_step()

        self.assertGreater(g_norm, 0.0)
        self.assertGreater(d_norm, 0.0)
        self.assertFalse(torch.equal(before, trainer.vocoder.conv.weight_v))
        self.assertFalse(torch.equal(d_before, next(trainer.mpd.parameters())))

    def test_an_all_gated_batch_steps_without_a_discriminator_pass(self) -> None:
        """Every sample dropped by the t gate: no waveforms, so nothing queued.

        This has to stay deadlock-free, because clip_and_step is a collective:
        a rank that returned early here while its peers reduced would hang.
        """
        trainer = make_trainer()
        trainer.zero_grad_all()
        self.assertIsNone(trainer.discriminator_backward(scale=1.0))
        g_norm, d_norm = trainer.clip_and_step()
        self.assertEqual(g_norm, 0.0)
        self.assertEqual(d_norm, 0.0)

    def test_the_gan_terms_ramp_in_from_zero(self) -> None:
        trainer = make_trainer(warmup_steps=100)
        self.assertEqual(trainer.gan_ramp(0), 0.0)
        self.assertAlmostEqual(trainer.gan_ramp(50), 0.5)
        self.assertEqual(trainer.gan_ramp(1000), 1.0)

    def test_pred_branch_is_not_judged_by_default(self) -> None:
        """gan_on='real' must keep the flow model out of the adversarial graph."""
        trainer, real_mel, wav_real, _ = self._batch()
        pred_mel = torch.randn(2, 128, 8, requires_grad=True)
        wav_pred = trainer.vocoder(pred_mel).squeeze(1)
        trainer.generator_loss(
            wav_real=wav_real,
            wav_from_real_mel=None,
            wav_from_pred_mel=wav_pred,
            real_mel_chunk=real_mel,
            global_step=10,
        )
        # No pair queued for D, and the adv/fm components stayed exactly zero.
        self.assertIsNone(trainer.discriminator_backward())
        self.assertEqual(trainer.last_components["adv"].item(), 0.0)
        self.assertEqual(trainer.last_components["fm"].item(), 0.0)
        # The predicted branch still gets its mel term.
        self.assertGreater(trainer.last_components["mel"].item(), 0.0)

    def test_an_unknown_gan_on_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "VOCODER_GAN_ON"):
            make_trainer(gan_on="sometimes")

    def test_training_state_round_trips(self) -> None:
        trainer, real_mel, wav_real, wav_fake = self._batch()
        loss = trainer.generator_loss(
            wav_real=wav_real,
            wav_from_real_mel=wav_fake,
            real_mel_chunk=real_mel,
            global_step=10,
        )
        loss.backward()
        trainer.discriminator_backward(scale=1.0)
        trainer.clip_and_step()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocoder_gan.pt"
            torch.save(trainer.training_state(), path)
            restored = make_trainer(warmup_steps=0)
            restored.load_training_state(torch.load(path, weights_only=False))

        for name, param in trainer.vocoder.named_parameters():
            self.assertTrue(
                torch.equal(param, dict(restored.vocoder.named_parameters())[name]),
                f"vocoder.{name} did not round-trip",
            )
        for name, param in trainer.mpd.named_parameters():
            self.assertTrue(
                torch.equal(param, dict(restored.mpd.named_parameters())[name]),
                f"mpd.{name} did not round-trip",
            )
        # Adam's moments matter as much as the weights: losing them restarts the
        # optimizer cold on a converged vocoder.
        self.assertEqual(
            trainer.optimizer_g.state_dict()["state"].keys(),
            restored.optimizer_g.state_dict()["state"].keys(),
        )
        self.assertEqual(
            trainer.scheduler_g.state_dict()["last_epoch"],
            restored.scheduler_g.state_dict()["last_epoch"],
        )

    def test_the_saved_vocoder_keeps_weight_norm(self) -> None:
        """BigVGAN trains *with* weight norm, so the fold must happen at load.

        A folded checkpoint cannot be resumed, and a weight-normed state dict
        cannot be loaded into an already-folded module -- which is why
        load_vocoder folds only after load_state_dict.
        """
        trainer = make_trainer()
        state = trainer.vocoder.state_dict()
        self.assertIn("conv.weight_v", state)
        self.assertIn("conv.weight_g", state)

        folded = TinyVocoder()
        folded.remove_weight_norm()
        with self.assertRaises(RuntimeError):
            folded.load_state_dict(state, strict=True)


class VocoderCheckpointRotationTest(unittest.TestCase):
    def test_vocoder_files_rotate_with_their_model_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for step in (1000, 2000, 3000, 4000, 10000, 20000):
                (output_dir / f"checkpoint-{step}").mkdir()
                (output_dir / f"s2mel_step{step}.pth").touch()
                (output_dir / f"bigvgan_step{step}.pt").touch()

            rotate_checkpoints(output_dir, keep_last=3, archive_interval=10000)

            vocoder_steps = sorted(
                int(path.stem.removeprefix("bigvgan_step"))
                for path in output_dir.glob("bigvgan_step*.pt")
            )
            model_steps = sorted(
                int(path.stem.removeprefix("s2mel_step"))
                for path in output_dir.glob("s2mel_step*.pth")
            )
            # A step that keeps its model must keep its vocoder, or the pair
            # stops being resumable together.
            self.assertEqual(vocoder_steps, model_steps)
            self.assertEqual(vocoder_steps, [2000, 3000, 4000, 10000, 20000])


class MelArgsContractTest(unittest.TestCase):
    """The vocoder's mel loss must use the training mel's exact convention.

    A different filterbank would optimise the vocoder against a target the model
    can never produce, and nothing would fail loudly.
    """

    def test_mel_args_match_the_v25_config(self) -> None:
        from omegaconf import OmegaConf

        from trainers.train_s2mel_zipformer import _mel_args_from_cfg

        cfg = OmegaConf.load(
            "configs/s2mel_dit_indextts25_v25_44k_vocoder_joint.yaml"
        )
        self.assertEqual(
            _mel_args_from_cfg(cfg),
            {
                "n_fft": 2048,
                "num_mels": 128,
                "sampling_rate": 44100,
                "hop_size": 512,
                "win_size": 2048,
                "fmin": 0.0,
                "fmax": None,
                "center": False,
            },
        )

    def test_yaml_none_for_fmax_stays_none(self) -> None:
        # `fmax: None` in yaml is the string "None", not null: float("None")
        # raises, and any fallback that quietly swallowed it would build the
        # wrong filterbank.
        from omegaconf import OmegaConf

        from trainers.train_s2mel_zipformer import _mel_args_from_cfg

        cfg = OmegaConf.create(
            {"preprocess_params": {"sr": 44100, "spect_params": {"fmax": "None"}}}
        )
        self.assertIsNone(_mel_args_from_cfg(cfg)["fmax"])
        cfg.preprocess_params.spect_params.fmax = 16000
        self.assertEqual(_mel_args_from_cfg(cfg)["fmax"], 16000.0)


class EnvDefaultsTest(unittest.TestCase):
    def test_joint_training_is_off_unless_asked_for(self) -> None:
        # The live run is restarted from its own launch script, so anything that
        # turned itself on by default would change that run's loss on a restart.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(VocoderGANTrainer.enabled())
        with mock.patch.dict(os.environ, {"VOCODER_TRAIN": "0"}):
            self.assertFalse(VocoderGANTrainer.enabled())
        with mock.patch.dict(os.environ, {"VOCODER_TRAIN": "1"}):
            self.assertTrue(VocoderGANTrainer.enabled())

    def test_from_env_defaults_match_the_documented_recipe(self) -> None:
        with mock.patch.dict(os.environ, {"VOCODER_TRAIN": "1"}, clear=True):
            trainer = VocoderGANTrainer.from_env(
                TinyVocoder(), mel_args=MEL_ARGS, world_size=1
            )
        self.assertAlmostEqual(trainer.optimizer_g.param_groups[0]["lr"], 1.0e-05)
        self.assertAlmostEqual(trainer.optimizer_d.param_groups[0]["lr"], 1.0e-04)
        self.assertEqual(trainer.mel_weight, 15.0)  # upstream lambda_melloss
        self.assertEqual(trainer.fm_weight, 2.0)
        self.assertEqual(trainer.grad_clip, 500.0)
        self.assertEqual(trainer.gan_on, "real")


if __name__ == "__main__":
    unittest.main()
