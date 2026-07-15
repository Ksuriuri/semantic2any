from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from trainers.train_s2mel_zipformer import (
    AsyncFeatureBatchBuilder,
    async_feature_extraction_enabled,
    step_requires_async_prefetch_barrier,
)


class AsyncFeatureBatchBuilderTest(unittest.TestCase):
    def test_prefetches_next_batch_and_preserves_order(self) -> None:
        second_started = threading.Event()
        release_second = threading.Event()

        def build(raw_batch):
            batch_id = raw_batch["id"]
            if batch_id == 1:
                second_started.set()
                if not release_second.wait(timeout=2):
                    raise TimeoutError("test did not release the second batch")
            return {"id": torch.tensor(batch_id)}

        builder = AsyncFeatureBatchBuilder(build, device=torch.device("cpu"))
        try:
            builder.submit({"id": 0})
            first = builder.get()
            self.assertEqual(first["id"].item(), 0)

            builder.submit({"id": 1})
            self.assertTrue(second_started.wait(timeout=2))
            release_second.set()
            second = builder.get()
            self.assertEqual(second["id"].item(), 1)
        finally:
            release_second.set()
            builder.close()

    def test_outputs_can_be_saved_for_autograd(self) -> None:
        builder = AsyncFeatureBatchBuilder(
            lambda _raw_batch: {"features": torch.ones(2, 3)},
            device=torch.device("cpu"),
        )
        try:
            builder.submit({"id": 0})
            features = builder.get()["features"]
            projection = torch.nn.Linear(3, 1)
            projection(features).sum().backward()

            self.assertIsNotNone(projection.weight.grad)
        finally:
            builder.close()

    def test_worker_exception_is_propagated_and_close_is_idempotent(self) -> None:
        def fail(_raw_batch):
            raise ValueError("feature failure")

        builder = AsyncFeatureBatchBuilder(fail, device=torch.device("cpu"))
        builder.submit({"id": 0})
        with self.assertRaisesRegex(ValueError, "feature failure"):
            builder.get()

        builder.close()
        builder.close()

    def test_rejects_more_than_one_in_flight_batch(self) -> None:
        release = threading.Event()

        def build(_raw_batch):
            if not release.wait(timeout=2):
                raise TimeoutError("test did not release the batch")
            return {"id": torch.tensor(0)}

        builder = AsyncFeatureBatchBuilder(build, device=torch.device("cpu"))
        try:
            builder.submit({"id": 0})
            with self.assertRaisesRegex(RuntimeError, "one feature batch"):
                builder.submit({"id": 1})
        finally:
            release.set()
            builder.close()


class AsyncFeatureExtractionConfigTest(unittest.TestCase):
    def test_enables_configured_online_cuda_extraction(self) -> None:
        cfg = OmegaConf.create(
            {
                "data": {
                    "preload_features": False,
                    "async_feature_extraction": True,
                }
            }
        )
        accelerator = SimpleNamespace(device=torch.device("cuda", 0))

        self.assertTrue(async_feature_extraction_enabled(cfg, accelerator))

    def test_disables_prefetch_for_preloaded_or_cpu_training(self) -> None:
        preloaded = OmegaConf.create({"data": {"preload_features": True}})
        online = OmegaConf.create({"data": {"preload_features": False}})
        cuda = SimpleNamespace(device=torch.device("cuda", 0))

        self.assertFalse(
            async_feature_extraction_enabled(
                preloaded,
                cuda,
            )
        )
        self.assertFalse(async_feature_extraction_enabled(online, cuda))
        self.assertFalse(
            async_feature_extraction_enabled(
                OmegaConf.create(
                    {
                        "data": {
                            "preload_features": False,
                            "async_feature_extraction": True,
                        }
                    }
                ),
                SimpleNamespace(device=torch.device("cpu")),
            )
        )

    def test_prefetch_barrier_covers_validation_checkpoint_and_stop(self) -> None:
        cfg = OmegaConf.create(
            {
                "train": {
                    "valid_interval": 5,
                    "save_interval": 7,
                    "archive_save_interval": 11,
                    "max_steps": 13,
                }
            }
        )

        self.assertFalse(
            step_requires_async_prefetch_barrier(
                cfg,
                sync_gradients=False,
                next_global_step=5,
                has_validation=True,
            )
        )
        for step in (5, 7, 11, 13):
            with self.subTest(step=step):
                self.assertTrue(
                    step_requires_async_prefetch_barrier(
                        cfg,
                        sync_gradients=True,
                        next_global_step=step,
                        has_validation=True,
                    )
                )
        self.assertFalse(
            step_requires_async_prefetch_barrier(
                cfg,
                sync_gradients=True,
                next_global_step=1,
                has_validation=True,
            )
        )


class AccelerateLookaheadTest(unittest.TestCase):
    def test_advancing_loader_inside_accumulate_preserves_sync_boundaries(self) -> None:
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=3)
        model = accelerator.prepare(torch.nn.Linear(1, 1))
        loader = accelerator.prepare(
            DataLoader(
                TensorDataset(torch.arange(5.0).view(-1, 1)),
                batch_size=1,
            )
        )
        iterator = iter(loader)
        current_batch = next(iterator)
        sync_sequence = []

        while current_batch is not None:
            with accelerator.accumulate(model):
                sync_sequence.append(accelerator.sync_gradients)
                try:
                    next_batch = next(iterator)
                except StopIteration:
                    next_batch = None
                model(current_batch[0]).sum().backward()
            current_batch = next_batch

        self.assertEqual(sync_sequence, [False, False, True, False, True])


if __name__ == "__main__":
    unittest.main()
