from __future__ import annotations

import unittest

import torch
from omegaconf import OmegaConf

from trainers.train_s2mel_zipformer import make_lr_scheduler


class CosineLrSchedulerTest(unittest.TestCase):
    def test_cosine_schedule_reaches_configured_minimum_learning_rate(self) -> None:
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=1.0e-4)
        cfg = OmegaConf.create(
            {
                "train": {
                    "learning_rate": 1.0e-4,
                    "min_learning_rate": 1.0e-5,
                    "warmup_steps": 10,
                    "lr_scheduler": "cosine",
                }
            }
        )

        scheduler = make_lr_scheduler(optimizer, cfg, num_training_steps=100)
        lr_lambda = scheduler.lr_lambdas[0]

        self.assertAlmostEqual(lr_lambda(10), 1.0)
        self.assertAlmostEqual(lr_lambda(100), 0.1)
        self.assertAlmostEqual(lr_lambda(200), 0.1)


if __name__ == "__main__":
    unittest.main()
