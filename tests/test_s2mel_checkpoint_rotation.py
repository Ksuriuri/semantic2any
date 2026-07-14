from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainers.train_s2mel_zipformer import rotate_checkpoints


class CheckpointRotationTest(unittest.TestCase):
    def test_archives_do_not_count_toward_recent_checkpoint_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for step in (1000, 2000, 3000, 4000, 10000, 20000):
                (output_dir / f"checkpoint-{step}").mkdir()
                (output_dir / f"s2mel_step{step}.pth").touch()

            rotate_checkpoints(output_dir, keep_last=3, archive_interval=10000)

            checkpoint_steps = sorted(
                int(path.name.removeprefix("checkpoint-"))
                for path in output_dir.glob("checkpoint-*")
            )
            weight_steps = sorted(
                int(path.stem.removeprefix("s2mel_step"))
                for path in output_dir.glob("s2mel_step*.pth")
            )
            self.assertEqual(checkpoint_steps, [2000, 3000, 4000, 10000, 20000])
            self.assertEqual(weight_steps, [2000, 3000, 4000, 10000, 20000])


if __name__ == "__main__":
    unittest.main()
