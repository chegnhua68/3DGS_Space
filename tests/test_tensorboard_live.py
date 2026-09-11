import tempfile
import unittest
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TensorBoardLiveTests(unittest.TestCase):
    def test_event_is_readable_before_writer_close_and_steps_increase(self):
        with tempfile.TemporaryDirectory() as directory:
            logdir = Path(directory)
            writer = SummaryWriter(log_dir=str(logdir), max_queue=100, flush_secs=5)
            try:
                writer.add_scalar("train/loss_total", 1.0, 1)
                writer.flush()
                accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
                accumulator.Reload()
                self.assertEqual(
                    [event.step for event in accumulator.Scalars("train/loss_total")],
                    [1],
                )
                writer.add_scalar("train/loss_total", 0.5, 50)
                writer.flush()
                accumulator.Reload()
                self.assertEqual(
                    [event.step for event in accumulator.Scalars("train/loss_total")],
                    [1, 50],
                )
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
