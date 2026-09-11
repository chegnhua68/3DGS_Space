import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from integrations.thermal3dgs.run_control import (
    UserStopRequested,
    raise_if_stop_requested,
    write_run_status,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e2_gd_runner", ROOT / "scripts" / "run_experiments.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class E2GDControlTests(unittest.TestCase):
    def test_stop_file_is_sticky_and_status_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop_file = root / "STOP_REQUESTED"
            stop_file.touch()
            with self.assertRaises(UserStopRequested) as raised:
                raise_if_stop_requested(stop_file, "val", 12)
            self.assertEqual(raised.exception.last_completed_iteration, 12)
            write_run_status(
                root,
                status="interrupted",
                phase="val",
                last_completed_iteration=12,
                stop_reason="user_request",
            )
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "interrupted")
            self.assertTrue(stop_file.exists())

    def test_runner_stops_current_worker_and_never_starts_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            stop_file = root / "STOP_REQUESTED"
            child = (
                "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
                "while_not=False\n"
            )
            child = (
                "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
                "\nwhile not p.exists(): time.sleep(0.02)\nraise SystemExit(130)"
            )
            command = [sys.executable, "-c", child, str(stop_file)]
            timer = threading.Timer(0.15, stop_file.touch)
            timer.start()
            try:
                status = RUNNER.run_experiment(
                    "stop-test",
                    output,
                    command,
                    [sys.executable, "-c", "raise SystemExit(9)"],
                    skip_train=False,
                    skip_render=False,
                    allow_existing=False,
                    stop_file=stop_file,
                )
            finally:
                timer.cancel()
            self.assertEqual(status, "interrupted")
            record = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(record["interrupted_phase"], "train")
            self.assertFalse((output / "render_started").exists())
            self.assertTrue(stop_file.exists())


if __name__ == "__main__":
    unittest.main()
