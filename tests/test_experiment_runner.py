import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_experiments", REPOSITORY_ROOT / "scripts" / "run_experiments.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ExperimentRunnerTests(unittest.TestCase):
    def test_build_commands_include_zero_loss_defaults_and_manifest_paths(self):
        matrix = {
            "source_path": "data/scene",
            "dataset_manifest": "data/scene/dataset.json",
            "output_root": "runs",
            "common_args": {
                "eval": True,
                "evaluation_partition": "val",
                "lambda_thermal": 0.0,
                "seed": 2027,
            },
        }
        experiment = {
            "name": "baseline",
            "split_manifest": "splits/full.json",
            "degradation_manifest": None,
            "args": {"iterations": 10},
        }
        output, train, render = RUNNER.build_commands(matrix, experiment)
        self.assertEqual(output.name, "baseline")
        self.assertIn("--eval", train)
        self.assertIn("--lambda_thermal", train)
        self.assertEqual(train[train.index("--lambda_thermal") + 1], "0.0")
        self.assertEqual(render[render.index("-m") + 1], str(output))
        self.assertIn("--skip_train", render)
        self.assertEqual(
            render[render.index("--evaluation_partition") + 1], "val"
        )
        self.assertIn("--seed", train)
        self.assertEqual(train[train.index("--seed") + 1], "2027")

        matrix["common_args"]["quiet"] = True
        _, _, quiet_render = RUNNER.build_commands(matrix, experiment)
        self.assertIn("--quiet", quiet_render)

    def test_invalid_evaluation_partition_is_rejected(self):
        matrix = {
            "source_path": "data/scene",
            "dataset_manifest": "data/scene/dataset.json",
            "output_root": "runs",
            "common_args": {"evaluation_partition": "validation"},
        }
        experiment = {
            "name": "invalid-partition",
            "split_manifest": "splits/full.json",
            "degradation_manifest": None,
            "args": {},
        }
        with self.assertRaisesRegex(RUNNER.MatrixError, "evaluation_partition"):
            RUNNER.build_commands(matrix, experiment)

    def test_matrix_rejects_duplicate_experiment_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            value = {
                "schema_version": 1,
                "source_path": "scene",
                "dataset_manifest": "dataset.json",
                "output_root": "runs",
                "common_args": {},
                "experiments": [
                    {"name": "same", "split_manifest": "a.json", "args": {}},
                    {"name": "same", "split_manifest": "b.json", "args": {}},
                ],
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.MatrixError, "duplicate experiment"):
                RUNNER.load_matrix(path)

    def test_environment_variables_are_expanded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            variable = "THERMAL3DGS_TEST_MATRIX_ROOT"
            old_value = os.environ.get(variable)
            os.environ[variable] = "expanded-scene"
            try:
                value = {
                    "schema_version": 1,
                    "source_path": "${" + variable + "}",
                    "dataset_manifest": "dataset.json",
                    "output_root": "runs",
                    "common_args": {},
                    "experiments": [
                        {"name": "one", "split_manifest": "a.json", "args": {}}
                    ],
                }
                path.write_text(json.dumps(value), encoding="utf-8")
                self.assertEqual(RUNNER.load_matrix(path)["source_path"], "expanded-scene")
            finally:
                if old_value is None:
                    del os.environ[variable]
                else:
                    os.environ[variable] = old_value

    def test_matrix_rejects_path_escape_and_reserved_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            value = {
                "schema_version": 1,
                "source_path": "scene",
                "dataset_manifest": "dataset.json",
                "output_root": "runs",
                "common_args": {},
                "experiments": [
                    {"name": "..", "split_manifest": "a.json", "args": {}}
                ],
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.MatrixError, "safe path"):
                RUNNER.load_matrix(path)

            value["experiments"][0] = {
                "name": "valid",
                "split_manifest": "a.json",
                "args": {"dataset_manifest": "other.json"},
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.MatrixError, "reserved"):
                RUNNER.load_matrix(path)

    def test_matrix_rejects_nonstandard_numeric_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(
                '{"schema_version":1,"source_path":"scene",'
                '"dataset_manifest":"dataset.json","output_root":"runs",'
                '"common_args":{"lambda_thermal":NaN},'
                '"experiments":[{"name":"one","split_manifest":"a.json",'
                '"args":{}}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RUNNER.MatrixError, "non-standard"):
                RUNNER.load_matrix(path)

    def test_run_experiment_writes_child_logs_and_manifest_references(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            command = [
                sys.executable,
                "-c",
                "import sys; print('child stdout'); print('child stderr', file=sys.stderr)",
            ]
            RUNNER.run_experiment(
                "logging",
                output,
                command,
                command,
                skip_train=False,
                skip_render=False,
                allow_existing=False,
            )

            self.assertIn("child stdout", (output / "stdout.log").read_text())
            self.assertIn("child stderr", (output / "stderr.log").read_text())
            record = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["log_files"], {"stdout": "stdout.log", "stderr": "stderr.log"})


if __name__ == "__main__":
    unittest.main()
