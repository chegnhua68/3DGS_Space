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

        matrix["common_args"].update(
            {
                "aux_loss_version": "filtered_edge",
                "edge_filter_kernel": 5,
                "edge_filter_sigma": 1.0,
            }
        )
        _, filtered_train, _ = RUNNER.build_commands(matrix, experiment)
        self.assertEqual(
            filtered_train[filtered_train.index("--aux_loss_version") + 1],
            "filtered_edge",
        )
        self.assertEqual(
            filtered_train[filtered_train.index("--edge_filter_kernel") + 1], "5"
        )
        self.assertEqual(
            filtered_train[filtered_train.index("--edge_filter_sigma") + 1], "1.0"
        )

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

    def test_matrix_rejects_invalid_filtered_edge_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            value = {
                "schema_version": 1,
                "source_path": "scene",
                "dataset_manifest": "dataset.json",
                "output_root": "runs",
                "common_args": {
                    "aux_loss_version": "filtered_edge",
                    "lambda_thermal": 0.1,
                },
                "experiments": [
                    {"name": "invalid", "split_manifest": "a.json", "args": {}}
                ],
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.MatrixError, "requires lambda_thermal"):
                RUNNER.load_matrix(path)

            value["common_args"] = {"edge_filter_kernel": 4}
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.MatrixError, "positive odd"):
                RUNNER.load_matrix(path)

    def test_priority1_matrix_freezes_inputs_and_forwards_four_methods(self):
        matrix = RUNNER.load_matrix(
            REPOSITORY_ROOT
            / "configs"
            / "experiment_matrix.priority1_loss_revision.json"
        )
        self.assertEqual(len(matrix["experiments"]), 4)
        expected_methods = {
            "B0_baseline_sparse25_noise03_seed2026_r1_7k": ("legacy", "0.0", "0.0", "0.0"),
            "O0_legacy_full_sparse25_noise03_seed2026_r1_7k": ("legacy", "0.1", "0.01", "0.001"),
            "E1_legacy_edge_sparse25_noise03_seed2026_r1_7k": ("legacy", "0.0", "0.001", "0.0"),
            "E2_filtered_edge_sparse25_noise03_seed2026_r1_7k": ("filtered_edge", "0.0", "0.001", "0.0"),
        }
        frozen_commands = []
        for experiment in matrix["experiments"]:
            output, train, render = RUNNER.build_commands(matrix, experiment)
            name = experiment["name"]
            version, thermal, edge, smooth = expected_methods[name]
            self.assertEqual(output.name, name)
            self.assertEqual(train[train.index("--aux_loss_version") + 1], version)
            self.assertEqual(train[train.index("--lambda_thermal") + 1], thermal)
            self.assertEqual(train[train.index("--lambda_edge") + 1], edge)
            self.assertEqual(train[train.index("--lambda_smooth") + 1], smooth)
            self.assertEqual(train[train.index("--iterations") + 1], "7000")
            self.assertEqual(train[train.index("--resolution") + 1], "1")
            self.assertEqual(train[train.index("--seed") + 1], "2026")
            self.assertEqual(train[train.index("--evaluation_partition") + 1], "val")
            self.assertEqual(render[render.index("--evaluation_partition") + 1], "val")
            frozen_commands.append(
                tuple(
                    train[train.index(flag) + 1]
                    for flag in (
                        "--dataset_manifest",
                        "--split_manifest",
                        "--degradation_manifest",
                    )
                )
            )
        self.assertEqual(len(set(frozen_commands)), 1)

    def test_expected_input_hash_mismatch_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "dataset.json"
            manifest.write_text("fixed input", encoding="utf-8")
            output = root / "run"
            command = [
                sys.executable,
                "-c",
                "pass",
                "--dataset_manifest",
                str(manifest),
            ]
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                RUNNER.run_experiment(
                    "hash-mismatch",
                    output,
                    command,
                    command,
                    skip_train=False,
                    skip_render=False,
                    allow_existing=False,
                    expected_input_hashes={"dataset_manifest": "0" * 64},
                )
            self.assertFalse(output.exists())

    def test_existing_output_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            sentinel = output / "sentinel.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                RUNNER.run_experiment(
                    "existing",
                    output,
                    [sys.executable, "-c", "raise SystemExit(9)"],
                    [sys.executable, "-c", "raise SystemExit(9)"],
                    skip_train=False,
                    skip_render=False,
                    allow_existing=False,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_run_experiment_writes_child_logs_and_manifest_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            input_manifest = root / "dataset.json"
            input_manifest.write_text("pinned", encoding="utf-8")
            command = [
                sys.executable,
                "-c",
                "import sys; print('child stdout'); print('child stderr', file=sys.stderr)",
                "--dataset_manifest",
                str(input_manifest),
            ]
            expected_hash = RUNNER._sha256_file(input_manifest)
            RUNNER.run_experiment(
                "logging",
                output,
                command,
                command,
                skip_train=False,
                skip_render=False,
                allow_existing=False,
                expected_input_hashes={"dataset_manifest": expected_hash},
            )

            self.assertIn("child stdout", (output / "stdout.log").read_text())
            self.assertIn("child stderr", (output / "stderr.log").read_text())
            record = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["log_files"], {"stdout": "stdout.log", "stderr": "stderr.log"})
            self.assertEqual(record["input_file_sha256"]["dataset_manifest"], expected_hash)
            self.assertEqual(
                record["expected_input_file_sha256"]["dataset_manifest"],
                expected_hash,
            )


if __name__ == "__main__":
    unittest.main()
