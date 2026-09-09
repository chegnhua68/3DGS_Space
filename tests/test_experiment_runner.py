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

FIXED_SOURCE_PATH = "data/TI-NSD/heated"
FIXED_DATASET_MANIFEST = "data/TI-NSD/heated/dataset_manifest.v1.json"
FIXED_SPLIT_MANIFEST = (
    "data/TI-NSD/heated/splits/seed2026/sparse_nested-random_25.json"
)
FIXED_DEGRADATION_MANIFEST = (
    "data/TI-NSD/heated/derived/noise03_sparse25_seed2026/"
    "degradation_manifest.v1.json"
)
FIXED_INPUT_HASHES = {
    "dataset_manifest": (
        "d4fe9c97ed110fa1ecdf0636a19f7a7062169d3cfbd719d5594158f9d615e919"
    ),
    "split_manifest": (
        "c9a1e0aa65e54e1558779b557539cb2eaba4d1faab765b786a4437aa519ba79e"
    ),
    "degradation_manifest": (
        "06d68ec82b7faa8d65168b9bd2b35f013207331f4683a2ec0ad883837be3f03b"
    ),
}


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
                "edge_filter_mode": "identity",
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
            filtered_train[filtered_train.index("--edge_filter_mode") + 1],
            "identity",
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

            value["common_args"] = {"edge_filter_mode": "median"}
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.MatrixError, "edge_filter_mode"):
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

        self.assertNotIn("edge_filter_mode", matrix["common_args"])
        e2 = next(
            experiment
            for experiment in matrix["experiments"]
            if experiment["name"].startswith("E2_")
        )
        resolved = RUNNER._resolved_auxiliary_loss_config(
            matrix["common_args"], e2["args"]
        )
        self.assertEqual(resolved["edge_filter_mode"], "gaussian")
        self.assertTrue(resolved["filter_active"])

    def _assert_frozen_step2_inputs(self, matrix):
        self.assertEqual(matrix["source_path"], FIXED_SOURCE_PATH)
        self.assertEqual(matrix["dataset_manifest"], FIXED_DATASET_MANIFEST)
        for experiment in matrix["experiments"]:
            self.assertEqual(experiment["split_manifest"], FIXED_SPLIT_MANIFEST)
            self.assertEqual(
                experiment["degradation_manifest"], FIXED_DEGRADATION_MANIFEST
            )
            self.assertEqual(
                experiment["expected_input_sha256"], FIXED_INPUT_HASHES
            )

    def test_edge_filter_ablation_matrix_freezes_methods_inputs_and_order(self):
        matrix = RUNNER.load_matrix(
            REPOSITORY_ROOT
            / "configs"
            / "experiment_matrix.edge_filter_ablation_7k.json"
        )
        self.assertEqual(matrix["output_root"], "runs/edge_filter_ablation_7k")
        self.assertEqual(
            [experiment["name"] for experiment in matrix["experiments"]],
            [
                "B0_trainseed2026",
                "E2_no_filter_trainseed2026",
                "E2_trainseed2026",
            ],
        )
        self.assertEqual(matrix["common_args"]["save_iterations"], [2000, 7000])
        self.assertEqual(matrix["common_args"]["test_iterations"], [2000, 7000])
        self._assert_frozen_step2_inputs(matrix)

        expected = {
            "B0_trainseed2026": ("legacy", "gaussian", "0.0", False),
            "E2_no_filter_trainseed2026": (
                "filtered_edge",
                "identity",
                "0.001",
                False,
            ),
            "E2_trainseed2026": (
                "filtered_edge",
                "gaussian",
                "0.001",
                True,
            ),
        }
        for experiment in matrix["experiments"]:
            with self.subTest(experiment=experiment["name"]):
                output, train, render = RUNNER.build_commands(matrix, experiment)
                version, mode, edge_weight, filter_active = expected[
                    experiment["name"]
                ]
                self.assertEqual(output.name, experiment["name"])
                self.assertEqual(
                    train[train.index("--aux_loss_version") + 1], version
                )
                self.assertEqual(train[train.index("--edge_filter_mode") + 1], mode)
                self.assertEqual(
                    train[train.index("--lambda_edge") + 1], edge_weight
                )
                self.assertEqual(train[train.index("--lambda_thermal") + 1], "0.0")
                self.assertEqual(train[train.index("--lambda_smooth") + 1], "0.0")
                self.assertEqual(train[train.index("--iterations") + 1], "7000")
                self.assertEqual(train[train.index("--seed") + 1], "2026")
                self.assertEqual(
                    render[render.index("--evaluation_partition") + 1], "val"
                )
                self.assertIn("--quiet", render)
                resolved = RUNNER._resolved_auxiliary_loss_config(
                    matrix["common_args"], experiment["args"]
                )
                self.assertEqual(resolved["filter_active"], filter_active)

    def test_edge_filter_30k_matrices_freeze_methods_seeds_and_schedule(self):
        matrix_cases = (
            (
                "experiment_matrix.edge_filter_validation_30k_seed2026.json",
                ["B0_trainseed2026", "E2_trainseed2026"],
                [2026, 2026],
            ),
            (
                "experiment_matrix.edge_filter_validation_30k_extra_seeds.json",
                [
                    "B0_trainseed2027",
                    "E2_trainseed2027",
                    "B0_trainseed2028",
                    "E2_trainseed2028",
                ],
                [2027, 2027, 2028, 2028],
            ),
        )
        milestones = [2000, 7000, 15000, 30000]
        for filename, expected_names, expected_seeds in matrix_cases:
            with self.subTest(matrix=filename):
                matrix = RUNNER.load_matrix(
                    REPOSITORY_ROOT / "configs" / filename
                )
                self.assertEqual(
                    matrix["output_root"], "runs/edge_filter_validation_30k"
                )
                self.assertEqual(
                    [experiment["name"] for experiment in matrix["experiments"]],
                    expected_names,
                )
                self.assertEqual(
                    matrix["common_args"]["save_iterations"], milestones
                )
                self.assertEqual(
                    matrix["common_args"]["test_iterations"], milestones
                )
                self._assert_frozen_step2_inputs(matrix)

                for experiment, expected_seed in zip(
                    matrix["experiments"], expected_seeds
                ):
                    _, train, render = RUNNER.build_commands(matrix, experiment)
                    is_e2 = experiment["name"].startswith("E2_")
                    self.assertEqual(
                        train[train.index("--aux_loss_version") + 1],
                        "filtered_edge" if is_e2 else "legacy",
                    )
                    self.assertEqual(
                        train[train.index("--edge_filter_mode") + 1], "gaussian"
                    )
                    self.assertEqual(
                        train[train.index("--lambda_edge") + 1],
                        "0.001" if is_e2 else "0.0",
                    )
                    self.assertEqual(
                        train[train.index("--lambda_thermal") + 1], "0.0"
                    )
                    self.assertEqual(
                        train[train.index("--lambda_smooth") + 1], "0.0"
                    )
                    self.assertEqual(
                        train[train.index("--iterations") + 1], "30000"
                    )
                    self.assertEqual(
                        train[train.index("--seed") + 1], str(expected_seed)
                    )
                    self.assertEqual(
                        render[render.index("--evaluation_partition") + 1], "val"
                    )

        extra_seeds = RUNNER.load_matrix(
            REPOSITORY_ROOT
            / "configs"
            / "experiment_matrix.edge_filter_validation_30k_extra_seeds.json"
        )
        self.assertNotIn("seed", extra_seeds["common_args"])

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
            auxiliary_loss_config = RUNNER._resolved_auxiliary_loss_config(
                {
                    "aux_loss_version": "filtered_edge",
                    "edge_filter_kernel": 5,
                    "edge_filter_mode": "gaussian",
                    "edge_filter_sigma": 1.0,
                },
                {
                    "lambda_edge": 0.001,
                    "lambda_smooth": 0.0,
                    "lambda_thermal": 0.0,
                },
            )
            RUNNER.run_experiment(
                "logging",
                output,
                command,
                command,
                skip_train=False,
                skip_render=False,
                allow_existing=False,
                expected_input_hashes={"dataset_manifest": expected_hash},
                auxiliary_loss_config=auxiliary_loss_config,
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
            self.assertEqual(
                record["auxiliary_loss_config"], auxiliary_loss_config
            )
            self.assertTrue(record["auxiliary_loss_config"]["filter_active"])
            for phase in ("train", "render"):
                timing = record["phase_timings"][phase]
                self.assertEqual(
                    set(timing),
                    {"started_at_utc", "finished_at_utc", "wall_seconds"},
                )
                self.assertIsInstance(timing["started_at_utc"], str)
                self.assertIsInstance(timing["finished_at_utc"], str)
                self.assertGreaterEqual(timing["wall_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
