from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DEPENDENCIES_AVAILABLE = True
DEPENDENCY_ERROR = None
try:
    import numpy as np
    from PIL import Image
    import matplotlib  # noqa: F401
    import torch

    from thermal3dgs_sparse_ir.edge_filter_ablation import (
        AblationAnalysisError,
        CHECKPOINTS,
        EXPECTED_INPUT_SHA256,
        EXPECTED_RUN_NAMES,
        METHODS,
        METRICS,
        analyze_edge_filter_ablation,
    )
    from thermal3dgs_sparse_ir.metrics import collect_metrics
except Exception as exc:
    DEPENDENCIES_AVAILABLE = False
    DEPENDENCY_ERROR = exc


SUMMARY_FIELDS = (
    "experiment",
    "num_views",
    "psnr",
    "ssim",
    "lpips",
    "t_mae",
    "e_mae",
    "gradient_preservation",
    "roi_mae",
)
PER_VIEW_FIELDS = (
    "experiment",
    "image",
    "psnr",
    "ssim",
    "lpips",
    "t_mae",
    "e_mae",
    "gradient_preservation",
    "roi_mae",
)
VAL_IDS = (
    "001.jpg", "010.jpg", "019.jpg", "028.jpg", "037.jpg", "046.jpg",
    "055.jpg", "065.jpg", "074.jpg", "083.jpg", "092.jpg", "101.jpg",
    "110.jpg", "119.jpg", "129.jpg", "138.jpg", "147.jpg", "156.jpg",
    "165.jpg", "174.jpg", "183.jpg", "193.jpg", "202.jpg", "211.jpg",
    "220.jpg", "229.jpg", "238.jpg", "247.jpg", "257.jpg", "266.jpg",
    "275.jpg", "284.jpg", "293.jpg", "302.jpg",
)


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "ablation dependencies unavailable: {}".format(DEPENDENCY_ERROR),
)
class EdgeFilterAblationTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, fields, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    @staticmethod
    def _write_image(path: Path, base: int, offset: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.full((30, 45), base, dtype=np.uint8)
        array[:, 22:] = np.clip(base + 35, 0, 255)
        array[8:22, 14:31] = np.clip(base + offset + 15, 0, 255)
        Image.fromarray(array, mode="L").save(str(path))

    @staticmethod
    def _commands(method: str, run_dir: Path):
        python = REPOSITORY_ROOT / ".venv" / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        source = REPOSITORY_ROOT / "data" / "TI-NSD" / "heated"
        version, filter_mode, lambda_edge = {
            "B0": ("legacy", "gaussian", "0.0"),
            "E2_no_filter": ("filtered_edge", "identity", "0.001"),
            "E2": ("filtered_edge", "gaussian", "0.001"),
        }[method]
        train = [
            str(python.resolve()),
            str((REPOSITORY_ROOT / "train.py").resolve()),
            "-s", str(source.resolve()),
            "-m", str(run_dir.resolve()),
            "--dataset_manifest", str((source / "dataset_manifest.v1.json").resolve()),
            "--split_manifest",
            str((source / "splits" / "seed2026" / "sparse_nested-random_25.json").resolve()),
            "--degradation_manifest",
            str(
                (
                    source
                    / "derived"
                    / "noise03_sparse25_seed2026"
                    / "degradation_manifest.v1.json"
                ).resolve()
            ),
            "--aux_loss_version", version,
            "--data_device", "cpu",
            "--edge_filter_kernel", "5",
            "--edge_filter_mode", filter_mode,
            "--edge_filter_sigma", "1.0",
            "--eval",
            "--evaluation_partition", "val",
            "--iterations", "7000",
            "--lambda_edge", lambda_edge,
            "--lambda_smooth", "0.0",
            "--lambda_thermal", "0.0",
            "--load2gpu_on_the_fly",
            "--log_interval", "500",
            "--quiet",
            "--resolution", "1",
            "--save_iterations", "2000", "7000",
            "--seed", "2026",
            "--test_iterations", "2000", "7000",
        ]
        render = [
            str(python.resolve()),
            str((REPOSITORY_ROOT / "render.py").resolve()),
            "-m", str(run_dir.resolve()),
            "--skip_train",
            "--evaluation_partition", "val",
            "--quiet",
        ]
        return train, render

    @classmethod
    def _make_fixture(cls, root: Path):
        metric_dirs = {}
        run_dirs = {}
        for method_index, method in enumerate(METHODS):
            run_dir = root / "runs" / method
            run_dirs[method] = run_dir
            cameras = [
                {
                    "id": index,
                    "img_name": view_id,
                    "width": 45,
                    "height": 30,
                    "position": [float(index), 0.0, 1.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "fy": 50.0,
                    "fx": 50.0,
                }
                for index, view_id in enumerate(VAL_IDS)
            ]
            cls._write_json(run_dir / "cameras.json", cameras)
            train_command, render_command = cls._commands(method, run_dir)
            cls._write_json(
                run_dir / "run_manifest.json",
                {
                    "schema_version": 1,
                    "experiment": EXPECTED_RUN_NAMES[method],
                    "status": "completed",
                    "train_command": train_command,
                    "render_command": render_command,
                    "phase_timings": {
                        "train": {
                            "started_at_utc": "2026-09-09T00:00:00+00:00",
                            "finished_at_utc": "2026-09-09T00:02:00+00:00",
                            "wall_seconds": 100.0 + method_index,
                        },
                        "render": {
                            "started_at_utc": "2026-09-09T00:02:00+00:00",
                            "finished_at_utc": "2026-09-09T00:02:04+00:00",
                            "wall_seconds": 3.0 + method_index * 0.1,
                        },
                    },
                    "git": {
                        "head": "a" * 40,
                        "diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "source_snapshot_sha256": "b" * 64,
                        "status": "",
                    },
                    "input_file_sha256": EXPECTED_INPUT_SHA256,
                    "expected_input_file_sha256": EXPECTED_INPUT_SHA256,
                    "auxiliary_loss_config": {
                        "aux_loss_version": (
                            "legacy" if method == "B0" else "filtered_edge"
                        ),
                        "auxiliary_enabled": method != "B0",
                        "edge_filter_mode": (
                            "identity" if method == "E2_no_filter" else "gaussian"
                        ),
                        "filter_active": method == "E2",
                        "edge_filter_kernel": 5,
                        "edge_filter_sigma": 1.0,
                        "lambda_thermal": 0.0,
                        "lambda_edge": 0.0 if method == "B0" else 0.001,
                        "lambda_smooth": 0.0,
                    },
                },
            )
            loss_rows = []
            for iteration in CHECKPOINTS:
                loss_rows.append(
                    {
                        "iteration": iteration,
                        "Gaussian_count": 1000 + method_index * 100 + iteration,
                        "avg_ms": 10.0 + method_index,
                        "peak_cuda_mb": 200.0 + method_index,
                    }
                )
                for role, relative, payload in (
                    (
                        "point_cloud",
                        Path("point_cloud") / "iteration_{}".format(iteration) / "point_cloud.ply",
                        b"ply-data",
                    ),
                    (
                        "atf",
                        Path("ATF") / "iteration_{}".format(iteration) / "ATF.pth",
                        b"atf-data",
                    ),
                    (
                        "tcm",
                        Path("TCM") / "iteration_{}".format(iteration) / "TCM.pth",
                        b"tcm-data",
                    ),
                ):
                    del role
                    path = run_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload + method.encode("ascii") + str(iteration).encode("ascii"))

                render_root = run_dir / "val" / "ours_{}".format(iteration)
                cls._write_json(
                    render_root / "render_timing.json",
                    {
                        "schema_version": 1,
                        "partition": "val",
                        "iteration": iteration,
                        "wall_seconds": 1.5,
                        "num_views": 34,
                        "forward_mean_ms": 10.0,
                        "forward_samples_total": 34,
                        "warmup_excluded": 5,
                        "forward_timed_samples": 29,
                        "wall_scope": (
                            "render_set loop including device transfer and PNG writes; "
                            "excludes model loading"
                        ),
                        "forward_scope": (
                            "synchronized ATF, Gaussian renderer, and TCM; excludes PNG I/O"
                        ),
                    },
                )
                (render_root / "render_time.txt").write_text(
                    "".join("10.00ms\n" for _ in range(34)) + "Mean time: 10.00ms\n",
                    encoding="utf-8",
                )
                for view_index in range(34):
                    filename = "{:05d}.png".format(view_index)
                    cls._write_image(render_root / "gt" / filename, 40 + view_index)
                    cls._write_image(
                        render_root / "renders" / filename,
                        40 + view_index,
                        offset=(3 - method_index) + (1 if iteration == 2000 else 0),
                    )
            cls._write_csv(
                run_dir / "loss_components.csv",
                ("iteration", "Gaussian_count", "avg_ms", "peak_cuda_mb"),
                loss_rows,
            )

        for iteration in CHECKPOINTS:
            metric_dir = root / "metrics" / "val_{}".format(iteration)
            metric_dirs[iteration] = metric_dir
            collect_metrics(
                experiments=[
                    (
                        method,
                        run_dirs[method] / "val" / "ours_{}".format(iteration),
                    )
                    for method in METHODS
                ],
                output_dir=metric_dir,
                skip_lpips=True,
                device=torch.device("cpu"),
            )
        return metric_dirs, run_dirs

    def test_generates_strict_deltas_resources_figures_and_hash_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            output_dir = root / "results"
            outputs = analyze_edge_filter_ablation(
                metric_dirs, run_dirs, output_dir, dpi=40
            )
            self.assertEqual(
                set(outputs),
                {
                    "metrics_deltas",
                    "per_view_deltas",
                    "resource_summary",
                    "qualitative_full",
                    "absolute_error",
                    "qualitative_zoom",
                    "manifest",
                },
            )
            with outputs["metrics_deltas"].open("r", encoding="utf-8", newline="") as source:
                deltas = list(csv.DictReader(source))
            with outputs["per_view_deltas"].open("r", encoding="utf-8", newline="") as source:
                per_view = list(csv.DictReader(source))
            with outputs["resource_summary"].open("r", encoding="utf-8", newline="") as source:
                resources = list(csv.DictReader(source))
            self.assertEqual(len(deltas), 30)
            self.assertEqual(len(per_view), 1020)
            self.assertEqual(len(resources), 6)
            primary = next(
                row
                for row in deltas
                if row["iteration"] == "7000"
                and row["comparison"] == "E2_minus_E2_no_filter"
                and row["metric"] == "psnr"
            )
            self.assertEqual(primary["mean_outcome"], "improved")
            self.assertEqual(primary["per_view_improved"], "34")
            lower = next(
                row
                for row in deltas
                if row["iteration"] == "7000"
                and row["comparison"] == "E2_minus_E2_no_filter"
                and row["metric"] == "t_mae"
            )
            self.assertEqual(lower["better_direction"], "lower")
            self.assertEqual(lower["mean_outcome"], "improved")
            self.assertEqual(per_view[0]["val_view_id"], "001.jpg")
            for key in ("qualitative_full", "absolute_error", "qualitative_zoom"):
                with Image.open(str(outputs[key])) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "edge-filter-ablation-results")
            self.assertIn(manifest["metric_contract"]["ssim_backend"], ("official", "fallback"))
            self.assertEqual(len(manifest["metric_contract"]["metrics_source_sha256"]), 64)
            self.assertEqual(
                len(manifest["metric_contract"]["collect_metrics_source_sha256"]), 64
            )
            self.assertIn(
                manifest["metric_contract"]["ssim_implementation"]["provider"],
                ("metrics_py", "source_file"),
            )
            self.assertEqual(
                len(manifest["metric_contract"]["ssim_implementation"]["sha256"]),
                64,
            )
            self.assertEqual(
                manifest["metric_contract"]["runtime"]["torch"], torch.__version__
            )
            input_kinds = [record["kind"] for record in manifest["inputs"]]
            self.assertEqual(input_kinds.count("metrics-manifest"), 2)
            self.assertIn("metrics-source", input_kinds)
            self.assertIn("metrics-cli-source", input_kinds)
            self.assertEqual(
                manifest["figure_protocol"]["val_view_ids"]["00022.png"], "202.jpg"
            )
            self.assertEqual(
                manifest["figure_protocol"]["crop_xyxy"]["00000.png"], [15, 10, 30, 20]
            )
            self.assertEqual(len(manifest["outputs"]), 6)
            for record in manifest["outputs"]:
                artifact_path = output_dir / record["path"]
                digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                self.assertEqual(record["sha256"], digest)
            with self.assertRaises(FileExistsError):
                analyze_edge_filter_ablation(metric_dirs, run_dirs, output_dir, dpi=40)

    def test_rejects_cross_method_gt_mismatch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            self._write_image(
                run_dirs["E2"] / "val" / "ours_7000" / "gt" / "00017.png",
                199,
            )
            output_dir = root / "results"
            with self.assertRaisesRegex(AblationAnalysisError, "ground-truth SHA256"):
                analyze_edge_filter_ablation(metric_dirs, run_dirs, output_dir, dpi=40)
            self.assertFalse(output_dir.exists())

    def test_rejects_summary_mismatch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            summary_path = metric_dirs[2000] / "metrics_summary.csv"
            with summary_path.open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            rows[0]["psnr"] = "99.00000000"
            self._write_csv(summary_path, SUMMARY_FIELDS, rows)
            metric_manifest_path = metric_dirs[2000] / "metrics_manifest.json"
            metric_manifest = json.loads(metric_manifest_path.read_text(encoding="utf-8"))
            metric_manifest["outputs"]["metrics_summary.csv"] = {
                "bytes": summary_path.stat().st_size,
                "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            }
            self._write_json(metric_manifest_path, metric_manifest)
            output_dir = root / "results"
            with self.assertRaisesRegex(AblationAnalysisError, "summary mismatch"):
                analyze_edge_filter_ablation(metric_dirs, run_dirs, output_dir, dpi=40)
            self.assertFalse(output_dir.exists())

    def test_rejects_non_frozen_input_hash_and_training_command(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            manifest_path = run_dirs["B0"] / "run_manifest.json"
            original = json.loads(manifest_path.read_text(encoding="utf-8"))

            changed = json.loads(json.dumps(original))
            changed["input_file_sha256"]["dataset_manifest"] = "0" * 64
            self._write_json(manifest_path, changed)
            with self.assertRaisesRegex(AblationAnalysisError, "frozen Step2 SHA256"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "hash-results", dpi=40
                )

            changed = json.loads(json.dumps(original))
            seed_index = changed["train_command"].index("--seed") + 1
            changed["train_command"][seed_index] = "2027"
            self._write_json(manifest_path, changed)
            with self.assertRaisesRegex(AblationAnalysisError, "train_command differs"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "command-results", dpi=40
                )
            self.assertFalse((root / "hash-results").exists())
            self.assertFalse((root / "command-results").exists())

    def test_rejects_noncanonical_val_id_and_camera_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            camera_paths = {
                method: run_dirs[method] / "cameras.json" for method in METHODS
            }
            originals = {
                method: json.loads(path.read_text(encoding="utf-8"))
                for method, path in camera_paths.items()
            }

            for method, path in camera_paths.items():
                changed = json.loads(json.dumps(originals[method]))
                changed[17]["img_name"] = "wrong-view.jpg"
                self._write_json(path, changed)
            with self.assertRaisesRegex(AblationAnalysisError, "frozen val view"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "view-results", dpi=40
                )

            for method, path in camera_paths.items():
                self._write_json(path, originals[method])
            changed = json.loads(json.dumps(originals["E2"]))
            changed[17]["position"][0] += 0.25
            self._write_json(camera_paths["E2"], changed)
            with self.assertRaisesRegex(AblationAnalysisError, "geometry/intrinsics"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "camera-results", dpi=40
                )
            self.assertFalse((root / "view-results").exists())
            self.assertFalse((root / "camera-results").exists())

    def test_rejects_nonselected_image_dimension_and_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            render_path = (
                run_dirs["E2"] / "val" / "ours_7000" / "renders" / "00017.png"
            )
            Image.fromarray(np.zeros((31, 45), dtype=np.uint8), mode="L").save(
                str(render_path)
            )
            with self.assertRaisesRegex(AblationAnalysisError, "dimensions"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "dimension-results", dpi=40
                )
            self.assertFalse((root / "dimension-results").exists())

    def test_rejects_metric_manifest_bound_to_stale_render_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            render_path = (
                run_dirs["E2"] / "val" / "ours_7000" / "renders" / "00017.png"
            )
            self._write_image(render_path, 199, offset=1)
            with self.assertRaisesRegex(AblationAnalysisError, "metric renders image SHA256"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "binding-results", dpi=40
                )
            self.assertFalse((root / "binding-results").exists())

    def test_rejects_stale_ssim_implementation_and_runtime_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metric_dirs, run_dirs = self._make_fixture(root)
            manifest_path = metric_dirs[2000] / "metrics_manifest.json"
            original = json.loads(manifest_path.read_text(encoding="utf-8"))

            changed = json.loads(json.dumps(original))
            changed["source"]["ssim_implementation"]["sha256"] = "0" * 64
            self._write_json(manifest_path, changed)
            with self.assertRaisesRegex(AblationAnalysisError, "SSIM implementation"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "ssim-results", dpi=40
                )

            changed = json.loads(json.dumps(original))
            changed["source"]["runtime"]["torch"] = "different"
            self._write_json(manifest_path, changed)
            with self.assertRaisesRegex(AblationAnalysisError, "metric runtime"):
                analyze_edge_filter_ablation(
                    metric_dirs, run_dirs, root / "runtime-results", dpi=40
                )
            self.assertFalse((root / "ssim-results").exists())
            self.assertFalse((root / "runtime-results").exists())

    def test_cli_parser_accepts_exact_named_inputs(self):
        from tools.analyze_edge_filter_ablation import build_parser

        args = build_parser().parse_args(
            (
                "--metrics-dir", "2000=results/val_2000",
                "--metrics-dir", "7000=results/val_7000",
                "--run-dir", "B0=runs/B0",
                "--run-dir", "E2_no_filter=runs/E2_no_filter",
                "--run-dir", "E2=runs/E2",
                "--output-dir", "results/edge_filter_ablation_7k",
            )
        )
        self.assertEqual(dict(args.metrics_dir)[2000], Path("results/val_2000"))
        self.assertEqual(dict(args.run_dir)["E2"], Path("runs/E2"))


if __name__ == "__main__":
    unittest.main()
