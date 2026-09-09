from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
from contextlib import redirect_stderr
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


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
    import torch

    from thermal3dgs_sparse_ir import metrics
except Exception as exc:  # Native torch/numpy availability depends on the host.
    DEPENDENCIES_AVAILABLE = False
    DEPENDENCY_ERROR = exc


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "metric dependencies unavailable: {}".format(DEPENDENCY_ERROR),
)
class MetricTests(unittest.TestCase):
    def test_identical_edge_image_has_ideal_metrics_without_lpips(self):
        image = torch.zeros((1, 16, 16), dtype=torch.float32)
        image[:, :, 8:] = 1.0
        values = metrics.evaluate_pair(image, image)
        self.assertTrue(math.isinf(values["psnr"]))
        self.assertAlmostEqual(values["ssim"], 1.0, places=6)
        self.assertIsNone(values["lpips"])
        self.assertEqual(values["t_mae"], 0.0)
        self.assertEqual(values["e_mae"], 0.0)
        self.assertAlmostEqual(values["gradient_preservation"], 1.0, places=6)
        self.assertIsNone(values["roi_mae"])

    def test_rgb_uses_fixed_bt601_luma_without_minmax(self):
        prediction = torch.zeros((3, 4, 5), dtype=torch.float32)
        prediction[0] = 1.0
        ground_truth = torch.zeros_like(prediction)
        self.assertAlmostEqual(
            metrics.thermal_mae(prediction, ground_truth), 0.299, places=6
        )
        expected_psnr = 10.0 * math.log10(1.0 / (0.299 ** 2))
        self.assertAlmostEqual(
            metrics.psnr(prediction, ground_truth), expected_psnr, places=5
        )

    def test_empty_roi_is_unavailable_and_nonempty_roi_is_normalized(self):
        prediction = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]])
        ground_truth = torch.zeros_like(prediction)
        empty = torch.zeros((1, 2, 2))
        self.assertIsNone(metrics.roi_mae(prediction, ground_truth, empty))
        soft_mask = torch.tensor([[[0.0, 1.0], [0.5, 0.0]]])
        expected = (0.5 * 1.0 + 1.0 * 0.5) / 1.5
        self.assertAlmostEqual(
            metrics.roi_mae(prediction, ground_truth, soft_mask), expected, places=6
        )

    def test_edge_mae_uses_raw_gt_edge_weights_and_support_denominator(self):
        ground_truth = torch.zeros((1, 1, 16, 16))
        ground_truth[:, :, :, 8:] = 1.0
        prediction = torch.roll(ground_truth, shifts=1, dims=3)
        weights = metrics._gt_edge_weights(ground_truth)
        half_strength_weights = metrics._gt_edge_weights(ground_truth * 0.5)
        self.assertTrue(torch.allclose(half_strength_weights, weights * 0.5))
        pred_x, pred_y = metrics._sobel_components(prediction)
        gt_x, gt_y = metrics._sobel_components(ground_truth)
        error = 0.5 * ((pred_x - gt_x).abs() + (pred_y - gt_y).abs())
        expected = (weights * error).sum() / weights.sum()
        self.assertAlmostEqual(
            metrics.edge_mae(prediction, ground_truth), expected.item(), places=6
        )

    def test_lpips_receives_three_identical_luma_channels(self):
        class CapturingLPIPS(torch.nn.Module):
            def __init__(self):
                super(CapturingLPIPS, self).__init__()
                self.last_prediction = None

            def forward(self, prediction, ground_truth):
                self.last_prediction = prediction.detach().clone()
                return (prediction - ground_truth).abs().mean().reshape(1, 1, 1, 1)

        prediction = torch.zeros((3, 5, 4))
        prediction[1] = 1.0
        ground_truth = torch.zeros_like(prediction)
        model = CapturingLPIPS()
        value = metrics.lpips_score(prediction, ground_truth, model)
        self.assertAlmostEqual(value, 0.587, places=6)
        self.assertEqual(tuple(model.last_prediction.shape), (1, 3, 5, 4))
        self.assertTrue(
            torch.equal(
                model.last_prediction[:, 0], model.last_prediction[:, 1]
            )
        )
        self.assertTrue(
            torch.equal(
                model.last_prediction[:, 1], model.last_prediction[:, 2]
            )
        )

    def test_out_of_range_tensor_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.thermal_mae(torch.full((1, 2, 2), 2.0), torch.zeros(1, 2, 2))


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "metric dependencies unavailable: {}".format(DEPENDENCY_ERROR),
)
class MetricCollectionTests(unittest.TestCase):
    @staticmethod
    def _write_png(path: Path, value: int) -> None:
        array = np.full((12, 12), value, dtype=np.uint8)
        Image.fromarray(array, mode="L").save(str(path))

    def _make_experiment(self, root: Path, names) -> Path:
        method_dir = root / "method"
        renders = method_dir / "renders"
        ground_truth = method_dir / "gt"
        renders.mkdir(parents=True)
        ground_truth.mkdir(parents=True)
        for index, name in enumerate(names):
            self._write_png(renders / name, 32 + index)
            self._write_png(ground_truth / name, 32 + index)
        return method_dir

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_pairs_are_exact_and_sorted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            method_dir = self._make_experiment(
                Path(temporary_dir), ("view_b.png", "view_a.png")
            )
            pairs = metrics.pair_experiment_images(method_dir)
            self.assertEqual([pair[0] for pair in pairs], ["view_a.png", "view_b.png"])
            self._write_png(method_dir / "renders" / "extra.png", 0)
            with self.assertRaisesRegex(ValueError, "filename mismatch"):
                metrics.pair_experiment_images(method_dir)

    def test_cli_accepts_repeated_named_experiments(self):
        from tools.collect_metrics import build_parser

        args = build_parser().parse_args(
            (
                "--experiment",
                "baseline=run/baseline",
                "--experiment",
                "ours=run/ours",
                "--skip-lpips",
            )
        )
        self.assertEqual(
            args.experiment,
            [
                ("baseline", Path("run/baseline")),
                ("ours", Path("run/ours")),
            ],
        )
        self.assertFalse(args.overwrite)

        overwrite_args = build_parser().parse_args(
            (
                "--experiment",
                "baseline=run/baseline",
                "--skip-lpips",
                "--overwrite",
            )
        )
        self.assertTrue(overwrite_args.overwrite)

    def test_cli_forwards_explicit_overwrite(self):
        from tools import collect_metrics as collect_metrics_cli

        with mock.patch.object(
            collect_metrics_cli, "collect_metrics", return_value={}
        ) as collector:
            exit_code = collect_metrics_cli.main(
                (
                    "--experiment",
                    "baseline=run/baseline",
                    "--skip-lpips",
                    "--device",
                    "cpu",
                    "--overwrite",
                )
            )
        self.assertEqual(exit_code, 0)
        self.assertTrue(collector.call_args.kwargs["overwrite"])

    def test_collection_writes_outputs_and_verifiable_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = self._make_experiment(root / "first", ("b.png", "a.png"))
            second = self._make_experiment(root / "second", ("a.png",))
            mask_dir = root / "masks"
            mask_dir.mkdir()
            self._write_png(mask_dir / "a.png", 0)
            self._write_png(mask_dir / "b.png", 0)
            output_dir = root / "results"
            outputs = metrics.collect_metrics(
                experiments=(("z_method", second), ("a_method", first)),
                output_dir=output_dir,
                roi_mask_dirs={"a_method": mask_dir},
                skip_lpips=True,
                device=torch.device("cpu"),
            )
            self.assertEqual(
                set(outputs),
                {"summary_csv", "summary_md", "per_view_csv", "manifest"},
            )
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            with outputs["summary_csv"].open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["experiment"] for row in rows], ["a_method", "z_method"])
            self.assertEqual(rows[0]["lpips"], "unavailable")
            self.assertEqual(rows[0]["roi_mae"], "unavailable")
            with outputs["per_view_csv"].open("r", encoding="utf-8", newline="") as source:
                per_view = list(csv.DictReader(source))
            self.assertEqual(
                [row["image"] for row in per_view if row["experiment"] == "a_method"],
                ["a.png", "b.png"],
            )
            markdown = outputs["summary_md"].read_text(encoding="utf-8")
            self.assertIn("no per-image min-max normalization", markdown)

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "thermal3dgs.metrics_manifest")
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["kind"], "metric_collection")
            self.assertEqual(manifest["parameters"]["device"], "cpu")
            self.assertTrue(manifest["parameters"]["skip_lpips"])
            self.assertIn(manifest["source"]["ssim_backend"], ("official", "fallback"))
            metrics_path = Path(manifest["source"]["metrics_py"]["path"])
            self.assertEqual(
                manifest["source"]["metrics_py"]["sha256"],
                self._sha256(metrics_path),
            )
            ssim_source = manifest["source"]["ssim_implementation"]
            ssim_path = Path(ssim_source["path"])
            self.assertEqual(ssim_source["sha256"], self._sha256(ssim_path))
            if manifest["source"]["ssim_backend"] == "official":
                self.assertEqual(ssim_source["provider"], "source_file")
            else:
                self.assertEqual(ssim_source["provider"], "metrics_py")
                self.assertEqual(ssim_path, metrics_path)
            self.assertEqual(
                manifest["source"]["runtime"]["python"],
                platform.python_version(),
            )
            self.assertEqual(
                manifest["source"]["runtime"]["torch"], torch.__version__
            )

            self.assertEqual(
                [item["name"] for item in manifest["experiments"]],
                ["a_method", "z_method"],
            )
            first_inventory = manifest["experiments"][0]
            self.assertEqual(first_inventory["method_dir"], str(first.resolve()))
            self.assertEqual(first_inventory["inventory"]["view_count"], 2)
            self.assertEqual(first_inventory["inventory"]["count"], 6)
            expected_names = {"a.png", "b.png"}
            self.assertEqual(
                set(first_inventory["renders"]["files"]), expected_names
            )
            self.assertEqual(set(first_inventory["gt"]["files"]), expected_names)
            self.assertEqual(
                set(first_inventory["roi_masks"]["files"]), expected_names
            )
            self.assertEqual(
                first_inventory["renders"]["files"]["a.png"],
                self._sha256(first / "renders" / "a.png"),
            )
            for output_name, record in manifest["outputs"].items():
                self.assertEqual(
                    record["sha256"], self._sha256(output_dir / output_name)
                )
                self.assertEqual(
                    record["bytes"], (output_dir / output_name).stat().st_size
                )

    def test_collection_refuses_existing_target_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            method_dir = self._make_experiment(root / "input", ("a.png",))
            output_dir = root / "results"
            output_dir.mkdir()
            existing = output_dir / "metrics_summary.csv"
            existing.write_text("preserve me\n", encoding="utf-8")

            with mock.patch.object(metrics, "evaluate_experiment") as evaluator:
                with self.assertRaises(FileExistsError):
                    metrics.collect_metrics(
                        experiments=(("baseline", method_dir),),
                        output_dir=output_dir,
                        skip_lpips=True,
                        device=torch.device("cpu"),
                    )
            evaluator.assert_not_called()
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve me\n")
            self.assertFalse((output_dir / "metrics_manifest.json").exists())

    def test_collection_rejects_non_boolean_overwrite(self):
        with self.assertRaisesRegex(TypeError, "overwrite must be boolean"):
            metrics.collect_metrics((), overwrite=1)

    def test_collection_rejects_inputs_changed_during_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            method_dir = self._make_experiment(root / "input", ("a.png",))
            output_dir = root / "results"
            real_evaluate = metrics.evaluate_experiment

            def evaluate_then_mutate(*args, **kwargs):
                result = real_evaluate(*args, **kwargs)
                self._write_png(method_dir / "renders" / "a.png", 99)
                return result

            with mock.patch.object(
                metrics, "evaluate_experiment", side_effect=evaluate_then_mutate
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "input files changed during evaluation"
                ):
                    metrics.collect_metrics(
                        experiments=(("baseline", method_dir),),
                        output_dir=output_dir,
                        skip_lpips=True,
                        device=torch.device("cpu"),
                    )

            self.assertFalse(
                any(
                    (output_dir / name).exists()
                    for name in metrics.METRIC_OUTPUT_FILENAMES.values()
                )
            )

    def test_cli_reports_output_path_os_errors_without_traceback(self):
        from tools import collect_metrics as collect_metrics_cli

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "not-a-directory"
            output_path.write_text("occupied\n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    collect_metrics_cli.main(
                        (
                            "--experiment",
                            "baseline=unused",
                            "--output-dir",
                            str(output_path),
                            "--skip-lpips",
                            "--device",
                            "cpu",
                        )
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("metric output path is not a directory", stderr.getvalue())

    def test_explicit_overwrite_replaces_only_managed_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            method_dir = self._make_experiment(root / "input", ("a.png",))
            output_dir = root / "results"
            first_outputs = metrics.collect_metrics(
                experiments=(("baseline", method_dir),),
                output_dir=output_dir,
                skip_lpips=True,
                device=torch.device("cpu"),
            )
            unrelated = output_dir / "keep.txt"
            unrelated.write_text("unrelated\n", encoding="utf-8")
            first_outputs["summary_csv"].write_text("stale\n", encoding="utf-8")

            second_outputs = metrics.collect_metrics(
                experiments=(("baseline", method_dir),),
                output_dir=output_dir,
                skip_lpips=True,
                device=torch.device("cpu"),
                overwrite=True,
            )

            self.assertEqual(unrelated.read_text(encoding="utf-8"), "unrelated\n")
            self.assertNotEqual(
                second_outputs["summary_csv"].read_text(encoding="utf-8"), "stale\n"
            )
            self.assertTrue(second_outputs["manifest"].is_file())
            self.assertFalse(any(output_dir.glob(".metrics-stage-*")))

    def test_publication_failure_restores_all_managed_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            method_dir = self._make_experiment(root / "input", ("a.png",))
            output_dir = root / "results"
            outputs = metrics.collect_metrics(
                experiments=(("baseline", method_dir),),
                output_dir=output_dir,
                skip_lpips=True,
                device=torch.device("cpu"),
            )
            original_bytes = {key: path.read_bytes() for key, path in outputs.items()}
            real_replace = metrics.os.replace
            replace_count = 0

            def fail_once_during_install(source, target):
                nonlocal replace_count
                replace_count += 1
                if replace_count == 6:
                    raise OSError("simulated publication failure")
                return real_replace(source, target)

            with mock.patch.object(
                metrics.os, "replace", side_effect=fail_once_during_install
            ):
                with self.assertRaisesRegex(OSError, "simulated publication failure"):
                    metrics.collect_metrics(
                        experiments=(("baseline", method_dir),),
                        output_dir=output_dir,
                        skip_lpips=True,
                        device=torch.device("cpu"),
                        overwrite=True,
                    )

            self.assertEqual(
                {key: path.read_bytes() for key, path in outputs.items()},
                original_bytes,
            )
            self.assertFalse(any(output_dir.glob(".metrics-stage-*")))


if __name__ == "__main__":
    unittest.main()
