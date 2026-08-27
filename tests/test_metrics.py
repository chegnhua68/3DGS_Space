from __future__ import annotations

import csv
import math
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

    def test_collection_writes_three_outputs_and_marks_skips_unavailable(self):
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
                set(outputs), {"summary_csv", "summary_md", "per_view_csv"}
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


if __name__ == "__main__":
    unittest.main()
