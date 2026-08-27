from __future__ import annotations

import csv
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

    from thermal3dgs_sparse_ir import figures
except Exception as exc:
    DEPENDENCIES_AVAILABLE = False
    DEPENDENCY_ERROR = exc


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "figure dependencies unavailable: {}".format(DEPENDENCY_ERROR),
)
class FigureTests(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path, offset: int) -> None:
        array = np.zeros((12, 16), dtype=np.uint8)
        array[:, 8:] = 160
        array[3:9, 5:11] = np.clip(80 + offset, 0, 255)
        Image.fromarray(array, mode="L").save(str(path))

    def _make_method(self, root: Path, render_offset: int) -> Path:
        method = root
        renders = method / "renders"
        ground_truth = method / "gt"
        renders.mkdir(parents=True)
        ground_truth.mkdir(parents=True)
        for index, name in enumerate(("view_a.png", "view_b.png")):
            self._write_image(ground_truth / name, index)
            self._write_image(renders / name, index + render_offset)
        return method

    @staticmethod
    def _write_metrics(path: Path) -> None:
        rows = (
            ("base_sparse", 21.0),
            ("base_dense", 24.0),
            ("ours_sparse", 22.5),
            ("ours_dense", 25.2),
            ("base_clean", 24.0),
            ("base_noisy", 19.0),
            ("ours_clean", 25.2),
            ("ours_noisy", 23.0),
        )
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=("experiment", "num_views", "psnr"))
            writer.writeheader()
            for experiment, psnr in rows:
                writer.writerow({"experiment": experiment, "num_views": 2, "psnr": psnr})

    def _spec(self, root: Path, first: Path, second: Path, metrics: Path):
        return {
            "schema_version": 1,
            "kind": "thermal-figure-spec",
            "output_dir": "figures",
            "dpi": 80,
            "image_comparison": {
                "filenames": ["view_a.png", "view_b.png"],
                "ground_truth_label": "Ground truth",
                "methods": [
                    {"label": "Baseline", "method_dir": first.relative_to(root).as_posix()},
                    {"label": "Proposed", "method_dir": second.relative_to(root).as_posix()},
                ],
                "qualitative": {"output": "qualitative_comparison.png"},
                "absolute_error": {
                    "output": "absolute_error_maps.png",
                    "cmap": "magma",
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "colorbar_label": "Absolute normalized intensity error",
                },
                "edge": {
                    "output": "thermal_edge_comparison.png",
                    "operator": "sobel",
                    "cmap": "gray",
                    "vmin": 0.0,
                    "vmax": 0.5,
                    "colorbar_label": "Sobel magnitude",
                },
            },
            "sparse_view_curve": {
                "output": "sparse_view_performance.png",
                "title": "Sparse-view performance",
                "metrics_csv": metrics.relative_to(root).as_posix(),
                "y_field": "psnr",
                "x_label": "Training views (%)",
                "y_label": "PSNR (dB)",
                "series": [
                    {
                        "label": "Baseline",
                        "points": [
                            {"x": 12.5, "experiment": "base_sparse"},
                            {"x": 100.0, "experiment": "base_dense"},
                        ],
                    },
                    {
                        "label": "Proposed",
                        "points": [
                            {"x": 12.5, "experiment": "ours_sparse"},
                            {"x": 100.0, "experiment": "ours_dense"},
                        ],
                    },
                ],
            },
            "degradation_curve": {
                "output": "degradation_robustness.png",
                "title": "Degradation robustness",
                "metrics_csv": metrics.relative_to(root).as_posix(),
                "y_field": "psnr",
                "x_label": "Noise standard deviation",
                "y_label": "PSNR (dB)",
                "series": [
                    {
                        "label": "Baseline",
                        "points": [
                            {"x": 0.0, "experiment": "base_clean"},
                            {"x": 0.1, "experiment": "base_noisy"},
                        ],
                    },
                    {
                        "label": "Proposed",
                        "points": [
                            {"x": 0.0, "experiment": "ours_clean"},
                            {"x": 0.1, "experiment": "ours_noisy"},
                        ],
                    },
                ],
            },
        }

    def _fixture(self, root: Path):
        first = self._make_method(root / "baseline", 8)
        second = self._make_method(root / "proposed", 3)
        metrics = root / "metrics_summary.csv"
        self._write_metrics(metrics)
        spec_path = root / "figure_spec.json"
        spec_path.write_text(
            json.dumps(self._spec(root, first, second, metrics), indent=2), encoding="utf-8"
        )
        return spec_path, first, second

    def test_generates_five_pngs_and_hash_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            spec_path, _, _ = self._fixture(root)
            outputs = figures.make_figures(spec_path)
            self.assertEqual(
                set(outputs),
                {
                    "qualitative",
                    "absolute_error",
                    "edge",
                    "sparse_view_curve",
                    "degradation_curve",
                    "manifest",
                },
            )
            for key, path in outputs.items():
                self.assertTrue(path.is_file(), key)
            for key in set(outputs) - {"manifest"}:
                with Image.open(str(outputs[key])) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 0)
                    self.assertGreater(image.height, 0)
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "thermal-figure-manifest")
            self.assertEqual(len(manifest["outputs"]), 5)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["outputs"]))
            self.assertEqual(manifest["parameters"]["image_comparison"]["absolute_error"]["vmax"], 1.0)
            self.assertEqual(manifest["parameters"]["sparse_view_curve"]["series"][0]["color"], "#0072B2")
            inventory_inputs = [item for item in manifest["inputs"] if item["kind"] == "paired-image-inventory"]
            self.assertEqual(len(inventory_inputs), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in inventory_inputs))
            with self.assertRaises(FileExistsError):
                figures.make_figures(spec_path)

    def test_rejects_render_gt_pairing_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            spec_path, first, _ = self._fixture(root)
            self._write_image(first / "renders" / "extra.png", 0)
            with self.assertRaisesRegex(figures.FigureSpecError, "filename mismatch"):
                figures.make_figures(spec_path)

    def test_rejects_different_ground_truth_across_methods(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            spec_path, _, second = self._fixture(root)
            self._write_image(second / "gt" / "view_a.png", 50)
            with self.assertRaisesRegex(figures.FigureSpecError, "ground truth content differs"):
                figures.make_figures(spec_path)

    def test_curve_metadata_is_explicit_and_missing_experiment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            spec_path, _, _ = self._fixture(root)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["sparse_view_curve"]["series"][0]["points"][0]["experiment"] = "not_present"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(figures.FigureSpecError, "missing experiment"):
                figures.make_figures(spec_path)

    def test_rejects_output_path_traversal_and_nonincreasing_x(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            spec_path, _, _ = self._fixture(root)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["image_comparison"]["qualitative"]["output"] = "../outside.png"
            with self.assertRaisesRegex(figures.FigureSpecError, "normalized relative"):
                figures.validate_figure_spec(spec)
            spec["image_comparison"]["qualitative"]["output"] = "qualitative.png"
            spec["degradation_curve"]["series"][0]["points"][1]["x"] = -1.0
            with self.assertRaisesRegex(figures.FigureSpecError, "strictly increasing"):
                figures.validate_figure_spec(spec)

    def test_rejects_boolean_schema_version_and_unknown_legend_location(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            spec_path, _, _ = self._fixture(root)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["schema_version"] = True
            with self.assertRaisesRegex(figures.FigureSpecError, "schema_version"):
                figures.validate_figure_spec(spec)
            spec["schema_version"] = 1
            spec["sparse_view_curve"]["legend_location"] = "somewhere"
            with self.assertRaisesRegex(figures.FigureSpecError, "legend_location"):
                figures.validate_figure_spec(spec)

    def test_cli_parser_requires_explicit_spec(self):
        from tools.make_figures import build_parser

        args = build_parser().parse_args(("--spec", "configs/figures.json", "--overwrite"))
        self.assertEqual(args.spec, Path("configs/figures.json"))
        self.assertTrue(args.overwrite)


if __name__ == "__main__":
    unittest.main()
