from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir import degradations


DEPENDENCIES_AVAILABLE = degradations.dependencies_available()
if DEPENDENCIES_AVAILABLE:
    import numpy as np
    from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DependencyIndependentTests(unittest.TestCase):
    def test_view_seed_is_stable_and_view_specific(self) -> None:
        first = degradations.derive_view_seed(17, "view-001")
        self.assertEqual(first, degradations.derive_view_seed(17, "view-001"))
        self.assertNotEqual(first, degradations.derive_view_seed(17, "view-002"))
        self.assertNotEqual(first, degradations.derive_view_seed(18, "view-001"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**64)

    def test_combined_transforms_preserve_explicit_order(self) -> None:
        transforms = degradations.normalize_transforms(
            [
                {"name": "contrast", "alpha": 0.5},
                {"name": "gaussian-noise", "sigma": 0.03},
                {"name": "blur", "sigma": 1.0, "kernel_size": 3},
            ]
        )
        self.assertEqual(
            [transform["name"] for transform in transforms],
            ["contrast", "gaussian-noise", "blur"],
        )


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    f"NumPy/Pillow unavailable: {degradations.dependency_error_message()}",
)
class DegradationTests(unittest.TestCase):
    def test_fixed_data_range_preserves_shape_dtype_and_identity(self) -> None:
        image = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
        result = degradations.apply_degradations(
            image,
            [{"name": "gaussian-noise", "sigma": 0.0}],
            [0, 65535],
            seed=5,
        )
        np.testing.assert_array_equal(result, image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.dtype("uint16"))

    def test_combined_order_is_explicit_and_observable(self) -> None:
        image = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
        noise_then_contrast = degradations.apply_degradations(
            image,
            [
                {"name": "gaussian-noise", "sigma": 0.2},
                {"name": "contrast", "alpha": 0.3},
            ],
            [0, 255],
            seed=1234,
        )
        contrast_then_noise = degradations.apply_degradations(
            image,
            [
                {"name": "contrast", "alpha": 0.3},
                {"name": "gaussian-noise", "sigma": 0.2},
            ],
            [0, 255],
            seed=1234,
        )
        self.assertFalse(np.array_equal(noise_then_contrast, contrast_then_noise))

    def test_out_of_global_range_is_rejected(self) -> None:
        image = np.array([[0, 128]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "outside the fixed data_range"):
            degradations.apply_degradations(
                image,
                [{"name": "contrast", "alpha": 0.5}],
                [0, 100],
                seed=0,
            )

    def test_png_and_tiff_io_preserve_uint8_and_uint16(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for suffix in (".png", ".tiff"):
                for dtype, upper in ((np.uint8, 255), (np.uint16, 65535)):
                    label = np.dtype(dtype).name
                    source = root / f"source-{label}{suffix}"
                    output = root / f"output-{label}{suffix}"
                    array = np.linspace(0, upper, 35).reshape(5, 7).astype(dtype)
                    source_image = Image.fromarray(array)
                    try:
                        source_image.save(source)
                    finally:
                        source_image.close()
                    loaded, source_format = degradations._load_supported_image(source)
                    transformed = degradations.apply_degradations(
                        loaded,
                        [{"name": "blur", "sigma": 0.8, "kernel_size": 3}],
                        [0, upper],
                        seed=1,
                    )
                    degradations._save_image_atomic(transformed, output, source_format)
                    roundtrip, _ = degradations._load_supported_image(output)
                    self.assertEqual(roundtrip.shape, array.shape)
                    self.assertEqual(roundtrip.dtype, np.dtype(dtype))

    def test_jpeg_source_is_degraded_to_lossless_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "source"
            image_dir.mkdir()
            views = []
            for sequence_index, view_id in enumerate(("train", "val", "test")):
                relative_path = f"source/{view_id}.jpg"
                path = root / relative_path
                array = np.full((5, 7, 3), 40 + sequence_index * 60, dtype=np.uint8)
                source_image = Image.fromarray(array, mode="RGB")
                try:
                    source_image.save(path, format="JPEG", quality=95)
                finally:
                    source_image.close()
                views.append(
                    {
                        "id": view_id,
                        "relative_image_path": relative_path,
                        "sequence_index": sequence_index,
                        "camera_ref": None,
                        "source_sha256": _sha256(path),
                    }
                )

            dataset = {
                "schema_version": 1,
                "kind": "thermal-dataset",
                "dataset_id": "jpeg-fixture",
                "image_domain": "thermal_intensity",
                "calibrated_temperature": False,
                "data_range": [0, 255],
                "views": views,
            }
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            split = {
                "schema_version": 1,
                "kind": "sparse-split",
                "split_id": "jpeg-fixture-sparse",
                "dataset_id": "jpeg-fixture",
                "dataset_sha256": degradations.manifest_sha256(dataset),
                "train_pool": ["train"],
                "val": ["val"],
                "test": ["test"],
                "requested_ratio": 1.0,
                "actual_ratio": 1.0,
                "train_selected": ["train"],
                "method": "nested-random",
                "seed": 7,
                "generator_version": "test/1",
                "source_base_split_sha256": "0" * 64,
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")

            manifest_path = degradations.degrade_dataset(
                dataset_path,
                split_path,
                root / "degraded",
                [{"name": "gaussian-noise", "sigma": 0.0}],
                global_seed=42,
            )
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = {record["id"]: record for record in result["views"]}

            train_record = records["train"]
            self.assertEqual(train_record["input_relative_image_path"], "source/train.jpg")
            self.assertEqual(
                train_record["output_relative_image_path"], "images/source/train.png"
            )
            generated_path = root / "degraded" / train_record["output_relative_image_path"]
            with Image.open(generated_path) as generated:
                self.assertEqual(generated.format, "PNG")
                self.assertEqual(generated.mode, "RGB")
            self.assertEqual(_sha256(generated_path), train_record["output_sha256"])

            for held_out_id in ("val", "test"):
                held_out = records[held_out_id]
                self.assertFalse(held_out["degraded"])
                self.assertEqual(held_out["output_path_base"], "dataset")
                self.assertEqual(
                    held_out["output_relative_image_path"], f"source/{held_out_id}.jpg"
                )
                self.assertFalse(
                    (root / "degraded/images/source" / f"{held_out_id}.png").exists()
                )

    def test_dataset_pipeline_degrades_train_only_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "source"
            image_dir.mkdir()
            arrays = {
                "train": np.arange(30, dtype=np.uint16).reshape(5, 6) * 100,
                "val": np.full((5, 6), 1200, dtype=np.uint16),
                "test": np.full((5, 6), 2200, dtype=np.uint16),
                "unused": np.full((5, 6), 3200, dtype=np.uint16),
            }
            views = []
            for sequence_index, (view_id, array) in enumerate(arrays.items()):
                relative_path = f"source/{view_id}.tiff"
                path = root / relative_path
                source_image = Image.fromarray(array)
                try:
                    source_image.save(path, format="TIFF")
                finally:
                    source_image.close()
                views.append(
                    {
                        "id": view_id,
                        "relative_image_path": relative_path,
                        "sequence_index": sequence_index,
                        "camera_ref": None,
                        "source_sha256": _sha256(path),
                    }
                )

            dataset = {
                "schema_version": 1,
                "kind": "thermal-dataset",
                "dataset_id": "fixture",
                "image_domain": "thermal_intensity",
                "calibrated_temperature": False,
                "data_range": [0, 65535],
                "views": views,
            }
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            split = {
                "schema_version": 1,
                "kind": "sparse-split",
                "split_id": "fixture-sparse",
                "dataset_id": "fixture",
                "dataset_sha256": degradations.manifest_sha256(dataset),
                "train_pool": ["train", "unused"],
                "val": ["val"],
                "test": ["test"],
                "requested_ratio": 0.5,
                "actual_ratio": 0.5,
                "train_selected": ["train"],
                "method": "nested-random",
                "seed": 7,
                "generator_version": "test/1",
                "source_base_split_sha256": "0" * 64,
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")

            output_dir = root / "degraded"
            manifest_path = degradations.degrade_dataset(
                dataset_path,
                split_path,
                output_dir,
                [{"name": "gaussian-noise", "sigma": 0.03}],
                global_seed=42,
            )
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = {record["id"]: record for record in result["views"]}

            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(result["kind"], "degradation-manifest")
            self.assertEqual(result["train_selected"], ["train"])
            self.assertTrue(records["train"]["degraded"])
            self.assertFalse(records["val"]["degraded"])
            self.assertFalse(records["test"]["degraded"])
            self.assertEqual(
                records["val"]["input_relative_image_path"],
                records["val"]["output_relative_image_path"],
            )
            self.assertEqual(records["val"]["input_sha256"], records["val"]["output_sha256"])
            self.assertEqual(records["test"]["input_sha256"], records["test"]["output_sha256"])
            self.assertFalse((output_dir / "images/source/val.tiff").exists())
            self.assertFalse((output_dir / "images/source/test.tiff").exists())
            self.assertFalse((output_dir / "images/source/unused.tiff").exists())

            generated_path = output_dir / records["train"]["output_relative_image_path"]
            generated_array, _ = degradations._load_supported_image(generated_path)
            self.assertEqual(generated_array.shape, arrays["train"].shape)
            self.assertEqual(generated_array.dtype, np.dtype("uint16"))
            self.assertEqual(_sha256(generated_path), records["train"]["output_sha256"])
            self.assertEqual(result["manifest_sha256"], degradations.manifest_sha256(result))

            replica_manifest = degradations.degrade_dataset(
                dataset_path,
                split_path,
                root / "replica",
                [{"name": "gaussian-noise", "sigma": 0.03}],
                global_seed=42,
            )
            replica = json.loads(replica_manifest.read_text(encoding="utf-8"))
            replica_train = next(
                record for record in replica["views"] if record["id"] == "train"
            )
            self.assertEqual(records["train"]["view_seed"], replica_train["view_seed"])
            self.assertEqual(records["train"]["output_sha256"], replica_train["output_sha256"])

            with self.assertRaises(FileExistsError):
                degradations.degrade_dataset(
                    dataset_path,
                    split_path,
                    output_dir,
                    [{"name": "gaussian-noise", "sigma": 0.03}],
                    global_seed=42,
                )


if __name__ == "__main__":
    unittest.main()
