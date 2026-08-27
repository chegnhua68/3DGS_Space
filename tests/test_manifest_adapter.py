import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from integrations.thermal3dgs.manifest_adapter import load_colmap_selection
from thermal3dgs_sparse_ir.contracts import (
    manifest_sha256,
    with_manifest_sha256,
    write_manifest,
)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


class ManifestAdapterTests(unittest.TestCase):
    def _write_manifests(self, root):
        images = root / "images"
        images.mkdir()
        payloads = {"000.png": b"zero", "001.png": b"one", "002.png": b"two"}
        views = []
        for index, (name, payload) in enumerate(payloads.items()):
            (images / name).write_bytes(payload)
            views.append(
                {
                    "id": name,
                    "relative_image_path": "images/" + name,
                    "sequence_index": index,
                    "camera_ref": name,
                    "source_sha256": _digest(payload),
                }
            )
        dataset = with_manifest_sha256(
            {
                "schema_version": 1,
                "kind": "thermal-dataset",
                "dataset_id": "fixture",
                "image_domain": "thermal_intensity",
                "calibrated_temperature": False,
                "data_range": [0, 255],
                "views": views,
            }
        )
        dataset_path = root / "dataset.json"
        write_manifest(dataset_path, dataset)
        split = with_manifest_sha256(
            {
                "schema_version": 1,
                "kind": "sparse-split",
                "split_id": "fixture:half",
                "dataset_id": "fixture",
                "dataset_sha256": manifest_sha256(dataset),
                "train_pool": ["001.png", "002.png"],
                "val": [],
                "test": ["000.png"],
                "requested_ratio": 0.5,
                "actual_ratio": 0.5,
                "train_selected": ["002.png"],
                "method": "nested-random",
                "seed": 7,
                "generator_version": "test/1",
                "source_base_split_sha256": "0" * 64,
            }
        )
        split_path = root / "split.json"
        write_manifest(split_path, split)
        return dataset, dataset_path, split, split_path

    def test_clean_selection_excludes_unselected_training_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, dataset_path, _, split_path = self._write_manifests(root)
            selection = load_colmap_selection(dataset_path, split_path)
            self.assertEqual(selection.train_view_ids, ["002.png"])
            self.assertEqual(selection.test_view_ids, ["000.png"])
            self.assertEqual(set(selection.active_camera_refs), {"000.png", "002.png"})
            self.assertEqual(selection.fid_by_camera_ref["002.png"], 1.0)

    def test_degraded_train_and_clean_test_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, dataset_path, split, split_path = self._write_manifests(root)
            derived = root / "derived"
            (derived / "images").mkdir(parents=True)
            degraded_payload = b"degraded-two"
            (derived / "images" / "002.png").write_bytes(degraded_payload)
            manifest = {
                "schema_version": 1,
                "kind": "degradation-manifest",
                "generator_version": "test/1",
                "dataset_id": "fixture",
                "dataset_manifest_sha256": manifest_sha256(dataset),
                "dataset_manifest_file_sha256": _digest(dataset_path.read_bytes()),
                "split_id": split["split_id"],
                "split_manifest_sha256": manifest_sha256(split),
                "split_manifest_file_sha256": _digest(split_path.read_bytes()),
                "data_range": [0, 255],
                "global_seed": 4,
                "seed_derivation": "test",
                "transforms": [{"name": "gaussian-noise", "sigma": 0.03}],
                "train_selected": ["002.png"],
                "val": [],
                "test": ["000.png"],
                "views": [
                    {
                        "id": "002.png",
                        "role": "train_selected",
                        "degraded": True,
                        "input_relative_image_path": "images/002.png",
                        "input_sha256": _digest(b"two"),
                        "output_relative_image_path": "images/002.png",
                        "output_path_base": "degradation_output",
                        "output_sha256": _digest(degraded_payload),
                        "view_seed": 1,
                        "shape": [1, 1],
                        "dtype": "uint8",
                    },
                    {
                        "id": "000.png",
                        "role": "test",
                        "degraded": False,
                        "input_relative_image_path": "images/000.png",
                        "input_sha256": _digest(b"zero"),
                        "output_relative_image_path": "images/000.png",
                        "output_path_base": "dataset",
                        "output_sha256": _digest(b"zero"),
                        "view_seed": None,
                    },
                ],
            }
            manifest["manifest_sha256"] = manifest_sha256(manifest)
            degradation_path = derived / "degradation.json"
            degradation_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = load_colmap_selection(
                dataset_path, split_path, degradation_path
            )
            self.assertEqual(
                selection.path_by_camera_ref["002.png"],
                derived / "images" / "002.png",
            )
            self.assertEqual(
                selection.path_by_camera_ref["000.png"], root / "images" / "000.png"
            )

    def test_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, dataset_path, _, split_path = self._write_manifests(root)
            (root / "images" / "002.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_colmap_selection(dataset_path, split_path)

    def test_empty_validation_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, dataset_path, _, split_path = self._write_manifests(root)
            with self.assertRaisesRegex(ValueError, "partition 'val' is empty"):
                load_colmap_selection(
                    dataset_path, split_path, evaluation_partition="val"
                )

    def test_fid_uses_full_sequence_extent_with_nonzero_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, dataset_path, split, split_path = self._write_manifests(root)
            for view, sequence_index in zip(dataset["views"], (10, 20, 30)):
                view["sequence_index"] = sequence_index
            dataset = with_manifest_sha256(dataset)
            write_manifest(dataset_path, dataset)
            split["dataset_sha256"] = manifest_sha256(dataset)
            split = with_manifest_sha256(split)
            write_manifest(split_path, split)

            selection = load_colmap_selection(dataset_path, split_path)
            self.assertEqual(selection.fid_by_camera_ref["000.png"], 0.0)
            self.assertEqual(selection.fid_by_camera_ref["002.png"], 1.0)

    def test_uint16_training_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, dataset_path, split, split_path = self._write_manifests(root)
            dataset["data_range"] = [0, 65535]
            dataset = with_manifest_sha256(dataset)
            write_manifest(dataset_path, dataset)
            split["dataset_sha256"] = manifest_sha256(dataset)
            split = with_manifest_sha256(split)
            write_manifest(split_path, split)

            with self.assertRaisesRegex(ValueError, "8-bit"):
                load_colmap_selection(dataset_path, split_path)


if __name__ == "__main__":
    unittest.main()
