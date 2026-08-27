from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir.contracts import (  # noqa: E402
    ManifestValidationError,
    load_sparse_split_manifest,
    manifest_sha256,
    validate_base_split_manifest,
    validate_dataset_manifest,
    write_manifest,
)
from thermal3dgs_sparse_ir.splits import (  # noqa: E402
    DEFAULT_RATIOS,
    create_sparse_split_files,
    generate_sparse_splits,
)


def make_dataset(count: int = 16) -> dict:
    views = []
    for index in range(count):
        views.append(
            {
                "id": "view-{:02d}".format(index),
                "relative_image_path": "thermal/frame-{:02d}.png".format(100 - index),
                "sequence_index": index * 10,
                "camera_ref": "colmap-image-id:{}".format(9000 + index),
                "source_sha256": None,
            }
        )
    return {
        "schema_version": 1,
        "kind": "thermal-dataset",
        "dataset_id": "test-scene",
        "image_domain": "thermal_intensity",
        "calibrated_temperature": False,
        "data_range": [0, 1],
        "views": views,
    }


def make_base(dataset: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": "base-split",
        "split_id": "official-v1",
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": manifest_sha256(dataset),
        "train_pool": [view["id"] for view in dataset["views"][:12]],
        "val": [view["id"] for view in dataset["views"][12:14]],
        "test": [view["id"] for view in dataset["views"][14:]],
    }


class ContractTests(unittest.TestCase):
    def test_canonical_hash_ignores_manifest_and_enumeration_order(self) -> None:
        dataset = make_dataset()
        reordered = copy.deepcopy(dataset)
        reordered["views"].reverse()
        reordered["manifest_sha256"] = manifest_sha256(reordered)
        self.assertEqual(manifest_sha256(dataset), manifest_sha256(reordered))

        base = make_base(dataset)
        reordered_base = copy.deepcopy(base)
        reordered_base["train_pool"].reverse()
        reordered_base["val"].reverse()
        self.assertEqual(manifest_sha256(base), manifest_sha256(reordered_base))

    def test_rejects_absolute_or_parent_image_paths(self) -> None:
        for invalid in ("C:/thermal/a.png", "/thermal/a.png", "../thermal/a.png"):
            with self.subTest(path=invalid):
                dataset = make_dataset()
                dataset["views"][0]["relative_image_path"] = invalid
                with self.assertRaises(ManifestValidationError):
                    validate_dataset_manifest(dataset)

    def test_rejects_duplicate_views_and_overlapping_base_sets(self) -> None:
        dataset = make_dataset()
        dataset["views"][1]["id"] = dataset["views"][0]["id"]
        with self.assertRaises(ManifestValidationError):
            validate_dataset_manifest(dataset)

        dataset = make_dataset()
        base = make_base(dataset)
        base["test"].append(base["train_pool"][0])
        with self.assertRaises(ManifestValidationError):
            validate_base_split_manifest(base, dataset)

    def test_dataset_hash_reference_is_verified(self) -> None:
        dataset = make_dataset()
        base = make_base(dataset)
        base["dataset_sha256"] = "0" * 64
        with self.assertRaises(ManifestValidationError):
            validate_base_split_manifest(base, dataset)

    def test_base_split_must_assign_every_dataset_view(self) -> None:
        dataset = make_dataset()
        base = make_base(dataset)
        base["test"].pop()
        with self.assertRaises(ManifestValidationError):
            validate_base_split_manifest(base, dataset)


class GenerationTests(unittest.TestCase):
    def test_nested_random_is_deterministic_nested_and_order_independent(self) -> None:
        dataset = make_dataset()
        base = make_base(dataset)
        first = generate_sparse_splits(dataset, base, seed=2026)
        second = generate_sparse_splits(dataset, base, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(
            [manifest["requested_ratio"] for manifest in first], list(DEFAULT_RATIOS)
        )

        selected = [set(manifest["train_selected"]) for manifest in first]
        for denser, sparser in zip(selected, selected[1:]):
            self.assertTrue(sparser <= denser)
        self.assertEqual(first[0]["train_selected"], sorted(base["train_pool"]))
        for manifest in first:
            self.assertEqual(manifest["val"], sorted(base["val"]))
            self.assertEqual(manifest["test"], sorted(base["test"]))

        shuffled_dataset = copy.deepcopy(dataset)
        shuffled_dataset["views"] = list(reversed(shuffled_dataset["views"]))
        shuffled_base = copy.deepcopy(base)
        shuffled_base["train_pool"] = list(reversed(shuffled_base["train_pool"]))
        shuffled_base["val"] = list(reversed(shuffled_base["val"]))
        self.assertEqual(
            first, generate_sparse_splits(shuffled_dataset, shuffled_base, seed=2026)
        )

    def test_seed_changes_nested_random_selection(self) -> None:
        dataset = make_dataset()
        base = make_base(dataset)
        first = generate_sparse_splits(dataset, base, ratios=(0.25,), seed=1)[0]
        second = generate_sparse_splits(dataset, base, ratios=(0.25,), seed=2)[0]
        self.assertNotEqual(first["train_selected"], second["train_selected"])

    def test_trajectory_uniform_uses_sequence_index_and_is_nested(self) -> None:
        dataset = make_dataset()
        base = make_base(dataset)
        splits = generate_sparse_splits(
            dataset, base, method="trajectory-uniform", seed=88
        )
        selected = [set(manifest["train_selected"]) for manifest in splits]
        for denser, sparser in zip(selected, selected[1:]):
            self.assertTrue(sparser <= denser)
        # With 12 train views, the 12.5% target rounds to two and the greedy
        # uniform prefix selects the two trajectory endpoints.
        self.assertEqual(selected[-1], {"view-00", "view-11"})

    def test_invalid_ratio_is_rejected(self) -> None:
        dataset = make_dataset()
        base = make_base(dataset)
        for ratios in ((), (0.0,), (1.1,), (0.5, 0.5)):
            with self.subTest(ratios=ratios):
                with self.assertRaises(ManifestValidationError):
                    generate_sparse_splits(dataset, base, ratios=ratios)

    def test_cli_service_writes_loadable_hashed_manifests(self) -> None:
        dataset = make_dataset()
        base = make_base(dataset)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = write_manifest(root / "dataset.json", dataset)
            base_path = write_manifest(root / "base.json", base)
            paths = create_sparse_split_files(
                dataset_path,
                base_path,
                output_dir=root / "splits",
                method="nested-random",
                seed=7,
            )
            self.assertEqual(len(paths), 4)
            for path in paths:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(raw["manifest_sha256"], manifest_sha256(raw))
                loaded = load_sparse_split_manifest(path, dataset, base)
                self.assertEqual(loaded, raw)


if __name__ == "__main__":
    unittest.main()
