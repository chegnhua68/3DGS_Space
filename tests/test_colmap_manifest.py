import hashlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir.colmap_manifest import (
    build_colmap_manifests,
    read_registered_images_binary,
    read_registered_images_text,
)
from thermal3dgs_sparse_ir.contracts import (
    manifest_sha256,
    validate_base_split_manifest,
    validate_dataset_manifest,
)


def _write_images_text(path, names):
    lines = ["# Image list with two lines of data per image"]
    for image_id, name in enumerate(names, start=1):
        lines.append("{} 1 0 0 0 0 0 0 1 {}".format(image_id, name))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_images_binary(path, names):
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(names)))
        for image_id, name in enumerate(names, start=1):
            stream.write(
                struct.pack(
                    "<idddddddi",
                    image_id,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1,
                )
            )
            stream.write(name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", 0))


class ColmapManifestTests(unittest.TestCase):
    def test_text_and_binary_readers_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["000.png", "folder/001.png"]
            text_path = root / "images.txt"
            binary_path = root / "images.bin"
            _write_images_text(text_path, names)
            _write_images_binary(binary_path, names)
            expected = [(1, names[0]), (2, names[1])]
            self.assertEqual(read_registered_images_text(text_path), expected)
            self.assertEqual(read_registered_images_binary(binary_path), expected)

    def test_builds_upstream_eval8_split_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "sparse" / "0").mkdir(parents=True)
            names = ["{:03d}.png".format(index) for index in range(10)]
            _write_images_text(root / "sparse" / "0" / "images.txt", names)
            for index, name in enumerate(names):
                (root / "images" / name).write_bytes(bytes([index, index + 1]))

            dataset, base_split = build_colmap_manifests(
                root, "fixture", (0, 255), test_every=8
            )

            validate_dataset_manifest(dataset)
            validate_base_split_manifest(base_split, dataset)
            self.assertEqual(base_split["test"], ["000.png", "008.png"])
            self.assertEqual(len(base_split["train_pool"]), 8)
            self.assertEqual(base_split["val"], [])
            first = next(view for view in dataset["views"] if view["id"] == "000.png")
            self.assertEqual(first["sequence_index"], 0)
            self.assertEqual(
                first["source_sha256"], hashlib.sha256(bytes([0, 1])).hexdigest()
            )
            self.assertEqual(base_split["dataset_sha256"], manifest_sha256(dataset))

    def test_validation_holdout_is_removed_from_training_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "sparse" / "0").mkdir(parents=True)
            names = ["{:03d}.png".format(index) for index in range(12)]
            _write_images_text(root / "sparse" / "0" / "images.txt", names)
            for name in names:
                (root / "images" / name).write_bytes(b"image")
            _, split = build_colmap_manifests(
                root, "fixture", (0, 255), test_every=4, val_every=3
            )
            train = set(split["train_pool"])
            val = set(split["val"])
            test = set(split["test"])
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
            self.assertEqual(len(train | val | test), len(names))

    def test_missing_registered_image_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "sparse" / "0").mkdir(parents=True)
            _write_images_text(root / "sparse" / "0" / "images.txt", ["missing.png"])
            with self.assertRaises(FileNotFoundError):
                build_colmap_manifests(root, "fixture", (0, 255))


if __name__ == "__main__":
    unittest.main()
