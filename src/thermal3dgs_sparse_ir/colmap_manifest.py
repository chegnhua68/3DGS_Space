"""Create neutral dataset and frozen base-split manifests from COLMAP views."""

import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple, Union

from .contracts import (
    BASE_SPLIT_KIND,
    DATASET_KIND,
    SCHEMA_VERSION,
    ManifestValidationError,
    manifest_sha256,
    validate_base_split_manifest,
    validate_dataset_manifest,
    with_manifest_sha256,
    write_manifest,
)


RegisteredImage = Tuple[int, str]


def _read_exact(stream: BinaryIO, byte_count: int, label: str) -> bytes:
    value = stream.read(byte_count)
    if len(value) != byte_count:
        raise ValueError("truncated COLMAP binary while reading {}".format(label))
    return value


def _read_c_string(stream: BinaryIO, label: str) -> str:
    value = bytearray()
    while True:
        byte = _read_exact(stream, 1, label)
        if byte == b"\0":
            break
        value.extend(byte)
        if len(value) > 1024 * 1024:
            raise ValueError("unreasonably long COLMAP image name")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("COLMAP image name is not UTF-8") from exc


def read_registered_images_binary(path: Union[Path, str]) -> List[RegisteredImage]:
    """Read ``(image_id, image_name)`` records from COLMAP ``images.bin``."""

    source = Path(path)
    records = []
    with source.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8, "image count"))[0]
        for index in range(count):
            header = struct.unpack(
                "<idddddddi",
                _read_exact(stream, struct.calcsize("<idddddddi"), "image header"),
            )
            image_id = int(header[0])
            name = _read_c_string(stream, "image name")
            point_count = struct.unpack(
                "<Q", _read_exact(stream, 8, "2D point count")
            )[0]
            point_bytes = point_count * struct.calcsize("<ddq")
            _read_exact(stream, point_bytes, "2D points for image {}".format(index))
            records.append((image_id, name))
        if stream.read(1):
            raise ValueError("COLMAP images.bin has trailing bytes")
    return records


def _next_metadata_line(stream: TextIO) -> Optional[str]:
    for line in stream:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None


def read_registered_images_text(path: Union[Path, str]) -> List[RegisteredImage]:
    """Read ``(image_id, image_name)`` records from COLMAP ``images.txt``."""

    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        while True:
            metadata = _next_metadata_line(stream)
            if metadata is None:
                break
            fields = metadata.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("invalid COLMAP image metadata line: {}".format(metadata))
            try:
                image_id = int(fields[0])
                for value in fields[1:8]:
                    float(value)
                int(fields[8])
            except ValueError as exc:
                raise ValueError(
                    "invalid numeric field in COLMAP image metadata: {}".format(metadata)
                ) from exc
            records.append((image_id, fields[9]))
            # The following line contains zero or more POINTS2D triples and may
            # legitimately be blank. Its contents are not needed by this tool.
            stream.readline()
    return records


def read_registered_images(sparse_dir: Union[Path, str]) -> List[RegisteredImage]:
    directory = Path(sparse_dir)
    binary_path = directory / "images.bin"
    text_path = directory / "images.txt"
    if binary_path.is_file():
        records = read_registered_images_binary(binary_path)
    elif text_path.is_file():
        records = read_registered_images_text(text_path)
    else:
        raise FileNotFoundError(
            "COLMAP camera list not found: expected {} or {}".format(
                binary_path, text_path
            )
        )
    if not records:
        raise ValueError("COLMAP camera list contains no registered images")
    return records


def sha256_file(path: Union[Path, str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ValueError("{} must be a non-empty relative POSIX path".format(label))
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("{} must be a normalized relative POSIX path".format(label))
    return value


def _normalized_colmap_name(name: str) -> str:
    return _relative_posix_path(name.replace("\\", "/"), "COLMAP image name")


def _resolve_below(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes dataset root: {}".format(relative_path)) from exc
    return candidate


def _validate_holdout(every: int, offset: int, label: str) -> None:
    if isinstance(every, bool) or not isinstance(every, int) or every < 0:
        raise ValueError("{}_every must be a non-negative integer".format(label))
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("{}_offset must be a non-negative integer".format(label))
    if every == 0 and offset != 0:
        raise ValueError("{}_offset must be zero when holdout is disabled".format(label))
    if every > 0 and offset >= every:
        raise ValueError("{}_offset must be smaller than {}_every".format(label, label))


def build_colmap_manifests(
    dataset_root: Union[Path, str],
    dataset_id: str,
    data_range: Sequence[float],
    images_dir: str = "images",
    sparse_dir: str = "sparse/0",
    image_domain: str = "thermal_intensity",
    calibrated_temperature: bool = False,
    test_every: int = 8,
    test_offset: int = 0,
    val_every: int = 0,
    val_offset: int = 0,
    split_id: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Build dataset and base split manifests matching upstream camera order."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("dataset root does not exist: {}".format(root))
    images_dir = _relative_posix_path(images_dir, "images_dir")
    sparse_dir = _relative_posix_path(sparse_dir, "sparse_dir")
    _validate_holdout(test_every, test_offset, "test")
    _validate_holdout(val_every, val_offset, "val")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    if len(data_range) != 2:
        raise ValueError("data_range must contain LOW and HIGH")

    registered = read_registered_images(_resolve_below(root, sparse_dir))
    normalized = [(image_id, _normalized_colmap_name(name)) for image_id, name in registered]
    names = [name for _, name in normalized]
    if len(names) != len(set(names)):
        raise ValueError("COLMAP camera list contains duplicate image names")
    stems = [PurePosixPath(name).stem for name in names]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "COLMAP image names must have unique stems for Thermal3D-GS compatibility"
        )
    normalized.sort(key=lambda item: (PurePosixPath(item[1]).stem, item[1], item[0]))

    views = []
    for sequence_index, (image_id, name) in enumerate(normalized):
        relative_path = (PurePosixPath(images_dir) / PurePosixPath(name)).as_posix()
        image_path = _resolve_below(root, relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                "registered COLMAP image does not exist: {}".format(image_path)
            )
        views.append(
            {
                "id": name,
                "relative_image_path": relative_path,
                "sequence_index": sequence_index,
                "camera_ref": name,
                "source_sha256": sha256_file(image_path),
            }
        )

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "dataset_id": dataset_id,
        "image_domain": image_domain,
        "calibrated_temperature": calibrated_temperature,
        "data_range": [float(data_range[0]), float(data_range[1])],
        "views": views,
    }
    validate_dataset_manifest(dataset)
    dataset = with_manifest_sha256(dataset)

    test = []
    remaining = []
    for index, view in enumerate(views):
        if test_every > 0 and index % test_every == test_offset:
            test.append(view["id"])
        else:
            remaining.append(view["id"])
    val = []
    train_pool = []
    for index, view_id in enumerate(remaining):
        if val_every > 0 and index % val_every == val_offset:
            val.append(view_id)
        else:
            train_pool.append(view_id)
    if not train_pool:
        raise ValueError("holdout settings leave no training views")

    if split_id is None:
        split_id = "{}:colmap-test{}".format(dataset_id, test_every)
        if val_every:
            split_id += "-val{}".format(val_every)
    base_split = {
        "schema_version": SCHEMA_VERSION,
        "kind": BASE_SPLIT_KIND,
        "split_id": split_id,
        "dataset_id": dataset_id,
        "dataset_sha256": manifest_sha256(dataset),
        "train_pool": train_pool,
        "val": val,
        "test": test,
    }
    validate_base_split_manifest(base_split, dataset)
    base_split = with_manifest_sha256(base_split)
    return dataset, base_split


def create_colmap_manifest_files(
    dataset_root: Union[Path, str],
    dataset_id: str,
    data_range: Sequence[float],
    dataset_manifest_name: str = "dataset_manifest.v1.json",
    base_split_name: str = "base_split.v1.json",
    overwrite: bool = False,
    **kwargs: object
) -> Tuple[Path, Path]:
    """Build and atomically write both manifests inside the dataset root."""

    root = Path(dataset_root).resolve()
    dataset_manifest_name = _relative_posix_path(
        dataset_manifest_name, "dataset_manifest_name"
    )
    base_split_name = _relative_posix_path(base_split_name, "base_split_name")
    dataset_path = _resolve_below(root, dataset_manifest_name)
    base_path = _resolve_below(root, base_split_name)
    existing = [path for path in (dataset_path, base_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing manifest(s): {}".format(
                ", ".join(str(path) for path in existing)
            )
        )
    dataset, base_split = build_colmap_manifests(
        root, dataset_id, data_range, **kwargs
    )
    write_manifest(dataset_path, dataset)
    write_manifest(base_path, base_split)
    return dataset_path, base_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create versioned dataset and base-split manifests from COLMAP."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--data-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        required=True,
        help="fixed encoded intensity range, e.g. 0 255 for uint8",
    )
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument(
        "--image-domain",
        choices=("thermal_intensity", "temperature"),
        default="thermal_intensity",
    )
    parser.add_argument("--calibrated-temperature", action="store_true")
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--split-id")
    parser.add_argument("--dataset-manifest-name", default="dataset_manifest.v1.json")
    parser.add_argument("--base-split-name", default="base_split.v1.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    options = {
        "images_dir": args.images_dir,
        "sparse_dir": args.sparse_dir,
        "image_domain": args.image_domain,
        "calibrated_temperature": args.calibrated_temperature,
        "test_every": args.test_every,
        "test_offset": args.test_offset,
        "val_every": args.val_every,
        "val_offset": args.val_offset,
        "split_id": args.split_id,
    }
    try:
        if args.dry_run:
            dataset, base_split = build_colmap_manifests(
                args.dataset_root, args.dataset_id, args.data_range, **options
            )
            print("dataset_sha256={}".format(manifest_sha256(dataset)))
            print("views={}".format(len(dataset["views"])))
            print("train_pool={}".format(len(base_split["train_pool"])))
            print("val={}".format(len(base_split["val"])))
            print("test={}".format(len(base_split["test"])))
        else:
            paths = create_colmap_manifest_files(
                args.dataset_root,
                args.dataset_id,
                args.data_range,
                dataset_manifest_name=args.dataset_manifest_name,
                base_split_name=args.base_split_name,
                overwrite=args.overwrite,
                **options
            )
            for path in paths:
                print(path)
    except (ManifestValidationError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


__all__ = [
    "build_colmap_manifests",
    "create_colmap_manifest_files",
    "read_registered_images",
    "read_registered_images_binary",
    "read_registered_images_text",
    "sha256_file",
]


if __name__ == "__main__":
    sys.exit(main())
