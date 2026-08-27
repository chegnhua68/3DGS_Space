"""Deterministic, manifest-driven degradations for thermal image datasets.

The public entry point is :func:`degrade_dataset`.  ``data_range`` is the
encoded intensity range of the source files (normally ``[0, 255]`` for uint8
or ``[0, 65535]`` for uint16), not the post-normalization range.  Image values
are mapped through that one declared range; no image is independently rescaled.
This is important for thermal-intensity experiments, where per-image
normalization would destroy cross-view radiometric meaning.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .contracts import (
    load_dataset_manifest,
    load_sparse_split_manifest,
    manifest_sha256 as _contract_manifest_sha256,
)

# Imports are deliberately lazy.  A mismatched NumPy binary can terminate the
# interpreter at native-extension import time, before Python can raise an
# ImportError.  The one-time subprocess probe keeps CLI help and diagnostics
# usable even in that environment.
np: Any = None
Image: Any = None
_DEPENDENCIES_CHECKED = False
_DEPENDENCY_ERROR: Optional[str] = None


SCHEMA_VERSION = 1
GENERATOR_VERSION = "thermal3dgs-sparse-ir-degradation/1.0"
SUPPORTED_TRANSFORMS = frozenset({"gaussian-noise", "contrast", "blur"})
LOSSLESS_OUTPUT_SUFFIXES = frozenset({".png", ".tif", ".tiff"})
SUPPORTED_SOURCE_SUFFIXES = LOSSLESS_OUTPUT_SUFFIXES | frozenset({".jpg", ".jpeg"})
_SHA256_HEX_LENGTH = 64


class ManifestError(ValueError):
    """Raised when an input manifest violates the degradation contract."""


class DependencyUnavailableError(RuntimeError):
    """Raised when NumPy or Pillow cannot be imported in the environment."""


def dependencies_available() -> bool:
    """Return whether the image-processing dependencies imported successfully."""

    _load_dependencies()
    return np is not None and Image is not None


def dependency_error_message() -> str:
    """Describe missing or broken image-processing dependencies."""

    _load_dependencies()
    return _DEPENDENCY_ERROR or ""


def _load_dependencies() -> None:
    global Image, np, _DEPENDENCIES_CHECKED, _DEPENDENCY_ERROR
    if _DEPENDENCIES_CHECKED:
        return
    _DEPENDENCIES_CHECKED = True

    probe = "import numpy; from PIL import Image"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _DEPENDENCY_ERROR = f"dependency import probe failed: {exc}"
        return
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or completed.stdout).split())
        if len(detail) > 500:
            detail = detail[:497] + "..."
        _DEPENDENCY_ERROR = (
            f"NumPy/Pillow import probe exited with status {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
        return

    try:
        np = importlib.import_module("numpy")
        Image = importlib.import_module("PIL.Image")
    except (ImportError, OSError) as exc:
        np = None
        Image = None
        _DEPENDENCY_ERROR = f"NumPy/Pillow import failed: {exc}"


def _require_dependencies() -> None:
    _load_dependencies()
    if np is None or Image is None:
        detail = dependency_error_message() or "unknown import error"
        raise DependencyUnavailableError(
            "Infrared degradation requires working NumPy and Pillow imports "
            f"({detail})."
        )


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest with the canonical project manifest contract."""

    return _contract_manifest_sha256(dict(manifest))


def derive_view_seed(global_seed: int, view_id: str) -> int:
    """Derive a stable uint64 seed from ``(global_seed, view_id)`` using SHA256."""

    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    if not isinstance(view_id, str) or not view_id:
        raise ValueError("view_id must be a non-empty string")
    payload = str(global_seed).encode("utf-8") + b"\0" + view_id.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ManifestError(f"{label} is missing required fields: {', '.join(missing)}")


def _validate_sha256(value: Any, label: str, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestError(f"{label} must be a lowercase SHA256 hex digest")


def _validate_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label} must be a non-empty relative path")
    if "\\" in value:
        raise ManifestError(f"{label} must use '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"{label} must be a normalized path below the dataset root")
    if path.parts and path.parts[0].endswith(":"):
        raise ManifestError(f"{label} must not contain a drive prefix")
    return path.as_posix()


def _resolve_below(root: Path, relative_path: str, label: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"{label} escapes its declared root: {relative_path}") from exc
    return candidate


def validate_dataset_manifest(
    manifest: Mapping[str, Any]
) -> Dict[str, Mapping[str, Any]]:
    """Validate a v1 thermal dataset manifest and index its views by id."""

    _require_keys(
        manifest,
        (
            "schema_version",
            "kind",
            "dataset_id",
            "image_domain",
            "calibrated_temperature",
            "data_range",
            "views",
        ),
        "dataset manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported dataset schema_version {manifest['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if manifest["kind"] != "thermal-dataset":
        raise ManifestError("dataset manifest kind must be 'thermal-dataset'")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"]:
        raise ManifestError("dataset_id must be a non-empty string")
    if manifest["image_domain"] not in {"thermal_intensity", "temperature"}:
        raise ManifestError("image_domain must be 'thermal_intensity' or 'temperature'")
    if not isinstance(manifest["calibrated_temperature"], bool):
        raise ManifestError("calibrated_temperature must be boolean")
    if manifest["image_domain"] == "temperature" and not manifest["calibrated_temperature"]:
        raise ManifestError("temperature image_domain requires calibrated_temperature=true")

    data_range = manifest["data_range"]
    if not isinstance(data_range, list) or len(data_range) != 2:
        raise ManifestError("data_range must be a two-element JSON array")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in data_range):
        raise ManifestError("data_range values must be finite numbers")
    low, high = float(data_range[0]), float(data_range[1])
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ManifestError("data_range must contain finite values with low < high")

    views = manifest["views"]
    if not isinstance(views, list) or not views:
        raise ManifestError("views must be a non-empty JSON array")
    by_id: Dict[str, Mapping[str, Any]] = {}
    paths: Set[str] = set()
    for index, view in enumerate(views):
        label = f"views[{index}]"
        if not isinstance(view, dict):
            raise ManifestError(f"{label} must be an object")
        _require_keys(
            view,
            ("id", "relative_image_path", "sequence_index", "camera_ref", "source_sha256"),
            label,
        )
        view_id = view["id"]
        if not isinstance(view_id, str) or not view_id:
            raise ManifestError(f"{label}.id must be a non-empty string")
        if view_id in by_id:
            raise ManifestError(f"Duplicate view id: {view_id!r}")
        relative_path = _validate_relative_path(
            view["relative_image_path"], f"{label}.relative_image_path"
        )
        if Path(relative_path).suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
            raise ManifestError(
                f"{label}.relative_image_path must be a PNG, TIFF, or JPEG image"
            )
        if relative_path in paths:
            raise ManifestError(f"Duplicate relative_image_path: {relative_path!r}")
        paths.add(relative_path)
        sequence_index = view["sequence_index"]
        if isinstance(sequence_index, bool) or not isinstance(sequence_index, int):
            raise ManifestError(f"{label}.sequence_index must be an integer")
        camera_ref = view["camera_ref"]
        if camera_ref is not None and not isinstance(camera_ref, str):
            raise ManifestError(f"{label}.camera_ref must be a string or null")
        _validate_sha256(view["source_sha256"], f"{label}.source_sha256", allow_none=True)
        by_id[view_id] = view
    return by_id


def _validate_id_list(
    manifest: Mapping[str, Any], key: str, known_view_ids: Set[str]
) -> List[str]:
    value = manifest[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ManifestError(f"split manifest {key} must be an array of view ids")
    if len(value) != len(set(value)):
        raise ManifestError(f"split manifest {key} contains duplicate view ids")
    unknown = sorted(set(value) - known_view_ids)
    if unknown:
        raise ManifestError(f"split manifest {key} contains unknown ids: {unknown}")
    return list(value)


def validate_sparse_split_manifest(
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
    dataset_views: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    """Validate split provenance and return train-selected, val, and test ids."""

    _require_keys(
        manifest,
        (
            "schema_version",
            "kind",
            "split_id",
            "dataset_id",
            "dataset_sha256",
            "train_pool",
            "val",
            "test",
            "requested_ratio",
            "actual_ratio",
            "train_selected",
            "method",
            "seed",
            "generator_version",
            "source_base_split_sha256",
        ),
        "split manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported split schema_version {manifest['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if manifest["kind"] != "sparse-split":
        raise ManifestError("split manifest kind must be 'sparse-split'")
    if manifest["dataset_id"] != dataset["dataset_id"]:
        raise ManifestError("split dataset_id does not match dataset manifest")
    _validate_sha256(manifest["dataset_sha256"], "split dataset_sha256")
    expected_dataset_hash = manifest_sha256(dataset)
    if manifest["dataset_sha256"] != expected_dataset_hash:
        raise ManifestError(
            "split dataset_sha256 does not match the canonical dataset manifest hash"
        )

    known = set(dataset_views)
    train_pool = _validate_id_list(manifest, "train_pool", known)
    train_selected = _validate_id_list(manifest, "train_selected", known)
    val = _validate_id_list(manifest, "val", known)
    test = _validate_id_list(manifest, "test", known)
    if not set(train_selected).issubset(train_pool):
        raise ManifestError("train_selected must be a subset of train_pool")
    groups = {
        "train_pool": set(train_pool),
        "val": set(val),
        "test": set(test),
    }
    for left, right in (("train_pool", "val"), ("train_pool", "test"), ("val", "test")):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise ManifestError(f"split sets {left} and {right} overlap: {overlap}")
    if not train_selected:
        raise ManifestError("train_selected must not be empty")
    return train_selected, val, test


def normalize_transforms(
    transforms: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Validate an ordered transform list and return canonical parameters."""

    if isinstance(transforms, (str, bytes)) or not isinstance(transforms, Sequence):
        raise ValueError("transforms must be an ordered sequence of mappings")
    if not transforms:
        raise ValueError("At least one degradation transform is required")

    normalized: List[Dict[str, Any]] = []
    for index, transform in enumerate(transforms):
        if not isinstance(transform, Mapping):
            raise ValueError(f"transforms[{index}] must be a mapping")
        name = transform.get("name")
        if name not in SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"transforms[{index}].name must be one of {sorted(SUPPORTED_TRANSFORMS)}"
            )
        if name == "gaussian-noise":
            allowed = {"name", "sigma"}
            unknown = set(transform) - allowed
            if unknown:
                raise ValueError(f"Unknown gaussian-noise parameters: {sorted(unknown)}")
            sigma = transform.get("sigma")
            if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
                raise ValueError("gaussian-noise sigma must be a finite non-negative number")
            sigma = float(sigma)
            if not math.isfinite(sigma) or sigma < 0:
                raise ValueError("gaussian-noise sigma must be a finite non-negative number")
            normalized.append({"name": name, "sigma": sigma})
        elif name == "contrast":
            allowed = {"name", "alpha"}
            unknown = set(transform) - allowed
            if unknown:
                raise ValueError(f"Unknown contrast parameters: {sorted(unknown)}")
            alpha = transform.get("alpha")
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
                raise ValueError("contrast alpha must be a finite number in [0, 1]")
            alpha = float(alpha)
            if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                raise ValueError("contrast alpha must be a finite number in [0, 1]")
            normalized.append({"name": name, "alpha": alpha})
        else:
            allowed = {"name", "sigma", "kernel_size"}
            unknown = set(transform) - allowed
            if unknown:
                raise ValueError(f"Unknown blur parameters: {sorted(unknown)}")
            sigma = transform.get("sigma")
            kernel_size = transform.get("kernel_size")
            if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
                raise ValueError("blur sigma must be a finite positive number")
            sigma = float(sigma)
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError("blur sigma must be a finite positive number")
            if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
                raise ValueError("blur kernel_size must be a positive odd integer")
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("blur kernel_size must be a positive odd integer")
            normalized.append(
                {"name": name, "sigma": sigma, "kernel_size": kernel_size}
            )
    return normalized


def _gaussian_kernel(sigma: float, kernel_size: int) -> Any:
    _require_dependencies()
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    return kernel / kernel.sum()


def _blur_axis(array: Any, kernel: Any, axis: int) -> Any:
    radius = len(kernel) // 2
    if radius == 0:
        return array.copy()
    padding = [(0, 0)] * array.ndim
    padding[axis] = (radius, radius)
    mode = "reflect" if array.shape[axis] > 1 else "edge"
    padded = np.pad(array, padding, mode=mode)
    result = np.zeros_like(array, dtype=np.float64)
    slices = [slice(None)] * array.ndim
    for offset, weight in enumerate(kernel):
        slices[axis] = slice(offset, offset + array.shape[axis])
        result += weight * padded[tuple(slices)]
    return result


def gaussian_blur_normalized(image: Any, sigma: float, kernel_size: int) -> Any:
    """Blur spatial axes of a normalized grayscale or multi-channel image."""

    _require_dependencies()
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ValueError("sigma must be a finite positive number")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("sigma must be a finite positive number")
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise ValueError("kernel_size must be a positive odd integer")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if image.ndim not in {2, 3}:
        raise ValueError("Images must have shape (H, W) or (H, W, C)")
    kernel = _gaussian_kernel(float(sigma), kernel_size)
    return _blur_axis(_blur_axis(image, kernel, axis=0), kernel, axis=1)


def apply_degradations(
    image: Any,
    transforms: Sequence[Mapping[str, Any]],
    data_range: Sequence[float],
    seed: int,
) -> Any:
    """Apply ordered degradations while preserving image shape and integer dtype.

    ``sigma`` for Gaussian noise is expressed as a fraction of the declared
    global data range.  Contrast is reduced around each channel's spatial mean.
    """

    _require_dependencies()
    canonical = normalize_transforms(transforms)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(f"Only uint8 and uint16 images are supported, got {image.dtype}")
    if image.ndim not in {2, 3} or (image.ndim == 3 and image.shape[2] not in {1, 2, 3, 4}):
        raise ValueError("Images must have shape (H, W) or (H, W, C), C in 1..4")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image height and width must be non-zero")
    if len(data_range) != 2:
        raise ValueError("data_range must contain exactly two values")
    low, high = float(data_range[0]), float(data_range[1])
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("data_range must contain finite values with low < high")
    observed_low = float(image.min())
    observed_high = float(image.max())
    tolerance = max(1.0, abs(low), abs(high)) * 1e-12
    if observed_low < low - tolerance or observed_high > high + tolerance:
        raise ValueError(
            f"Image values [{observed_low}, {observed_high}] fall outside the fixed "
            f"data_range [{low}, {high}]"
        )

    working = (image.astype(np.float64) - low) / (high - low)
    rng = np.random.default_rng(seed)
    spatial_axes = (0, 1)
    for transform in canonical:
        name = transform["name"]
        if name == "gaussian-noise":
            working = working + rng.normal(0.0, transform["sigma"], size=working.shape)
        elif name == "contrast":
            center = working.mean(axis=spatial_axes, keepdims=True)
            working = center + transform["alpha"] * (working - center)
        else:
            working = gaussian_blur_normalized(
                working, transform["sigma"], transform["kernel_size"]
            )
        working = np.clip(working, 0.0, 1.0)

    restored = np.rint(low + working * (high - low))
    dtype_info = np.iinfo(image.dtype)
    restored = np.clip(restored, dtype_info.min, dtype_info.max)
    return restored.astype(image.dtype)


def _load_supported_image(path: Path) -> Tuple[Any, str]:
    _require_dependencies()
    if path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    try:
        with Image.open(path) as opened:
            opened.load()
            source_format = opened.format or ""
            source_mode = opened.mode
            array = np.asarray(opened)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode image {path}: {exc}") from exc

    if array.dtype == np.dtype("int32") and source_mode in {"I", "I;16", "I;16B", "I;16L"}:
        if array.size and (array.min() < 0 or array.max() > 65535):
            raise TypeError(f"32-bit integer image values cannot be represented as uint16: {path}")
        array = array.astype(np.uint16)
    elif array.dtype.kind == "u" and array.dtype.itemsize == 2:
        # Pillow may expose big-endian I;16B TIFF data as dtype('>u2').
        array = array.astype(np.uint16)
    if array.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(f"Only uint8 and uint16 images are supported: {path}")
    if array.ndim not in {2, 3}:
        raise ValueError(f"Unsupported image shape {array.shape}: {path}")
    return np.array(array, copy=True), source_format


def _save_image_atomic(array: Any, path: Path, source_format: str) -> None:
    _require_dependencies()
    if array.dtype == np.dtype("uint16") and array.ndim != 2:
        raise TypeError("Pillow only supports this pipeline's uint16 output as grayscale")
    suffix = path.suffix.lower()
    if suffix not in LOSSLESS_OUTPUT_SUFFIXES:
        raise ValueError(f"Output image must use PNG or TIFF, got: {path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array)
    output_format = "PNG" if suffix == ".png" else "TIFF"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        image.save(temporary_path, format=output_format)
        os.replace(temporary_path, path)
    finally:
        image.close()
        if temporary_path.exists():
            temporary_path.unlink()


def _lossless_output_relative(relative_source: str) -> str:
    source_path = PurePosixPath(relative_source)
    if source_path.suffix.lower() not in LOSSLESS_OUTPUT_SUFFIXES:
        source_path = source_path.with_suffix(".png")
    return (PurePosixPath("images") / source_path).as_posix()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _held_out_record(
    view_id: str,
    role: str,
    view: Mapping[str, Any],
    dataset_root: Path,
) -> Dict[str, Any]:
    relative_path = _validate_relative_path(
        view["relative_image_path"], f"view {view_id!r} relative_image_path"
    )
    source_path = _resolve_below(dataset_root, relative_path, f"view {view_id!r}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image does not exist: {source_path}")
    actual_hash = sha256_file(source_path)
    declared_hash = view["source_sha256"]
    if declared_hash is not None and declared_hash != actual_hash:
        raise ManifestError(
            f"Source SHA256 mismatch for view {view_id!r}: "
            f"declared {declared_hash}, actual {actual_hash}"
        )
    return {
        "id": view_id,
        "role": role,
        "degraded": False,
        "input_relative_image_path": relative_path,
        "input_sha256": actual_hash,
        "output_relative_image_path": relative_path,
        "output_path_base": "dataset",
        "output_sha256": actual_hash,
        "view_seed": None,
    }


def degrade_dataset(
    dataset_manifest_path: Union[str, Path],
    split_manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    transforms: Sequence[Mapping[str, Any]],
    global_seed: int = 0,
    overwrite: bool = False,
    manifest_name: str = "degradation_manifest.v1.json",
) -> Path:
    """Degrade selected training images and write a provenance manifest.

    Source paths are resolved relative to the dataset manifest.  Only
    ``train_selected`` images are written below ``output_dir/images``.  ``val``
    and ``test`` records retain their original dataset-relative path and SHA256.
    Existing output targets are rejected unless ``overwrite=True``.
    """

    _require_dependencies()
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    manifest_name = _validate_relative_path(manifest_name, "manifest_name")
    if Path(manifest_name).suffix.lower() != ".json":
        raise ValueError("manifest_name must end in .json")
    canonical_transforms = normalize_transforms(transforms)

    dataset_path = Path(dataset_manifest_path).resolve()
    split_path = Path(split_manifest_path).resolve()
    output_root = Path(output_dir).resolve()
    try:
        dataset = load_dataset_manifest(dataset_path)
        split = load_sparse_split_manifest(split_path, dataset=dataset)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ManifestError(f"Cannot load input manifests: {exc}") from exc
    dataset_views = validate_dataset_manifest(dataset)
    train_selected, val_ids, test_ids = validate_sparse_split_manifest(
        split, dataset, dataset_views
    )
    dataset_root = dataset_path.parent

    manifest_path = _resolve_below(output_root, manifest_name, "output manifest")
    if manifest_path in {dataset_path, split_path}:
        raise ValueError("Output manifest must not replace an input manifest")
    output_paths: Dict[str, Tuple[str, Path]] = {}
    for view_id in train_selected:
        relative_source = _validate_relative_path(
            dataset_views[view_id]["relative_image_path"],
            f"view {view_id!r} relative_image_path",
        )
        output_relative = _lossless_output_relative(relative_source)
        target = _resolve_below(output_root, output_relative, f"view {view_id!r} output")
        source = _resolve_below(dataset_root, relative_source, f"view {view_id!r} source")
        if target == source:
            raise ValueError(f"Output for view {view_id!r} would replace its source image")
        if target in (path for _, path in output_paths.values()):
            raise ManifestError(f"Multiple selected views map to output path {output_relative!r}")
        output_paths[view_id] = (output_relative, target)

    existing = [path for _, path in output_paths.values() if path.exists()]
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not overwrite:
        displayed = ", ".join(str(path) for path in existing[:3])
        extra = " ..." if len(existing) > 3 else ""
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output target(s): {displayed}{extra}"
        )

    # Validate held-out files before creating any degraded output.
    records: List[Dict[str, Any]] = []
    held_out_records = [
        _held_out_record(view_id, "val", dataset_views[view_id], dataset_root)
        for view_id in val_ids
    ] + [
        _held_out_record(view_id, "test", dataset_views[view_id], dataset_root)
        for view_id in test_ids
    ]

    for view_id in train_selected:
        view = dataset_views[view_id]
        relative_source = _validate_relative_path(
            view["relative_image_path"], f"view {view_id!r} relative_image_path"
        )
        source_path = _resolve_below(dataset_root, relative_source, f"view {view_id!r}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Source image does not exist: {source_path}")
        input_hash = sha256_file(source_path)
        declared_hash = view["source_sha256"]
        if declared_hash is not None and declared_hash != input_hash:
            raise ManifestError(
                f"Source SHA256 mismatch for view {view_id!r}: "
                f"declared {declared_hash}, actual {input_hash}"
            )
        source_image, source_format = _load_supported_image(source_path)
        view_seed = derive_view_seed(global_seed, view_id)
        degraded = apply_degradations(
            source_image, canonical_transforms, dataset["data_range"], view_seed
        )
        if degraded.shape != source_image.shape:
            raise RuntimeError(f"Degradation changed shape for view {view_id!r}")
        output_relative, target = output_paths[view_id]
        _save_image_atomic(degraded, target, source_format)
        output_hash = sha256_file(target)
        records.append(
            {
                "id": view_id,
                "role": "train_selected",
                "degraded": True,
                "input_relative_image_path": relative_source,
                "input_sha256": input_hash,
                "output_relative_image_path": output_relative,
                "output_path_base": "degradation_output",
                "output_sha256": output_hash,
                "view_seed": view_seed,
                "shape": list(source_image.shape),
                "dtype": str(source_image.dtype),
            }
        )
    records.extend(held_out_records)

    output_manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "degradation-manifest",
        "generator_version": GENERATOR_VERSION,
        "dataset_id": dataset["dataset_id"],
        "dataset_manifest_sha256": manifest_sha256(dataset),
        "dataset_manifest_file_sha256": sha256_file(dataset_path),
        "split_id": split["split_id"],
        "split_manifest_sha256": manifest_sha256(split),
        "split_manifest_file_sha256": sha256_file(split_path),
        "data_range": list(dataset["data_range"]),
        "global_seed": global_seed,
        "seed_derivation": "uint64_be(sha256(utf8(global_seed) || NUL || utf8(view_id))[0:8])",
        "transforms": canonical_transforms,
        "train_selected": list(train_selected),
        "val": list(val_ids),
        "test": list(test_ids),
        "views": records,
    }
    output_manifest["manifest_sha256"] = manifest_sha256(output_manifest)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output manifest: {manifest_path}")
    _write_json_atomic(manifest_path, output_manifest)
    return manifest_path


__all__ = [
    "DependencyUnavailableError",
    "GENERATOR_VERSION",
    "ManifestError",
    "SCHEMA_VERSION",
    "SUPPORTED_TRANSFORMS",
    "apply_degradations",
    "degrade_dataset",
    "dependencies_available",
    "dependency_error_message",
    "derive_view_seed",
    "gaussian_blur_normalized",
    "manifest_sha256",
    "normalize_transforms",
    "sha256_file",
    "validate_dataset_manifest",
    "validate_sparse_split_manifest",
]
