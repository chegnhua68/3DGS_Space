"""Versioned JSON contracts used by the sparse infrared tooling.

The manifests deliberately contain explicit view-to-image and view-to-camera
bindings.  Consumers must not infer those bindings from COLMAP file names.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union


SCHEMA_VERSION = 1
DATASET_KIND = "thermal-dataset"
BASE_SPLIT_KIND = "base-split"
SPARSE_SPLIT_KIND = "sparse-split"
MANIFEST_HASH_FIELD = "manifest_sha256"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DATASET_KEYS = {
    "schema_version",
    "kind",
    "dataset_id",
    "image_domain",
    "calibrated_temperature",
    "data_range",
    "views",
}
_VIEW_KEYS = {
    "id",
    "relative_image_path",
    "sequence_index",
    "camera_ref",
    "source_sha256",
}
_BASE_SPLIT_KEYS = {
    "schema_version",
    "kind",
    "split_id",
    "dataset_id",
    "dataset_sha256",
    "train_pool",
    "val",
    "test",
}
_SPARSE_SPLIT_KEYS = _BASE_SPLIT_KEYS | {
    "requested_ratio",
    "actual_ratio",
    "train_selected",
    "method",
    "seed",
    "generator_version",
    "source_base_split_sha256",
}
_SPLIT_METHODS = {"nested-random", "trajectory-uniform"}


class ManifestValidationError(ValueError):
    """Raised when a manifest violates its versioned contract."""


def _duplicate_rejecting_object(
    pairs: Sequence[Tuple[str, Any]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError("duplicate JSON object key: {!r}".format(key))
        result[key] = value
    return result


def read_json(path: Union[os.PathLike, str]) -> Dict[str, Any]:
    """Read one UTF-8 JSON object while rejecting duplicate object keys."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_duplicate_rejecting_object)
    except ManifestValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(
            "could not read JSON manifest {}: {}".format(source, exc)
        ) from exc
    if not isinstance(value, dict):
        raise ManifestValidationError("manifest root must be a JSON object")
    return value


def _require_object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError("{} must be an object".format(location))
    if not all(isinstance(key, str) for key in value):
        raise ManifestValidationError("{} must have string keys".format(location))
    return value


def _require_keys(
    value: Mapping[str, Any],
    required: Set[str],
    location: str,
    optional: Optional[Set[str]] = None,
) -> None:
    optional = optional or set()
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ManifestValidationError(
            "{} is missing required field(s): {}".format(
                location, ", ".join(sorted(missing))
            )
        )
    if unknown:
        raise ManifestValidationError(
            "{} has unknown field(s): {}".format(
                location, ", ".join(sorted(unknown))
            )
        )


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError("{} must be a non-empty string".format(location))
    return value


def _require_integer(value: Any, location: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestValidationError("{} must be an integer".format(location))
    if minimum is not None and value < minimum:
        raise ManifestValidationError(
            "{} must be at least {}".format(location, minimum)
        )
    return value


def _require_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError("{} must be a number".format(location))
    number = float(value)
    if not math.isfinite(number):
        raise ManifestValidationError("{} must be finite".format(location))
    return number


def _require_sha256(value: Any, location: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        suffix = " or null" if nullable else ""
        raise ManifestValidationError(
            "{} must be a lowercase SHA-256 hex digest{}".format(location, suffix)
        )


def _require_relative_posix_path(value: Any, location: str) -> str:
    path_text = _require_string(value, location)
    if "\\" in path_text:
        raise ManifestValidationError(
            "{} must use forward slashes".format(location)
        )
    posix_path = PurePosixPath(path_text)
    windows_path = PureWindowsPath(path_text)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or path_text in {".", ".."}
        or ".." in posix_path.parts
        or posix_path.as_posix() != path_text
    ):
        raise ManifestValidationError(
            "{} must be a normalized relative POSIX path".format(location)
        )
    return path_text


def _validate_envelope(
    manifest: Any, kind: str, fields: Set[str]
) -> Mapping[str, Any]:
    obj = _require_object(manifest, "manifest")
    _require_keys(obj, fields, "manifest", {MANIFEST_HASH_FIELD})
    version = _require_integer(obj["schema_version"], "schema_version")
    if version != SCHEMA_VERSION:
        raise ManifestValidationError(
            "unsupported schema_version {}; expected {}".format(version, SCHEMA_VERSION)
        )
    if obj["kind"] != kind:
        raise ManifestValidationError(
            "kind must be {!r}, got {!r}".format(kind, obj["kind"])
        )
    if MANIFEST_HASH_FIELD in obj:
        _require_sha256(obj[MANIFEST_HASH_FIELD], MANIFEST_HASH_FIELD)
    return obj


def _validate_manifest_hash(manifest: Mapping[str, Any]) -> None:
    recorded = manifest.get(MANIFEST_HASH_FIELD)
    if recorded is not None:
        calculated = manifest_sha256(manifest)
        if recorded != calculated:
            raise ManifestValidationError(
                "{} does not match canonical manifest content: expected {}".format(
                    MANIFEST_HASH_FIELD, calculated
                )
            )


def validate_dataset_manifest(manifest: Any) -> None:
    """Validate a schema-v1 thermal dataset manifest."""

    obj = _validate_envelope(manifest, DATASET_KIND, _DATASET_KEYS)
    _require_string(obj["dataset_id"], "dataset_id")
    if obj["image_domain"] not in {"thermal_intensity", "temperature"}:
        raise ManifestValidationError(
            "image_domain must be 'thermal_intensity' or 'temperature'"
        )
    if not isinstance(obj["calibrated_temperature"], bool):
        raise ManifestValidationError("calibrated_temperature must be a boolean")
    if obj["image_domain"] == "temperature" and not obj["calibrated_temperature"]:
        raise ManifestValidationError(
            "temperature image_domain requires calibrated_temperature=true"
        )

    data_range = obj["data_range"]
    if not isinstance(data_range, list) or len(data_range) != 2:
        raise ManifestValidationError("data_range must be a two-element array")
    low = _require_number(data_range[0], "data_range[0]")
    high = _require_number(data_range[1], "data_range[1]")
    if not low < high:
        raise ManifestValidationError("data_range must be strictly increasing")

    views = obj["views"]
    if not isinstance(views, list) or not views:
        raise ManifestValidationError("views must be a non-empty array")
    view_ids: Set[str] = set()
    sequence_indices: Set[int] = set()
    for index, raw_view in enumerate(views):
        location = "views[{}]".format(index)
        view = _require_object(raw_view, location)
        # camera_ref and source_sha256 may be omitted by hand-written inputs;
        # normalization writes both explicitly as null.
        _require_keys(
            view,
            _VIEW_KEYS - {"camera_ref", "source_sha256"},
            location,
            {"camera_ref", "source_sha256"},
        )
        view_id = _require_string(view["id"], location + ".id")
        if view_id in view_ids:
            raise ManifestValidationError("duplicate view id: {!r}".format(view_id))
        view_ids.add(view_id)
        _require_relative_posix_path(
            view["relative_image_path"], location + ".relative_image_path"
        )
        sequence_index = _require_integer(
            view["sequence_index"], location + ".sequence_index", minimum=0
        )
        if sequence_index in sequence_indices:
            raise ManifestValidationError(
                "duplicate sequence_index: {}".format(sequence_index)
            )
        sequence_indices.add(sequence_index)
        camera_ref = view.get("camera_ref")
        if camera_ref is not None:
            if isinstance(camera_ref, bool) or not isinstance(camera_ref, (str, int)):
                raise ManifestValidationError(
                    location + ".camera_ref must be a non-empty string, integer, or null"
                )
            if isinstance(camera_ref, str) and not camera_ref.strip():
                raise ManifestValidationError(
                    location + ".camera_ref must be a non-empty string, integer, or null"
                )
        _require_sha256(
            view.get("source_sha256"), location + ".source_sha256", nullable=True
        )
    _validate_manifest_hash(obj)


def _validate_id_list(value: Any, location: str, *, non_empty: bool) -> List[str]:
    if not isinstance(value, list):
        raise ManifestValidationError("{} must be an array".format(location))
    if non_empty and not value:
        raise ManifestValidationError("{} must not be empty".format(location))
    result: List[str] = []
    seen: Set[str] = set()
    for index, item in enumerate(value):
        view_id = _require_string(item, "{}[{}]".format(location, index))
        if view_id in seen:
            raise ManifestValidationError(
                "{} contains duplicate view id {!r}".format(location, view_id)
            )
        seen.add(view_id)
        result.append(view_id)
    return result


def _validate_disjoint_partitions(
    train_pool: Sequence[str], val: Sequence[str], test: Sequence[str]
) -> None:
    named = (("train_pool", set(train_pool)), ("val", set(val)), ("test", set(test)))
    for left_index, (left_name, left_values) in enumerate(named):
        for right_name, right_values in named[left_index + 1 :]:
            overlap = left_values & right_values
            if overlap:
                raise ManifestValidationError(
                    "{} and {} overlap at: {}".format(
                        left_name, right_name, ", ".join(sorted(overlap))
                    )
                )


def _validate_dataset_reference(
    manifest: Mapping[str, Any], dataset: Optional[Mapping[str, Any]]
) -> None:
    if dataset is None:
        return
    validate_dataset_manifest(dataset)
    if manifest["dataset_id"] != dataset["dataset_id"]:
        raise ManifestValidationError("dataset_id does not match dataset manifest")
    expected_hash = manifest_sha256(dataset)
    if manifest["dataset_sha256"] != expected_hash:
        raise ManifestValidationError(
            "dataset_sha256 does not match dataset manifest; expected {}".format(
                expected_hash
            )
        )
    known_ids = {view["id"] for view in dataset["views"]}
    referenced_ids = (
        set(manifest["train_pool"]) | set(manifest["val"]) | set(manifest["test"])
    )
    missing = referenced_ids - known_ids
    if missing:
        raise ManifestValidationError(
            "split references unknown dataset view id(s): {}".format(
                ", ".join(sorted(missing))
            )
        )
    unassigned = known_ids - referenced_ids
    if unassigned:
        raise ManifestValidationError(
            "split does not assign dataset view id(s): {}".format(
                ", ".join(sorted(unassigned))
            )
        )


def validate_base_split_manifest(
    manifest: Any, dataset: Optional[Mapping[str, Any]] = None
) -> None:
    """Validate a base split and, when supplied, its dataset reference."""

    obj = _validate_envelope(manifest, BASE_SPLIT_KIND, _BASE_SPLIT_KEYS)
    _require_string(obj["split_id"], "split_id")
    _require_string(obj["dataset_id"], "dataset_id")
    _require_sha256(obj["dataset_sha256"], "dataset_sha256")
    train_pool = _validate_id_list(obj["train_pool"], "train_pool", non_empty=True)
    val = _validate_id_list(obj["val"], "val", non_empty=False)
    test = _validate_id_list(obj["test"], "test", non_empty=False)
    _validate_disjoint_partitions(train_pool, val, test)
    _validate_dataset_reference(obj, dataset)
    _validate_manifest_hash(obj)


def validate_sparse_split_manifest(
    manifest: Any,
    dataset: Optional[Mapping[str, Any]] = None,
    base_split: Optional[Mapping[str, Any]] = None,
) -> None:
    """Validate a generated sparse split and its optional source manifests."""

    obj = _validate_envelope(manifest, SPARSE_SPLIT_KIND, _SPARSE_SPLIT_KEYS)
    _require_string(obj["split_id"], "split_id")
    _require_string(obj["dataset_id"], "dataset_id")
    _require_sha256(obj["dataset_sha256"], "dataset_sha256")
    train_pool = _validate_id_list(obj["train_pool"], "train_pool", non_empty=True)
    val = _validate_id_list(obj["val"], "val", non_empty=False)
    test = _validate_id_list(obj["test"], "test", non_empty=False)
    selected = _validate_id_list(
        obj["train_selected"], "train_selected", non_empty=True
    )
    _validate_disjoint_partitions(train_pool, val, test)
    unknown_selected = set(selected) - set(train_pool)
    if unknown_selected:
        raise ManifestValidationError(
            "train_selected is not a subset of train_pool: {}".format(
                ", ".join(sorted(unknown_selected))
            )
        )

    requested_ratio = _require_number(obj["requested_ratio"], "requested_ratio")
    actual_ratio = _require_number(obj["actual_ratio"], "actual_ratio")
    if not 0.0 < requested_ratio <= 1.0:
        raise ManifestValidationError("requested_ratio must be in (0, 1]")
    expected_actual = len(selected) / len(train_pool)
    if not 0.0 < actual_ratio <= 1.0 or not math.isclose(
        actual_ratio, expected_actual, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ManifestValidationError(
            "actual_ratio must equal len(train_selected) / len(train_pool)"
        )
    if obj["method"] not in _SPLIT_METHODS:
        raise ManifestValidationError(
            "method must be 'nested-random' or 'trajectory-uniform'"
        )
    _require_integer(obj["seed"], "seed")
    _require_string(obj["generator_version"], "generator_version")
    _require_sha256(
        obj["source_base_split_sha256"], "source_base_split_sha256"
    )
    _validate_dataset_reference(obj, dataset)

    if base_split is not None:
        validate_base_split_manifest(base_split, dataset)
        if obj["source_base_split_sha256"] != manifest_sha256(base_split):
            raise ManifestValidationError(
                "source_base_split_sha256 does not match base split manifest"
            )
        for field in ("dataset_id", "dataset_sha256"):
            if obj[field] != base_split[field]:
                raise ManifestValidationError(
                    "{} does not match base split manifest".format(field)
                )
        for field in ("train_pool", "val", "test"):
            if set(obj[field]) != set(base_split[field]):
                raise ManifestValidationError(
                    "{} is not frozen from the base split manifest".format(field)
                )
    _validate_manifest_hash(obj)


def validate_manifest(
    manifest: Any,
    *,
    dataset: Optional[Mapping[str, Any]] = None,
    base_split: Optional[Mapping[str, Any]] = None,
) -> None:
    """Dispatch validation according to the manifest ``kind`` field."""

    obj = _require_object(manifest, "manifest")
    kind = obj.get("kind")
    if kind == DATASET_KIND:
        if dataset is not None or base_split is not None:
            raise ManifestValidationError(
                "dataset/base_split context is not valid for a dataset manifest"
            )
        validate_dataset_manifest(obj)
    elif kind == BASE_SPLIT_KIND:
        if base_split is not None:
            raise ManifestValidationError(
                "base_split context is not valid for a base split manifest"
            )
        validate_base_split_manifest(obj, dataset)
    elif kind == SPARSE_SPLIT_KIND:
        validate_sparse_split_manifest(obj, dataset, base_split)
    else:
        raise ManifestValidationError("unknown manifest kind: {!r}".format(kind))


def _normalize_number(value: Any) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def normalize_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the canonical, enumeration-order-independent manifest form."""

    normalized = dict(manifest)
    kind = normalized.get("kind")
    if kind == DATASET_KIND:
        normalized["schema_version"] = int(normalized["schema_version"])
        normalized["data_range"] = [
            _normalize_number(normalized["data_range"][0]),
            _normalize_number(normalized["data_range"][1]),
        ]
        views = []
        for raw_view in normalized["views"]:
            view = dict(raw_view)
            view.setdefault("camera_ref", None)
            view.setdefault("source_sha256", None)
            view["sequence_index"] = int(view["sequence_index"])
            views.append(view)
        normalized["views"] = sorted(views, key=lambda view: view["id"])
    elif kind in {BASE_SPLIT_KIND, SPARSE_SPLIT_KIND}:
        normalized["schema_version"] = int(normalized["schema_version"])
        for field in ("train_pool", "val", "test"):
            normalized[field] = sorted(normalized[field])
        if kind == SPARSE_SPLIT_KIND:
            normalized["train_selected"] = sorted(normalized["train_selected"])
            normalized["requested_ratio"] = _normalize_number(
                normalized["requested_ratio"]
            )
            normalized["actual_ratio"] = _normalize_number(
                normalized["actual_ratio"]
            )
            normalized["seed"] = int(normalized["seed"])
    return normalized


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical JSON, normalizing known manifest set-like arrays."""

    if isinstance(value, Mapping) and value.get("kind") in {
        DATASET_KIND,
        BASE_SPLIT_KIND,
        SPARSE_SPLIT_KIND,
    }:
        value = normalize_manifest(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError("value is not canonical JSON: {}".format(exc)) from exc
    return text.encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash canonical content, excluding the top-level self-hash field."""

    content = dict(manifest)
    content.pop(MANIFEST_HASH_FIELD, None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Return a SHA-256 digest for canonical JSON content.

    For a recognized manifest this has the same self-hash behavior as
    :func:`manifest_sha256`; for other JSON values it hashes the complete value.
    """

    if isinstance(value, Mapping) and value.get("kind") in {
        DATASET_KIND,
        BASE_SPLIT_KIND,
        SPARSE_SPLIT_KIND,
    }:
        return manifest_sha256(value)
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_manifest_sha256(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a manifest and attach its canonical self hash."""

    content = dict(manifest)
    content.pop(MANIFEST_HASH_FIELD, None)
    normalized = normalize_manifest(content)
    normalized[MANIFEST_HASH_FIELD] = manifest_sha256(normalized)
    return normalized


def write_manifest(
    path: Union[os.PathLike, str],
    manifest: Mapping[str, Any],
    *,
    add_manifest_hash: bool = True,
) -> Path:
    """Validate and atomically write a canonical, versioned JSON manifest."""

    content = dict(manifest)
    content.pop(MANIFEST_HASH_FIELD, None)
    validate_manifest(content)
    output = with_manifest_sha256(content) if add_manifest_hash else normalize_manifest(content)
    validate_manifest(output)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(destination.parent),
            prefix="." + destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(serialized)
        os.replace(temporary_name, destination)
    except OSError:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
        raise
    return destination


def load_dataset_manifest(path: Union[os.PathLike, str]) -> Dict[str, Any]:
    manifest = read_json(path)
    validate_dataset_manifest(manifest)
    return normalize_manifest(manifest)


def load_base_split_manifest(
    path: Union[os.PathLike, str],
    dataset: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = read_json(path)
    validate_base_split_manifest(manifest, dataset)
    return normalize_manifest(manifest)


def load_sparse_split_manifest(
    path: Union[os.PathLike, str],
    dataset: Optional[Mapping[str, Any]] = None,
    base_split: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = read_json(path)
    validate_sparse_split_manifest(manifest, dataset, base_split)
    return normalize_manifest(manifest)


load_json = read_json
canonical_hash = canonical_sha256
