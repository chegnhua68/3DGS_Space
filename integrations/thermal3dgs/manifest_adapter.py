"""Translate neutral experiment manifests into Thermal3D-GS camera inputs."""

import hashlib
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, NamedTuple, Optional, Set, Tuple, Union


_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from thermal3dgs_sparse_ir.contracts import (  # noqa: E402
    load_dataset_manifest,
    load_sparse_split_manifest,
    manifest_sha256,
    read_json,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ColmapManifestSelection(NamedTuple):
    train_view_ids: List[str]
    test_view_ids: List[str]
    dataset_camera_refs: Set[str]
    path_by_camera_ref: Dict[str, Path]
    view_id_by_camera_ref: Dict[str, str]
    fid_by_camera_ref: Dict[str, float]

    @property
    def active_camera_refs(self):
        return set(self.path_by_camera_ref)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_ref(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty COLMAP image name".format(label))
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != normalized:
        raise ValueError("{} must be a normalized relative path".format(label))
    return normalized


def _resolve_below(root: Path, relative_path: str, label: str) -> Path:
    root = root.resolve()
    relative = PurePosixPath(relative_path)
    candidate = (root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("{} escapes its declared root".format(label)) from exc
    return candidate


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA-256 digest".format(label))
    return value


def _verify_file(path: Path, expected_hash: Optional[str], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(label, path))
    if expected_hash is not None:
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                "{} SHA-256 mismatch: expected {}, got {}".format(
                    label, expected_hash, actual_hash
                )
            )


def _load_degradation_records(
    path: Path,
    dataset: Mapping[str, object],
    split: Mapping[str, object],
    dataset_manifest_path: Path,
    split_manifest_path: Path,
) -> Tuple[Mapping[str, Mapping[str, object]], Path]:
    manifest = read_json(path)
    required = {
        "schema_version",
        "kind",
        "generator_version",
        "dataset_id",
        "dataset_manifest_sha256",
        "dataset_manifest_file_sha256",
        "split_id",
        "split_manifest_sha256",
        "split_manifest_file_sha256",
        "data_range",
        "global_seed",
        "seed_derivation",
        "transforms",
        "train_selected",
        "val",
        "test",
        "views",
        "manifest_sha256",
    }
    missing = required - set(manifest)
    unknown = set(manifest) - required
    if missing or unknown:
        raise ValueError(
            "invalid degradation manifest fields; missing={}, unknown={}".format(
                sorted(missing), sorted(unknown)
            )
        )
    if manifest["schema_version"] != 1 or manifest["kind"] != "degradation-manifest":
        raise ValueError("unsupported degradation manifest schema or kind")
    if manifest["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("degradation dataset_id does not match dataset manifest")
    if manifest["split_id"] != split["split_id"]:
        raise ValueError("degradation split_id does not match sparse split")
    if manifest["dataset_manifest_sha256"] != manifest_sha256(dataset):
        raise ValueError("degradation dataset manifest hash does not match")
    if manifest["split_manifest_sha256"] != manifest_sha256(split):
        raise ValueError("degradation split manifest hash does not match")
    if _require_sha256(
        manifest["dataset_manifest_file_sha256"],
        "degradation dataset_manifest_file_sha256",
    ) != _sha256_file(dataset_manifest_path):
        raise ValueError("degradation dataset manifest file hash does not match")
    if _require_sha256(
        manifest["split_manifest_file_sha256"],
        "degradation split_manifest_file_sha256",
    ) != _sha256_file(split_manifest_path):
        raise ValueError("degradation split manifest file hash does not match")
    if [float(value) for value in manifest["data_range"]] != [
        float(value) for value in dataset["data_range"]
    ]:
        raise ValueError("degradation data_range does not match dataset manifest")
    recorded_hash = _require_sha256(
        manifest["manifest_sha256"], "degradation manifest_sha256"
    )
    if recorded_hash != manifest_sha256(manifest):
        raise ValueError("degradation manifest_sha256 does not match its content")
    for field in ("train_selected", "val", "test"):
        if set(manifest[field]) != set(split[field]):
            raise ValueError("degradation {} set does not match split".format(field))

    records = manifest["views"]
    if not isinstance(records, list):
        raise ValueError("degradation views must be an array")
    by_id = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("invalid degradation view record at index {}".format(index))
        view_id = record["id"]
        if view_id in by_id:
            raise ValueError("duplicate degradation view id: {}".format(view_id))
        by_id[view_id] = record
    expected_ids = set(split["train_selected"]) | set(split["val"]) | set(split["test"])
    if set(by_id) != expected_ids:
        raise ValueError("degradation view records do not cover the frozen split")
    roles = {}
    roles.update({view_id: "train_selected" for view_id in split["train_selected"]})
    roles.update({view_id: "val" for view_id in split["val"]})
    roles.update({view_id: "test" for view_id in split["test"]})
    for view_id, expected_role in roles.items():
        record = by_id[view_id]
        is_training = expected_role == "train_selected"
        if record.get("role") != expected_role:
            raise ValueError("degradation role mismatch for {}".format(view_id))
        if bool(record.get("degraded")) != is_training:
            raise ValueError("only selected training views may be degraded")
        input_hash = _require_sha256(
            record.get("input_sha256"), "input_sha256 for {}".format(view_id)
        )
        output_hash = _require_sha256(
            record.get("output_sha256"), "output_sha256 for {}".format(view_id)
        )
        if not is_training:
            if record.get("output_path_base") != "dataset":
                raise ValueError("held-out views must resolve to the dataset root")
            if input_hash != output_hash:
                raise ValueError("held-out view hashes must remain unchanged")
    return by_id, path.parent.resolve()


def load_colmap_selection(
    dataset_manifest_path: Union[Path, str],
    split_manifest_path: Union[Path, str],
    degradation_manifest_path: Optional[Union[Path, str]] = None,
    evaluation_partition: str = "test",
    verify_hashes: bool = True,
) -> ColmapManifestSelection:
    """Load, cross-check, and resolve manifest-selected COLMAP image paths."""

    dataset_path = Path(dataset_manifest_path).resolve()
    split_path = Path(split_manifest_path).resolve()
    dataset = load_dataset_manifest(dataset_path)
    split = load_sparse_split_manifest(split_path, dataset=dataset)
    if [float(value) for value in dataset["data_range"]] != [0.0, 255.0]:
        raise ValueError(
            "the pinned Thermal3D-GS loader only supports 8-bit encoded data_range [0,255]"
        )
    if evaluation_partition not in ("val", "test"):
        raise ValueError("evaluation_partition must be 'val' or 'test'")
    if not split[evaluation_partition]:
        raise ValueError(
            "evaluation partition {!r} is empty".format(evaluation_partition)
        )
    dataset_root = dataset_path.parent.resolve()
    views_by_id = {view["id"]: view for view in dataset["views"]}

    degradation_records = None
    degradation_root = None
    if degradation_manifest_path:
        degradation_path = Path(degradation_manifest_path).resolve()
        degradation_records, degradation_root = _load_degradation_records(
            degradation_path, dataset, split, dataset_path, split_path
        )

    evaluation_ids = split[evaluation_partition]
    active_ids = set(split["train_selected"]) | set(evaluation_ids)
    sequence_min = min(view["sequence_index"] for view in dataset["views"])
    sequence_max = max(view["sequence_index"] for view in dataset["views"])
    sequence_scale = float(sequence_max - sequence_min)
    if sequence_scale <= 0:
        sequence_scale = 1.0
    path_by_ref = {}
    view_id_by_ref = {}
    fid_by_ref = {}
    dataset_camera_refs = set()
    for view in dataset["views"]:
        camera_ref = _normalized_ref(view.get("camera_ref"), "camera_ref")
        if camera_ref in dataset_camera_refs:
            raise ValueError("duplicate camera_ref in dataset: {}".format(camera_ref))
        dataset_camera_refs.add(camera_ref)
    for view_id in active_ids:
        view = views_by_id[view_id]
        camera_ref = _normalized_ref(view.get("camera_ref"), "camera_ref")
        if camera_ref in view_id_by_ref:
            raise ValueError("duplicate camera_ref in active views: {}".format(camera_ref))

        expected_hash = view.get("source_sha256")
        if degradation_records is None:
            relative_path = view["relative_image_path"]
            image_path = _resolve_below(dataset_root, relative_path, "dataset image")
        else:
            record = degradation_records[view_id]
            is_train = view_id in set(split["train_selected"])
            expected_role = "train_selected" if is_train else evaluation_partition
            if record.get("role") != expected_role:
                raise ValueError("degradation role mismatch for {}".format(view_id))
            if bool(record.get("degraded")) != is_train:
                raise ValueError("only selected training views may be degraded")
            base = record.get("output_path_base")
            if base == "dataset":
                path_root = dataset_root
            elif base == "degradation_output" and is_train:
                path_root = degradation_root
            else:
                raise ValueError("invalid output_path_base for {}".format(view_id))
            relative_path = record.get("output_relative_image_path")
            if not isinstance(relative_path, str):
                raise ValueError("missing output image path for {}".format(view_id))
            image_path = _resolve_below(path_root, relative_path, "degradation image")
            expected_hash = _require_sha256(
                record.get("output_sha256"), "output_sha256 for {}".format(view_id)
            )

        _verify_file(
            image_path,
            expected_hash if verify_hashes else None,
            "image for view {}".format(view_id),
        )
        path_by_ref[camera_ref] = image_path
        view_id_by_ref[camera_ref] = view_id
        fid_by_ref[camera_ref] = (
            float(view["sequence_index"] - sequence_min) / sequence_scale
        )

    order_key = lambda view_id: (views_by_id[view_id]["sequence_index"], view_id)
    train_ids = sorted(split["train_selected"], key=order_key)
    test_ids = sorted(evaluation_ids, key=order_key)
    return ColmapManifestSelection(
        train_view_ids=train_ids,
        test_view_ids=test_ids,
        dataset_camera_refs=dataset_camera_refs,
        path_by_camera_ref=path_by_ref,
        view_id_by_camera_ref=view_id_by_ref,
        fid_by_camera_ref=fid_by_ref,
    )
