#!/usr/bin/env python3
"""Build a manifest-pinned BM3D float32 soft-target cache for train_selected views."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir.contracts import (  # noqa: E402
    load_dataset_manifest,
    load_sparse_split_manifest,
    manifest_sha256,
)
from integrations.thermal3dgs.soft_target import mix_soft_target  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if "\\" in value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("path must be normalized and relative: {}".format(value))
    return value


def _resolve_below(root: Path, relative: str, label: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(*PurePosixPath(_safe_relative(relative)).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("{} escapes its declared root".format(label)) from exc
    return candidate


def _load_degradation(path: Path, dataset: Mapping[str, Any], split: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    required = {
        "schema_version", "kind", "dataset_id", "dataset_manifest_sha256",
        "split_id", "split_manifest_sha256", "train_selected", "val", "test",
        "views", "manifest_sha256", "data_range", "transforms",
    }
    if set(manifest) < required:
        raise ValueError("degradation manifest is missing required fields")
    if manifest["schema_version"] != 1 or manifest["kind"] != "degradation-manifest":
        raise ValueError("unsupported degradation manifest")
    if manifest["dataset_id"] != dataset["dataset_id"] or manifest["split_id"] != split["split_id"]:
        raise ValueError("degradation manifest dataset/split mismatch")
    if manifest["dataset_manifest_sha256"] != manifest_sha256(dataset):
        raise ValueError("degradation dataset manifest hash mismatch")
    if manifest["split_manifest_sha256"] != manifest_sha256(split):
        raise ValueError("degradation split manifest hash mismatch")
    if list(manifest["data_range"]) != list(dataset["data_range"]):
        raise ValueError("degradation data_range mismatch")
    payload = dict(manifest)
    recorded_hash = payload.pop("manifest_sha256")
    if recorded_hash != manifest_sha256(payload):
        raise ValueError("degradation manifest hash mismatch")
    for field in ("train_selected", "val", "test"):
        if set(manifest[field]) != set(split[field] if field != "train_selected" else split["train_selected"]):
            raise ValueError("degradation {} set mismatch".format(field))
    records: Dict[str, Mapping[str, Any]] = {}
    for record in manifest["views"]:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("invalid degradation view record")
        if record["id"] in records:
            raise ValueError("duplicate degradation view id")
        records[record["id"]] = record
    if set(records) != set(split["train_selected"]) | set(split["val"]) | set(split["test"]):
        raise ValueError("degradation view coverage mismatch")
    for view_id in split["train_selected"]:
        record = records[view_id]
        if record.get("role") != "train_selected" or not record.get("degraded"):
            raise ValueError("training view is not a degraded selected view: {}".format(view_id))
        if record.get("output_path_base") != "degradation_output":
            raise ValueError("training output must be below degradation root")
    return records


def _decode_observed(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode not in ("L", "LA", "RGB", "RGBA"):
            raise ValueError("unsupported observed image mode {}".format(source.mode))
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255.0))


def _dependency_metadata() -> Dict[str, Any]:
    import bm3d
    import bm4d

    packages = {}
    for name, module in (("bm3d", bm3d), ("bm4d", bm4d), ("numpy", np)):
        location = Path(getattr(module, "__file__", "")).resolve()
        packages[name] = {
            "version": importlib.metadata.version(name) if name != "numpy" else np.__version__,
            "module": str(location),
            "module_sha256": _sha256_file(location) if location.is_file() else None,
        }
    return packages


def _denoise(image: np.ndarray, sigma: float, threads: int) -> np.ndarray:
    from bm3d import BM3DStages, bm3d
    from bm3d.profiles import BM3DProfile

    profile = BM3DProfile()
    profile.num_threads = int(threads)
    channels = []
    for channel in range(image.shape[2]):
        result = bm3d(
            image[:, :, channel],
            sigma_psd=float(sigma),
            profile=profile,
            stage_arg=BM3DStages.ALL_STAGES,
            blockmatches=(False, False),
        )
        channels.append(np.asarray(result, dtype=np.float32))
    return np.stack(channels, axis=2).astype(np.float32, copy=False)


def _check_stop(stop_file: Optional[Path]) -> None:
    if stop_file is not None and stop_file.is_file():
        raise InterruptedError("stop request exists: {}".format(stop_file))


def build_cache(args: argparse.Namespace) -> Path:
    dataset_path = args.dataset_manifest.resolve()
    split_path = args.split_manifest.resolve()
    degradation_path = args.degradation_manifest.resolve()
    output = args.output_dir.resolve()
    stop_file = args.stop_file.resolve() if args.stop_file else None
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite non-empty cache: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_manifest(dataset_path)
    split = load_sparse_split_manifest(split_path, dataset=dataset)
    records = _load_degradation(degradation_path, dataset, split)
    train_ids = list(split["train_selected"])
    views_by_id = {view["id"]: view for view in dataset["views"]}
    if set(train_ids) != set(records) & set(train_ids):
        raise ValueError("train_selected IDs are inconsistent")
    if float(args.sigma) != 0.03 or float(args.rho) != 0.75:
        raise ValueError("this frozen cache builder requires sigma=0.03 and rho=0.75")
    dependency = _dependency_metadata()
    started = time.perf_counter()
    view_rows = []
    repeat_max_abs = None
    for index, view_id in enumerate(train_ids):
        _check_stop(stop_file)
        record = records[view_id]
        observed_path = _resolve_below(
            degradation_path.parent,
            record["output_relative_image_path"],
            "degraded output for {}".format(view_id),
        )
        if not observed_path.is_file():
            raise FileNotFoundError("degraded training image missing: {}".format(observed_path))
        if _sha256_file(observed_path) != record["output_sha256"]:
            raise ValueError("degraded image hash mismatch for {}".format(view_id))
        observed = _decode_observed(observed_path)
        expected_shape = tuple(int(value) for value in record["shape"])
        if tuple(observed.shape) != expected_shape:
            raise ValueError("decoded shape mismatch for {}".format(view_id))
        if index == 0:
            repeat = _denoise(observed, args.sigma, args.threads)
        raw_teacher = _denoise(observed, args.sigma, args.threads)
        if index == 0:
            repeat_max_abs = float(np.max(np.abs(raw_teacher - repeat)))
        if not np.isfinite(raw_teacher).all():
            raise ValueError("BM3D produced non-finite output for {}".format(view_id))
        raw_min = float(raw_teacher.min())
        raw_max = float(raw_teacher.max())
        out_of_range_fraction = float(np.mean((raw_teacher < 0.0) | (raw_teacher > 1.0)))
        teacher = np.clip(raw_teacher, 0.0, 1.0).astype(np.float32, copy=False)
        target = mix_soft_target(observed, teacher, args.rho)
        teacher_relative = "teacher_float/{}.npy".format(view_id)
        target_relative = "target_float/{}.npy".format(view_id)
        teacher_path = _resolve_below(output, teacher_relative, "teacher output")
        target_path = _resolve_below(output, target_relative, "target output")
        _atomic_save_npy(teacher_path, teacher)
        _atomic_save_npy(target_path, target)
        teacher_check = np.load(teacher_path, allow_pickle=False)
        target_check = np.load(target_path, allow_pickle=False)
        if teacher_check.dtype != np.dtype("float32") or target_check.dtype != np.dtype("float32"):
            raise ValueError("cache dtype verification failed for {}".format(view_id))
        view_rows.append({
            "id": view_id,
            "observed_relative_path": str(observed_path.relative_to(degradation_path.parent).as_posix()),
            "observed_file_sha256": _sha256_file(observed_path),
            "observed_tensor_sha256": _tensor_hash(observed),
            "shape": list(observed.shape),
            "teacher_relative_path": teacher_relative,
            "teacher_file_sha256": _sha256_file(teacher_path),
            "teacher_tensor_sha256": _tensor_hash(teacher),
            "target_relative_path": target_relative,
            "target_file_sha256": _sha256_file(target_path),
            "target_tensor_sha256": _tensor_hash(target),
            "teacher_raw_min": raw_min,
            "teacher_raw_max": raw_max,
            "teacher_raw_out_of_range_fraction": out_of_range_fraction,
            "target_min": float(target.min()),
            "target_max": float(target.max()),
        })
    elapsed = time.perf_counter() - started
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "denoised-soft-target-cache",
        "cache_status": "completed",
        "mode": "denoised_soft_target",
        "cache_key": "",
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": manifest_sha256(dataset),
        "split_manifest": str(split_path),
        "split_manifest_sha256": manifest_sha256(split),
        "degradation_manifest": str(degradation_path),
        "degradation_manifest_sha256": _sha256_file(degradation_path),
        "train_selected": train_ids,
        "rho": float(args.rho),
        "teacher_sigma": float(args.sigma),
        "teacher_profile": "np",
        "teacher_profile_name": "BM3DProfile (normal)",
        "teacher_stages": "ALL_STAGES",
        "teacher_channel_policy": "independent_rgb_2d",
        "teacher_range_policy": "clip_finite_output_to_0_1",
        "dtype": "float32",
        "layout": "HWC",
        "data_range": [0.0, 1.0],
        "shape": [479, 715, 3],
        "threads": int(args.threads),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency,
        "command": list(sys.argv),
        "repeat_check_first_view_max_abs": repeat_max_abs,
        "preparation_wall_seconds": elapsed,
        "preparation_per_view_seconds": elapsed / len(train_ids),
        "views": view_rows,
    }
    payload["cache_key"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key not in {"cache_key", "command", "preparation_wall_seconds", "preparation_per_view_seconds", "views"}}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["manifest_sha256"] = manifest_sha256(payload)
    manifest_path = output / "supervision_manifest.v1.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + manifest_path.name + ".", suffix=".tmp", dir=str(output))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(str(temporary_path), str(manifest_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a frozen BM3D denoised soft-target cache from train_selected degradation outputs.")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--degradation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--rho", type=float, default=0.75)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    try:
        print(build_cache(args))
    except InterruptedError as exc:
        raise SystemExit("interrupted: {}".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
