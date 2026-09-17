"""Strict loader for frozen denoised soft-target supervision caches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


SCHEMA_VERSION = 1
MODE = "denoised_soft_target"
_SHA256 = set("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be a non-empty relative path".format(label))
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("{} must be normalized and relative".format(label))
    return value


def _resolve_below(root: Path, relative: str, label: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("{} escapes cache root".format(label)) from exc
    return candidate


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(label))
    return value


def _canonical_tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def mix_soft_target(observed: np.ndarray, teacher: np.ndarray, rho: float) -> np.ndarray:
    """Return the frozen float32 target ``(1-rho)*Y + rho*Z``."""

    observed = np.asarray(observed, dtype=np.float32)
    teacher = np.asarray(teacher, dtype=np.float32)
    if observed.shape != teacher.shape:
        raise ValueError("observed and teacher must have matching shapes")
    if not np.isfinite(observed).all() or not np.isfinite(teacher).all():
        raise ValueError("observed and teacher must be finite")
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    target = np.asarray(
        np.float32(1.0 - rho) * observed + np.float32(rho) * teacher,
        dtype=np.float32,
    )
    if not np.isfinite(target).all() or float(target.min()) < 0.0 or float(target.max()) > 1.0:
        raise ValueError("soft target must be finite and in [0, 1]")
    return target


def load_manifest(path: os.PathLike[str] | str) -> Dict[str, Any]:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("soft-target manifest root must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported soft-target manifest schema")
    if manifest.get("cache_status") != "completed":
        raise ValueError("soft-target cache is not completed")
    if manifest.get("mode") != MODE:
        raise ValueError("soft-target manifest mode mismatch")
    if float(manifest.get("rho", -1.0)) != 0.75:
        raise ValueError("soft-target manifest rho must be 0.75")
    if manifest.get("dtype") != "float32" or manifest.get("layout") != "HWC":
        raise ValueError("soft-target cache must be float32 HWC")
    if list(manifest.get("data_range", [])) != [0.0, 1.0]:
        raise ValueError("soft-target cache must use data_range [0, 1]")
    records = manifest.get("views")
    train_ids = manifest.get("train_selected")
    if not isinstance(records, list) or not isinstance(train_ids, list) or not train_ids:
        raise ValueError("soft-target manifest must contain train_selected and views")
    by_id: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("invalid soft-target view record")
        view_id = record["id"]
        if view_id in by_id:
            raise ValueError("duplicate soft-target view id: {}".format(view_id))
        by_id[view_id] = record
    if set(by_id) != set(train_ids):
        raise ValueError("soft-target records do not exactly match train_selected")
    expected_manifest_hash = manifest.get("manifest_sha256")
    if expected_manifest_hash:
        payload = dict(manifest)
        payload.pop("manifest_sha256", None)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if expected_manifest_hash != hashlib.sha256(encoded).hexdigest():
            raise ValueError("soft-target manifest_sha256 does not match content")
    return manifest


class SoftTargetStore:
    """Validate and CPU-cache frozen T arrays without touching observed images."""

    def __init__(self, manifest_path: os.PathLike[str] | str, *, rho: float = 0.75):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = load_manifest(self.manifest_path)
        if float(rho) != float(self.manifest["rho"]):
            raise ValueError("requested ds_rho does not match soft-target manifest")
        self._records = {record["id"]: record for record in self.manifest["views"]}
        self._cache: Dict[str, np.ndarray] = {}
        self._validate_records()

    @property
    def cache_key(self) -> str:
        return str(self.manifest.get("cache_key", ""))

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest.get("manifest_sha256", ""))

    @property
    def train_ids(self):
        return tuple(self.manifest["train_selected"])

    def _validate_records(self) -> None:
        for view_id, record in self._records.items():
            target_path = _resolve_below(
                self.root,
                _safe_relative_path(record.get("target_relative_path"), "target_relative_path"),
                "target path for {}".format(view_id),
            )
            if not target_path.is_file():
                raise FileNotFoundError("soft-target file does not exist: {}".format(target_path))
            expected_hash = _require_sha256(record.get("target_file_sha256"), "target_file_sha256")
            if _sha256_file(target_path) != expected_hash:
                raise ValueError("soft-target file hash mismatch for {}".format(view_id))
            shape = record.get("shape")
            if not isinstance(shape, list) or len(shape) != 3 or any(int(value) < 1 for value in shape):
                raise ValueError("invalid soft-target shape for {}".format(view_id))
            array = np.load(target_path, allow_pickle=False)
            self._validate_array(array, shape, view_id)
            expected_tensor_hash = _require_sha256(record.get("target_tensor_sha256"), "target_tensor_sha256")
            if _canonical_tensor_hash(array) != expected_tensor_hash:
                raise ValueError("soft-target tensor hash mismatch for {}".format(view_id))
            self._cache[view_id] = np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _validate_array(array: np.ndarray, shape, view_id: str) -> None:
        if array.dtype != np.dtype("float32") or tuple(array.shape) != tuple(int(value) for value in shape):
            raise ValueError("soft-target dtype/shape mismatch for {}".format(view_id))
        if not np.isfinite(array).all():
            raise ValueError("soft-target contains non-finite values for {}".format(view_id))
        if float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6:
            raise ValueError("soft-target range mismatch for {}".format(view_id))

    def target_for(self, view_id: str, observed: torch.Tensor) -> torch.Tensor:
        if view_id not in self._cache:
            raise KeyError("view is absent from soft-target cache: {}".format(view_id))
        array = self._cache[view_id]
        target = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if tuple(target.shape) != tuple(observed.shape):
            raise ValueError(
                "soft-target shape {} does not match observed {} for {}".format(
                    tuple(target.shape), tuple(observed.shape), view_id
                )
            )
        return target.to(device=observed.device, dtype=torch.float32).detach()


__all__ = ["MODE", "SCHEMA_VERSION", "SoftTargetStore", "load_manifest", "mix_soft_target"]
