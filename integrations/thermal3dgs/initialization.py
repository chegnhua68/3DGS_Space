"""Shared step-0 initialization packages and deterministic state audits."""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from integrations.thermal3dgs.run_control import write_json_atomic


GAUSSIAN_TENSORS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
    "max_radii2D",
    "xyz_gradient_accum",
    "denom",
)
GAUSSIAN_ATTRIBUTES = (
    "active_sh_degree",
    "max_sh_degree",
    "spatial_lr_scale",
    "percent_dense",
    "og_number_points",
)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().cpu()
    return tensor.view(torch.uint8).numpy().tobytes()


def tensor_digest(value: torch.Tensor) -> Dict[str, Any]:
    tensor = value.detach().contiguous().cpu()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
    }


def _state_dict_cpu(module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_dict_audit(state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    return {name: tensor_digest(state[name]) for name in sorted(state)}


def _simple_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.device):
        return str(value)
    return repr(value)


def optimizer_audit(optimizer) -> Dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                str(key): _simple_value(value)
                for key, value in sorted(group.items())
                if key != "params"
            }
        )
    state = {}
    for parameter, values in optimizer.state.items():
        entry = {}
        for key, value in sorted(values.items()):
            entry[key] = tensor_digest(value) if torch.is_tensor(value) else _simple_value(value)
        state[str(id(parameter))] = entry
    return {
        "param_groups": groups,
        "state_entries": len(state),
        "state": state,
        "state_status": "empty_at_step0" if not state else "nonempty",
    }


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    for device_index, device_state in enumerate(state.get("torch_cuda", [])):
        torch.cuda.set_rng_state(device_state.cpu(), device=device_index)


def rng_state_hashes(state: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "python": hashlib.sha256(pickle.dumps(state["python"], protocol=4)).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(state["numpy"], protocol=4)).hexdigest(),
        "torch_cpu": hashlib.sha256(state["torch_cpu"].cpu().numpy().tobytes()).hexdigest(),
        "torch_cuda": hashlib.sha256(
            b"".join(item.cpu().numpy().tobytes() for item in state.get("torch_cuda", []))
        ).hexdigest(),
    }


def _gaussian_snapshot(gaussians) -> Dict[str, Any]:
    tensors = {
        name: getattr(gaussians, name).detach().cpu().clone()
        for name in GAUSSIAN_TENSORS
        if hasattr(gaussians, name) and torch.is_tensor(getattr(gaussians, name))
    }
    attributes = {
        name: _simple_value(getattr(gaussians, name))
        for name in GAUSSIAN_ATTRIBUTES
        if hasattr(gaussians, name)
    }
    return {
        "tensors": tensors,
        "attributes": attributes,
        "tensor_audit": _state_dict_audit(tensors),
    }


def _module_snapshot(model) -> Dict[str, Any]:
    module = model.ATF if hasattr(model, "ATF") else model.TCM
    state = _state_dict_cpu(module)
    return {
        "state_dict": state,
        "tensor_audit": _state_dict_audit(state),
        "training": bool(module.training),
    }


def audit_components(gaussians, atf, tcm, optimizers=None) -> Dict[str, Any]:
    optimizers = optimizers or {
        "gaussian": gaussians.optimizer,
        "atf": atf.optimizer,
        "tcm": tcm.optimizer,
    }
    gaussian = _gaussian_snapshot(gaussians)
    atf_snapshot = _module_snapshot(atf)
    tcm_snapshot = _module_snapshot(tcm)
    result = {
        "gaussian": {
            "attributes": gaussian["attributes"],
            "tensors": gaussian["tensor_audit"],
        },
        "atf": {
            "training": atf_snapshot["training"],
            "tensors": atf_snapshot["tensor_audit"],
        },
        "tcm": {
            "training": tcm_snapshot["training"],
            "tensors": tcm_snapshot["tensor_audit"],
        },
        "optimizer": {
            name: optimizer_audit(value) for name, value in sorted(optimizers.items())
        },
    }
    return result


def _snapshot_payload(gaussians, atf, tcm) -> Dict[str, Any]:
    gaussian = _gaussian_snapshot(gaussians)
    atf_snapshot = _module_snapshot(atf)
    tcm_snapshot = _module_snapshot(tcm)
    return {
        "gaussian": gaussian,
        "atf": atf_snapshot,
        "tcm": tcm_snapshot,
        "optimizer": {
            "gaussian": optimizer_audit(gaussians.optimizer),
            "atf": optimizer_audit(atf.optimizer),
            "tcm": optimizer_audit(tcm.optimizer),
        },
    }


def save_initialization_package(output_dir, *, seed: int, gaussians, atf, tcm, metadata=None) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng_state = capture_rng_state()
    snapshot = _snapshot_payload(gaussians, atf, tcm)
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "snapshot": snapshot,
        "rng_state": rng_state,
        "rng_state_hashes": rng_state_hashes(rng_state),
        "metadata": dict(metadata or {}),
    }
    package_path = output_dir / "init_step0.pt"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".init_step0.", suffix=".tmp", dir=str(output_dir)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, str(temporary_path))
        os.replace(str(temporary_path), str(package_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    package_hash = hashlib.sha256(package_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "seed": int(seed),
        "package": "init_step0.pt",
        "package_sha256": package_hash,
        "tensor_audit": audit_components(gaussians, atf, tcm),
        "rng_state_hashes": payload["rng_state_hashes"],
        "metadata": dict(metadata or {}),
    }
    write_json_atomic(output_dir / "init_manifest.json", manifest)
    return manifest


def _load_package(path: Path) -> Dict[str, Any]:
    try:
        package = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        package = torch.load(str(path), map_location="cpu")
    if not isinstance(package, dict) or package.get("schema_version") != 1:
        raise ValueError("invalid shared initialization package: {}".format(path))
    return package


def _copy_tensor(target: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if tuple(target.shape) != tuple(source.shape) or target.dtype != source.dtype:
        raise ValueError(
            "initialization tensor mismatch for {}: target {} {}, source {} {}".format(
                name, tuple(target.shape), target.dtype, tuple(source.shape), source.dtype
            )
        )
    target.data.copy_(source.to(device=target.device))


def load_initialization_package(path, *, seed: int, gaussians, atf, tcm) -> Dict[str, Any]:
    path = Path(path)
    package = _load_package(path / "init_step0.pt" if path.is_dir() else path)
    if int(package.get("seed", -1)) != int(seed):
        raise ValueError("shared initialization seed mismatch")
    snapshot = package["snapshot"]
    for name, source in snapshot["gaussian"]["tensors"].items():
        if not hasattr(gaussians, name):
            raise ValueError("initialization Gaussian field is unavailable: {}".format(name))
        _copy_tensor(getattr(gaussians, name), source, "gaussian." + name)
    for name, value in snapshot["gaussian"]["attributes"].items():
        if hasattr(gaussians, name) and _simple_value(getattr(gaussians, name)) != value:
            setattr(gaussians, name, value)
    atf.ATF.load_state_dict(snapshot["atf"]["state_dict"], strict=True)
    tcm.TCM.load_state_dict(snapshot["tcm"]["state_dict"], strict=True)
    atf.ATF.train(snapshot["atf"]["training"])
    tcm.TCM.train(snapshot["tcm"]["training"])
    expected_optimizers = snapshot["optimizer"]
    actual_optimizers = {
        "gaussian": optimizer_audit(gaussians.optimizer),
        "atf": optimizer_audit(atf.optimizer),
        "tcm": optimizer_audit(tcm.optimizer),
    }
    for name in expected_optimizers:
        expected = expected_optimizers[name]
        actual = actual_optimizers[name]
        if expected["param_groups"] != actual["param_groups"] or expected["state_status"] != actual["state_status"]:
            raise ValueError("initialization optimizer mismatch: {}".format(name))
    restore_rng_state(package["rng_state"])
    loaded_audit = audit_components(gaussians, atf, tcm)
    expected_audit = package["snapshot"]
    comparison = {
        "gaussian": loaded_audit["gaussian"] == {
            "attributes": expected_audit["gaussian"]["attributes"],
            "tensors": expected_audit["gaussian"]["tensor_audit"],
        },
        "atf": loaded_audit["atf"] == {
            "training": expected_audit["atf"]["training"],
            "tensors": expected_audit["atf"]["tensor_audit"],
        },
        "tcm": loaded_audit["tcm"] == {
            "training": expected_audit["tcm"]["training"],
            "tensors": expected_audit["tcm"]["tensor_audit"],
        },
        "optimizer": all(
            loaded_audit["optimizer"][name]["param_groups"]
            == expected_audit["optimizer"][name]["param_groups"]
            and loaded_audit["optimizer"][name]["state_status"]
            == expected_audit["optimizer"][name]["state_status"]
            for name in expected_audit["optimizer"]
        ),
        "rng": rng_state_hashes(package["rng_state"]) == package["rng_state_hashes"],
    }
    if not all(comparison.values()):
        raise ValueError("shared initialization post-load audit mismatch: {}".format(comparison))
    return {
        "schema_version": 1,
        "package": str(path),
        "seed": int(seed),
        "package_sha256": hashlib.sha256((path / "init_step0.pt" if path.is_dir() else path).read_bytes()).hexdigest(),
        "package_tensor_audit": {
            "gaussian": expected_audit["gaussian"]["tensor_audit"],
            "atf": expected_audit["atf"]["tensor_audit"],
            "tcm": expected_audit["tcm"]["tensor_audit"],
        },
        "loaded_audit": loaded_audit,
        "comparison": comparison,
        "rng_state_hashes": package["rng_state_hashes"],
        "optimizer_state": {
            name: expected_audit["optimizer"][name]["state_status"]
            for name in expected_audit["optimizer"]
        },
    }
