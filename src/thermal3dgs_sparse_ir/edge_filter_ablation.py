"""Strict, reproducible analysis for the Step2 edge-filter ablation.

The analyzer intentionally consumes already collected validation metrics.  It
checks their relationship to the immutable render products before deriving
paired deltas, resource summaries, and method-independent qualitative figures.
"""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePath
import platform
import re
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


METHODS = ("B0", "E2_no_filter", "E2")
CHECKPOINTS = (2000, 7000)
METRICS = ("psnr", "ssim", "t_mae", "e_mae", "gradient_preservation")
HIGHER_IS_BETTER = frozenset(("psnr", "ssim", "gradient_preservation"))
COMPARISONS = (
    ("E2_minus_E2_no_filter", "E2", "E2_no_filter"),
    ("E2_minus_B0", "E2", "B0"),
    ("E2_no_filter_minus_B0", "E2_no_filter", "B0"),
)
EXPECTED_VIEWS = 34
SELECTED_FILENAMES = ("00000.png", "00011.png", "00022.png", "00033.png")
EXPECTED_SELECTED_VIEW_IDS = {
    "00000.png": "001.jpg",
    "00011.png": "101.jpg",
    "00022.png": "202.jpg",
    "00033.png": "302.jpg",
}
SUMMARY_TOLERANCE = Decimal("1e-8")
RESULT_KIND = "edge-filter-ablation-results"
RESULT_SCHEMA_VERSION = 1
GENERATOR_VERSION = "1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INPUT_SHA256 = {
    "dataset_manifest": "d4fe9c97ed110fa1ecdf0636a19f7a7062169d3cfbd719d5594158f9d615e919",
    "split_manifest": "c9a1e0aa65e54e1558779b557539cb2eaba4d1faab765b786a4437aa519ba79e",
    "degradation_manifest": "06d68ec82b7faa8d65168b9bd2b35f013207331f4683a2ec0ad883837be3f03b",
}
EXPECTED_RUN_NAMES = {
    "B0": "B0_trainseed2026",
    "E2_no_filter": "E2_no_filter_trainseed2026",
    "E2": "E2_trainseed2026",
}
EXPECTED_VAL_VIEW_IDS = (
    "001.jpg", "010.jpg", "019.jpg", "028.jpg", "037.jpg", "046.jpg",
    "055.jpg", "065.jpg", "074.jpg", "083.jpg", "092.jpg", "101.jpg",
    "110.jpg", "119.jpg", "129.jpg", "138.jpg", "147.jpg", "156.jpg",
    "165.jpg", "174.jpg", "183.jpg", "193.jpg", "202.jpg", "211.jpg",
    "220.jpg", "229.jpg", "238.jpg", "247.jpg", "257.jpg", "266.jpg",
    "275.jpg", "284.jpg", "293.jpg", "302.jpg",
)
EXPECTED_RENDER_FILENAMES = tuple(
    "{:05d}.png".format(index) for index in range(EXPECTED_VIEWS)
)
FIXED_SOURCE_PATH = REPOSITORY_ROOT / "data" / "TI-NSD" / "heated"
FIXED_DATASET_MANIFEST = FIXED_SOURCE_PATH / "dataset_manifest.v1.json"
FIXED_SPLIT_MANIFEST = (
    FIXED_SOURCE_PATH / "splits" / "seed2026" / "sparse_nested-random_25.json"
)
FIXED_DEGRADATION_MANIFEST = (
    FIXED_SOURCE_PATH
    / "derived"
    / "noise03_sparse25_seed2026"
    / "degradation_manifest.v1.json"
)

SUMMARY_FIELDS = (
    "experiment",
    "num_views",
    "psnr",
    "ssim",
    "lpips",
    "t_mae",
    "e_mae",
    "gradient_preservation",
    "roi_mae",
)
PER_VIEW_FIELDS = (
    "experiment",
    "image",
    "psnr",
    "ssim",
    "lpips",
    "t_mae",
    "e_mae",
    "gradient_preservation",
    "roi_mae",
)
DELTA_FIELDS = (
    "iteration",
    "comparison",
    "lhs",
    "rhs",
    "metric",
    "better_direction",
    "lhs_mean",
    "rhs_mean",
    "delta",
    "mean_outcome",
    "num_views",
    "per_view_improved",
    "per_view_regressed",
    "per_view_tied",
)
PER_VIEW_DELTA_FIELDS = (
    "iteration",
    "comparison",
    "lhs",
    "rhs",
    "image",
    "val_view_id",
    "metric",
    "better_direction",
    "lhs_value",
    "rhs_value",
    "delta",
    "oriented_delta",
    "outcome",
)
RESOURCE_FIELDS = (
    "experiment",
    "iteration",
    "run_status",
    "train_wall_seconds",
    "runner_final_render_wall_seconds",
    "gaussian_count",
    "core_avg_ms",
    "cuda_allocated_peak_mb",
    "point_cloud_bytes",
    "atf_bytes",
    "tcm_bytes",
    "model_total_bytes",
    "render_wall_seconds",
    "render_forward_mean_ms",
    "render_samples_total",
    "render_warmup_excluded",
    "render_timed_samples",
)
LOSSLESS_SUFFIXES = frozenset(
    (".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm", ".pbm", ".pnm")
)
LOSSY_SUFFIXES = frozenset((".jpg", ".jpeg", ".jpe", ".webp"))
OUTPUT_NAMES = {
    "metrics_deltas": "metrics_deltas.csv",
    "per_view_deltas": "per_view_deltas.csv",
    "resource_summary": "resource_summary.csv",
    "qualitative_full": "qualitative_full.png",
    "absolute_error": "absolute_error_maps.png",
    "qualitative_zoom": "qualitative_center_zoom.png",
    "manifest": "results_manifest.json",
}


class AblationAnalysisError(ValueError):
    """Raised when an input violates the frozen Step2 analysis contract."""


class AblationDependencyError(RuntimeError):
    """Raised when qualitative-figure dependencies are unavailable."""


def _duplicate_rejecting_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AblationAnalysisError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError("required JSON file does not exist: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source, object_pairs_hook=_duplicate_rejecting_object)
    except json.JSONDecodeError as exc:
        raise AblationAnalysisError("invalid JSON in {}: {}".format(path, exc))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_file_record(kind: str, path: Path, **metadata: Any) -> Dict[str, Any]:
    resolved = path.resolve()
    record = {
        "kind": kind,
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    record.update(metadata)
    return record


def _inventory_sha256(hashes: Mapping[str, str]) -> str:
    payload = "".join(
        "{}\0{}\n".format(name, hashes[name]) for name in sorted(hashes)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path, required_fields: Sequence[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError("required CSV file does not exist: {}".format(path))
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if fields is None:
            raise AblationAnalysisError("CSV has no header: {}".format(path))
        if len(fields) != len(set(fields)):
            raise AblationAnalysisError("CSV has duplicate header fields: {}".format(path))
        missing = [field for field in required_fields if field not in fields]
        if missing:
            raise AblationAnalysisError(
                "CSV {} is missing field(s): {}".format(path, ", ".join(missing))
            )
        rows: List[Dict[str, str]] = []
        for index, raw in enumerate(reader, start=2):
            if None in raw:
                raise AblationAnalysisError(
                    "CSV {} row {} has too many columns".format(path, index)
                )
            row = {key: (value or "").strip() for key, value in raw.items()}
            if not any(row.values()):
                raise AblationAnalysisError(
                    "CSV {} contains a blank row at {}".format(path, index)
                )
            rows.append(row)
    if not rows:
        raise AblationAnalysisError("CSV contains no data rows: {}".format(path))
    return rows


def _decimal(value: Any, location: str, positive: bool = False) -> Decimal:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise AblationAnalysisError("{} must be numeric".format(location))
    if not number.is_finite():
        raise AblationAnalysisError("{} must be finite".format(location))
    if positive and number <= 0:
        raise AblationAnalysisError("{} must be positive".format(location))
    return number


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise AblationAnalysisError("{} must be an integer".format(location))
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise AblationAnalysisError("{} must be an integer".format(location))
    number = int(text)
    if number < minimum:
        raise AblationAnalysisError("{} must be at least {}".format(location, minimum))
    return number


def _unavailable(value: str, location: str) -> None:
    if value != "unavailable":
        raise AblationAnalysisError("{} must be 'unavailable'".format(location))


def _fixed(value: Decimal) -> str:
    return "{:.8f}".format(value)


def _finite_float(value: Any, location: str, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AblationAnalysisError("{} must be numeric".format(location))
    if not math.isfinite(number):
        raise AblationAnalysisError("{} must be finite".format(location))
    if positive and number <= 0.0:
        raise AblationAnalysisError("{} must be positive".format(location))
    return number


def _validate_named_paths(
    values: Mapping[Any, Union[os.PathLike, str]], expected: Sequence[Any], label: str
) -> Dict[Any, Path]:
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(label))
    if set(values) != set(expected):
        raise AblationAnalysisError(
            "{} names must be exactly {}; got {}".format(
                label, ", ".join(str(item) for item in expected), ", ".join(str(item) for item in values)
            )
        )
    return {key: Path(values[key]).resolve() for key in expected}


def _append_frozen_arguments(command: List[str], arguments: Mapping[str, Any]) -> None:
    for key in sorted(arguments):
        value = arguments[key]
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif isinstance(value, (tuple, list)):
            command.append(flag)
            command.extend(str(item) for item in value)
        else:
            command.extend((flag, str(value)))


def _expected_frozen_commands(
    method: str, run_dir: Path
) -> Tuple[List[str], List[str]]:
    auxiliary = {
        "B0": ("legacy", "gaussian", 0.0),
        "E2_no_filter": ("filtered_edge", "identity", 0.001),
        "E2": ("filtered_edge", "gaussian", 0.001),
    }
    version, filter_mode, lambda_edge = auxiliary[method]
    arguments: Dict[str, Any] = {
        "aux_loss_version": version,
        "data_device": "cpu",
        "edge_filter_kernel": 5,
        "edge_filter_mode": filter_mode,
        "edge_filter_sigma": 1.0,
        "evaluation_partition": "val",
        "eval": True,
        "iterations": 7000,
        "load2gpu_on_the_fly": True,
        "log_interval": 500,
        "quiet": True,
        "resolution": 1,
        "save_iterations": (2000, 7000),
        "seed": 2026,
        "test_iterations": (2000, 7000),
        "lambda_edge": lambda_edge,
        "lambda_smooth": 0.0,
        "lambda_thermal": 0.0,
    }
    train_tail = [
        str((REPOSITORY_ROOT / "train.py").resolve()),
        "-s",
        str(FIXED_SOURCE_PATH.resolve()),
        "-m",
        str(run_dir.resolve()),
        "--dataset_manifest",
        str(FIXED_DATASET_MANIFEST.resolve()),
        "--split_manifest",
        str(FIXED_SPLIT_MANIFEST.resolve()),
        "--degradation_manifest",
        str(FIXED_DEGRADATION_MANIFEST.resolve()),
    ]
    _append_frozen_arguments(train_tail, arguments)
    render_tail = [
        str((REPOSITORY_ROOT / "render.py").resolve()),
        "-m",
        str(run_dir.resolve()),
        "--skip_train",
        "--evaluation_partition",
        "val",
        "--quiet",
    ]
    return train_tail, render_tail


def _validate_frozen_command(
    value: Any, expected_tail: Sequence[str], label: str, manifest_path: Path
) -> None:
    if (
        not isinstance(value, list)
        or len(value) < 2
        or any(not isinstance(token, str) or not token for token in value)
    ):
        raise AblationAnalysisError(
            "{}.{} must be a non-empty string array".format(manifest_path, label)
        )
    expected_python = (
        REPOSITORY_ROOT
        / ".venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ).resolve()
    if Path(value[0]).resolve() != expected_python:
        raise AblationAnalysisError(
            "{}.{} must use the repository .venv Python".format(manifest_path, label)
        )
    actual_tail = value[1:]
    if actual_tail != list(expected_tail):
        difference = next(
            (
                index
                for index, pair in enumerate(
                    zip(actual_tail, expected_tail), start=1
                )
                if pair[0] != pair[1]
            ),
            min(len(actual_tail), len(expected_tail)) + 1,
        )
        raise AblationAnalysisError(
            "{}.{} differs from the frozen Step2 command at token {}".format(
                manifest_path, label, difference
            )
        )


def _fixed_hash_mapping(value: Any, label: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_INPUT_SHA256):
        raise AblationAnalysisError(
            "{} must contain exactly {}".format(
                label, ", ".join(EXPECTED_INPUT_SHA256)
            )
        )
    normalized: Dict[str, str] = {}
    for key in EXPECTED_INPUT_SHA256:
        digest = value[key]
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
        ):
            raise AblationAnalysisError("{}.{} must be a SHA256 digest".format(label, key))
        normalized[key] = digest.lower()
    if normalized != EXPECTED_INPUT_SHA256:
        raise AblationAnalysisError("{} does not match the frozen Step2 SHA256 values".format(label))
    return normalized


def _metric_implementation_provenance() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from . import metrics as metrics_module

    metrics_path = (REPOSITORY_ROOT / "src" / "thermal3dgs_sparse_ir" / "metrics.py").resolve()
    collector_path = (REPOSITORY_ROOT / "tools" / "collect_metrics.py").resolve()
    loaded_path = Path(metrics_module.__file__).resolve()
    if loaded_path != metrics_path:
        raise AblationAnalysisError(
            "loaded metrics module is not the frozen repository source: {}".format(loaded_path)
        )
    official_ssim = getattr(metrics_module, "_official_ssim", None)
    backend = "official" if official_ssim is not None else "fallback"
    if official_ssim is None:
        implementation_path = metrics_path
        implementation_provider = "metrics_py"
    else:
        raw_implementation_path = inspect.getsourcefile(official_ssim)
        if raw_implementation_path is None:
            raise AblationAnalysisError("unable to resolve the official SSIM source file")
        implementation_path = Path(raw_implementation_path).resolve()
        implementation_provider = "source_file"
        if not implementation_path.is_file():
            raise AblationAnalysisError(
                "official SSIM source is not a regular file: {}".format(
                    implementation_path
                )
            )
    implementation = {
        "provider": implementation_provider,
        "path": str(implementation_path),
        "sha256": _sha256_file(implementation_path),
    }
    runtime = {
        "python": platform.python_version(),
        "torch": metrics_module.torch.__version__,
    }
    details = {
        "metrics_source_sha256": _sha256_file(metrics_path),
        "collect_metrics_source_sha256": _sha256_file(collector_path),
        "ssim_backend": backend,
        "ssim_implementation": implementation,
        "runtime": runtime,
    }
    inputs = [
        _input_file_record("metrics-source", metrics_path, ssim_backend=backend),
        _input_file_record("metrics-cli-source", collector_path),
    ]
    if implementation_path != metrics_path:
        inputs.append(
            _input_file_record(
                "ssim-implementation-source",
                implementation_path,
                provider=implementation_provider,
            )
        )
    return details, inputs


def _manifest_hash_map(value: Any, label: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_RENDER_FILENAMES):
        raise AblationAnalysisError(
            "{} must contain exactly the frozen 34 render filenames".format(label)
        )
    result: Dict[str, str] = {}
    for name in EXPECTED_RENDER_FILENAMES:
        digest = value[name]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise AblationAnalysisError("{}[{}] must be a SHA256 digest".format(label, name))
        result[name] = digest.lower()
    return result


def _validate_metric_manifest(
    metric_dir: Path,
    iteration: int,
    metric_provenance: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    path = metric_dir / "metrics_manifest.json"
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise AblationAnalysisError("metrics_manifest must contain an object: {}".format(path))
    if (
        value.get("schema") != "thermal3dgs.metrics_manifest"
        or value.get("schema_version") != 1
        or value.get("kind") != "metric_collection"
    ):
        raise AblationAnalysisError("invalid metrics_manifest identity: {}".format(path))

    source = value.get("source")
    if not isinstance(source, Mapping):
        raise AblationAnalysisError("metrics_manifest source is missing: {}".format(path))
    metrics_source = source.get("metrics_py")
    expected_metrics_path = (
        REPOSITORY_ROOT / "src" / "thermal3dgs_sparse_ir" / "metrics.py"
    ).resolve()
    if not isinstance(metrics_source, Mapping):
        raise AblationAnalysisError("metrics_manifest metrics_py is missing: {}".format(path))
    try:
        recorded_metrics_path = Path(str(metrics_source.get("path"))).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise AblationAnalysisError("invalid metrics.py path in {}: {}".format(path, exc))
    if recorded_metrics_path != expected_metrics_path:
        raise AblationAnalysisError("metrics_manifest references a different metrics.py: {}".format(path))
    if metrics_source.get("sha256") != metric_provenance["metrics_source_sha256"]:
        raise AblationAnalysisError("metrics.py SHA256 differs from current source: {}".format(path))
    if source.get("ssim_backend") != metric_provenance["ssim_backend"]:
        raise AblationAnalysisError("SSIM backend differs from current evaluator: {}".format(path))
    implementation = source.get("ssim_implementation")
    expected_implementation = metric_provenance["ssim_implementation"]
    if not isinstance(implementation, Mapping):
        raise AblationAnalysisError("SSIM implementation provenance is missing: {}".format(path))
    if implementation.get("provider") != expected_implementation["provider"]:
        raise AblationAnalysisError("SSIM implementation provider mismatch: {}".format(path))
    try:
        implementation_path = Path(str(implementation.get("path"))).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise AblationAnalysisError("invalid SSIM implementation path in {}: {}".format(path, exc))
    if (
        implementation_path != Path(expected_implementation["path"]).resolve()
        or implementation.get("sha256") != expected_implementation["sha256"]
        or not implementation_path.is_file()
        or _sha256_file(implementation_path) != expected_implementation["sha256"]
    ):
        raise AblationAnalysisError("SSIM implementation path/SHA256 mismatch: {}".format(path))
    runtime = source.get("runtime")
    if not isinstance(runtime, Mapping) or dict(runtime) != metric_provenance["runtime"]:
        raise AblationAnalysisError("metric runtime differs from current Python/Torch: {}".format(path))

    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping):
        raise AblationAnalysisError("metrics_manifest parameters are missing: {}".format(path))
    if parameters.get("skip_lpips") is not True or parameters.get("lpips_net") != "vgg":
        raise AblationAnalysisError("metrics must use skip_lpips=true and lpips_net=vgg: {}".format(path))
    device = parameters.get("device")
    if device not in ("cpu", "cuda"):
        raise AblationAnalysisError("metrics device must be recorded as cpu or cuda: {}".format(path))

    output_names = ("metrics_summary.csv", "metrics_summary.md", "metrics_per_view.csv")
    outputs = value.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(output_names):
        raise AblationAnalysisError("metrics_manifest outputs are incomplete: {}".format(path))
    inputs = [_input_file_record("metrics-manifest", path, iteration=iteration)]
    for name in output_names:
        output_path = metric_dir / name
        record = outputs[name]
        if not isinstance(record, Mapping):
            raise AblationAnalysisError("invalid metrics output record for {}".format(name))
        expected_bytes = _integer(record.get("bytes"), "metrics output bytes", 1)
        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
            raise AblationAnalysisError("invalid metrics output SHA256 for {}".format(name))
        if not output_path.is_file():
            raise FileNotFoundError("required metrics output does not exist: {}".format(output_path))
        if output_path.stat().st_size != expected_bytes or _sha256_file(output_path) != expected_sha.lower():
            raise AblationAnalysisError("metrics output hash/size mismatch: {}".format(output_path))
        if name == "metrics_summary.md":
            inputs.append(_input_file_record("metrics-summary-markdown", output_path, iteration=iteration))

    experiments = value.get("experiments")
    if not isinstance(experiments, list) or any(not isinstance(item, Mapping) for item in experiments):
        raise AblationAnalysisError("metrics_manifest experiments must be an array of objects")
    names = [item.get("name") for item in experiments]
    if names != sorted(METHODS):
        raise AblationAnalysisError(
            "metrics_manifest experiments must be exactly {}".format(", ".join(sorted(METHODS)))
        )
    return dict(value), inputs


def _validate_metric_image_binding(
    manifests: Mapping[int, Mapping[str, Any]],
    products: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Path]]]],
    run_dirs: Mapping[str, Path],
) -> None:
    for iteration in CHECKPOINTS:
        by_method = {
            str(record["name"]): record
            for record in manifests[iteration]["experiments"]
        }
        for method in METHODS:
            record = by_method[method]
            method_dir = (run_dirs[method] / "val" / "ours_{}".format(iteration)).resolve()
            try:
                recorded_method_dir = Path(str(record.get("method_dir"))).resolve()
            except (OSError, TypeError, ValueError) as exc:
                raise AblationAnalysisError(
                    "invalid metric method_dir for {} iteration {}: {}".format(method, iteration, exc)
                )
            if recorded_method_dir != method_dir:
                raise AblationAnalysisError(
                    "metric method_dir mismatch for {} iteration {}".format(method, iteration)
                )
            if record.get("roi_masks", object()) is not None:
                raise AblationAnalysisError(
                    "ROI masks must be unavailable for {} iteration {}".format(method, iteration)
                )

            role_hashes: Dict[str, Dict[str, str]] = {}
            for role in ("renders", "gt"):
                role_record = record.get(role)
                if not isinstance(role_record, Mapping):
                    raise AblationAnalysisError(
                        "metric inventory is missing {} for {} iteration {}".format(
                            role, method, iteration
                        )
                    )
                expected_directory = (method_dir / role).resolve()
                if Path(str(role_record.get("directory"))).resolve() != expected_directory:
                    raise AblationAnalysisError(
                        "metric {} directory mismatch for {} iteration {}".format(
                            role, method, iteration
                        )
                    )
                if _integer(role_record.get("count"), "metric image count", 1) != EXPECTED_VIEWS:
                    raise AblationAnalysisError("metric image count must be 34")
                hashes = _manifest_hash_map(
                    role_record.get("files"),
                    "metric {} files for {} iteration {}".format(role, method, iteration),
                )
                actual = {
                    name: _sha256_file(products[iteration][method][role][name])
                    for name in EXPECTED_RENDER_FILENAMES
                }
                if hashes != actual:
                    differing = [name for name in EXPECTED_RENDER_FILENAMES if hashes[name] != actual[name]]
                    raise AblationAnalysisError(
                        "metric {} image SHA256 mismatch for {} iteration {}: {}".format(
                            role, method, iteration, ", ".join(differing)
                        )
                    )
                role_hashes[role] = hashes

            inventory = record.get("inventory")
            if not isinstance(inventory, Mapping):
                raise AblationAnalysisError("metric inventory summary is missing")
            payload = {
                "renders": role_hashes["renders"],
                "gt": role_hashes["gt"],
                "roi_masks": None,
            }
            digest = hashlib.sha256(
                json.dumps(
                    payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            ).hexdigest()
            if (
                inventory.get("algorithm") != "sha256"
                or inventory.get("digest") != digest
                or _integer(inventory.get("count"), "metric inventory count", 1) != 68
                or _integer(inventory.get("view_count"), "metric view count", 1) != EXPECTED_VIEWS
            ):
                raise AblationAnalysisError(
                    "metric inventory summary mismatch for {} iteration {}".format(method, iteration)
                )


def _validate_metrics(
    metric_dirs: Mapping[int, Union[os.PathLike, str]],
    metric_provenance: Mapping[str, str],
) -> Tuple[
    Dict[int, Dict[str, Dict[str, Decimal]]],
    Dict[int, Dict[str, Dict[str, Dict[str, Decimal]]]],
    Dict[int, Sequence[str]],
    Dict[int, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    directories = _validate_named_paths(metric_dirs, CHECKPOINTS, "metric checkpoints")
    summaries: Dict[int, Dict[str, Dict[str, Decimal]]] = {}
    per_views: Dict[int, Dict[str, Dict[str, Dict[str, Decimal]]]] = {}
    names_by_checkpoint: Dict[int, Sequence[str]] = {}
    manifests: Dict[int, Dict[str, Any]] = {}
    inputs: List[Dict[str, Any]] = []
    reference_names: Optional[Tuple[str, ...]] = None

    for iteration in CHECKPOINTS:
        summary_path = directories[iteration] / "metrics_summary.csv"
        per_view_path = directories[iteration] / "metrics_per_view.csv"
        metric_manifest, manifest_inputs = _validate_metric_manifest(
            directories[iteration], iteration, metric_provenance
        )
        manifests[iteration] = metric_manifest
        summary_rows = _read_csv(summary_path, SUMMARY_FIELDS)
        per_view_rows = _read_csv(per_view_path, PER_VIEW_FIELDS)
        inputs.extend(
            (
                _input_file_record("metrics-summary", summary_path, iteration=iteration),
                _input_file_record("metrics-per-view", per_view_path, iteration=iteration),
            )
        )
        inputs.extend(manifest_inputs)

        summary: Dict[str, Dict[str, Decimal]] = {}
        for row_index, row in enumerate(summary_rows, start=2):
            method = row["experiment"]
            if method in summary:
                raise AblationAnalysisError(
                    "duplicate summary experiment {!r} at iteration {}".format(method, iteration)
                )
            if method not in METHODS:
                raise AblationAnalysisError(
                    "unexpected summary experiment {!r} at iteration {}".format(method, iteration)
                )
            if _integer(row["num_views"], "summary num_views", 1) != EXPECTED_VIEWS:
                raise AblationAnalysisError(
                    "{} iteration {} must report exactly {} views".format(
                        method, iteration, EXPECTED_VIEWS
                    )
                )
            _unavailable(row["lpips"], "summary lpips")
            _unavailable(row["roi_mae"], "summary roi_mae")
            summary[method] = {
                metric: _decimal(
                    row[metric],
                    "summary {} {} {}".format(iteration, method, metric),
                )
                for metric in METRICS
            }
        if set(summary) != set(METHODS):
            raise AblationAnalysisError(
                "iteration {} summary must contain exactly {}".format(iteration, ", ".join(METHODS))
            )

        per_view: Dict[str, Dict[str, Dict[str, Decimal]]] = {
            method: {} for method in METHODS
        }
        for row_index, row in enumerate(per_view_rows, start=2):
            method = row["experiment"]
            if method not in per_view:
                raise AblationAnalysisError(
                    "unexpected per-view experiment {!r} at iteration {}".format(method, iteration)
                )
            image_name = row["image"]
            if (
                not image_name
                or PurePath(image_name).name != image_name
                or Path(image_name).suffix.lower() != ".png"
            ):
                raise AblationAnalysisError(
                    "invalid per-view image name at iteration {} row {}: {!r}".format(
                        iteration, row_index, image_name
                    )
                )
            if image_name in per_view[method]:
                raise AblationAnalysisError(
                    "duplicate per-view row for {} {} at iteration {}".format(
                        method, image_name, iteration
                    )
                )
            _unavailable(row["lpips"], "per-view lpips")
            _unavailable(row["roi_mae"], "per-view roi_mae")
            per_view[method][image_name] = {
                metric: _decimal(
                    row[metric],
                    "per-view {} {} {} {}".format(iteration, method, image_name, metric),
                )
                for metric in METRICS
            }

        method_names: Optional[Tuple[str, ...]] = None
        for method in METHODS:
            names = tuple(sorted(per_view[method]))
            if len(names) != EXPECTED_VIEWS:
                raise AblationAnalysisError(
                    "{} iteration {} has {} views, expected {}".format(
                        method, iteration, len(names), EXPECTED_VIEWS
                    )
                )
            if method_names is not None and names != method_names:
                raise AblationAnalysisError(
                    "iteration {} methods have different per-view filename sets".format(iteration)
                )
            method_names = names
            for metric in METRICS:
                recomputed = sum(
                    (per_view[method][name][metric] for name in names), Decimal(0)
                ) / Decimal(EXPECTED_VIEWS)
                difference = abs(recomputed - summary[method][metric])
                if difference > SUMMARY_TOLERANCE:
                    raise AblationAnalysisError(
                        "summary mismatch for iteration {} {} {}: difference {} exceeds {}".format(
                            iteration, method, metric, difference, SUMMARY_TOLERANCE
                        )
                    )
        assert method_names is not None
        if reference_names is not None and method_names != reference_names:
            raise AblationAnalysisError("2k and 7k per-view filename sets differ")
        reference_names = method_names
        summaries[iteration] = summary
        per_views[iteration] = per_view
        names_by_checkpoint[iteration] = method_names
    return summaries, per_views, names_by_checkpoint, manifests, inputs


def _list_png_images(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError("image directory does not exist: {}".format(directory))
    images: Dict[str, Path] = {}
    invalid_images: List[str] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".png":
            images[path.name] = path
        elif suffix in LOSSLESS_SUFFIXES or suffix in LOSSY_SUFFIXES:
            invalid_images.append(path.name)
    if invalid_images:
        raise AblationAnalysisError(
            "only PNG render products are accepted in {}: {}".format(
                directory, ", ".join(sorted(invalid_images))
            )
        )
    if not images:
        raise AblationAnalysisError("no PNG images found in {}".format(directory))
    return images


def _png_signature(path: Path) -> Tuple[int, int, str]:
    try:
        from PIL import Image
    except Exception as exc:
        raise AblationDependencyError(
            "PNG validation requires Pillow: {}".format(exc)
        )
    try:
        with Image.open(str(path)) as image:
            image.load()
            if image.format != "PNG":
                raise AblationAnalysisError(
                    "image has a .png name but is not encoded as PNG: {}".format(path)
                )
            width, height = image.size
            mode = image.mode
    except AblationAnalysisError:
        raise
    except (OSError, ValueError) as exc:
        raise AblationAnalysisError("could not validate PNG {}: {}".format(path, exc))
    if width < 1 or height < 1 or not mode:
        raise AblationAnalysisError("PNG has invalid dimensions or mode: {}".format(path))
    return width, height, mode


def _validate_render_products(
    run_dirs: Mapping[str, Path],
    names_by_checkpoint: Mapping[int, Sequence[str]],
    camera_dimensions: Mapping[str, Tuple[int, int]],
) -> Tuple[
    Dict[int, Dict[str, Dict[str, Dict[str, Path]]]],
    Dict[str, str],
    List[Dict[str, Any]],
]:
    products: Dict[int, Dict[str, Dict[str, Dict[str, Path]]]] = {}
    inputs: List[Dict[str, Any]] = []
    reference_gt_hashes: Optional[Dict[str, str]] = None

    for iteration in CHECKPOINTS:
        expected_names = set(names_by_checkpoint[iteration])
        if tuple(sorted(expected_names)) != EXPECTED_RENDER_FILENAMES:
            raise AblationAnalysisError(
                "iteration {} render filenames do not match the frozen 34-view sequence".format(
                    iteration
                )
            )
        products[iteration] = {}
        for method in METHODS:
            method_dir = run_dirs[method] / "val" / "ours_{}".format(iteration)
            renders = _list_png_images(method_dir / "renders")
            ground_truth = _list_png_images(method_dir / "gt")
            if set(renders) != expected_names or set(ground_truth) != expected_names:
                raise AblationAnalysisError(
                    "{} iteration {} render/GT filenames do not match metrics".format(method, iteration)
                )
            render_hashes = {name: _sha256_file(renders[name]) for name in sorted(renders)}
            gt_hashes = {
                name: _sha256_file(ground_truth[name]) for name in sorted(ground_truth)
            }
            for name in sorted(expected_names):
                render_signature = _png_signature(renders[name])
                gt_signature = _png_signature(ground_truth[name])
                expected_size = camera_dimensions[name]
                if render_signature[:2] != expected_size or gt_signature[:2] != expected_size:
                    raise AblationAnalysisError(
                        "{} iteration {} {} image dimensions do not match its camera {}".format(
                            method, iteration, name, expected_size
                        )
                    )
                if render_signature[2] != gt_signature[2]:
                    raise AblationAnalysisError(
                        "{} iteration {} {} render/GT image modes differ".format(
                            method, iteration, name
                        )
                    )
            if reference_gt_hashes is not None and gt_hashes != reference_gt_hashes:
                different = sorted(
                    name
                    for name in expected_names
                    if gt_hashes.get(name) != reference_gt_hashes.get(name)
                )
                raise AblationAnalysisError(
                    "ground-truth SHA256 differs across methods/checkpoints: {}".format(
                        ", ".join(different)
                    )
                )
            reference_gt_hashes = gt_hashes
            products[iteration][method] = {"renders": renders, "gt": ground_truth}
            inputs.extend(
                (
                    {
                        "kind": "render-image-inventory",
                        "method": method,
                        "iteration": iteration,
                        "path": (method_dir / "renders").resolve().as_posix(),
                        "file_count": len(render_hashes),
                        "sha256": _inventory_sha256(render_hashes),
                    },
                    {
                        "kind": "ground-truth-image-inventory",
                        "method": method,
                        "iteration": iteration,
                        "path": (method_dir / "gt").resolve().as_posix(),
                        "file_count": len(gt_hashes),
                        "sha256": _inventory_sha256(gt_hashes),
                    },
                )
            )
    assert reference_gt_hashes is not None
    return products, reference_gt_hashes, inputs


def _validate_camera_record(record: Mapping[str, Any], path: Path, camera_id: int) -> None:
    required = {"id", "img_name", "width", "height", "position", "rotation", "fy", "fx"}
    missing = required - set(record)
    if missing:
        raise AblationAnalysisError(
            "camera {} in {} is missing field(s): {}".format(
                camera_id, path, ", ".join(sorted(missing))
            )
        )
    _integer(record["width"], "camera width", 1)
    _integer(record["height"], "camera height", 1)
    for field in ("fx", "fy"):
        if isinstance(record[field], bool):
            raise AblationAnalysisError("camera {} must be numeric".format(field))
        _finite_float(record[field], "camera {}".format(field), positive=True)
    position = record["position"]
    rotation = record["rotation"]
    if not isinstance(position, list) or len(position) != 3:
        raise AblationAnalysisError("camera position must contain three values: {}".format(path))
    if (
        not isinstance(rotation, list)
        or len(rotation) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
    ):
        raise AblationAnalysisError("camera rotation must be 3x3: {}".format(path))
    for label, numbers in (("position", position), ("rotation", sum(rotation, []))):
        for value in numbers:
            if isinstance(value, bool):
                raise AblationAnalysisError("camera {} must be numeric".format(label))
            _finite_float(value, "camera {}".format(label))


def _camera_mapping(
    run_dirs: Mapping[str, Path], expected_names: Sequence[str]
) -> Tuple[Dict[str, str], Dict[str, Tuple[int, int]], List[Dict[str, Any]]]:
    if tuple(expected_names) != EXPECTED_RENDER_FILENAMES:
        raise AblationAnalysisError("metrics do not use the frozen 34 render filenames")
    reference: Optional[Dict[str, str]] = None
    reference_records: Optional[Dict[str, Dict[str, Any]]] = None
    reference_dimensions: Optional[Dict[str, Tuple[int, int]]] = None
    inputs: List[Dict[str, Any]] = []
    for method in METHODS:
        path = run_dirs[method] / "cameras.json"
        value = _read_json(path)
        if not isinstance(value, list):
            raise AblationAnalysisError("cameras.json must contain an array: {}".format(path))
        by_id: Dict[int, Dict[str, Any]] = {}
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                raise AblationAnalysisError("invalid camera record {} in {}".format(index, path))
            camera_id = _integer(raw.get("id"), "camera id", 0)
            image_name = raw.get("img_name")
            if not isinstance(image_name, str) or not image_name:
                raise AblationAnalysisError("camera img_name must be non-empty in {}".format(path))
            if camera_id in by_id:
                raise AblationAnalysisError("duplicate camera id {} in {}".format(camera_id, path))
            by_id[camera_id] = dict(raw)
        mapping: Dict[str, str] = {}
        records: Dict[str, Dict[str, Any]] = {}
        dimensions: Dict[str, Tuple[int, int]] = {}
        for render_index, image in enumerate(expected_names):
            if render_index not in by_id:
                raise AblationAnalysisError(
                    "cameras.json has no mapping for render index {}".format(render_index)
                )
            record = by_id[render_index]
            _validate_camera_record(record, path, render_index)
            image_name = record["img_name"]
            expected_view_id = EXPECTED_VAL_VIEW_IDS[render_index]
            if image_name != expected_view_id:
                raise AblationAnalysisError(
                    "camera {} must map to frozen val view {}, got {}".format(
                        render_index, expected_view_id, image_name
                    )
                )
            mapping[image] = image_name
            records[image] = record
            dimensions[image] = (int(record["width"]), int(record["height"]))
        if reference is not None and mapping != reference:
            raise AblationAnalysisError("validation camera mapping differs across methods")
        if reference_records is not None and records != reference_records:
            raise AblationAnalysisError(
                "validation camera geometry/intrinsics differ across methods"
            )
        reference = mapping
        reference_records = records
        reference_dimensions = dimensions
        inputs.append(_input_file_record("camera-map", path, method=method))
    assert reference is not None and reference_dimensions is not None
    selected = {name: reference[name] for name in SELECTED_FILENAMES}
    if selected != EXPECTED_SELECTED_VIEW_IDS:
        raise AblationAnalysisError(
            "fixed qualitative mapping changed: expected {}, got {}".format(
                EXPECTED_SELECTED_VIEW_IDS, selected
            )
        )
    return reference, reference_dimensions, inputs


def _phase_wall_seconds(manifest: Mapping[str, Any], phase: str, path: Path) -> float:
    timings = manifest.get("phase_timings")
    if not isinstance(timings, Mapping):
        raise AblationAnalysisError("{} is missing phase_timings".format(path))
    record = timings.get(phase)
    if not isinstance(record, Mapping):
        raise AblationAnalysisError("{} is missing phase_timings.{}".format(path, phase))
    timestamps = []
    for field in ("started_at_utc", "finished_at_utc"):
        raw = record.get(field)
        if not isinstance(raw, str) or not raw:
            raise AblationAnalysisError(
                "{}.phase_timings.{} is missing {}".format(path, phase, field)
            )
        try:
            timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            raise AblationAnalysisError(
                "{}.phase_timings.{}.{} is not ISO-8601".format(path, phase, field)
            )
        if timestamp.tzinfo is None:
            raise AblationAnalysisError(
                "{}.phase_timings.{}.{} must include a timezone".format(path, phase, field)
            )
        timestamps.append(timestamp)
    if timestamps[1] < timestamps[0]:
        raise AblationAnalysisError(
            "{}.phase_timings.{} finishes before it starts".format(path, phase)
        )
    return _finite_float(
        record.get("wall_seconds"),
        "{}.phase_timings.{}.wall_seconds".format(path, phase),
        positive=True,
    )


def _parse_render_time_text(path: Path) -> Tuple[List[float], float]:
    if not path.is_file():
        raise FileNotFoundError("required render timing file does not exist: {}".format(path))
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != EXPECTED_VIEWS + 1:
        raise AblationAnalysisError(
            "{} must contain {} samples and one mean line".format(path, EXPECTED_VIEWS)
        )
    samples: List[float] = []
    for index, line in enumerate(lines[:-1]):
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)ms", line)
        if match is None:
            raise AblationAnalysisError("invalid render sample {} in {}".format(index, path))
        samples.append(_finite_float(match.group(1), "render sample", positive=True))
    match = re.fullmatch(r"Mean time: ([0-9]+(?:\.[0-9]+)?)ms", lines[-1])
    if match is None:
        raise AblationAnalysisError("invalid render mean footer in {}".format(path))
    footer_mean = _finite_float(match.group(1), "render mean", positive=True)
    recomputed = sum(samples[5:]) / len(samples[5:])
    if abs(recomputed - footer_mean) > 0.02:
        raise AblationAnalysisError("render mean footer mismatch in {}".format(path))
    return samples, footer_mean


def _parse_render_timing(path: Path, iteration: int) -> Dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise AblationAnalysisError("render_timing.json must contain an object: {}".format(path))
    required = (
        "schema_version",
        "partition",
        "iteration",
        "wall_seconds",
        "num_views",
        "forward_mean_ms",
        "forward_samples_total",
        "warmup_excluded",
        "forward_timed_samples",
        "wall_scope",
        "forward_scope",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise AblationAnalysisError(
            "{} is missing timing field(s): {}".format(path, ", ".join(missing))
        )
    if value["schema_version"] != 1:
        raise AblationAnalysisError("render_timing.json schema_version must be 1: {}".format(path))
    if value["partition"] != "val":
        raise AblationAnalysisError("render_timing.json partition must be val: {}".format(path))
    if _integer(value["iteration"], "render iteration", 1) != iteration:
        raise AblationAnalysisError("render_timing.json iteration mismatch: {}".format(path))
    expected_wall_scope = "render_set loop including device transfer and PNG writes; excludes model loading"
    expected_forward_scope = "synchronized ATF, Gaussian renderer, and TCM; excludes PNG I/O"
    if value["wall_scope"] != expected_wall_scope or value["forward_scope"] != expected_forward_scope:
        raise AblationAnalysisError("render timing scope changed in {}".format(path))
    parsed = {
        "wall_seconds": _finite_float(value["wall_seconds"], "render wall_seconds", positive=True),
        "num_views": _integer(value["num_views"], "render num_views", 1),
        "forward_mean_ms": _finite_float(
            value["forward_mean_ms"], "render forward_mean_ms", positive=True
        ),
        "forward_samples_total": _integer(
            value["forward_samples_total"], "render forward_samples_total", 1
        ),
        "warmup_excluded": _integer(value["warmup_excluded"], "render warmup_excluded", 0),
        "forward_timed_samples": _integer(
            value["forward_timed_samples"], "render forward_timed_samples", 1
        ),
    }
    if parsed["num_views"] != EXPECTED_VIEWS:
        raise AblationAnalysisError("render_timing.json must report 34 views: {}".format(path))
    if parsed["forward_samples_total"] != EXPECTED_VIEWS:
        raise AblationAnalysisError("render_timing.json must report 34 forward samples: {}".format(path))
    if parsed["warmup_excluded"] != 5 or parsed["forward_timed_samples"] != 29:
        raise AblationAnalysisError(
            "render_timing.json must report five excluded warm-up views and 29 timed views: {}".format(path)
        )
    return parsed


def _validate_run_metadata(
    run_dirs: Mapping[str, Path]
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    inputs: List[Dict[str, Any]] = []
    reference_snapshot: Optional[Tuple[Any, Any, Any]] = None
    reference_hashes: Optional[Mapping[str, Any]] = None
    common_git: Dict[str, Any] = {}
    expected_auxiliary = {
        "B0": ("legacy", "gaussian", False, False, 0.0, 0.0, 0.0),
        "E2_no_filter": ("filtered_edge", "identity", True, False, 0.0, 0.001, 0.0),
        "E2": ("filtered_edge", "gaussian", True, True, 0.0, 0.001, 0.0),
    }
    for method in METHODS:
        path = run_dirs[method] / "run_manifest.json"
        manifest = _read_json(path)
        if not isinstance(manifest, Mapping):
            raise AblationAnalysisError("run_manifest must contain an object: {}".format(path))
        if manifest.get("schema_version") != 1:
            raise AblationAnalysisError("{} run_manifest schema_version must be 1".format(method))
        if manifest.get("experiment") != EXPECTED_RUN_NAMES[method]:
            raise AblationAnalysisError(
                "{} run_manifest experiment must be {}".format(
                    method, EXPECTED_RUN_NAMES[method]
                )
            )
        if manifest.get("status") != "completed":
            raise AblationAnalysisError("{} run status is not completed".format(method))
        train_wall = _phase_wall_seconds(manifest, "train", path)
        render_wall = _phase_wall_seconds(manifest, "render", path)
        git = manifest.get("git")
        if not isinstance(git, Mapping):
            raise AblationAnalysisError("{} is missing Git metadata".format(path))
        snapshot = (git.get("head"), git.get("diff_sha256"), git.get("source_snapshot_sha256"))
        if any(not isinstance(item, str) or not item for item in snapshot):
            raise AblationAnalysisError("{} contains incomplete Git snapshot metadata".format(path))
        if reference_snapshot is not None and snapshot != reference_snapshot:
            raise AblationAnalysisError("run Git snapshots differ across methods")
        if git.get("status") != "":
            raise AblationAnalysisError("{} run did not start from a clean Git worktree".format(method))
        reference_snapshot = snapshot
        input_hashes = _fixed_hash_mapping(
            manifest.get("input_file_sha256"),
            "{}.input_file_sha256".format(path),
        )
        if reference_hashes is not None and dict(input_hashes) != dict(reference_hashes):
            raise AblationAnalysisError("run input hashes differ across methods")
        reference_hashes = input_hashes
        expected_hashes = _fixed_hash_mapping(
            manifest.get("expected_input_file_sha256"),
            "{}.expected_input_file_sha256".format(path),
        )
        if expected_hashes != input_hashes:
            raise AblationAnalysisError("{} expected and actual input hashes differ".format(method))
        auxiliary = manifest.get("auxiliary_loss_config")
        if not isinstance(auxiliary, Mapping):
            raise AblationAnalysisError("{} is missing auxiliary_loss_config".format(path))
        for field in ("auxiliary_enabled", "filter_active"):
            if not isinstance(auxiliary.get(field), bool):
                raise AblationAnalysisError("{}.{} must be boolean".format(path, field))
        actual_auxiliary = (
            auxiliary.get("aux_loss_version"),
            auxiliary.get("edge_filter_mode"),
            auxiliary.get("auxiliary_enabled"),
            auxiliary.get("filter_active"),
            _finite_float(auxiliary.get("lambda_thermal"), "lambda_thermal"),
            _finite_float(auxiliary.get("lambda_edge"), "lambda_edge"),
            _finite_float(auxiliary.get("lambda_smooth"), "lambda_smooth"),
        )
        if actual_auxiliary != expected_auxiliary[method]:
            raise AblationAnalysisError(
                "{} auxiliary loss configuration does not match its method label".format(method)
            )
        if _integer(auxiliary.get("edge_filter_kernel"), "edge_filter_kernel", 1) != 5:
            raise AblationAnalysisError("{} edge_filter_kernel must remain 5".format(method))
        if _finite_float(auxiliary.get("edge_filter_sigma"), "edge_filter_sigma", positive=True) != 1.0:
            raise AblationAnalysisError("{} edge_filter_sigma must remain 1".format(method))
        train_tail, render_tail = _expected_frozen_commands(method, run_dirs[method])
        _validate_frozen_command(manifest.get("train_command"), train_tail, "train_command", path)
        _validate_frozen_command(
            manifest.get("render_command"), render_tail, "render_command", path
        )
        if (run_dirs[method] / "test").exists():
            raise AblationAnalysisError("{} contains forbidden test render output".format(method))
        metadata[method] = {
            "status": "completed",
            "train_wall_seconds": train_wall,
            "runner_render_wall_seconds": render_wall,
            "manifest": manifest,
            "auxiliary_loss_config": dict(auxiliary),
        }
        inputs.append(_input_file_record("run-manifest", path, method=method))
    assert reference_snapshot is not None and reference_hashes is not None
    common_git = {
        "head": reference_snapshot[0],
        "diff_sha256": reference_snapshot[1],
        "source_snapshot_sha256": reference_snapshot[2],
        "input_file_sha256": dict(reference_hashes),
        "auxiliary_loss_configs": {
            method: metadata[method]["auxiliary_loss_config"] for method in METHODS
        },
    }
    return metadata, inputs, common_git


def _resource_rows(
    run_dirs: Mapping[str, Path], run_metadata: Mapping[str, Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    inputs: List[Dict[str, Any]] = []
    for method in METHODS:
        loss_path = run_dirs[method] / "loss_components.csv"
        loss_rows = _read_csv(
            loss_path, ("iteration", "Gaussian_count", "avg_ms", "peak_cuda_mb")
        )
        by_iteration: Dict[int, Mapping[str, str]] = {}
        for raw in loss_rows:
            iteration = _integer(raw["iteration"], "loss iteration", 1)
            if iteration in by_iteration:
                raise AblationAnalysisError(
                    "duplicate loss_components iteration {} for {}".format(iteration, method)
                )
            by_iteration[iteration] = raw
        inputs.append(_input_file_record("loss-components", loss_path, method=method))

        for iteration in CHECKPOINTS:
            if iteration not in by_iteration:
                raise AblationAnalysisError(
                    "loss_components for {} is missing iteration {}".format(method, iteration)
                )
            loss = by_iteration[iteration]
            gaussian_count = _integer(loss["Gaussian_count"], "Gaussian_count", 1)
            core_avg_ms = _finite_float(loss["avg_ms"], "avg_ms", positive=True)
            peak_mb = _finite_float(loss["peak_cuda_mb"], "peak_cuda_mb", positive=True)

            model_paths = {
                "point_cloud": run_dirs[method]
                / "point_cloud"
                / "iteration_{}".format(iteration)
                / "point_cloud.ply",
                "atf": run_dirs[method]
                / "ATF"
                / "iteration_{}".format(iteration)
                / "ATF.pth",
                "tcm": run_dirs[method]
                / "TCM"
                / "iteration_{}".format(iteration)
                / "TCM.pth",
            }
            sizes: Dict[str, int] = {}
            for role, model_path in model_paths.items():
                if not model_path.is_file():
                    raise FileNotFoundError("model file does not exist: {}".format(model_path))
                sizes[role] = model_path.stat().st_size
                if sizes[role] <= 0:
                    raise AblationAnalysisError("model file is empty: {}".format(model_path))
                inputs.append(
                    _input_file_record(
                        "model-file", model_path, method=method, iteration=iteration, role=role
                    )
                )

            render_root = run_dirs[method] / "val" / "ours_{}".format(iteration)
            timing_path = render_root / "render_timing.json"
            text_path = render_root / "render_time.txt"
            timing = _parse_render_timing(timing_path, iteration)
            samples, footer_mean = _parse_render_time_text(text_path)
            if abs(timing["forward_mean_ms"] - footer_mean) > 0.02:
                raise AblationAnalysisError(
                    "render_timing.json and render_time.txt mean differ for {} iteration {}".format(
                        method, iteration
                    )
                )
            recomputed_mean = sum(samples[5:]) / len(samples[5:])
            if abs(timing["forward_mean_ms"] - recomputed_mean) > 0.02:
                raise AblationAnalysisError(
                    "render forward mean cannot be reproduced for {} iteration {}".format(
                        method, iteration
                    )
                )
            inputs.extend(
                (
                    _input_file_record(
                        "render-timing-json", timing_path, method=method, iteration=iteration
                    ),
                    _input_file_record(
                        "render-timing-text", text_path, method=method, iteration=iteration
                    ),
                )
            )
            runner_render = run_metadata[method]["runner_render_wall_seconds"]
            if iteration == CHECKPOINTS[-1] and runner_render + 0.1 < timing["wall_seconds"]:
                raise AblationAnalysisError(
                    "runner render wall time is shorter than render-set wall time for {}".format(method)
                )
            rows.append(
                {
                    "experiment": method,
                    "iteration": iteration,
                    "run_status": run_metadata[method]["status"],
                    "train_wall_seconds": "{:.8f}".format(
                        run_metadata[method]["train_wall_seconds"]
                    ),
                    "runner_final_render_wall_seconds": (
                        "{:.8f}".format(runner_render)
                        if iteration == CHECKPOINTS[-1]
                        else "unavailable"
                    ),
                    "gaussian_count": gaussian_count,
                    "core_avg_ms": "{:.8f}".format(core_avg_ms),
                    "cuda_allocated_peak_mb": "{:.8f}".format(peak_mb),
                    "point_cloud_bytes": sizes["point_cloud"],
                    "atf_bytes": sizes["atf"],
                    "tcm_bytes": sizes["tcm"],
                    "model_total_bytes": sum(sizes.values()),
                    "render_wall_seconds": "{:.8f}".format(timing["wall_seconds"]),
                    "render_forward_mean_ms": "{:.8f}".format(timing["forward_mean_ms"]),
                    "render_samples_total": timing["forward_samples_total"],
                    "render_warmup_excluded": timing["warmup_excluded"],
                    "render_timed_samples": timing["forward_timed_samples"],
                }
            )
    return rows, inputs


def _outcome(oriented_delta: Decimal) -> str:
    if oriented_delta > 0:
        return "improved"
    if oriented_delta < 0:
        return "regressed"
    return "tied"


def _delta_rows(
    summaries: Mapping[int, Mapping[str, Mapping[str, Decimal]]],
    per_views: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Decimal]]]],
    camera_mapping: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for iteration in CHECKPOINTS:
        image_names = sorted(per_views[iteration][METHODS[0]])
        for comparison, lhs, rhs in COMPARISONS:
            for metric in METRICS:
                direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
                multiplier = Decimal(1) if direction == "higher" else Decimal(-1)
                counts = {"improved": 0, "regressed": 0, "tied": 0}
                for image in image_names:
                    lhs_value = per_views[iteration][lhs][image][metric]
                    rhs_value = per_views[iteration][rhs][image][metric]
                    delta = lhs_value - rhs_value
                    oriented = delta * multiplier
                    result = _outcome(oriented)
                    counts[result] += 1
                    detail_rows.append(
                        {
                            "iteration": iteration,
                            "comparison": comparison,
                            "lhs": lhs,
                            "rhs": rhs,
                            "image": image,
                            "val_view_id": camera_mapping[image],
                            "metric": metric,
                            "better_direction": direction,
                            "lhs_value": _fixed(lhs_value),
                            "rhs_value": _fixed(rhs_value),
                            "delta": _fixed(delta),
                            "oriented_delta": _fixed(oriented),
                            "outcome": result,
                        }
                    )
                lhs_mean = summaries[iteration][lhs][metric]
                rhs_mean = summaries[iteration][rhs][metric]
                mean_delta = lhs_mean - rhs_mean
                summary_rows.append(
                    {
                        "iteration": iteration,
                        "comparison": comparison,
                        "lhs": lhs,
                        "rhs": rhs,
                        "metric": metric,
                        "better_direction": direction,
                        "lhs_mean": _fixed(lhs_mean),
                        "rhs_mean": _fixed(rhs_mean),
                        "delta": _fixed(mean_delta),
                        "mean_outcome": _outcome(mean_delta * multiplier),
                        "num_views": EXPECTED_VIEWS,
                        "per_view_improved": counts["improved"],
                        "per_view_regressed": counts["regressed"],
                        "per_view_tied": counts["tied"],
                    }
                )
    if len(summary_rows) != 30 or len(detail_rows) != 1020:
        raise AssertionError("unexpected Step2 delta row count")
    return summary_rows, detail_rows


def _plotting_dependencies() -> Tuple[Any, Any, Any]:
    try:
        import numpy as np
        from PIL import Image
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return np, Image, plt
    except Exception as exc:
        raise AblationDependencyError(
            "qualitative figures require numpy, Pillow, and matplotlib: {}".format(exc)
        )


def _load_luma(path: Path, np: Any, Image: Any) -> Any:
    try:
        with Image.open(str(path)) as image:
            image.load()
            mode = image.mode
            if mode in ("1", "L"):
                array = np.array(image.convert("L"), dtype=np.float32, copy=True) / 255.0
            elif mode == "LA":
                array = np.array(image.getchannel("L"), dtype=np.float32, copy=True) / 255.0
            elif mode.startswith("I;16") or mode == "I":
                array = np.array(image, dtype=np.float32, copy=True) / 65535.0
            elif mode == "F":
                array = np.array(image, dtype=np.float32, copy=True)
            else:
                rgb = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
                array = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    except (OSError, ValueError) as exc:
        raise AblationAnalysisError("could not load image {}: {}".format(path, exc))
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 3:
        raise AblationAnalysisError("image has invalid dimensions: {}".format(path))
    if not bool(np.isfinite(array).all()):
        raise AblationAnalysisError("image contains non-finite values: {}".format(path))
    if float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6:
        raise AblationAnalysisError("image lies outside fixed [0,1] range: {}".format(path))
    return array


def _row_label(axis: Any, image: str, view_id: str) -> None:
    axis.text(
        -0.03,
        0.5,
        "{} / {}".format(image, view_id),
        transform=axis.transAxes,
        fontsize=7,
        rotation=90,
        horizontalalignment="right",
        verticalalignment="center",
    )


def _save_figure(figure: Any, path: Path, dpi: int, plt: Any) -> None:
    figure.savefig(
        str(path),
        format="png",
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "thermal3dgs_sparse_ir.edge_filter_ablation/{}".format(GENERATOR_VERSION)},
    )
    plt.close(figure)


def _render_figures(
    staging_dir: Path,
    products: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Path]]]],
    camera_mapping: Mapping[str, str],
    dpi: int,
) -> Tuple[Dict[str, Path], Dict[str, List[int]], Dict[str, str]]:
    np, Image, plt = _plotting_dependencies()
    iteration = CHECKPOINTS[-1]
    method_labels = ("B0", "E2 no filter", "E2 Gaussian")
    cache: Dict[Path, Any] = {}

    def load(path: Path) -> Any:
        resolved = path.resolve()
        if resolved not in cache:
            cache[resolved] = _load_luma(resolved, np, Image)
        return cache[resolved]

    gt_maps: Dict[str, Any] = {}
    render_maps: Dict[str, Dict[str, Any]] = {method: {} for method in METHODS}
    crops: Dict[str, List[int]] = {}
    for name in SELECTED_FILENAMES:
        gt = load(products[iteration]["B0"]["gt"][name])
        gt_maps[name] = gt
        height, width = gt.shape
        crop = [
            width // 3,
            height // 3,
            (2 * width + 2) // 3,
            (2 * height + 2) // 3,
        ]
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise AblationAnalysisError("fixed center crop is empty for {}".format(name))
        crops[name] = crop
        for method in METHODS:
            rendered = load(products[iteration][method]["renders"][name])
            if rendered.shape != gt.shape:
                raise AblationAnalysisError(
                    "render/GT dimensions differ for {} {}".format(method, name)
                )
            render_maps[method][name] = rendered

    outputs: Dict[str, Path] = {}
    full_path = staging_dir / OUTPUT_NAMES["qualitative_full"]
    figure, axes = plt.subplots(
        len(SELECTED_FILENAMES), 4, squeeze=False, figsize=(12.0, 8.8), constrained_layout=True
    )
    for row, name in enumerate(SELECTED_FILENAMES):
        maps = [gt_maps[name]] + [render_maps[method][name] for method in METHODS]
        for column, image in enumerate(maps):
            axes[row, column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_axis_off()
        _row_label(axes[row, 0], name, camera_mapping[name])
    for column, title in enumerate(("Ground truth",) + method_labels):
        axes[0, column].set_title(title, fontsize=10)
    _save_figure(figure, full_path, dpi, plt)
    outputs["qualitative_full"] = full_path

    error_path = staging_dir / OUTPUT_NAMES["absolute_error"]
    figure, axes = plt.subplots(
        len(SELECTED_FILENAMES), 3, squeeze=False, figsize=(9.2, 8.8), constrained_layout=True
    )
    artist = None
    for row, name in enumerate(SELECTED_FILENAMES):
        for column, method in enumerate(METHODS):
            error = np.abs(render_maps[method][name] - gt_maps[name])
            artist = axes[row, column].imshow(
                error, cmap="magma", vmin=0.0, vmax=0.25
            )
            axes[row, column].set_axis_off()
        _row_label(axes[row, 0], name, camera_mapping[name])
    for column, title in enumerate(method_labels):
        axes[0, column].set_title(title, fontsize=10)
    assert artist is not None
    figure.colorbar(
        artist,
        ax=axes.ravel().tolist(),
        label="Absolute normalized intensity error",
        fraction=0.025,
        pad=0.02,
    )
    _save_figure(figure, error_path, dpi, plt)
    outputs["absolute_error"] = error_path

    zoom_path = staging_dir / OUTPUT_NAMES["qualitative_zoom"]
    figure, axes = plt.subplots(
        len(SELECTED_FILENAMES), 4, squeeze=False, figsize=(12.0, 8.8), constrained_layout=True
    )
    for row, name in enumerate(SELECTED_FILENAMES):
        x0, y0, x1, y1 = crops[name]
        maps = [gt_maps[name]] + [render_maps[method][name] for method in METHODS]
        for column, image in enumerate(maps):
            axes[row, column].imshow(
                image[y0:y1, x0:x1], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest"
            )
            axes[row, column].set_axis_off()
        _row_label(axes[row, 0], name, camera_mapping[name])
    for column, title in enumerate(("Ground truth",) + method_labels):
        axes[0, column].set_title(title, fontsize=10)
    _save_figure(figure, zoom_path, dpi, plt)
    outputs["qualitative_zoom"] = zoom_path
    return outputs, crops, EXPECTED_SELECTED_VIEW_IDS.copy()


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(serialized)


def analyze_edge_filter_ablation(
    metric_dirs: Mapping[int, Union[os.PathLike, str]],
    run_dirs: Mapping[str, Union[os.PathLike, str]],
    output_dir: Union[os.PathLike, str],
    overwrite: bool = False,
    dpi: int = 200,
) -> Dict[str, Path]:
    """Validate frozen Step2 inputs and create all derived result artifacts."""

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    output_root = Path(output_dir).resolve()
    destinations = {key: output_root / name for key, name in OUTPUT_NAMES.items()}
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing ablation output(s): {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    normalized_runs = _validate_named_paths(run_dirs, METHODS, "run directories")
    for method, path in normalized_runs.items():
        if not path.is_dir():
            raise FileNotFoundError("run directory does not exist for {}: {}".format(method, path))
    metric_provenance, metric_source_inputs = _metric_implementation_provenance()
    (
        summaries,
        per_views,
        names_by_checkpoint,
        metric_manifests,
        metric_inputs,
    ) = _validate_metrics(metric_dirs, metric_provenance)
    camera_mapping, camera_dimensions, camera_inputs = _camera_mapping(
        normalized_runs, names_by_checkpoint[CHECKPOINTS[-1]]
    )
    products, gt_hashes, image_inputs = _validate_render_products(
        normalized_runs, names_by_checkpoint, camera_dimensions
    )
    _validate_metric_image_binding(metric_manifests, products, normalized_runs)
    run_metadata, run_inputs, common_run = _validate_run_metadata(normalized_runs)
    resource_rows, resource_inputs = _resource_rows(normalized_runs, run_metadata)
    delta_rows, per_view_delta_rows = _delta_rows(summaries, per_views, camera_mapping)
    _plotting_dependencies()

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".edge-filter-ablation.", dir=str(output_root.parent)
    ) as temporary_dir:
        staging = Path(temporary_dir)
        staged = {
            "metrics_deltas": staging / OUTPUT_NAMES["metrics_deltas"],
            "per_view_deltas": staging / OUTPUT_NAMES["per_view_deltas"],
            "resource_summary": staging / OUTPUT_NAMES["resource_summary"],
        }
        _write_csv(staged["metrics_deltas"], DELTA_FIELDS, delta_rows)
        _write_csv(staged["per_view_deltas"], PER_VIEW_DELTA_FIELDS, per_view_delta_rows)
        _write_csv(staged["resource_summary"], RESOURCE_FIELDS, resource_rows)
        figure_paths, crops, selected_mapping = _render_figures(
            staging, products, camera_mapping, dpi
        )
        staged.update(figure_paths)

        all_inputs = (
            metric_inputs
            + image_inputs
            + camera_inputs
            + run_inputs
            + resource_inputs
            + metric_source_inputs
        )
        all_inputs.append(
            _input_file_record("analysis-source", Path(__file__).resolve())
        )
        output_records = []
        for key in (
            "metrics_deltas",
            "per_view_deltas",
            "resource_summary",
            "qualitative_full",
            "absolute_error",
            "qualitative_zoom",
        ):
            path = staged[key]
            output_records.append(
                {
                    "artifact": key,
                    "path": OUTPUT_NAMES[key],
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "kind": RESULT_KIND,
            "generator_version": GENERATOR_VERSION,
            "methods": list(METHODS),
            "checkpoints": list(CHECKPOINTS),
            "comparisons": [
                {"name": name, "lhs": lhs, "rhs": rhs} for name, lhs, rhs in COMPARISONS
            ],
            "metric_contract": {
                "metrics": list(METRICS),
                "higher_is_better": sorted(HIGHER_IS_BETTER),
                "lower_is_better": [metric for metric in METRICS if metric not in HIGHER_IS_BETTER],
                "delta": "lhs_minus_rhs",
                "tie_rule": "exact equality after reading the published 8-decimal CSV values",
                "summary_recompute_absolute_tolerance": str(SUMMARY_TOLERANCE),
                "lpips": "unavailable",
                "roi_mae": "unavailable",
                **metric_provenance,
            },
            "figure_protocol": {
                "iteration": CHECKPOINTS[-1],
                "selection_rule": (
                    "sort 34 validation views by frozen sequence order and select "
                    "floor(k*(N-1)/3) for k=0,1,2,3"
                ),
                "filenames": list(SELECTED_FILENAMES),
                "val_view_ids": selected_mapping,
                "crop_rule": (
                    "center one third, half-open [floor(W/3), floor(H/3), "
                    "ceil(2W/3), ceil(2H/3)]"
                ),
                "crop_xyxy": crops,
                "display_range": [0.0, 1.0],
                "absolute_error": {"cmap": "magma", "vmin": 0.0, "vmax": 0.25},
                "normalization": "fixed encoded range; no per-image min-max",
                "dpi": dpi,
            },
            "resource_contract": {
                "train_wall_seconds": "whole train subprocess from run_manifest.phase_timings.train",
                "runner_final_render_wall_seconds": (
                    "whole final render subprocess from run_manifest.phase_timings.render; "
                    "reported only for iteration 7000"
                ),
                "render_wall_seconds": "per-checkpoint render-set wall time including image writes",
                "render_forward_mean_ms": (
                    "synchronized ATF + Gaussian renderer + TCM forward time; first five views excluded; "
                    "PNG I/O excluded"
                ),
                "core_avg_ms": "cumulative trainer CUDA-event core average, not full iteration wall time",
                "cuda_allocated_peak_mb": "PyTorch CUDA allocator peak, not whole-card usage",
                "model_total_bytes": "point_cloud.ply + ATF.pth + TCM.pth",
            },
            "run_snapshot": common_run,
            "ground_truth_sha256": {name: gt_hashes[name] for name in sorted(gt_hashes)},
            "inputs": sorted(
                all_inputs,
                key=lambda item: (
                    str(item.get("kind", "")),
                    str(item.get("method", "")),
                    int(item.get("iteration", 0)),
                    str(item.get("path", "")),
                ),
            ),
            "outputs": output_records,
        }
        staged_manifest = staging / OUTPUT_NAMES["manifest"]
        _write_json(staged_manifest, manifest)
        staged["manifest"] = staged_manifest

        output_root.mkdir(parents=True, exist_ok=True)
        for key in OUTPUT_NAMES:
            os.replace(str(staged[key]), str(destinations[key]))
    return destinations


__all__ = [
    "AblationAnalysisError",
    "AblationDependencyError",
    "CHECKPOINTS",
    "COMPARISONS",
    "METHODS",
    "METRICS",
    "SELECTED_FILENAMES",
    "analyze_edge_filter_ablation",
]
