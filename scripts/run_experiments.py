#!/usr/bin/env python3
"""Run a manifest-pinned Thermal3D-GS experiment matrix."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESERVED_ARGUMENTS = frozenset(
    (
        "source_path",
        "model_path",
        "dataset_manifest",
        "split_manifest",
        "degradation_manifest",
        "stop_file",
        "initialization_path",
        "prepare_initialization",
    )
)
INPUT_HASH_KEYS = frozenset(
    ("dataset_manifest", "split_manifest", "degradation_manifest")
)


class MatrixError(ValueError):
    pass


class ExperimentInterrupted(RuntimeError):
    def __init__(self, phase: str, reason: str = "user_request"):
        self.phase = phase
        self.reason = reason
        super().__init__("experiment interrupted during {} ({})".format(phase, reason))


def _validate_aux_loss_arguments(arguments: Mapping[str, object], label: str) -> None:
    version = arguments.get("aux_loss_version", "legacy")
    if version not in ("legacy", "filtered_edge"):
        raise MatrixError(
            "{}.aux_loss_version must be 'legacy' or 'filtered_edge'".format(label)
        )
    filter_mode = arguments.get("edge_filter_mode", "gaussian")
    if filter_mode not in ("gaussian", "identity"):
        raise MatrixError(
            "{}.edge_filter_mode must be 'gaussian' or 'identity'".format(label)
        )
    weights = {}
    for name in ("lambda_thermal", "lambda_edge", "lambda_smooth"):
        value = arguments.get(name, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise MatrixError("{}.{} must be finite and non-negative".format(label, name))
        weights[name] = float(value)
    kernel = arguments.get("edge_filter_kernel", 5)
    if isinstance(kernel, bool) or not isinstance(kernel, int):
        raise MatrixError("{}.edge_filter_kernel must be an integer".format(label))
    if kernel < 1 or kernel % 2 == 0:
        raise MatrixError(
            "{}.edge_filter_kernel must be a positive odd integer".format(label)
        )
    sigma = arguments.get("edge_filter_sigma", 1.0)
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise MatrixError("{}.edge_filter_sigma must be numeric".format(label))
    if not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise MatrixError(
            "{}.edge_filter_sigma must be finite and positive".format(label)
        )
    if version == "filtered_edge":
        for name in ("lambda_thermal", "lambda_smooth"):
            if weights[name] != 0:
                raise MatrixError(
                    "filtered_edge requires lambda_thermal == 0 and lambda_smooth == 0"
                )
    lambda_detail = arguments.get("lambda_detail", 0.0)
    if (
        isinstance(lambda_detail, bool)
        or not isinstance(lambda_detail, (int, float))
        or not math.isfinite(float(lambda_detail))
        or float(lambda_detail) != 0.0
    ):
        raise MatrixError("{}.lambda_detail must remain exactly 0 on the E2/GD branch".format(label))


def _validate_gd_arguments(arguments: Mapping[str, object], label: str) -> None:
    max_rate = arguments.get("gd_max_rate", 0.0)
    if (
        isinstance(max_rate, bool)
        or not isinstance(max_rate, (int, float))
        or not math.isfinite(float(max_rate))
        or not 0.0 <= float(max_rate) < 1.0
    ):
        raise MatrixError("{}.gd_max_rate must be finite and in [0, 1)".format(label))
    warmup = arguments.get("gd_warmup_iterations", 1000)
    ramp_end = arguments.get("gd_ramp_end", 3000)
    gd_seed = arguments.get("gd_seed", 104729)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (warmup, ramp_end, gd_seed)):
        raise MatrixError("{}.gd warmup, ramp_end, and seed must be integers".format(label))
    if warmup < 0 or ramp_end <= warmup or gd_seed < 0:
        raise MatrixError("{}.gd schedule or seed is invalid".format(label))
    for name in ("log_interval", "tb_log_interval", "tb_flush_secs"):
        value = arguments.get(name, {"log_interval": 500, "tb_log_interval": 50, "tb_flush_secs": 5}[name])
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise MatrixError("{}.{} must be a positive integer".format(label, name))


def _resolved_auxiliary_loss_config(
    common_args: Mapping[str, object], experiment_args: Mapping[str, object]
) -> Dict[str, object]:
    arguments = dict(common_args)
    arguments.update(experiment_args)
    version = str(arguments.get("aux_loss_version", "legacy"))
    filter_mode = str(arguments.get("edge_filter_mode", "gaussian"))
    weights = {
        name: float(arguments.get(name, 0.0))
        for name in ("lambda_thermal", "lambda_edge", "lambda_smooth")
    }
    auxiliary_enabled = any(value > 0 for value in weights.values())
    return {
        "aux_loss_version": version,
        "auxiliary_enabled": auxiliary_enabled,
        "edge_filter_mode": filter_mode,
        "filter_active": bool(
            auxiliary_enabled
            and version == "filtered_edge"
            and weights["lambda_edge"] > 0
            and filter_mode == "gaussian"
        ),
        "edge_filter_kernel": int(arguments.get("edge_filter_kernel", 5)),
        "edge_filter_sigma": float(arguments.get("edge_filter_sigma", 1.0)),
        **weights,
    }


def _validate_expected_hashes(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or not value:
        raise MatrixError("{} must be a non-empty object".format(label))
    unknown = set(value) - INPUT_HASH_KEYS
    if unknown:
        raise MatrixError(
            "{} contains unknown input(s): {}".format(label, ", ".join(sorted(unknown)))
        )
    for name, digest in value.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise MatrixError("{}.{} must be a SHA256 hex digest".format(label, name))


def _read_json(path: Path) -> Dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MatrixError("duplicate JSON key: {}".format(key))
            result[key] = value
        return result

    def reject_constant(value):
        raise MatrixError("non-standard JSON numeric constant: {}".format(value))

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict):
        raise MatrixError("experiment matrix root must be an object")
    return value


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def load_matrix(path: Path) -> Dict[str, object]:
    matrix = _expand(_read_json(path))
    required = {
        "schema_version",
        "source_path",
        "dataset_manifest",
        "output_root",
        "common_args",
        "experiments",
    }
    allowed_root_fields = required | {
        "initialization_root",
        "results_root",
        "evaluation_iterations",
    }
    unknown = set(matrix) - allowed_root_fields
    missing = required - set(matrix)
    if missing or unknown:
        raise MatrixError(
            "invalid matrix fields; missing={}, unknown={}".format(
                sorted(missing), sorted(unknown)
            )
        )
    if matrix["schema_version"] != 1:
        raise MatrixError("only experiment matrix schema_version 1 is supported")
    for key in ("source_path", "dataset_manifest", "output_root"):
        if not isinstance(matrix[key], str) or not matrix[key]:
            raise MatrixError("{} must be a non-empty path string".format(key))
    for key in ("initialization_root", "results_root"):
        if key in matrix and (not isinstance(matrix[key], str) or not matrix[key]):
            raise MatrixError("{} must be a non-empty path string".format(key))
    evaluation_iterations = matrix.get("evaluation_iterations")
    if evaluation_iterations is not None:
        if (
            not isinstance(evaluation_iterations, list)
            or not evaluation_iterations
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in evaluation_iterations
            )
        ):
            raise MatrixError("evaluation_iterations must be a non-empty list of positive integers")
    if not isinstance(matrix["common_args"], dict):
        raise MatrixError("common_args must be an object")
    if not isinstance(matrix["experiments"], list) or not matrix["experiments"]:
        raise MatrixError("experiments must be a non-empty array")

    names = set()
    for index, experiment in enumerate(matrix["experiments"]):
        if not isinstance(experiment, dict):
            raise MatrixError("experiments[{}] must be an object".format(index))
        allowed = {
            "name",
            "split_manifest",
            "degradation_manifest",
            "expected_input_sha256",
            "initialization_seed",
            "args",
        }
        missing_experiment = {"name", "split_manifest", "args"} - set(experiment)
        unknown_experiment = set(experiment) - allowed
        if missing_experiment or unknown_experiment:
            raise MatrixError(
                "invalid experiments[{}] fields; missing={}, unknown={}".format(
                    index, sorted(missing_experiment), sorted(unknown_experiment)
                )
            )
        name = experiment["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or Path(name).name != name
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in name
            )
        ):
            raise MatrixError("experiment name must be one safe path component")
        if name in names:
            raise MatrixError("duplicate experiment name: {}".format(name))
        names.add(name)
        if not isinstance(experiment["split_manifest"], str):
            raise MatrixError("split_manifest must be a path string")
        degradation = experiment.get("degradation_manifest")
        if degradation is not None and not isinstance(degradation, str):
            raise MatrixError("degradation_manifest must be null or a path string")
        if not isinstance(experiment["args"], dict):
            raise MatrixError("experiment args must be an object")
        if "initialization_seed" in experiment and (
            isinstance(experiment["initialization_seed"], bool)
            or not isinstance(experiment["initialization_seed"], int)
            or experiment["initialization_seed"] < 0
        ):
            raise MatrixError("initialization_seed must be a non-negative integer")
        _validate_expected_hashes(
            experiment.get("expected_input_sha256"),
            "experiments[{}].expected_input_sha256".format(index),
        )
        for label, arguments in (
            ("common_args", matrix["common_args"]),
            ("experiments[{}].args".format(index), experiment["args"]),
        ):
            reserved = RESERVED_ARGUMENTS & set(arguments)
            if reserved:
                raise MatrixError(
                    "{} cannot override reserved argument(s): {}".format(
                        label, ", ".join(sorted(reserved))
                    )
                )
        resolved_arguments = dict(matrix["common_args"])
        resolved_arguments.update(experiment["args"])
        _validate_aux_loss_arguments(
            resolved_arguments, "experiments[{}]".format(index)
        )
        _validate_gd_arguments(
            resolved_arguments, "experiments[{}]".format(index)
        )
    return matrix


def _append_arguments(command: List[str], arguments: Mapping[str, object]) -> None:
    for key in sorted(arguments):
        if not key or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in key):
            raise MatrixError("invalid CLI argument name: {!r}".format(key))
        value = arguments[key]
        if key in RESERVED_ARGUMENTS:
            raise MatrixError("cannot override reserved CLI argument: {}".format(key))
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif value is None:
            continue
        elif isinstance(value, list):
            command.append(flag)
            command.extend(str(item) for item in value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise MatrixError("{} must be finite".format(flag))
            command.extend((flag, str(value)))
        elif isinstance(value, (str, int)):
            command.extend((flag, str(value)))
        else:
            raise MatrixError("unsupported value for {}: {!r}".format(flag, value))


def build_commands(
    matrix: Mapping[str, object], experiment: Mapping[str, object]
) -> Tuple[Path, List[str], List[str]]:
    output_path = _resolve_path(str(matrix["output_root"])) / str(experiment["name"])
    train_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "train.py"),
        "-s",
        str(_resolve_path(str(matrix["source_path"]))),
        "-m",
        str(output_path),
        "--dataset_manifest",
        str(_resolve_path(str(matrix["dataset_manifest"]))),
        "--split_manifest",
        str(_resolve_path(str(experiment["split_manifest"]))),
    ]
    degradation = experiment.get("degradation_manifest")
    if degradation:
        train_command.extend(
            ("--degradation_manifest", str(_resolve_path(str(degradation))))
        )
    arguments = dict(matrix["common_args"])
    arguments.update(experiment["args"])
    _append_arguments(train_command, arguments)
    initialization_root = matrix.get("initialization_root")
    if initialization_root:
        seed = experiment.get("initialization_seed", arguments.get("seed"))
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise MatrixError(
                "shared initialization requires a non-negative initialization_seed or args.seed"
            )
        train_command.extend(
            (
                "--initialization_path",
                str(_resolve_path(str(initialization_root)) / ("seed_{}".format(seed))),
            )
        )
    stop_file = _resolve_path(str(matrix["output_root"])) / "STOP_REQUESTED"
    train_command.extend(("--stop_file", str(stop_file)))
    evaluation_partition = arguments.get("evaluation_partition", "test")
    if evaluation_partition not in ("val", "test"):
        raise MatrixError("evaluation_partition must be 'val' or 'test'")
    render_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "render.py"),
        "-m",
        str(output_path),
        "--skip_train",
        "--evaluation_partition",
        evaluation_partition,
    ]
    if arguments.get("quiet") is True:
        render_command.append("--quiet")
    render_command.extend(("--stop_file", str(stop_file)))
    return output_path, train_command, render_command


def build_initialization_command(
    matrix: Mapping[str, object], experiment: Mapping[str, object], output_path: Path
) -> List[str]:
    """Build the native train.py command that prepares one step-0 package."""
    arguments = dict(matrix["common_args"])
    arguments.update(experiment["args"])
    seed = experiment.get("initialization_seed", arguments.get("seed"))
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MatrixError("shared initialization requires a non-negative seed")
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "train.py"),
        "-s",
        str(_resolve_path(str(matrix["source_path"]))),
        "-m",
        str(output_path),
        "--dataset_manifest",
        str(_resolve_path(str(matrix["dataset_manifest"]))),
        "--split_manifest",
        str(_resolve_path(str(experiment["split_manifest"]))),
    ]
    degradation = experiment.get("degradation_manifest")
    if degradation:
        command.extend(("--degradation_manifest", str(_resolve_path(str(degradation)))))
    _append_arguments(command, arguments)
    command.extend(("--prepare_initialization", str(output_path)))
    return command


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_ROOTS = (
    "train.py",
    "render.py",
    "arguments",
    "scene",
    "gaussian_renderer",
    "utils",
    "losses",
    "integrations",
    "scripts",
    "tools",
)
SOURCE_SUFFIXES = frozenset((".py", ".c", ".cc", ".cpp", ".cu", ".h", ".hh", ".hpp"))


def _training_source_files() -> List[Path]:
    paths = []
    for relative_root in SOURCE_ROOTS:
        root = REPOSITORY_ROOT / relative_root
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = root.rglob("*")
        else:
            candidates = []
        for path in candidates:
            if (
                path.is_file()
                and path.suffix.lower() in SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix())


def _training_source_metadata() -> Dict[str, object]:
    digest = hashlib.sha256()
    files = []
    for path in _training_source_files():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        content_hash = _sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(content_hash))
        files.append({"path": relative, "sha256": content_hash})
    return {"sha256": digest.hexdigest(), "files": files}


def _git_metadata() -> Dict[str, object]:
    metadata = {
        "head": None,
        "status": None,
        "diff_sha256": None,
        "source_snapshot_sha256": None,
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPOSITORY_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(REPOSITORY_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=str(REPOSITORY_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(REPOSITORY_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        snapshot = hashlib.sha256()
        for encoded_path in sorted(item for item in listed.split(b"\0") if item):
            relative_path = os.fsdecode(encoded_path)
            source_path = REPOSITORY_ROOT / relative_path
            if not source_path.is_file():
                continue
            snapshot.update(encoded_path)
            snapshot.update(b"\0")
            snapshot.update(bytes.fromhex(_sha256_file(source_path)))
        metadata = {
            "head": head.decode("ascii"),
            "status": status.decode("utf-8", errors="replace"),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_snapshot_sha256": snapshot.hexdigest(),
        }
        training_source = _training_source_metadata()
        metadata["training_source_sha256"] = training_source["sha256"]
        metadata["training_source_files"] = training_source["files"]
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        pass
    return metadata


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _input_hashes(train_command: Sequence[str]) -> Dict[str, str]:
    hashes = {}
    for flag in ("--dataset_manifest", "--split_manifest", "--degradation_manifest"):
        if flag not in train_command:
            continue
        path = Path(train_command[train_command.index(flag) + 1])
        if not path.is_file():
            raise FileNotFoundError("required input does not exist: {}".format(path))
        hashes[flag[2:]] = _sha256_file(path)
    return hashes


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_child(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_stream,
    stderr_stream,
    stop_file: Optional[Path],
    phase: str,
    record: Dict[str, object],
    run_manifest_path: Path,
) -> None:
    if stop_file is not None and stop_file.is_file():
        raise ExperimentInterrupted(phase)
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=stdout_stream,
        stderr=stderr_stream,
    )
    record.setdefault("processes", {})[phase] = {"pid": process.pid}
    _write_json_atomic(run_manifest_path, record)
    keyboard_stop = False
    while True:
        try:
            return_code = process.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            keyboard_stop = True
            if stop_file is not None:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch(exist_ok=True)
            continue
    record.setdefault("processes", {})[phase]["returncode"] = return_code
    if return_code == 130 or keyboard_stop or (stop_file is not None and stop_file.is_file()):
        raise ExperimentInterrupted(phase)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def run_experiment(
    name: str,
    output_path: Path,
    train_command: List[str],
    render_command: List[str],
    skip_train: bool,
    skip_render: bool,
    allow_existing: bool,
    expected_input_hashes: Optional[Mapping[str, str]] = None,
    auxiliary_loss_config: Optional[Mapping[str, object]] = None,
    stop_file: Optional[Path] = None,
    evaluation_iterations: Optional[Sequence[int]] = None,
    config_sha256: Optional[str] = None,
) -> str:
    if output_path.exists() and not allow_existing:
        raise FileExistsError(
            "refusing to reuse existing experiment output: {}".format(output_path)
        )
    input_hashes = _input_hashes(train_command)
    if expected_input_hashes:
        mismatches = {
            name: (expected.lower(), input_hashes.get(name))
            for name, expected in expected_input_hashes.items()
            if input_hashes.get(name) != expected.lower()
        }
        if mismatches:
            raise ValueError("input SHA256 mismatch: {}".format(mismatches))
    output_path.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_path / "run_manifest.json"
    stdout_path = output_path / "stdout.log"
    stderr_path = output_path / "stderr.log"
    record = {
        "schema_version": 1,
        "experiment": name,
        "status": "running",
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_metadata(),
        "config_sha256": config_sha256,
        "input_file_sha256": input_hashes,
        "expected_input_file_sha256": {
            name: digest.lower()
            for name, digest in (expected_input_hashes or {}).items()
        },
        "train_command": train_command,
        "render_command": render_command,
        "render_commands": [],
        "evaluation_iterations": list(evaluation_iterations or []),
        "auxiliary_loss_config": (
            dict(auxiliary_loss_config) if auxiliary_loss_config is not None else None
        ),
        "phase_timings": {"train": None, "render": None},
        "log_files": {"stdout": "stdout.log", "stderr": "stderr.log"},
    }
    _write_json_atomic(run_manifest_path, record)
    try:
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_stream, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_stream:
            if not skip_train:
                print("[START] {} train".format(name), flush=True)
                phase_started = _utc_now()
                phase_clock = time.perf_counter()
                try:
                    _run_child(
                        train_command,
                        cwd=str(REPOSITORY_ROOT),
                        stdout_stream=stdout_stream,
                        stderr_stream=stderr_stream,
                        stop_file=stop_file,
                        phase="train",
                        record=record,
                        run_manifest_path=run_manifest_path,
                    )
                finally:
                    record["phase_timings"]["train"] = {
                        "started_at_utc": phase_started,
                        "finished_at_utc": _utc_now(),
                        "wall_seconds": time.perf_counter() - phase_clock,
                    }
                print("[DONE] {} train".format(name), flush=True)
            if not skip_render:
                render_iterations = list(evaluation_iterations or [])
                render_commands = []
                if render_iterations:
                    render_commands = [
                        list(render_command) + ["--iteration", str(iteration)]
                        for iteration in render_iterations
                    ]
                else:
                    render_commands = [list(render_command)]
                record["render_commands"] = render_commands
                phase_started = _utc_now()
                phase_clock = time.perf_counter()
                try:
                    for render_index, command in enumerate(render_commands):
                        iteration_label = (
                            " iteration {}".format(render_iterations[render_index])
                            if render_iterations
                            else ""
                        )
                        print("[START] {} render{}".format(name, iteration_label), flush=True)
                        _run_child(
                            command,
                            cwd=str(REPOSITORY_ROOT),
                            stdout_stream=stdout_stream,
                            stderr_stream=stderr_stream,
                            stop_file=stop_file,
                            phase=(
                                "render_{}".format(render_iterations[render_index])
                                if render_iterations
                                else "render"
                            ),
                            record=record,
                            run_manifest_path=run_manifest_path,
                        )
                        print("[DONE] {} render{}".format(name, iteration_label), flush=True)
                finally:
                    record["phase_timings"]["render"] = {
                        "started_at_utc": phase_started,
                        "finished_at_utc": _utc_now(),
                        "wall_seconds": time.perf_counter() - phase_clock,
                    }
        record["status"] = "completed"
        return record["status"]
    except ExperimentInterrupted as exc:
        record["status"] = "interrupted"
        record["stop_reason"] = exc.reason
        record["interrupted_phase"] = exc.phase
        print(
            "[INTERRUPTED] {} phase={} stop_reason={} (see stdout.log/stderr.log)".format(
                name, exc.phase, exc.reason
            ),
            flush=True,
        )
        return record["status"]
    except (OSError, subprocess.CalledProcessError) as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        print("[FAILED] {}: {} (see stdout.log/stderr.log)".format(name, exc), flush=True)
        raise
    finally:
        record["finished_at_utc"] = _utc_now()
        _write_json_atomic(run_manifest_path, record)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _prepare_shared_initializations(
    matrix: Mapping[str, object], experiments: Sequence[Mapping[str, object]], stop_file: Path,
    allow_existing: bool,
) -> List[Dict[str, object]]:
    initialization_root_value = matrix.get("initialization_root")
    if not initialization_root_value:
        return []
    initialization_root = _resolve_path(str(initialization_root_value))
    results_root = _resolve_path(
        str(matrix.get("results_root", "results/" + initialization_root.name))
    )
    checks_root = results_root / "checks"
    checks_root.mkdir(parents=True, exist_ok=True)
    by_seed = {}
    for experiment in experiments:
        arguments = dict(matrix["common_args"])
        arguments.update(experiment["args"])
        seed = experiment.get("initialization_seed", arguments.get("seed"))
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise MatrixError("shared initialization requires a non-negative seed")
        by_seed.setdefault(seed, experiment)
    initialization_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed, experiment in sorted(by_seed.items()):
        if stop_file.is_file():
            raise ExperimentInterrupted("initialization")
        output_path = initialization_root / ("seed_{}".format(seed))
        package_path = output_path / "init_step0.pt"
        manifest_path = output_path / "init_manifest.json"
        command = build_initialization_command(matrix, experiment, output_path)
        entry = {
            "seed": seed,
            "path": str(output_path),
            "command": command,
            "stdout": str(checks_root / ("initialization_seed{}.stdout.log".format(seed))),
            "stderr": str(checks_root / ("initialization_seed{}.stderr.log".format(seed))),
        }
        if output_path.exists() and not allow_existing:
            raise FileExistsError(
                "refusing to reuse existing initialization output: {}".format(output_path)
            )
        if package_path.is_file() and manifest_path.is_file() and allow_existing:
            entry["status"] = "reused"
            entries.append(entry)
            continue
        if output_path.exists() and any(output_path.iterdir()):
            raise FileExistsError(
                "initialization output is non-empty and cannot be reused: {}".format(output_path)
            )
        stdout_path = Path(entry["stdout"])
        stderr_path = Path(entry["stderr"])
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_stream, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_stream:
            process = subprocess.Popen(
                command,
                cwd=str(REPOSITORY_ROOT),
                stdout=stdout_stream,
                stderr=stderr_stream,
            )
            entry["pid"] = process.pid
            return_code = process.wait()
        if return_code != 0:
            entry["status"] = "failed"
            raise subprocess.CalledProcessError(return_code, command)
        if not package_path.is_file() or not manifest_path.is_file():
            entry["status"] = "failed"
            raise FileNotFoundError("initialization package missing: {}".format(output_path))
        entry["status"] = "completed"
        entries.append(entry)
    _write_json_atomic(
        initialization_root / "initialization_manifest.json",
        {"schema_version": 1, "entries": entries},
    )
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        matrix_path = args.matrix.resolve()
        matrix = load_matrix(matrix_path)
        matrix_sha256 = _sha256_file(matrix_path)
        selected = set(args.only) if args.only else None
        known_names = {experiment["name"] for experiment in matrix["experiments"]}
        if selected is not None and selected - known_names:
            raise MatrixError(
                "unknown experiment(s): {}".format(", ".join(sorted(selected - known_names)))
            )
        experiments = [
            experiment
            for experiment in matrix["experiments"]
            if selected is None or experiment["name"] in selected
        ]
        batch_root = _resolve_path(str(matrix["output_root"]))
        stop_file = batch_root / "STOP_REQUESTED"
        batch_manifest_path = batch_root / "batch_manifest.json"
        batch_record = {
            "schema_version": 1,
            "status": "planned" if args.dry_run else "running",
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "stop_file": str(stop_file),
            "matrix_sha256": matrix_sha256,
            "initializations": [],
            "experiments": [
                {"name": experiment["name"], "status": "not_started"}
                for experiment in experiments
            ],
        }
        if not args.dry_run:
            if stop_file.exists():
                raise MatrixError(
                    "stop request already exists; refusing to start a new batch: {}".format(
                        stop_file
                    )
                )
            batch_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(batch_manifest_path, batch_record)
            if matrix.get("initialization_root"):
                batch_record["initializations"] = _prepare_shared_initializations(
                    matrix, experiments, stop_file, args.allow_existing
                )
                _write_json_atomic(batch_manifest_path, batch_record)
        elif matrix.get("initialization_root"):
            by_seed = {}
            for experiment in experiments:
                arguments = dict(matrix["common_args"])
                arguments.update(experiment["args"])
                seed = experiment.get("initialization_seed", arguments.get("seed"))
                by_seed.setdefault(seed, experiment)
            for seed, experiment in sorted(by_seed.items()):
                output_path = _resolve_path(str(matrix["initialization_root"])) / ("seed_{}".format(seed))
                print(
                    "[initialization_seed{}] {}".format(
                        seed,
                        subprocess.list2cmdline(
                            build_initialization_command(matrix, experiment, output_path)
                        ),
                    )
                )
        for index, experiment in enumerate(experiments):
            name = experiment["name"]
            output, train_command, render_command = build_commands(matrix, experiment)
            if args.dry_run:
                if not args.skip_train:
                    print("[{}] {}".format(name, subprocess.list2cmdline(train_command)))
                if not args.skip_render:
                    print("[{}] {}".format(name, subprocess.list2cmdline(render_command)))
                continue
            if stop_file.exists():
                batch_record["status"] = "interrupted"
                batch_record["stop_reason"] = "user_request"
                for pending in batch_record["experiments"][index:]:
                    pending["status"] = "not_started_due_to_user_stop"
                batch_record["finished_at_utc"] = _utc_now()
                _write_json_atomic(batch_manifest_path, batch_record)
                print("[INTERRUPTED] batch stop requested before {}".format(name), flush=True)
                return 130
            batch_record["experiments"][index]["status"] = "running"
            batch_record["experiments"][index]["started_at_utc"] = _utc_now()
            _write_json_atomic(batch_manifest_path, batch_record)
            try:
                status = run_experiment(
                    name,
                    output,
                    train_command,
                    render_command,
                    args.skip_train,
                    args.skip_render,
                    args.allow_existing,
                    experiment.get("expected_input_sha256"),
                    _resolved_auxiliary_loss_config(
                        matrix["common_args"], experiment["args"]
                    ),
                    stop_file=stop_file,
                    evaluation_iterations=matrix.get("evaluation_iterations"),
                    config_sha256=matrix_sha256,
                )
            except Exception:
                batch_record["experiments"][index]["status"] = "failed"
                batch_record["experiments"][index]["finished_at_utc"] = _utc_now()
                batch_record["status"] = "failed"
                batch_record["finished_at_utc"] = _utc_now()
                _write_json_atomic(batch_manifest_path, batch_record)
                raise
            batch_record["experiments"][index]["status"] = status
            batch_record["experiments"][index]["finished_at_utc"] = _utc_now()
            if status == "interrupted":
                batch_record["status"] = "interrupted"
                batch_record["stop_reason"] = "user_request"
                for pending in batch_record["experiments"][index + 1:]:
                    pending["status"] = "not_started_due_to_user_stop"
                batch_record["finished_at_utc"] = _utc_now()
                _write_json_atomic(batch_manifest_path, batch_record)
                return 130
            _write_json_atomic(batch_manifest_path, batch_record)
        if not args.dry_run:
            batch_record["status"] = "completed"
            batch_record["finished_at_utc"] = _utc_now()
            _write_json_atomic(batch_manifest_path, batch_record)
    except ExperimentInterrupted as exc:
        print("[INTERRUPTED] {}".format(exc), flush=True)
        return INTERRUPTED_EXIT_CODE
    except (MatrixError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
