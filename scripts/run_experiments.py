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
    )
)
INPUT_HASH_KEYS = frozenset(
    ("dataset_manifest", "split_manifest", "degradation_manifest")
)


class MatrixError(ValueError):
    pass


def _validate_aux_loss_arguments(arguments: Mapping[str, object], label: str) -> None:
    version = arguments.get("aux_loss_version", "legacy")
    if version not in ("legacy", "filtered_edge"):
        raise MatrixError(
            "{}.aux_loss_version must be 'legacy' or 'filtered_edge'".format(label)
        )
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
            value = arguments.get(name, 0.0)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise MatrixError("{}.{} must be finite and numeric".format(label, name))
            if float(value) != 0:
                raise MatrixError(
                    "filtered_edge requires lambda_thermal == 0 and lambda_smooth == 0"
                )


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
    unknown = set(matrix) - required
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
    return matrix


def _append_arguments(command: List[str], arguments: Mapping[str, object]) -> None:
    for key in sorted(arguments):
        if not key or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in key):
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
    return output_path, train_command, render_command


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def run_experiment(
    name: str,
    output_path: Path,
    train_command: List[str],
    render_command: List[str],
    skip_train: bool,
    skip_render: bool,
    allow_existing: bool,
    expected_input_hashes: Optional[Mapping[str, str]] = None,
) -> None:
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
        "input_file_sha256": input_hashes,
        "expected_input_file_sha256": {
            name: digest.lower()
            for name, digest in (expected_input_hashes or {}).items()
        },
        "train_command": train_command,
        "render_command": render_command,
        "log_files": {"stdout": "stdout.log", "stderr": "stderr.log"},
    }
    _write_json_atomic(run_manifest_path, record)
    try:
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_stream, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_stream:
            if not skip_train:
                print("[{}] train started".format(name), flush=True)
                subprocess.run(
                    train_command,
                    cwd=str(REPOSITORY_ROOT),
                    check=True,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                )
                print("[{}] train completed".format(name), flush=True)
            if not skip_render:
                print("[{}] render started".format(name), flush=True)
                subprocess.run(
                    render_command,
                    cwd=str(REPOSITORY_ROOT),
                    check=True,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                )
                print("[{}] render completed".format(name), flush=True)
        record["status"] = "completed"
    except (OSError, subprocess.CalledProcessError) as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        print("[{}] failed: {} (see stdout.log/stderr.log)".format(name, exc), flush=True)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        matrix = load_matrix(args.matrix.resolve())
        selected = set(args.only) if args.only else None
        known_names = {experiment["name"] for experiment in matrix["experiments"]}
        if selected is not None and selected - known_names:
            raise MatrixError(
                "unknown experiment(s): {}".format(", ".join(sorted(selected - known_names)))
            )
        for experiment in matrix["experiments"]:
            name = experiment["name"]
            if selected is not None and name not in selected:
                continue
            output, train_command, render_command = build_commands(matrix, experiment)
            if args.dry_run:
                if not args.skip_train:
                    print("[{}] {}".format(name, subprocess.list2cmdline(train_command)))
                if not args.skip_render:
                    print("[{}] {}".format(name, subprocess.list2cmdline(render_command)))
            else:
                run_experiment(
                    name,
                    output,
                    train_command,
                    render_command,
                    args.skip_train,
                    args.skip_render,
                    args.allow_existing,
                    experiment.get("expected_input_sha256"),
                )
    except (MatrixError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
