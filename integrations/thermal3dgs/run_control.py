"""Shared sticky stop-file and compact status helpers."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional


INTERRUPTED_EXIT_CODE = 130


class UserStopRequested(RuntimeError):
    def __init__(self, phase: str, last_completed_iteration: int):
        self.phase = phase
        self.last_completed_iteration = int(last_completed_iteration)
        super().__init__(
            "user stop requested during {} after iteration {}".format(
                phase, last_completed_iteration
            )
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_stop_file(value) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser().resolve()


def stop_requested(stop_file) -> bool:
    path = resolve_stop_file(stop_file)
    return path is not None and path.is_file()


def raise_if_stop_requested(stop_file, phase: str, last_completed_iteration: int) -> None:
    if stop_requested(stop_file):
        raise UserStopRequested(phase, last_completed_iteration)


def write_json_atomic(path, value: Mapping[str, object]) -> None:
    path = Path(path)
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


def write_run_status(
    model_path,
    *,
    status: str,
    phase: str,
    last_completed_iteration: int,
    stop_reason=None,
    error=None,
) -> None:
    write_json_atomic(
        Path(model_path) / "status.json",
        {
            "schema_version": 1,
            "status": status,
            "phase": phase,
            "last_completed_iteration": int(last_completed_iteration),
            "updated_at_utc": utc_now(),
            "pid": os.getpid(),
            "stop_reason": stop_reason,
            "error": error,
        },
    )
