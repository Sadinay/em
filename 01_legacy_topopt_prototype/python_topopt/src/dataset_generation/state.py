from __future__ import annotations

from enum import StrEnum
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid


class RunStatus(StrEnum):
    CREATED = "created"
    AWAITING_REFERENCE = "awaiting_reference"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class CandidateStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    VALID = "valid"
    QUEUED = "queued"
    COMPLETED = "completed"
    FAILED = "failed"


class SampleStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    FEMM_RUNNING = "femm_running"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    CLASSIFIED = "classified"
    FEMM_FAILED = "femm_failed"
    TIMEOUT = "timeout"
    INVALID_RESULT = "invalid_result"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class AngleStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


RUN_TRANSITIONS: dict[str, set[str]] = {
    RunStatus.CREATED: {RunStatus.AWAITING_REFERENCE, RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.AWAITING_REFERENCE: {RunStatus.RUNNING, RunStatus.PAUSED, RunStatus.FAILED},
    RunStatus.RUNNING: {RunStatus.PAUSING, RunStatus.PAUSED, RunStatus.COMPLETED, RunStatus.FAILED},
    RunStatus.PAUSING: {RunStatus.PAUSED, RunStatus.FAILED},
    RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.FAILED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
}


def validate_run_transition(current: str, target: str) -> None:
    if target == current:
        return
    if target not in RUN_TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal run status transition {current!r} -> {target!r}")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes, *, keep_backup: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if keep_backup and path.is_file():
        backup = path.with_suffix(path.suffix + ".bak")
        backup_temp = backup.with_name(f".{backup.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(path, backup_temp)
        # Windows requires a writable descriptor for fsync in some runtimes.
        with backup_temp.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(backup_temp, backup)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    keep_backup: bool = False,
    sort_keys: bool = True,
) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=sort_keys,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, data, keep_backup=keep_backup)


def read_json_with_backup(path: Path) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.is_file():
            raise
        return json.loads(backup.read_text(encoding="utf-8"))
