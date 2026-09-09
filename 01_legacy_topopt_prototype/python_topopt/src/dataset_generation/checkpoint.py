from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import json_sha256
import json

from .state import atomic_write_json


CHECKPOINT_SCHEMA_VERSION = 1


def make_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "payload_hash": json_sha256(payload),
        "payload": payload,
    }


def write_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    envelope = make_checkpoint(payload)
    atomic_write_json(path, envelope, keep_backup=True)
    return envelope


def read_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path)
    failures: list[Exception] = []
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        if not candidate.is_file():
            continue
        try:
            envelope = json.loads(candidate.read_text(encoding="utf-8"))
            if envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("unsupported dataset checkpoint schema")
            payload = envelope.get("payload")
            if not isinstance(payload, dict) or json_sha256(payload) != envelope.get(
                "payload_hash"
            ):
                raise ValueError("dataset checkpoint hash mismatch")
            return payload
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(exc)
    if failures:
        raise ValueError(f"no valid dataset checkpoint: {failures[-1]}") from failures[-1]
    raise FileNotFoundError(path)
