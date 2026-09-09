from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    """Return a stable UTF-8 JSON representation for persistent hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_sample_hash(
    *,
    chromosome_hash: str,
    config_hash: str,
    geometry_mode: str,
    angles_deg: Iterable[float],
) -> str:
    return json_sha256(
        {
            "chromosome_hash": chromosome_hash,
            "config_hash": config_hash,
            "geometry_mode": geometry_mode,
            "angles_deg": [float(value) for value in angles_deg],
        }
    )


def source_manifest_hash(project_root: Path) -> str:
    """Hash runtime Python sources when a Git commit is unavailable."""

    root = Path(project_root).resolve()
    files: list[Path] = []
    for relative_root in ("src", "scripts"):
        folder = root / relative_root
        if folder.is_dir():
            files.extend(folder.rglob("*.py"))
    manifest = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(files)
        if "__pycache__" not in path.parts
    ]
    return json_sha256(manifest)
