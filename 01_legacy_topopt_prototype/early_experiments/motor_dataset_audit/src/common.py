from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("motor_dataset_audit")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(output_dir / "audit.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def iso_time(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def quick_hash(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        h.update(handle.read(block_size))
        if size > block_size:
            handle.seek(max(0, size - block_size))
            h.update(handle.read(block_size))
    return h.hexdigest()


def full_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def canonical_stem(stem: str, suffixes: list[str]) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", stem.casefold()).strip("_")
    parts = [p for p in value.split("_") if p]
    while parts and parts[-1] in suffixes:
        parts.pop()
    return "_".join(parts)


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<!\d)\d{2,}(?!\d)", text.casefold()))


def keyword_matches(text: str, keywords: Iterable[str]) -> list[str]:
    """Case-insensitive matching with camelCase boundaries and safe short tokens."""
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    normalized = re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
    tokens = set(normalized.split("_"))
    matches = []
    for keyword in keywords:
        key = re.sub(r"[^a-z0-9]+", "_", str(keyword).casefold()).strip("_")
        if not key:
            continue
        if (len(key) <= 3 and "_" not in key and key in tokens) or (len(key) > 3 or "_" in key) and key in normalized:
            matches.append(str(keyword))
    return sorted(set(matches))


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=json_default)
