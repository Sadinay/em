from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import full_sha256, iso_time, quick_hash, safe_rel, write_csv


INVENTORY_FIELDS = [
    "relative_path", "absolute_path", "filename", "stem", "extension",
    "size_bytes", "modified_time", "created_time", "quick_hash_sha256",
    "sha256", "parent_directory", "readable",
]


def scan_dataset(root: Path, output: Path, config: dict[str, Any], logger) -> list[dict[str, Any]]:
    root = root.resolve()
    output = output.resolve()
    quick_bytes = int(config.get("hash", {}).get("quick_bytes", 65536))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    directories: list[Path] = []

    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        dirnames.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        directories.append(current_path)
        for filename in filenames:
            path = current_path / filename
            if output == path or output in path.parents:
                continue
            try:
                stat = path.stat()
                qhash = quick_hash(path, quick_bytes)
                row = {
                    "relative_path": safe_rel(path, root),
                    "absolute_path": str(path.resolve()),
                    "filename": path.name,
                    "stem": path.stem,
                    "extension": path.suffix.casefold(),
                    "size_bytes": stat.st_size,
                    "modified_time": iso_time(stat.st_mtime),
                    "created_time": iso_time(getattr(stat, "st_ctime", None)),
                    "quick_hash_sha256": qhash,
                    "sha256": "",
                    "parent_directory": safe_rel(path.parent, root),
                    "readable": True,
                }
                rows.append(row)
            except Exception as exc:
                errors.append({"path": str(path), "stage": "scan", "error": f"{type(exc).__name__}: {exc}"})

    by_size_hash: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_size_hash[(int(row["size_bytes"]), str(row["quick_hash_sha256"]))].append(row)

    duplicate_rows: list[dict[str, Any]] = []
    group_number = 0
    for candidates in by_size_hash.values():
        if len(candidates) < 2:
            continue
        by_full: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            try:
                digest = full_sha256(Path(str(row["absolute_path"])))
                row["sha256"] = digest
                by_full[digest].append(row)
            except Exception as exc:
                errors.append({"path": str(row["absolute_path"]), "stage": "sha256", "error": f"{type(exc).__name__}: {exc}"})
        for digest, identical in by_full.items():
            if len(identical) < 2:
                continue
            group_number += 1
            for row in identical:
                duplicate_rows.append({
                    "duplicate_group": f"DUP_{group_number:04d}",
                    "sha256": digest,
                    "size_bytes": row["size_bytes"],
                    "relative_path": row["relative_path"],
                })

    ext_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"file_count": 0, "total_bytes": 0})
    for row in rows:
        ext = str(row["extension"]) or "<none>"
        ext_stats[ext]["file_count"] += 1
        ext_stats[ext]["total_bytes"] += int(row["size_bytes"])
    extension_rows = [
        {"extension": ext, **values}
        for ext, values in sorted(ext_stats.items(), key=lambda item: (-item[1]["file_count"], item[0]))
    ]

    write_csv(output / "file_inventory.csv", rows, INVENTORY_FIELDS)
    write_csv(output / "extension_summary.csv", extension_rows, ["extension", "file_count", "total_bytes"])
    write_csv(output / "duplicate_files.csv", duplicate_rows, ["duplicate_group", "sha256", "size_bytes", "relative_path"])
    write_csv(output / "scan_errors.csv", errors, ["path", "stage", "error"])
    write_tree(root, output / "directory_tree.txt", rows, directories)
    logger.info("Scanned %d files in %d directories (%.3f GB)", len(rows), len(directories), sum(int(r["size_bytes"]) for r in rows) / 1e9)
    return rows


def write_tree(root: Path, destination: Path, rows: list[dict[str, Any]], directories: list[Path]) -> None:
    counts = Counter(str(row["parent_directory"]) for row in rows)
    lines = [f"{root.name}/"]
    for directory in sorted(directories, key=lambda p: safe_rel(p, root).casefold()):
        if directory == root:
            continue
        rel = safe_rel(directory, root)
        depth = len(Path(rel).parts)
        lines.append(f"{'  ' * depth}{Path(rel).name}/  [{counts.get(rel, 0)} files]")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
