from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

from .common import compact_json, keyword_matches, write_csv


SUMMARY_FIELDS = ["relative_path", "absolute_path", "reader", "version", "read_status", "variable_count", "candidate_count", "error"]
VARIABLE_FIELDS = ["relative_path", "variable_path", "variable_name", "python_type", "shape", "dtype", "min", "max", "has_nan", "has_inf", "is_empty", "preview"]
CANDIDATE_FIELDS = ["file", "variable_path", "variable_name", "shape", "dtype", "reason"]


def inspect_mat_files(root: Path, output: Path, inventory: list[dict[str, Any]], config: dict[str, Any], logger) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keywords = [str(x).casefold() for x in config.get("mat", {}).get("candidate_keywords", [])]
    summaries: list[dict[str, Any]] = []
    variables: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    mat_rows = [r for r in inventory if str(r["extension"]).casefold() == ".mat"]
    for index, file_row in enumerate(mat_rows, 1):
        path = Path(str(file_row["absolute_path"]))
        rel = str(file_row["relative_path"])
        local: list[dict[str, Any]] = []
        reader = ""
        version = ""
        error = ""
        try:
            with path.open("rb") as handle:
                header = handle.read(128)
            if h5py.is_hdf5(path):
                reader, version = "h5py", "MATLAB v7.3/HDF5"
                with h5py.File(path, "r") as handle:
                    walk_hdf5(handle, "", rel, local)
            else:
                reader = "scipy.io"
                version = "MATLAB v5-v7.2" if header.startswith(b"MATLAB") else "pre-v7.3/unknown"
                # whosmat validates the directory cheaply; loadmat is needed for nested metadata.
                whosmat(path)
                data = loadmat(path, struct_as_record=False, squeeze_me=True, chars_as_strings=True)
                for name, value in data.items():
                    if name.startswith("__"):
                        continue
                    walk_value(value, name, name, rel, local, set(), depth=0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"relative_path": rel, "reader": reader or "unknown", "error": error})
        variables.extend(local)
        file_candidates = []
        for row in local:
            path_name = str(row["variable_path"]) + " " + str(row["variable_name"])
            matched = keyword_matches(path_name, keywords)
            if matched:
                candidate = {
                    "file": rel,
                    "variable_path": row["variable_path"],
                    "variable_name": row["variable_name"],
                    "shape": row["shape"],
                    "dtype": row["dtype"],
                    "reason": "变量名/完整路径包含性能或设计关键词: " + ", ".join(matched),
                }
                candidates.append(candidate)
                file_candidates.append(candidate)
        summaries.append({
            "relative_path": rel,
            "absolute_path": str(path),
            "reader": reader,
            "version": version,
            "read_status": "ok" if not error else "error",
            "variable_count": len(local),
            "candidate_count": len(file_candidates),
            "error": error,
        })
        if index % 50 == 0:
            logger.info("Inspected %d/%d MAT files", index, len(mat_rows))
    write_csv(output / "mat_file_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(output / "mat_variable_inventory.csv", variables, VARIABLE_FIELDS)
    write_csv(output / "mat_read_errors.csv", errors, ["relative_path", "reader", "error"])
    write_csv(output / "candidate_output_variables.csv", candidates, CANDIDATE_FIELDS)
    logger.info("MAT: %d files, %d readable, %d variable records, %d candidates", len(summaries), sum(s["read_status"] == "ok" for s in summaries), len(variables), len(candidates))
    return summaries, variables


def walk_hdf5(group: h5py.Group, prefix: str, rel: str, rows: list[dict[str, Any]]) -> None:
    for name, item in group.items():
        path_name = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Group):
            rows.append(base_row(rel, path_name, name, "h5py.Group", "", "", "", "", "", "", False, ""))
            walk_hdf5(item, path_name, rel, rows)
        else:
            value = item
            stats = array_stats(value, allow_read=value.size <= 1_000_000)
            rows.append(base_row(rel, path_name, name, "h5py.Dataset", str(value.shape), str(value.dtype), **stats))


def walk_value(value: Any, path_name: str, name: str, rel: str, rows: list[dict[str, Any]], seen: set[int], depth: int) -> None:
    if depth > 12:
        return
    identity = id(value)
    if identity in seen:
        return
    if not isinstance(value, (str, bytes, int, float, complex, np.generic)):
        seen.add(identity)
    if hasattr(value, "_fieldnames"):
        rows.append(base_row(rel, path_name, name, type(value).__name__, "", "struct", "", "", "", "", False, "fields=" + ",".join(value._fieldnames or [])))
        for field in value._fieldnames or []:
            child = getattr(value, field)
            walk_value(child, f"{path_name}.{field}", field, rel, rows, seen, depth + 1)
        return
    if isinstance(value, dict):
        rows.append(base_row(rel, path_name, name, "dict", "", "object", "", "", "", "", not value, "keys=" + ",".join(map(str, list(value)[:20]))))
        for key, child in list(value.items())[:500]:
            walk_value(child, f"{path_name}.{key}", str(key), rel, rows, seen, depth + 1)
        return
    if isinstance(value, np.ndarray) and value.dtype == object:
        rows.append(base_row(rel, path_name, name, "ndarray", str(value.shape), str(value.dtype), "", "", "", "", value.size == 0, "object/cell array"))
        for i, child in enumerate(value.flat):
            if i >= 500:
                break
            walk_value(child, f"{path_name}[{i}]", f"{name}[{i}]", rel, rows, seen, depth + 1)
        return
    arr = np.asarray(value)
    stats = array_stats(arr, allow_read=True)
    rows.append(base_row(rel, path_name, name, type(value).__name__, str(arr.shape), str(arr.dtype), **stats))


def array_stats(value: Any, allow_read: bool) -> dict[str, Any]:
    try:
        shape = tuple(int(x) for x in value.shape)
        size = int(np.prod(shape)) if shape else 1
        empty = size == 0
        if empty:
            return {"min": "", "max": "", "has_nan": False, "has_inf": False, "is_empty": True, "preview": ""}
        if not allow_read:
            return {"min": "", "max": "", "has_nan": "", "has_inf": "", "is_empty": False, "preview": "stats skipped: large HDF5 dataset"}
        arr = np.asarray(value[()] if isinstance(value, h5py.Dataset) else value)
        preview = compact_json(arr.reshape(-1)[:8].tolist())[:500]
        if np.issubdtype(arr.dtype, np.number):
            finite = np.asarray(arr, dtype=np.complex128 if np.iscomplexobj(arr) else np.float64)
            has_nan = bool(np.isnan(finite).any())
            has_inf = bool(np.isinf(finite).any())
            valid = finite[~np.isnan(finite)]
            if valid.size == 0:
                return {"min": "", "max": "", "has_nan": has_nan, "has_inf": has_inf, "is_empty": False, "preview": preview}
            if np.iscomplexobj(finite):
                magnitudes = np.abs(valid)
                minimum = float(np.min(magnitudes))
                maximum = float(np.max(magnitudes))
            else:
                minimum = float(np.min(valid))
                maximum = float(np.max(valid))
            return {"min": minimum, "max": maximum, "has_nan": has_nan, "has_inf": has_inf, "is_empty": False, "preview": preview}
        return {"min": "", "max": "", "has_nan": False, "has_inf": False, "is_empty": False, "preview": preview}
    except Exception as exc:
        return {"min": "", "max": "", "has_nan": "", "has_inf": "", "is_empty": False, "preview": f"stats error: {type(exc).__name__}"}


def base_row(rel: str, variable_path: str, variable_name: str, python_type: str, shape: str, dtype: str, min: Any = "", max: Any = "", has_nan: Any = "", has_inf: Any = "", is_empty: Any = "", preview: str = "") -> dict[str, Any]:
    return {
        "relative_path": rel, "variable_path": variable_path, "variable_name": variable_name,
        "python_type": python_type, "shape": shape, "dtype": dtype, "min": min, "max": max,
        "has_nan": has_nan, "has_inf": has_inf, "is_empty": is_empty, "preview": preview,
    }
