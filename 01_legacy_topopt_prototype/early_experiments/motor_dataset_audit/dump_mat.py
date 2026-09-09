from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def convert(value: Any) -> Any:
    """Convert a scipy-loaded MATLAB value without truncating array contents."""
    if isinstance(value, dict):
        return {str(key): convert(child) for key, child in value.items() if not str(key).startswith("__")}
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": convert(value.tolist()),
        }
    if isinstance(value, np.generic):
        return convert(value.item())
    if isinstance(value, complex):
        return {"__kind__": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, float):
        if math.isnan(value):
            return {"__kind__": "float", "value": "NaN"}
        if math.isinf(value):
            return {"__kind__": "float", "value": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, bytes):
        return {"__kind__": "bytes", "hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [convert(child) for child in value]
    if hasattr(value, "_fieldnames"):
        return {
            str(field): convert(getattr(value, field))
            for field in value._fieldnames or []
        }
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully expand a MATLAB v5-v7.2 MAT file into JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    stat = source.stat()
    loaded = loadmat(
        source,
        struct_as_record=False,
        squeeze_me=False,
        chars_as_strings=True,
        simplify_cells=True,
    )
    variables = {
        key: convert(value)
        for key, value in loaded.items()
        if not key.startswith("__")
    }
    result = {
        "source_file": str(source),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256(source),
        "top_level_variable_names": list(variables),
        "variables": variables,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote complete dump: {args.output.resolve()}")


if __name__ == "__main__":
    main()
