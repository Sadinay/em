from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TARGET_COLUMNS = {"t_ripple", "t_avg", "torque_ratio", "historical_j"}


def load_config(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training configuration must be a mapping")
    config = deepcopy(data)
    config["_config_path"] = str(resolved)
    required = {"data", "split", "model", "training", "output"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"training configuration missing sections: {sorted(missing)}")
    runs = config["data"].get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("data.runs must contain at least one SQLite run directory")
    targets = config["data"].get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("data.targets must contain at least one target")
    names: set[str] = set()
    for target in targets:
        if not isinstance(target, dict) or not {"name", "column"} <= set(target):
            raise ValueError("each target needs name and column")
        if target["column"] not in REQUIRED_TARGET_COLUMNS:
            raise ValueError(f"unsupported target column {target['column']}")
        if target["name"] in names:
            raise ValueError(f"duplicate target name {target['name']}")
        names.add(str(target["name"]))
    categories = tuple(int(value) for value in config["data"].get("material_categories", []))
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("material_categories must be unique and non-empty")
    if tuple(config["data"].get("network_matrix_shape", [])) != (18, 10):
        raise ValueError("network_matrix_shape must match the audited [18, 10] matrix")
    ratios = [float(config["split"][key]) for key in ("train_fraction", "val_fraction", "test_fraction")]
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must be positive and sum to one")
    if config["split"].get("group_by") not in {
        "generation", "parent", "root_parent", "topology_family"
    }:
        raise ValueError("unsupported split.group_by")
    reuse_from = config["split"].get("reuse_from")
    if reuse_from is not None and not isinstance(reuse_from, str):
        raise ValueError("split.reuse_from must be a path string when provided")
    if int(config["training"]["batch_size"]) <= 0 or int(config["training"]["max_epochs"]) <= 0:
        raise ValueError("batch size and max epochs must be positive")
    return config


def resolved_run_paths(config: dict[str, Any], project_root: Path) -> list[Path]:
    result = []
    for value in config["data"]["runs"]:
        path = Path(value)
        result.append((project_root / path).resolve() if not path.is_absolute() else path.resolve())
    return result
