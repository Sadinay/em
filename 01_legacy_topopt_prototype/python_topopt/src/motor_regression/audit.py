from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import SampleRecord, group_value


def _distribution(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "missing_or_nonfinite": int(values.size - finite.size),
        "min": float(np.min(finite)) if finite.size else None,
        "q05": float(np.quantile(finite, 0.05)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "median": float(np.quantile(finite, 0.5)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
        "q95": float(np.quantile(finite, 0.95)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def audit_records(
    records: Sequence[SampleRecord],
    *,
    target_names: Sequence[str],
    expected_categories: Sequence[int],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot audit an empty dataset")
    matrices = np.stack([record.matrix_18x10 for record in records])
    targets = np.stack([record.targets for record in records])
    hashes = Counter(record.chromosome_hash for record in records)
    matrix_keys = Counter(matrix.tobytes() for matrix in matrices)
    band_counts = Counter(record.fitness_band for record in records)
    category_counts = {
        str(int(value)): int(np.count_nonzero(matrices == int(value)))
        for value in np.unique(matrices)
    }
    group_stats = {}
    for mode in ("generation", "parent", "root_parent", "topology_family"):
        counts = Counter(group_value(record, mode) for record in records)
        group_stats[mode] = {
            "group_count": len(counts),
            "minimum_group_size": min(counts.values()),
            "median_group_size": float(np.median(list(counts.values()))),
            "maximum_group_size": max(counts.values()),
            "largest_groups": dict(counts.most_common(15)),
        }
    return {
        "sample_count": len(records),
        "run_counts": dict(Counter(record.run_id for record in records)),
        "fitness_band_counts": {
            f"B{index}": int(band_counts.get(f"B{index}", 0)) for index in range(1, 8)
        },
        "stored_matrix_shape": [18, 10],
        "network_tensor_shape_per_sample": [len(expected_categories), 18, 10],
        "material_categories_observed": [int(value) for value in np.unique(matrices)],
        "material_categories_expected": [int(value) for value in expected_categories],
        "material_cell_counts": category_counts,
        "unknown_material_cell_count": int(
            np.count_nonzero(~np.isin(matrices, np.asarray(expected_categories)))
        ),
        "missing_matrix_values": int(np.count_nonzero(~np.isfinite(matrices))),
        "duplicate_chromosome_hash_groups": sum(count > 1 for count in hashes.values()),
        "duplicate_chromosome_samples": sum(count for count in hashes.values() if count > 1),
        "duplicate_matrix_groups": sum(count > 1 for count in matrix_keys.values()),
        "duplicate_matrix_samples": sum(count for count in matrix_keys.values() if count > 1),
        "targets": {
            name: _distribution(targets[:, index]) for index, name in enumerate(target_names)
        },
        "groups": group_stats,
    }


def write_audit_report(report: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Motor topology regression data audit",
        "",
        f"- Samples: {report['sample_count']}",
        f"- Stored matrix: `{report['stored_matrix_shape']}`",
        f"- Network sample tensor: `{report['network_tensor_shape_per_sample']}`",
        f"- Observed materials: `{report['material_categories_observed']}`",
        f"- Missing matrix values: {report['missing_matrix_values']}",
        f"- Unknown material cells: {report['unknown_material_cell_count']}",
        f"- Duplicate chromosome samples: {report['duplicate_chromosome_samples']}",
        f"- Duplicate matrix samples: {report['duplicate_matrix_samples']}",
        "",
        "## Run counts",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in report["run_counts"].items())
    lines.extend(["", "## Fitness bands", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in report["fitness_band_counts"].items())
    lines.extend(["", "## Targets", ""])
    for name, values in report["targets"].items():
        lines.append(
            f"- `{name}`: mean={values['mean']:.8g}, std={values['std']:.8g}, "
            f"min={values['min']:.8g}, median={values['median']:.8g}, max={values['max']:.8g}, "
            f"missing={values['missing_or_nonfinite']}"
        )
    lines.extend(["", "## Group fields", ""])
    for mode, values in report["groups"].items():
        lines.append(
            f"- `{mode}`: {values['group_count']} groups, sizes "
            f"{values['minimum_group_size']} / {values['median_group_size']:.1f} / "
            f"{values['maximum_group_size']} (min/median/max)"
        )
    (output_directory / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
