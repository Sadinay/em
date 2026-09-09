"""Audit the 120 SPMSM gene positions against the supplied FEM template."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from spmsm_mapping import DEFAULT_MAT, DEFAULT_TEMPLATE, file_sha256, match_material_positions


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "femm_zone" / "results"
REPORTS = ROOT / "reports"


def fem_header(text: str) -> dict:
    result = {}
    for key in ("Format", "Frequency", "Precision", "MinAngle", "Depth", "Coordinates", "ProblemType"):
        match = re.search(rf"\[{key}\]\s*=\s*([^\r\n]+)", text)
        if match:
            result[key] = match.group(1).strip()
    for key in ("BdryProps", "BlockProps", "CircuitProps", "NumPoints", "NumSegments", "NumArcSegments", "NumHoles", "NumBlockLabels"):
        match = re.search(rf"\[{key}\]\s*=\s*(\d+)", text)
        if match:
            result[key] = int(match.group(1))
    return result


def save_figure(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    grid = np.arange(1, 121).reshape(20, 6).T
    axes[0].imshow(np.zeros_like(grid), origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for radial in range(6):
        for angular in range(20):
            axes[0].text(angular, radial, str(grid[radial, angular]), ha="center", va="center", fontsize=7)
    axes[0].set_xlabel("angular index (physical angle decreases left to right)")
    axes[0].set_ylabel("radial index (radius increases bottom to top)")
    axes[0].set_title("120 gene indices: 6 radial × 20 angular")

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for copy_index in range(1, 5):
        copy_rows = [row for row in rows if row["copy_index_1based"] == copy_index]
        axes[1].scatter(
            [row["x_mm"] for row in copy_rows],
            [row["y_mm"] for row in copy_rows],
            s=15,
            color=colors[copy_index - 1],
            label=f"copy {copy_index}",
        )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].set_title("MaterialPosition mapped to four FEM symmetry copies")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    data = loadmat(
        DEFAULT_MAT,
        variable_names=["MaterialPosition", "population_all", "population_noChange_all", "VolumePM_all"],
        squeeze_me=True,
    )
    positions = np.asarray(data["MaterialPosition"], dtype=np.float64)
    corrected = np.asarray(data["population_all"], dtype=np.uint8)
    before = np.asarray(data["population_noChange_all"], dtype=np.uint8)
    volume = np.asarray(data["VolumePM_all"], dtype=np.int64)
    template_text = DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    rows = match_material_positions(template_text, positions)

    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "gene_to_fem_labels.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first = [row for row in rows if row["copy_index_1based"] == 1]
    angles = np.asarray([row["polar_angle_deg"] for row in first]).reshape(20, 6)[:, 0]
    radii = np.asarray([row["radius_mm"] for row in first]).reshape(20, 6)[0]
    ones = corrected.sum(axis=1)
    same_before_after = np.all(corrected == before, axis=1)
    summary = {
        "status": "mapping_audited",
        "mat": {"path": str(DEFAULT_MAT.resolve()), "sha256": file_sha256(DEFAULT_MAT)},
        "fem": {"path": str(DEFAULT_TEMPLATE.resolve()), "sha256": file_sha256(DEFAULT_TEMPLATE)},
        "fem_header": fem_header(template_text),
        "gene_count": 120,
        "gene_values": [0, 1],
        "grid": {
            "radial_cells": 6,
            "angular_cells": 20,
            "index_formula_0based": "gene = angular_index * 6 + radial_index",
            "radial_centers_mm": radii.tolist(),
            "first_copy_angular_centers_deg": angles.tolist(),
            "angular_step_deg": float(np.median(np.diff(angles))),
        },
        "fem_mapping": {
            "copies_per_gene": 4,
            "mapped_labels": len(rows),
            "distinct_labels": len({row["label_index_1based"] for row in rows}),
            "maximum_coordinate_error_mm": max(row["coordinate_error_mm"] for row in rows),
            "label_material_names": "gene N maps to cN in all four copies",
        },
        "material_semantics": {
            "0": "Air",
            "1": "N38 permanent magnet",
            "evidence": "VolumePM equals the number of one-bits; c1..c120 are the variable material properties",
            "final_state_exact_matches": int(np.count_nonzero(ones[:, -1] == volume[:, -1])),
            "final_state_records": int(volume.shape[0]),
            "full_history_exact_matches": int(np.count_nonzero(ones == volume)),
            "full_history_records": int(volume.size),
        },
        "history_source": {
            "recommended_femm_input": "population_all",
            "unchanged_before_after_records": int(np.count_nonzero(same_before_after)),
            "total_records": int(same_before_after.size),
            "validation_candidates_restricted_to_unchanged_records": True,
        },
        "artifacts": {
            "mapping_csv": str(csv_path.resolve()),
            "mapping_figure": str((REPORTS / "gene_to_structure_mapping.png").resolve()),
        },
    }
    (RESULTS / "spmsm_structure_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_figure(REPORTS / "gene_to_structure_mapping.png", rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
