"""Analyze the relation between workspace material cells and IPMSM.fem labels."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEM = ROOT / "femm_zone" / "models" / "IPMSM.fem"
DEFAULT_MAT = ROOT / "data_zone" / "raw" / "workspace_600.mat"
DEFAULT_RESULTS = ROOT / "femm_zone" / "results"
DEFAULT_REPORTS = ROOT / "reports"


def parse_number(text: str) -> float | str:
    text = text.strip().strip('"')
    try:
        return float(text)
    except ValueError:
        return text


def parse_fem(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    properties: list[dict[str, Any]] = []
    for block in re.findall(r"<BeginBlock>(.*?)<EndBlock>", text, flags=re.S):
        item: dict[str, Any] = {}
        for key, value in re.findall(r"<([^>]+)>\s*=\s*([^\r\n]+)", block):
            item[key] = parse_number(value)
        properties.append(item)

    match = re.search(r"\[NumBlockLabels\]\s*=\s*(\d+)\s*\r?\n", text)
    if not match:
        raise ValueError("[NumBlockLabels] was not found")
    label_count = int(match.group(1))
    label_lines = text[match.end() :].splitlines()[:label_count]
    labels: list[dict[str, Any]] = []
    for line in label_lines:
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed block-label line: {line}")
        labels.append(
            {
                "x_mm": float(fields[0]),
                "y_mm": float(fields[1]),
                "block_type_1based": int(fields[2]),
                "max_area": float(fields[3]),
                "circuit_1based": int(fields[4]),
                "magnetization_deg": float(fields[5]),
                "group": int(fields[6]),
                "turns": float(fields[7]),
                "external": int(fields[8]),
            }
        )

    header: dict[str, Any] = {}
    for key in ("Format", "Frequency", "Precision", "MinAngle", "Depth", "Coordinates", "ProblemType"):
        header_match = re.search(rf"\[{key}\]\s*=\s*([^\r\n]+)", text)
        if header_match:
            header[key] = parse_number(header_match.group(1))
    for key in ("BdryProps", "BlockProps", "CircuitProps", "NumPoints", "NumSegments", "NumArcSegments", "NumHoles", "NumBlockLabels"):
        count_match = re.search(rf"\[{key}\]\s*=\s*(\d+)", text)
        if count_match:
            header[key] = int(count_match.group(1))
    return properties, labels, header


def physical_class(prop: dict[str, Any]) -> str:
    name = str(prop.get("BlockName", ""))
    hc = float(prop.get("H_c", 0.0))
    mux = float(prop.get("Mu_x", math.nan))
    muy = float(prop.get("Mu_y", math.nan))
    if abs(hc) > 1e-9:
        return "permanent_magnet"
    if name.lower() == "pure iron" or (mux > 100 and muy > 100):
        return "iron"
    if abs(mux - 1.0) < 1e-9 and abs(muy - 1.0) < 1e-9:
        return "air"
    return "other"


def match_design_cells(
    positions: np.ndarray,
    properties: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    tolerance_mm: float = 1e-8,
) -> list[dict[str, Any]]:
    label_xy = np.array([[label["x_mm"], label["y_mm"]] for label in labels])
    cells: list[dict[str, Any]] = []
    for cell_index, row in enumerate(np.asarray(positions, dtype=float)):
        copies: list[dict[str, Any]] = []
        for copy_index in range(4):
            point = row[2 * copy_index : 2 * copy_index + 2]
            distance = np.linalg.norm(label_xy - point, axis=1)
            candidate = int(np.argmin(distance))
            if distance[candidate] > tolerance_mm:
                raise AssertionError(
                    f"Cell {cell_index + 1}, copy {copy_index + 1} has no exact FEM label; "
                    f"nearest distance={distance[candidate]:.6g} mm"
                )
            label = labels[candidate]
            prop_index = label["block_type_1based"] - 1
            prop = properties[prop_index]
            copies.append(
                {
                    "copy_index_1based": copy_index + 1,
                    "label_index_1based": candidate + 1,
                    "x_mm": label["x_mm"],
                    "y_mm": label["y_mm"],
                    "block_type_1based": label["block_type_1based"],
                    "block_name": prop.get("BlockName", ""),
                    "physical_class": physical_class(prop),
                    "magnetization_deg": label["magnetization_deg"],
                    "group": label["group"],
                }
            )
        classes = {copy["physical_class"] for copy in copies}
        if len(classes) != 1:
            raise AssertionError(f"Symmetry copies of cell {cell_index + 1} have different classes: {classes}")
        angular_index = cell_index // 10
        radial_index = cell_index % 10
        base_x, base_y = row[6], row[7]
        cells.append(
            {
                "cell_index_1based": cell_index + 1,
                "angular_index_0based": angular_index,
                "radial_index_0based": radial_index,
                "base_angle_deg": math.degrees(math.atan2(base_y, base_x)),
                "base_radius_mm": math.hypot(base_x, base_y),
                "physical_class": next(iter(classes)),
                "copies": copies,
            }
        )
    return cells


def closest_historical_mapping(data: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    class_value = {"air": 0, "permanent_magnet": 1, "iron": 2, "other": 3}
    target = np.array([class_value[cell["physical_class"]] for cell in cells], dtype=np.uint8)
    results: dict[str, Any] = {}
    for source_name, variable in (
        ("raw_pre_repair", "population_noChange_all"),
        ("corrected_femm_input", "population_all"),
    ):
        bits = np.asarray(data[variable], dtype=np.uint8)
        codes = (2 * bits[:, 0::2, :] + bits[:, 1::2, :]).transpose(2, 0, 1).reshape(-1, 100)
        # Code 1 is independently tied to VolumePM, so retain code1=PM while
        # trying every physical assignment for the remaining code values.
        mappings = []
        for c0, c2, c3 in itertools.product((0, 1, 2), repeat=3):
            lookup = np.array([c0, 1, c2, c3], dtype=np.uint8)
            best_distance = 101
            best_flat_index = -1
            for start in range(0, len(codes), 20_000):
                predicted = lookup[codes[start : start + 20_000]]
                distances = np.count_nonzero(predicted != target, axis=1)
                local = int(np.argmin(distances))
                if int(distances[local]) < best_distance:
                    best_distance = int(distances[local])
                    best_flat_index = start + local
            mappings.append((best_distance, tuple(int(x) for x in lookup), best_flat_index))
        mappings.sort()
        distance, lookup, flat_index = mappings[0]
        state, population_row = divmod(flat_index, 504)
        results[source_name] = {
            "source_variable": variable,
            "exact_match_found": distance == 0,
            "minimum_cell_mismatches": distance,
            "best_simple_mapping_code_to_class": {
                str(i): ("air", "permanent_magnet", "iron")[lookup[i]] for i in range(4)
            },
            "nearest_state_index_0based": state,
            "nearest_population_row_0based": population_row,
        }
    results["interpretation"] = (
        "The reference FEM is not an exact saved historical chromosome. Material identity therefore relies on "
        "VolumePM plus the corrected one-topology FEMM replay, not on an exact-reference-genome match."
    )
    return results


def save_cells_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        for copy in cell["copies"]:
            rows.append({key: value for key, value in cell.items() if key != "copies"} | copy)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_mapping_figure(path: Path, cells: list[dict[str, Any]]) -> None:
    class_value = {"air": 0, "permanent_magnet": 1, "iron": 2, "other": 3}
    # Vector ordering is angular-major; plot rows as radius and columns as angle.
    grid = np.array([class_value[cell["physical_class"]] for cell in cells]).reshape(10, 10).T
    colors = ["#dceeff", "#e44b4b", "#6a7480", "#f0b429"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    image = axes[0].imshow(grid, origin="lower", cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5)
    for radial in range(10):
        for angular in range(10):
            index = angular * 10 + radial + 1
            axes[0].text(angular, radial, str(index), ha="center", va="center", fontsize=7,
                         color="white" if grid[radial, angular] in (1, 2) else "black")
    axes[0].set_title("100 gene cells in the 10x10 design domain")
    axes[0].set_xlabel("Angular index (1.125° to 21.375°)")
    axes[0].set_ylabel("Radial index (16.55 mm to 30.05 mm)")
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))

    for cell in cells:
        for copy in cell["copies"]:
            cls = copy["physical_class"]
            axes[1].scatter(copy["x_mm"], copy["y_mm"], color=colors[class_value[cls]], s=18)
    axes[1].set_aspect("equal")
    axes[1].set_title("Four FEM label copies for every gene cell")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].grid(alpha=0.2)

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=colors[i], label=name)
               for i, name in enumerate(("Air", "Permanent magnet", "Iron", "Other"))]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle("Genome-to-IPMSM.fem spatial relationship")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(fem_path: Path, mat_path: Path, results_dir: Path, reports_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    properties, labels, header = parse_fem(fem_path)
    cells = match_design_cells(np.asarray(data["MaterialPosition"]), properties, labels)
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["physical_class"]] = counts.get(cell["physical_class"], 0) + 1
    comparison = closest_historical_mapping(data, cells)

    result = {
        "fem_path": str(fem_path.resolve()),
        "fem_header_and_counts": header,
        "workspace_material_position_shape": list(np.asarray(data["MaterialPosition"]).shape),
        "mapped_gene_cells": len(cells),
        "mapped_fem_block_labels": sum(len(cell["copies"]) for cell in cells),
        "all_four_copies_found_exactly": True,
        "example_physical_class_counts": counts,
        "cell_order": "cell=(angular_index*10 + radial_index)+1",
        "base_copy": "MaterialPosition columns 7-8, spanning 1.125°..21.375°",
        "history_comparison": comparison,
        "cells": cells,
    }
    with (results_dir / "ipmsm_structure_analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    save_cells_csv(results_dir / "gene_to_fem_labels.csv", cells)
    save_mapping_figure(reports_dir / "gene_to_structure_mapping.png", cells)

    print(f"Mapped {len(cells)} gene cells to {sum(len(c['copies']) for c in cells)} exact FEM labels")
    print(f"Example physical classes: {counts}")
    print(
        "History reference-match distances: "
        f"raw={comparison['raw_pre_repair']['minimum_cell_mismatches']}, "
        f"corrected={comparison['corrected_femm_input']['minimum_cell_mismatches']}"
    )
    print(f"Outputs: {results_dir} and {reports_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem", type=Path, default=DEFAULT_FEM)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    analyze(args.fem, args.mat, args.results_dir, args.reports_dir)


if __name__ == "__main__":
    main()
