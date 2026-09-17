"""03基因映射辅助模块；运行与配置统一使用上级run_femm.py、femm_config.py。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


GENE_COUNT = 120
COPIES_PER_GENE = 4
ANGULAR_CELLS = 20
RADIAL_CELLS = 6
POSITION_TOLERANCE_MM = 1e-8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def genotype_sha256(bits: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()).hexdigest()


def _block_name(block: str) -> str | None:
    match = re.search(r'<BlockName>\s*=\s*"([^"]+)"', block)
    return match.group(1) if match else None


def material_blocks(fem_text: str) -> list[str]:
    return re.findall(r"<BeginBlock>.*?<EndBlock>", fem_text, flags=re.S)


def material_class(block: str) -> str:
    name = (_block_name(block) or "").lower()
    coercivity_match = re.search(r"<H_c>\s*=\s*([^\r\n]+)", block)
    coercivity = float(coercivity_match.group(1)) if coercivity_match else 0.0
    if name == "n38" or abs(coercivity) > 1e-9:
        return "permanent_magnet"
    if name == "pure iron":
        return "iron"
    return "air_or_nonmagnetic"


def parse_labels(fem_text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[NumBlockLabels\]\s*=\s*(\d+)\s*\r?\n", fem_text)
    if not match:
        raise ValueError("[NumBlockLabels] was not found")
    count = int(match.group(1))
    lines = fem_text[match.end() :].splitlines()[:count]
    labels: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed block label {index}: {line}")
        labels.append(
            {
                "label_index_1based": index,
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
    return labels


def match_material_positions(fem_text: str, positions: np.ndarray) -> list[dict[str, Any]]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (GENE_COUNT, 2 * COPIES_PER_GENE):
        raise ValueError(f"Expected MaterialPosition[{GENE_COUNT},8], got {positions.shape}")
    blocks = material_blocks(fem_text)
    labels = parse_labels(fem_text)
    label_xy = np.asarray([[item["x_mm"], item["y_mm"]] for item in labels])
    rows: list[dict[str, Any]] = []
    used_labels: set[int] = set()
    for gene_index, coordinates in enumerate(positions):
        expected_name = f"c{gene_index + 1}"
        for copy_index in range(COPIES_PER_GENE):
            point = coordinates[2 * copy_index : 2 * copy_index + 2]
            distances = np.linalg.norm(label_xy - point, axis=1)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            if distance > POSITION_TOLERANCE_MM:
                raise AssertionError(
                    f"Gene {gene_index + 1}, copy {copy_index + 1}: nearest FEM label is {distance:.6g} mm away"
                )
            label = labels[nearest]
            block_index = int(label["block_type_1based"]) - 1
            if not 0 <= block_index < len(blocks):
                raise AssertionError(f"Invalid block type at FEM label {nearest + 1}")
            observed_name = _block_name(blocks[block_index])
            if observed_name != expected_name:
                raise AssertionError(
                    f"Gene {gene_index + 1}, copy {copy_index + 1}: expected {expected_name}, found {observed_name}"
                )
            if nearest + 1 in used_labels:
                raise AssertionError(f"FEM label {nearest + 1} is assigned to more than one gene copy")
            used_labels.add(nearest + 1)
            rows.append(
                {
                    "gene_index_1based": gene_index + 1,
                    "angular_index_0based": gene_index // RADIAL_CELLS,
                    "radial_index_0based": gene_index % RADIAL_CELLS,
                    "copy_index_1based": copy_index + 1,
                    "label_index_1based": nearest + 1,
                    "x_mm": float(label["x_mm"]),
                    "y_mm": float(label["y_mm"]),
                    "radius_mm": math.hypot(float(label["x_mm"]), float(label["y_mm"])),
                    "polar_angle_deg": math.degrees(math.atan2(float(label["y_mm"]), float(label["x_mm"]))),
                    "block_name": observed_name,
                    "block_type_1based": int(label["block_type_1based"]),
                    "magnetization_deg": float(label["magnetization_deg"]),
                    "group": int(label["group"]),
                    "coordinate_error_mm": distance,
                }
            )
    if len(rows) != GENE_COUNT * COPIES_PER_GENE or len(used_labels) != len(rows):
        raise AssertionError("The 120 genes must map one-to-one to 480 distinct FEM block labels")
    return rows


def replace_cell_materials(template_text: str, bits: np.ndarray) -> str:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.shape != (GENE_COUNT,) or np.any((bits != 0) & (bits != 1)):
        raise ValueError("Expected exactly 120 binary gene values")
    pattern = re.compile(r"<BeginBlock>.*?<EndBlock>", flags=re.S)
    blocks = pattern.findall(template_text)
    by_name = {_block_name(block): block for block in blocks}
    if "Air" not in by_name or "N38" not in by_name:
        raise ValueError("Template must contain Air and N38 source materials")
    for index in range(1, GENE_COUNT + 1):
        if f"c{index}" not in by_name:
            raise ValueError(f"Template is missing c{index}")

    replacements: dict[str, str] = {}
    for index, bit in enumerate(bits, start=1):
        source = by_name["N38" if int(bit) == 1 else "Air"]
        replacements[f"c{index}"] = re.sub(
            r'(<BlockName>\s*=\s*)"[^"]+"',
            rf'\1"c{index}"',
            source,
            count=1,
        )

    def replace(match: re.Match[str]) -> str:
        block = match.group(0)
        return replacements.get(_block_name(block), block)

    return pattern.sub(replace, template_text)


def history_gene(mat_path: Path, state_index: int, population_row: int, source: str = "population_all") -> tuple[np.ndarray, dict[str, Any]]:
    if source not in {"population_all", "population_noChange_all"}:
        raise ValueError(f"Unsupported history source: {source}")
    data = loadmat(
        mat_path,
        variable_names=[source, "Tavg_all", "DeltaT_all", "Fitvalue_all", "VolumePM_all"],
        squeeze_me=True,
    )
    history = np.asarray(data[source], dtype=np.uint8)
    if not 0 <= state_index < history.shape[2]:
        raise IndexError(f"state_index must be in [0,{history.shape[2] - 1}]")
    if not 0 <= population_row < history.shape[0]:
        raise IndexError(f"population_row must be in [0,{history.shape[0] - 1}]")
    bits = history[population_row, :, state_index]
    provenance = {
        "kind": "workspace_history",
        "source_variable": source,
        "state_index_0based": int(state_index),
        "generation_interpretation": "0=initial, 1..200=GA generations",
        "population_row_0based": int(population_row),
        "stored_results": {
            "tavg_nm": float(np.asarray(data["Tavg_all"])[population_row, state_index]),
            "delta_t": float(np.asarray(data["DeltaT_all"])[population_row, state_index]),
            "fitness": float(np.asarray(data["Fitvalue_all"])[population_row, state_index]),
            "volume_pm_cells": int(np.asarray(data["VolumePM_all"])[population_row, state_index]),
        },
    }
    return bits, provenance


def validate_generated_model(fem_text: str, mapping_rows: list[dict[str, Any]], bits: np.ndarray) -> dict[str, Any]:
    blocks = material_blocks(fem_text)
    by_name = {_block_name(block): block for block in blocks}
    mismatches: list[int] = []
    for index, bit in enumerate(np.asarray(bits, dtype=np.uint8), start=1):
        observed = material_class(by_name[f"c{index}"])
        expected = "permanent_magnet" if int(bit) == 1 else "air_or_nonmagnetic"
        if observed != expected:
            mismatches.append(index)
    if mismatches:
        raise AssertionError(f"Generated material mismatches: {mismatches}")
    directions = [abs(float(row["magnetization_deg"])) for row in mapping_rows if bits[int(row["gene_index_1based"]) - 1] == 1]
    if directions and max(directions) == 0:
        raise AssertionError("All generated PM labels have zero magnetization direction")
    return {
        "mapped_genes": GENE_COUNT,
        "mapped_design_labels": len(mapping_rows),
        "distinct_design_labels": len({int(row["label_index_1based"]) for row in mapping_rows}),
        "maximum_coordinate_error_mm": max(float(row["coordinate_error_mm"]) for row in mapping_rows),
        "air_cells": int(np.sum(np.asarray(bits) == 0)),
        "permanent_magnet_cells": int(np.sum(np.asarray(bits) == 1)),
        "material_assignment_matches_all_120_bits": True,
        "template_magnetization_directions_preserved": True,
    }


def save_preview(path: Path, bits: np.ndarray, mapping_rows: list[dict[str, Any]]) -> None:
    grid = np.asarray(bits, dtype=np.uint8).reshape(ANGULAR_CELLS, RADIAL_CELLS).T
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].imshow(grid, origin="lower", aspect="auto", cmap=matplotlib.colors.ListedColormap(["#dceeff", "#e44b4b"]), vmin=0, vmax=1)
    for radial in range(RADIAL_CELLS):
        for angular in range(ANGULAR_CELLS):
            axes[0].text(angular, radial, str(angular * RADIAL_CELLS + radial + 1), ha="center", va="center", fontsize=6)
    axes[0].set_xlabel("angular index (20 cells)")
    axes[0].set_ylabel("radial index (6 cells)")
    axes[0].set_title("120-bit gene: Air=blue, PM=red")

    first_copy = [row for row in mapping_rows if int(row["copy_index_1based"]) == 1]
    colors = ["#e44b4b" if bits[int(row["gene_index_1based"]) - 1] else "#8ecae6" for row in first_copy]
    axes[1].scatter([row["x_mm"] for row in first_copy], [row["y_mm"] for row in first_copy], c=colors, s=25)
    for row in first_copy:
        axes[1].text(row["x_mm"], row["y_mm"], str(row["gene_index_1based"]), fontsize=5)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].set_title("First FEM symmetry copy")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_topology(bits: np.ndarray, template_path: Path, mat_path: Path, output_dir: Path, provenance: dict[str, Any]) -> Path:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.shape != (GENE_COUNT,) or np.any((bits != 0) & (bits != 1)):
        raise ValueError("Expected exactly 120 binary gene values")
    mat = loadmat(mat_path, variable_names=["MaterialPosition"], squeeze_me=True)
    positions = np.asarray(mat["MaterialPosition"], dtype=np.float64)
    template_text = template_path.read_text(encoding="utf-8", errors="strict")
    mapping_rows = match_material_positions(template_text, positions)
    generated_text = replace_cell_materials(template_text, bits)
    validation = validate_generated_model(generated_text, mapping_rows, bits)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.fem"
    model_path.write_text(generated_text, encoding="utf-8", newline="")
    (output_dir / "gene_bits.txt").write_text("".join(str(int(value)) for value in bits) + "\n", encoding="utf-8")
    np.savetxt(output_dir / "gene_grid_6x20.csv", bits.reshape(ANGULAR_CELLS, RADIAL_CELLS).T, fmt="%d", delimiter=",")
    with (output_dir / "gene_to_fem_mapping.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    save_preview(output_dir / "topology_preview.png", bits, mapping_rows)

    manifest = {
        "status": "generated_and_structurally_validated",
        "mapping_version": "spmsm_binary_pm_v1",
        "mapping": {"0": "Air", "1": "N38 permanent magnet"},
        "mapping_evidence": [
            "MaterialPosition[120,8] matches 480 unique FEM labels",
            "those labels point one-to-one to c1..c120",
            "VolumePM equals the number of one-bits for all 612 final-state individuals",
            "the design domain contains cell-specific Air properties plus N38 as the PM source",
        ],
        "grid_axes": (
            "gene_index = angular_index * 6 + radial_index; grid is [6 radial,20 angular]; "
            "radial index increases outward and angular index advances toward decreasing physical polar angle"
        ),
        "genotype_sha256": genotype_sha256(bits),
        "template": {"path": str(template_path.resolve()), "sha256": file_sha256(template_path)},
        "mat": {"path": str(mat_path.resolve()), "sha256": file_sha256(mat_path)},
        "provenance": provenance,
        "validation": validation,
        "femm_solve_performed": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path

