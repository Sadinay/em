"""Build a FEMM-ready IPMSM topology from one 200-bit chromosome.

The template and MAT source are read-only.  The generated model uses the
best-supported three-material interpretation of the four genetic codes:
0=Air, 1=permanent magnet, 2=iron, 3=iron.  Codes 2 and 3 therefore have the
same physical topology hash while retaining distinct genotype hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from analyze_ipmsm_structure import match_design_cells, parse_fem, physical_class, save_mapping_figure


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "femm_zone" / "models" / "IPMSM.fem"
DEFAULT_MAT = ROOT / "data_zone" / "raw" / "workspace_600.mat"
DEFAULT_OUTPUT = ROOT / "femm_zone" / "results" / "generated_topology"

MAPPING_VERSION = "inferred_three_material_v1"
CODE_TO_CLASS = np.array([0, 1, 2, 2], dtype=np.uint8)
MAPPING_CANDIDATES = {
    "air0": np.array([0, 1, 2, 2], dtype=np.uint8),
    "air2": np.array([2, 1, 0, 2], dtype=np.uint8),
    "air3": np.array([2, 1, 2, 0], dtype=np.uint8),
}
CLASS_NAMES = {0: "air", 1: "permanent_magnet", 2: "iron"}
CLASS_TO_BIT_PAIR = {0: "00", 1: "01", 2: "10"}


def expected_pm_magnetization_deg(x_mm: float, y_mm: float) -> float:
    """Return the radial PM direction used by the audited 90-degree template.

    The 45..90 degree pole is radially outward.  The 0..45 degree pole is
    radially inward.  Values above 202.5 degrees are represented by their
    equivalent negative angle to match the template's convention.
    """
    position_angle = math.degrees(math.atan2(y_mm, x_mm)) % 360.0
    if not -1e-9 <= position_angle <= 90.0 + 1e-9:
        raise ValueError(f"Expected a first-quadrant design label, got angle {position_angle}")
    direction = position_angle if position_angle >= 45.0 else position_angle + 180.0
    if direction > 202.5:
        direction -= 360.0
    return direction


def angular_distance_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_bits(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.shape != (200,):
        raise ValueError(f"A chromosome must contain exactly 200 bits, got {bits.size}")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Chromosome values must be only 0 or 1")
    return 2 * bits[0::2] + bits[1::2]


def physical_classes(material_codes: np.ndarray, code_to_class: np.ndarray = CODE_TO_CLASS) -> np.ndarray:
    codes = np.asarray(material_codes, dtype=np.uint8).reshape(-1)
    if codes.shape != (100,) or np.any(codes > 3):
        raise ValueError("Material codes must be a length-100 vector in {0,1,2,3}")
    mapping = np.asarray(code_to_class, dtype=np.uint8)
    if mapping.shape != (4,) or np.any(mapping > 2):
        raise ValueError("code_to_class must contain four class codes in {0,1,2}")
    return mapping[codes]


def genotype_hash(bits: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()).hexdigest()


def topology_hash(classes: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(classes, dtype=np.uint8).tobytes()).hexdigest()


def canonical_bits_from_classes(classes: np.ndarray) -> np.ndarray:
    pairs = "".join(CLASS_TO_BIT_PAIR[int(value)] for value in np.asarray(classes).reshape(-1))
    return np.fromiter((int(char) for char in pairs), dtype=np.uint8, count=200)


def _extract_bits_from_mapping(mapping: dict[str, Any]) -> np.ndarray:
    for key in (
        "genome_bits_corrected",
        "genome_bits_raw",
        "bits",
        "gene",
        "chromosome",
    ):
        if key in mapping:
            return np.asarray(mapping[key], dtype=np.uint8).reshape(-1)
    raise KeyError("No recognized chromosome key was found")


def read_gene_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _extract_bits_from_mapping(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".npz":
        with np.load(path) as data:
            for key in ("genome_bits_corrected", "genome_bits_raw", "bits", "gene", "chromosome"):
                if key in data:
                    values = np.asarray(data[key])
                    if values.ndim == 1:
                        return values.astype(np.uint8)
                    if values.ndim == 2 and values.shape[0] == 1:
                        return values[0].astype(np.uint8)
        raise KeyError("No single 200-bit chromosome was found in NPZ")
    text = path.read_text(encoding="utf-8-sig")
    tokens = re.findall(r"(?<!\d)[01](?!\d)", text)
    if len(tokens) == 200:
        return np.asarray(tokens, dtype=np.uint8)
    compact = re.sub(r"[^01]", "", text)
    if len(compact) == 200:
        return np.fromiter((int(char) for char in compact), dtype=np.uint8, count=200)
    raise ValueError(f"Could not identify exactly 200 bits in {path}; found {len(tokens)} separated tokens")


def gene_from_mat(
    mat_path: Path,
    state_index: int,
    population_row: int | None,
    gene_source: str = "after_correction",
) -> tuple[np.ndarray, dict[str, Any]]:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    source_variables = {
        "before_correction": "population_noChange_all",
        "after_correction": "population_all",
    }
    if gene_source not in source_variables:
        raise ValueError(f"Unknown gene_source {gene_source!r}")
    source_variable = source_variables[gene_source]
    history = np.asarray(data[source_variable], dtype=np.uint8)
    state_count = history.shape[2]
    if not 0 <= state_index < state_count:
        raise IndexError(f"state_index must be in [0,{state_count - 1}]")
    if population_row is None:
        fitness = np.asarray(data["Fitvalue_all"], dtype=np.float64)[:, state_index]
        population_row = int(np.argmax(fitness))
    if not 0 <= population_row < history.shape[0]:
        raise IndexError(f"population_row must be in [0,{history.shape[0] - 1}]")
    bits = history[population_row, :, state_index]
    provenance = {
        "kind": "mat_history",
        "mat_path": str(mat_path.resolve()),
        "mat_sha256": file_sha256(mat_path),
        "state_index_0based": state_index,
        "generation_interpretation": "0=initial, 1..600=GA generations",
        "population_row_0based": population_row,
        "gene_source": gene_source,
        "source_variable": source_variable,
        "stored_results": {
            "t_avg_nm": float(np.asarray(data["Tavg_all"])[population_row, state_index]),
            "delta_t": float(np.asarray(data["DeltaT_all"])[population_row, state_index]),
            "fitness": float(np.asarray(data["Fitvalue_all"])[population_row, state_index]),
        },
    }
    return bits, provenance


def _block_name(body: str) -> str | None:
    match = re.search(r'<BlockName>\s*=\s*"([^"]+)"', body)
    return match.group(1) if match else None


def replace_cell_material_properties(template_text: str, classes: np.ndarray) -> str:
    pattern = re.compile(r"<BeginBlock>.*?<EndBlock>", flags=re.S)
    blocks = pattern.findall(template_text)
    by_name = {_block_name(block): block for block in blocks}
    if "Air" not in by_name or "N38" not in by_name:
        raise ValueError("Template must contain source material blocks named Air and N38")
    for cell_index in range(1, 101):
        if f"c{cell_index}" not in by_name:
            raise ValueError(f"Template is missing cell material c{cell_index}")

    replacements: dict[str, str] = {}
    for cell_index, class_code in enumerate(classes, start=1):
        # Iron labels point to generic Pure Iron.  Reset the unused cN property
        # to Air so the generated file is deterministic and contains no stale PM.
        source_name = "N38" if int(class_code) == 1 else "Air"
        source = by_name[source_name]
        replacement = re.sub(
            r'(<BlockName>\s*=\s*)"[^"]+"',
            rf'\1"c{cell_index}"',
            source,
            count=1,
        )
        replacements[f"c{cell_index}"] = replacement

    def callback(match: re.Match[str]) -> str:
        body = match.group(0)
        name = _block_name(body)
        return replacements.get(name, body)

    return pattern.sub(callback, template_text)


def replace_design_block_labels(
    fem_text: str,
    positions: np.ndarray,
    classes: np.ndarray,
) -> tuple[str, list[dict[str, Any]]]:
    properties, labels, _ = parse_fem_text(fem_text)
    cells = match_design_cells(np.asarray(positions), properties, labels)
    design_label_to_cell: dict[int, tuple[int, int]] = {}
    for cell_index, cell in enumerate(cells):
        for copy in cell["copies"]:
            design_label_to_cell[int(copy["label_index_1based"])] = (
                cell_index,
                int(copy["copy_index_1based"]),
            )
    if len(design_label_to_cell) != 400:
        raise AssertionError("Expected exactly 400 design-domain block labels")

    header_match = re.search(r"\[NumBlockLabels\]\s*=\s*(\d+)\s*\r?\n", fem_text)
    if not header_match:
        raise ValueError("[NumBlockLabels] not found")
    label_count = int(header_match.group(1))
    tail = fem_text[header_match.end() :]
    tail_lines = tail.splitlines(keepends=True)
    if len(tail_lines) < label_count:
        raise ValueError("FEM label section is truncated")

    manifest_rows: list[dict[str, Any]] = []
    for label_index in range(1, label_count + 1):
        if label_index not in design_label_to_cell:
            continue
        line = tail_lines[label_index - 1]
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        fields = line.rstrip("\r\n").split()
        if len(fields) < 9:
            raise ValueError(f"Malformed block label {label_index}")
        cell_index, copy_index = design_label_to_cell[label_index]
        class_code = int(classes[cell_index])
        # Air and PM use the cell-specific cN property at block index 4+N.
        # Iron uses generic Pure Iron at block index 4.
        block_type = 4 if class_code == 2 else 5 + cell_index
        fields[2] = str(block_type)
        if class_code == 1:
            # A label that was Air/Iron in the reference snapshot often has a
            # zero magnetization angle.  When the chromosome turns that cell
            # into a PM, derive the correct pole-dependent radial direction.
            fields[5] = f"{expected_pm_magnetization_deg(float(fields[0]), float(fields[1])):.17g}"
        tail_lines[label_index - 1] = "\t".join(fields) + newline
        manifest_rows.append(
            {
                "cell_index_1based": cell_index + 1,
                "copy_index_1based": copy_index,
                "label_index_1based": label_index,
                "x_mm": float(fields[0]),
                "y_mm": float(fields[1]),
                "block_type_1based": block_type,
                "physical_class": CLASS_NAMES[class_code],
                "magnetization_deg": float(fields[5]),
                "group": int(fields[6]),
            }
        )
    return fem_text[: header_match.end()] + "".join(tail_lines), manifest_rows


def parse_fem_text(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use the audited parser without writing an intermediate file."""
    # The parser accepts a path, so reproduce only its two required sections.
    properties: list[dict[str, Any]] = []
    for block in re.findall(r"<BeginBlock>(.*?)<EndBlock>", text, flags=re.S):
        item: dict[str, Any] = {}
        for key, value in re.findall(r"<([^>]+)>\s*=\s*([^\r\n]+)", block):
            raw = value.strip().strip('"')
            try:
                item[key] = float(raw)
            except ValueError:
                item[key] = raw
        properties.append(item)
    match = re.search(r"\[NumBlockLabels\]\s*=\s*(\d+)\s*\r?\n", text)
    if not match:
        raise ValueError("[NumBlockLabels] not found")
    count = int(match.group(1))
    labels = []
    for line in text[match.end() :].splitlines()[:count]:
        fields = line.split()
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
    return properties, labels, {}


def validate_generated_model(fem_text: str, positions: np.ndarray, expected_classes: np.ndarray) -> dict[str, Any]:
    properties, labels, _ = parse_fem_text(fem_text)
    cells = match_design_cells(np.asarray(positions), properties, labels)
    observed_names = [cell["physical_class"] for cell in cells]
    expected_names = [CLASS_NAMES[int(value)] for value in expected_classes]
    mismatches = [index + 1 for index, (left, right) in enumerate(zip(observed_names, expected_names)) if left != right]
    copy_consistency = all(
        len({copy["physical_class"] for copy in cell["copies"]}) == 1 for cell in cells
    )
    if mismatches or not copy_consistency:
        raise AssertionError(f"Generated FEM validation failed; mismatched cells={mismatches}")
    pm_direction_errors = []
    for cell in cells:
        if cell["physical_class"] != "permanent_magnet":
            continue
        for copy in cell["copies"]:
            expected_direction = expected_pm_magnetization_deg(copy["x_mm"], copy["y_mm"])
            pm_direction_errors.append(angular_distance_deg(copy["magnetization_deg"], expected_direction))
    maximum_pm_direction_error = max(pm_direction_errors, default=0.0)
    if maximum_pm_direction_error > 1e-9:
        raise AssertionError(f"Generated PM magnetization direction error: {maximum_pm_direction_error} deg")
    counts = {name: observed_names.count(name) for name in ("air", "permanent_magnet", "iron")}
    return {
        "mapped_cells": len(cells),
        "mapped_design_labels": sum(len(cell["copies"]) for cell in cells),
        "four_copy_material_consistency": copy_consistency,
        "physical_class_counts": counts,
        "all_expected_materials_match": True,
        "pm_label_count": len(pm_direction_errors),
        "maximum_pm_magnetization_direction_error_deg": maximum_pm_direction_error,
        "all_pm_magnetization_directions_match_template_rule": True,
    }


def build_topology(
    bits: np.ndarray,
    template_path: Path,
    mat_path: Path,
    output_dir: Path,
    provenance: dict[str, Any],
    code_to_class: np.ndarray = CODE_TO_CLASS,
    mapping_version: str = MAPPING_VERSION,
) -> Path:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    material_codes = decode_bits(bits)
    code_to_class = np.asarray(code_to_class, dtype=np.uint8)
    classes = physical_classes(material_codes, code_to_class)
    mat_data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    positions = np.asarray(mat_data["MaterialPosition"], dtype=np.float64)
    if positions.shape != (100, 8):
        raise ValueError(f"Expected MaterialPosition[100,8], got {positions.shape}")

    template_text = template_path.read_text(encoding="utf-8", errors="strict")
    generated_text = replace_cell_material_properties(template_text, classes)
    generated_text, mapping_rows = replace_design_block_labels(generated_text, positions, classes)
    validation = validate_generated_model(generated_text, positions, classes)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.fem"
    model_path.write_text(generated_text, encoding="utf-8", newline="")
    generated_properties, generated_labels, _ = parse_fem_text(generated_text)
    generated_cells = match_design_cells(positions, generated_properties, generated_labels)
    save_mapping_figure(output_dir / "topology_preview.png", generated_cells)

    canonical_bits = canonical_bits_from_classes(classes)
    grid_codes = material_codes.reshape(10, 10).T
    grid_classes = classes.reshape(10, 10).T
    np.savetxt(output_dir / "material_codes_10x10.csv", grid_codes, fmt="%d", delimiter=",")
    np.savetxt(output_dir / "physical_classes_10x10.csv", grid_classes, fmt="%d", delimiter=",")
    with (output_dir / "gene_bits.txt").open("w", encoding="utf-8") as stream:
        stream.write("".join(str(int(value)) for value in bits) + "\n")
    with (output_dir / "canonical_physical_bits.txt").open("w", encoding="utf-8") as stream:
        stream.write("".join(str(int(value)) for value in canonical_bits) + "\n")

    with (output_dir / "gene_to_fem_mapping.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)

    manifest = {
        "status": "generated_and_structurally_validated",
        "femm_solve_performed": False,
        "mapping_mode": mapping_version,
        "mapping_evidence_status": "strongly inferred; matching MATLAB assignment source is absent",
        "code_mapping": {str(code): CLASS_NAMES[int(code_to_class[code])] for code in range(4)},
        "grid_axes": "[radial_index, angular_index]",
        "symmetry": "each of 100 cells is written to four audited FEM block-label coordinates",
        "genotype_sha256": genotype_hash(bits),
        "physical_topology_sha256": topology_hash(classes),
        "template": {
            "path": str(template_path.resolve()),
            "sha256": file_sha256(template_path),
        },
        "material_position_source": {
            "path": str(mat_path.resolve()),
            "sha256": file_sha256(mat_path),
        },
        "provenance": provenance,
        "material_code_counts": {str(code): int(np.sum(material_codes == code)) for code in range(4)},
        "validation": validation,
        "files": {
            "model": "model.fem",
            "original_gene": "gene_bits.txt",
            "canonical_three_material_gene": "canonical_physical_bits.txt",
            "material_code_grid": "material_codes_10x10.csv",
            "physical_class_grid": "physical_classes_10x10.csv",
            "cell_copy_mapping": "gene_to_fem_mapping.csv",
            "topology_preview": "topology_preview.png",
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return model_path


def run_hidden_mesh_check(model_path: Path) -> dict[str, Any]:
    """Open the generated model in FEMM and create a mesh without solving."""
    import femm

    started = time.perf_counter()
    opened = False
    try:
        femm.openfemm(1)
        opened = True
        femm.opendocument(str(model_path.resolve()))
        opened_at = time.perf_counter()
        mesh_result = femm.mi_createmesh()
        meshed_at = time.perf_counter()
        return {
            "status": "passed",
            "mode": "hidden FEMM open + mi_createmesh; no solve",
            "femm_open_seconds": opened_at - started,
            "mesh_seconds": meshed_at - opened_at,
            "mesh_return_value": mesh_result,
        }
    finally:
        if opened:
            try:
                femm.mi_close()
            except Exception:
                pass
            try:
                femm.closefemm()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--gene", help="A 200-character binary string")
    source.add_argument("--gene-file", type=Path, help="JSON, NPZ, TXT, or CSV containing one chromosome")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT, help="Source of MaterialPosition and optional history gene")
    parser.add_argument("--state-index", type=int, default=600, help="MAT state: 0=initial, 1..600=generations")
    parser.add_argument("--population-row", type=int, default=None, help="MAT population row; default=best fitness in state")
    parser.add_argument(
        "--gene-source",
        choices=("before_correction", "after_correction"),
        default="after_correction",
        help="Which MAT history tensor supplies the chromosome",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh-check", action="store_true", help="Open FEMM hidden and verify mesh generation; does not solve")
    parser.add_argument(
        "--mapping-candidate",
        choices=sorted(MAPPING_CANDIDATES),
        default="air0",
        help="Uniform candidate mapping: code 1 is PM; named code is Air; the remaining codes are Iron",
    )
    args = parser.parse_args()

    if args.gene is not None:
        compact = re.sub(r"[^01]", "", args.gene)
        bits = np.fromiter((int(char) for char in compact), dtype=np.uint8)
        provenance = {"kind": "command_line_gene"}
    elif args.gene_file is not None:
        bits = read_gene_file(args.gene_file)
        provenance = {"kind": "gene_file", "path": str(args.gene_file.resolve()), "sha256": file_sha256(args.gene_file)}
    else:
        bits, provenance = gene_from_mat(args.mat, args.state_index, args.population_row, args.gene_source)

    model_path = build_topology(
        bits,
        args.template,
        args.mat,
        args.output_dir,
        provenance,
        code_to_class=MAPPING_CANDIDATES[args.mapping_candidate],
        mapping_version=f"candidate_{args.mapping_candidate}_v1",
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.mesh_check:
        manifest["mesh_validation"] = run_hidden_mesh_check(model_path)
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"Generated FEM model: {model_path}")
    print(f"Genotype hash: {manifest['genotype_sha256']}")
    print(f"Physical topology hash: {manifest['physical_topology_sha256']}")
    print(f"Physical counts: {manifest['validation']['physical_class_counts']}")
    print("Structural validation: 100 cells / 400 symmetric labels matched")
    if args.mesh_check:
        print(f"FEMM mesh validation: passed; return={manifest['mesh_validation']['mesh_return_value']}")
    print("FEMM solve performed: no")


if __name__ == "__main__":
    main()
