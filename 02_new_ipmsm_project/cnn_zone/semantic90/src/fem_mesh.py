"""Build deterministic pixel lookup tables from a solved FEMM triangle mesh."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.tri as mtri
import numpy as np
from scipy.io import loadmat

from femm_zone.scripts.analyze_ipmsm_structure import match_design_cells, parse_fem, physical_class


FIXED_BACKGROUND = 0
FIXED_AIR = 1
FIXED_IRON = 2
FIXED_WINDING = 3
FIXED_OTHER = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ans_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return FEMM solution nodes, triangles and zero-based block-label IDs."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        solution_line = lines.index("[Solution]")
    except ValueError as exc:
        raise ValueError(f"No [Solution] section in {path}") from exc
    node_count = int(lines[solution_line + 1])
    node_start = solution_line + 2
    nodes = np.empty((node_count, 2), dtype=np.float64)
    for index, line in enumerate(lines[node_start : node_start + node_count]):
        fields = line.split()
        nodes[index] = (float(fields[0]), float(fields[1]))
    element_count_line = node_start + node_count
    element_count = int(lines[element_count_line])
    element_start = element_count_line + 1
    triangles = np.empty((element_count, 3), dtype=np.int32)
    block_label = np.empty(element_count, dtype=np.int32)
    for index, line in enumerate(lines[element_start : element_start + element_count]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed FEMM element row {index + 1}")
        triangles[index] = (int(fields[0]), int(fields[1]), int(fields[2]))
        block_label[index] = int(fields[3])
    if triangles.min() < 0 or triangles.max() >= node_count:
        raise AssertionError("Triangle node index is outside the node table")
    return nodes, triangles, block_label


def _fixed_class_for_label(label: dict[str, Any], properties: list[dict[str, Any]]) -> int:
    prop = properties[label["block_type_1based"] - 1]
    name = str(prop.get("BlockName", "")).strip().lower()
    if name == "copper" or label["circuit_1based"] > 0:
        return FIXED_WINDING
    cls = physical_class(prop)
    if cls == "air":
        return FIXED_AIR
    if cls == "iron":
        return FIXED_IRON
    return FIXED_OTHER


def build_lookup(
    fem_path: Path,
    mat_path: Path,
    ans_path: Path,
    image_size: int,
    output_path: Path,
) -> dict[str, Any]:
    if image_size not in (128, 224):
        raise ValueError("Supported image sizes are 128 and 224")
    properties, labels, fem_header = parse_fem(fem_path)
    ans_properties, ans_labels, _ = parse_fem(ans_path)
    if len(labels) != len(ans_labels):
        raise AssertionError("Reference FEM and mesh solution have different block-label counts")
    fem_xy = np.asarray([[item["x_mm"], item["y_mm"]] for item in labels])
    ans_xy = np.asarray([[item["x_mm"], item["y_mm"]] for item in ans_labels])
    if not np.allclose(fem_xy, ans_xy, atol=1e-10, rtol=0):
        raise AssertionError("The solution mesh does not share the reference FEM block-label ordering")
    data = loadmat(mat_path, variable_names=["MaterialPosition"], squeeze_me=True)
    cells = match_design_cells(np.asarray(data["MaterialPosition"]), properties, labels)
    label_to_gene = np.full(len(labels), -1, dtype=np.int16)
    label_to_replica = np.full(len(labels), -1, dtype=np.int8)
    design_label_indices: list[int] = []
    for gene_index, cell in enumerate(cells):
        for copy in cell["copies"]:
            label_index = int(copy["label_index_1based"]) - 1
            if label_to_gene[label_index] != -1:
                raise AssertionError("A FEM block label was assigned to two genes")
            label_to_gene[label_index] = gene_index
            # MaterialPosition copies are ordered 67.5..90, 45..67.5,
            # 22.5..45, 0..22.5.  Replica IDs follow increasing physical angle.
            label_to_replica[label_index] = 4 - int(copy["copy_index_1based"])
            design_label_indices.append(label_index)
    if len(design_label_indices) != 400 or len(set(design_label_indices)) != 400:
        raise AssertionError("Expected 400 distinct design-region labels")
    if not np.all(np.bincount(label_to_gene[label_to_gene >= 0], minlength=100) == 4):
        raise AssertionError("Each gene must map to exactly four FEM regions")

    nodes, triangles, element_label = parse_ans_mesh(ans_path)
    if element_label.min() < 0 or element_label.max() >= len(labels):
        raise AssertionError("Mesh element references an invalid block-label index")
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    finder = triangulation.get_trifinder()
    lower = float(min(0.0, nodes[:, 0].min(), nodes[:, 1].min()))
    upper = float(max(nodes[:, 0].max(), nodes[:, 1].max()))
    spacing = (upper - lower) / image_size
    axis = lower + (np.arange(image_size, dtype=np.float64) + 0.5) * spacing
    x_grid, y_grid = np.meshgrid(axis, axis)
    triangle_image = finder(x_grid, y_grid).astype(np.int32)
    geometry_mask = triangle_image >= 0
    pixel_block_label = np.full((image_size, image_size), -1, dtype=np.int16)
    pixel_block_label[geometry_mask] = element_label[triangle_image[geometry_mask]]
    pixel_gene_id = np.full_like(pixel_block_label, -1)
    pixel_replica_id = np.full((image_size, image_size), -1, dtype=np.int8)
    pixel_gene_id[geometry_mask] = label_to_gene[pixel_block_label[geometry_mask]]
    pixel_replica_id[geometry_mask] = label_to_replica[pixel_block_label[geometry_mask]]
    design_mask = pixel_gene_id >= 0

    fixed_class_by_label = np.asarray(
        [_fixed_class_for_label(label, properties) for label in labels], dtype=np.uint8
    )
    fixed_material_map = np.zeros((image_size, image_size), dtype=np.uint8)
    fixed_pixels = geometry_mask & ~design_mask
    fixed_material_map[fixed_pixels] = fixed_class_by_label[pixel_block_label[fixed_pixels]]
    magnet_polarity_map = np.zeros((image_size, image_size), dtype=np.int8)
    physical_angle = np.degrees(np.arctan2(y_grid, x_grid))
    magnet_polarity_map[design_mask & (physical_angle < 45.0)] = -1
    magnet_polarity_map[design_mask & (physical_angle >= 45.0)] = 1
    boundary_map = np.zeros((image_size, image_size), dtype=np.uint8)
    boundary_map[:, 1:] |= pixel_block_label[:, 1:] != pixel_block_label[:, :-1]
    boundary_map[1:, :] |= pixel_block_label[1:, :] != pixel_block_label[:-1, :]
    boundary_map &= geometry_mask.astype(np.uint8)

    # Strong geometric validation at all 400 exact block-label centres.
    centre_triangles = finder(fem_xy[design_label_indices, 0], fem_xy[design_label_indices, 1])
    centre_found_labels = element_label[centre_triangles]
    centre_match = centre_found_labels == np.asarray(design_label_indices)
    if not np.all(centre_triangles >= 0) or not np.all(centre_match):
        bad = int(np.sum(~centre_match))
        raise AssertionError(f"{bad} design block-label centres map to the wrong mesh region")
    represented_gene_ids = np.unique(pixel_gene_id[design_mask])
    if not np.array_equal(represented_gene_ids, np.arange(100)):
        raise AssertionError("Rasterization lost one or more gene regions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pixel_gene_id=pixel_gene_id,
        pixel_replica_id=pixel_replica_id,
        fixed_material_map=fixed_material_map,
        geometry_mask=geometry_mask.astype(np.uint8),
        magnet_polarity_map=magnet_polarity_map,
        boundary_map=boundary_map,
        pixel_block_label=pixel_block_label,
        x_axis_mm=axis,
        y_axis_mm=axis,
        label_to_gene=label_to_gene,
        label_to_replica=label_to_replica,
    )
    pixels_per_gene = np.bincount(pixel_gene_id[design_mask], minlength=100)
    summary = {
        "lookup_path": str(output_path.resolve()),
        "image_size": image_size,
        "coordinate_bounds_mm": [lower, upper],
        "pixel_spacing_mm": spacing,
        "mesh_nodes": int(len(nodes)),
        "mesh_triangles": int(len(triangles)),
        "fem_block_labels": len(labels),
        "design_block_labels": 400,
        "mapped_gene_count": 100,
        "copies_per_gene": 4,
        "all_400_label_centres_match_mesh_regions": True,
        "design_pixels": int(np.sum(design_mask)),
        "minimum_pixels_per_gene": int(pixels_per_gene.min()),
        "maximum_pixels_per_gene": int(pixels_per_gene.max()),
        "fixed_pixel_counts": {
            "air": int(np.sum(fixed_material_map == FIXED_AIR)),
            "iron": int(np.sum(fixed_material_map == FIXED_IRON)),
            "winding": int(np.sum(fixed_material_map == FIXED_WINDING)),
            "other": int(np.sum(fixed_material_map == FIXED_OTHER)),
        },
        "replica_definition": {
            "0": "0..22.5 deg, base",
            "1": "22.5..45 deg, mirrored",
            "2": "45..67.5 deg, base",
            "3": "67.5..90 deg, mirrored",
        },
        "pm_polarity_definition": {"-1": "radially inward, 0..45 deg", "1": "radially outward, 45..90 deg"},
        "source": {
            "fem": str(fem_path.resolve()),
            "fem_sha256": file_sha256(fem_path),
            "mat": str(mat_path.resolve()),
            "mat_sha256": file_sha256(mat_path),
            "ans_mesh": str(ans_path.resolve()),
            "ans_mesh_sha256": file_sha256(ans_path),
            "fem_counts": fem_header,
            "ans_material_count": len(ans_properties),
        },
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

