"""Build Cartesian and polar 224x224 lookups from the real SPMSM FEM mesh."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.tri as mtri
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from femm_zone.scripts.spmsm_mapping import (  # noqa: E402
    _block_name,
    match_material_positions,
    material_blocks,
    parse_labels,
)


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
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        solution_line = lines.index("[Solution]")
    except ValueError as exc:
        raise ValueError(f"No [Solution] section in {path}") from exc
    node_count = int(lines[solution_line + 1])
    node_start = solution_line + 2
    nodes = np.asarray(
        [[float(value) for value in line.split()[:2]] for line in lines[node_start : node_start + node_count]],
        dtype=np.float64,
    )
    element_count_line = node_start + node_count
    element_count = int(lines[element_count_line])
    element_start = element_count_line + 1
    triangles = np.empty((element_count, 3), dtype=np.int32)
    block_label = np.empty(element_count, dtype=np.int32)
    for index, line in enumerate(lines[element_start : element_start + element_count]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed FEMM element row {index + 1}")
        triangles[index] = [int(value) for value in fields[:3]]
        block_label[index] = int(fields[3])
    if triangles.min() < 0 or triangles.max() >= node_count:
        raise AssertionError("Triangle node index is outside the node table")
    return nodes, triangles, block_label


def _fixed_class_for_label(label: dict[str, Any], blocks: list[str]) -> int:
    block_index = int(label["block_type_1based"]) - 1
    name = (_block_name(blocks[block_index]) or "").strip().lower()
    if name == "copper" or int(label["circuit_1based"]) > 0:
        return FIXED_WINDING
    if name == "air":
        return FIXED_AIR
    if name == "pure iron":
        return FIXED_IRON
    return FIXED_OTHER


def _query_grid(nodes: np.ndarray, image_size: int, coordinate_system: str):
    outer_radius = float(np.hypot(nodes[:, 0], nodes[:, 1]).max())
    if coordinate_system == "xy":
        lower = float(min(0.0, nodes[:, 0].min(), nodes[:, 1].min()))
        upper = float(max(nodes[:, 0].max(), nodes[:, 1].max()))
        axis = lower + (np.arange(image_size, dtype=np.float64) + 0.5) * (upper - lower) / image_size
        x_grid, y_grid = np.meshgrid(axis, axis)
        axes = {"x_axis_mm": axis, "y_axis_mm": axis}
        bounds = {"x_mm": [lower, upper], "y_mm": [lower, upper]}
    elif coordinate_system == "polar90":
        radius = (np.arange(image_size, dtype=np.float64) + 0.5) * outer_radius / image_size
        theta = (np.arange(image_size, dtype=np.float64) + 0.5) * 90.0 / image_size
        theta_grid, radius_grid = np.meshgrid(np.deg2rad(theta), radius)
        x_grid = radius_grid * np.cos(theta_grid)
        y_grid = radius_grid * np.sin(theta_grid)
        axes = {"theta_axis_deg": theta, "radius_axis_mm": radius}
        bounds = {"theta_deg": [0.0, 90.0], "radius_mm": [0.0, outer_radius]}
    else:
        raise ValueError("coordinate_system must be 'xy' or 'polar90'")
    return x_grid, y_grid, axes, bounds, outer_radius


def build_lookup(
    fem_path: Path,
    mat_path: Path,
    ans_path: Path,
    image_size: int,
    coordinate_system: str,
    output_path: Path,
) -> dict[str, Any]:
    if image_size != 224:
        raise ValueError("This V3 specification uses 224x224 inputs")
    fem_text = fem_path.read_text(encoding="utf-8", errors="strict")
    ans_text = ans_path.read_text(encoding="utf-8", errors="replace")
    labels = parse_labels(fem_text)
    ans_labels = parse_labels(ans_text)
    if len(labels) != len(ans_labels):
        raise AssertionError("Reference FEM and solved mesh have different block-label counts")
    label_xy = np.asarray([[row["x_mm"], row["y_mm"]] for row in labels])
    ans_xy = np.asarray([[row["x_mm"], row["y_mm"]] for row in ans_labels])
    if not np.allclose(label_xy, ans_xy, atol=1e-10, rtol=0):
        raise AssertionError("Solved mesh does not preserve the reference FEM label ordering")

    positions = np.asarray(loadmat(mat_path, variable_names=["MaterialPosition"], squeeze_me=True)["MaterialPosition"])
    mapping = match_material_positions(fem_text, positions)
    label_to_gene = np.full(len(labels), -1, dtype=np.int16)
    label_to_replica = np.full(len(labels), -1, dtype=np.int8)
    polarity_by_label = np.zeros(len(labels), dtype=np.int8)
    design_labels: list[int] = []
    for row in mapping:
        label_index = int(row["label_index_1based"]) - 1
        gene_index = int(row["gene_index_1based"]) - 1
        copy_index = int(row["copy_index_1based"]) - 1
        if label_to_gene[label_index] >= 0:
            raise AssertionError("A design label is assigned twice")
        label_to_gene[label_index] = gene_index
        label_to_replica[label_index] = copy_index
        radial_angle = math.degrees(math.atan2(float(row["y_mm"]), float(row["x_mm"])))
        difference = math.radians(float(row["magnetization_deg"]) - radial_angle)
        polarity_by_label[label_index] = 1 if math.cos(difference) >= 0 else -1
        design_labels.append(label_index)
    if len(set(design_labels)) != 480:
        raise AssertionError("Expected 480 unique design labels")
    if not np.all(np.bincount(label_to_gene[label_to_gene >= 0], minlength=120) == 4):
        raise AssertionError("Each of 120 genes must map to four physical regions")

    nodes, triangles, element_label = parse_ans_mesh(ans_path)
    if element_label.min() < 0 or element_label.max() >= len(labels):
        raise AssertionError("Mesh element references an invalid block-label index")
    finder = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles).get_trifinder()
    x_grid, y_grid, axes, bounds, outer_radius = _query_grid(nodes, image_size, coordinate_system)
    triangle_image = finder(x_grid, y_grid).astype(np.int32)
    geometry_mask = triangle_image >= 0
    pixel_block_label = np.full((image_size, image_size), -1, dtype=np.int16)
    pixel_block_label[geometry_mask] = element_label[triangle_image[geometry_mask]]
    pixel_gene_id = np.full_like(pixel_block_label, -1)
    pixel_replica_id = np.full((image_size, image_size), -1, dtype=np.int8)
    pixel_gene_id[geometry_mask] = label_to_gene[pixel_block_label[geometry_mask]]
    pixel_replica_id[geometry_mask] = label_to_replica[pixel_block_label[geometry_mask]]
    design_mask = pixel_gene_id >= 0

    blocks = material_blocks(fem_text)
    fixed_class_by_label = np.asarray([_fixed_class_for_label(label, blocks) for label in labels], dtype=np.uint8)
    fixed_material_map = np.zeros((image_size, image_size), dtype=np.uint8)
    fixed_pixels = geometry_mask & ~design_mask
    fixed_material_map[fixed_pixels] = fixed_class_by_label[pixel_block_label[fixed_pixels]]
    magnet_polarity_map = np.zeros((image_size, image_size), dtype=np.int8)
    magnet_polarity_map[design_mask] = polarity_by_label[pixel_block_label[design_mask]]
    boundary_map = np.zeros((image_size, image_size), dtype=np.uint8)
    boundary_map[:, 1:] |= pixel_block_label[:, 1:] != pixel_block_label[:, :-1]
    boundary_map[1:, :] |= pixel_block_label[1:, :] != pixel_block_label[:-1, :]
    boundary_map &= geometry_mask.astype(np.uint8)

    centre_triangles = finder(label_xy[design_labels, 0], label_xy[design_labels, 1])
    if np.any(centre_triangles < 0):
        raise AssertionError("A design label centre is outside the solved mesh")
    if not np.array_equal(element_label[centre_triangles], np.asarray(design_labels)):
        raise AssertionError("One or more design label centres map to the wrong mesh region")
    represented = np.unique(pixel_gene_id[design_mask])
    if not np.array_equal(represented, np.arange(120)):
        missing = np.setdiff1d(np.arange(120), represented).tolist()
        raise AssertionError(f"224 raster lost gene regions: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pixel_gene_id": pixel_gene_id,
        "pixel_replica_id": pixel_replica_id,
        "fixed_material_map": fixed_material_map,
        "geometry_mask": geometry_mask.astype(np.uint8),
        "magnet_polarity_map": magnet_polarity_map,
        "boundary_map": boundary_map,
        "pixel_block_label": pixel_block_label,
        "label_to_gene": label_to_gene,
        "label_to_replica": label_to_replica,
        "coordinate_system": np.asarray(coordinate_system),
    }
    payload.update(axes)
    np.savez_compressed(output_path, **payload)
    pixels_per_gene = np.bincount(pixel_gene_id[design_mask], minlength=120)
    summary = {
        "status": "ready",
        "lookup_path": str(output_path.resolve()),
        "coordinate_system": coordinate_system,
        "image_shape": [image_size, image_size],
        "coverage": "complete 0-90 degree FEM motor quadrant",
        "coordinate_bounds": bounds,
        "outer_radius_mm": outer_radius,
        "mesh_nodes": int(len(nodes)),
        "mesh_triangles": int(len(triangles)),
        "fem_block_labels": len(labels),
        "design_block_labels": 480,
        "mapped_gene_count": 120,
        "copies_per_gene": 4,
        "design_pixels": int(design_mask.sum()),
        "geometry_pixels": int(geometry_mask.sum()),
        "minimum_pixels_per_gene": int(pixels_per_gene.min()),
        "maximum_pixels_per_gene": int(pixels_per_gene.max()),
        "fixed_pixel_counts": {
            "air": int(np.sum(fixed_material_map == FIXED_AIR)),
            "iron": int(np.sum(fixed_material_map == FIXED_IRON)),
            "winding": int(np.sum(fixed_material_map == FIXED_WINDING)),
            "other": int(np.sum(fixed_material_map == FIXED_OTHER)),
        },
        "polar_layout_note": (
            "for polar90, columns are physical angle 0..90 degrees and rows are radius 0..outer radius; "
            "the complete motor is included and no design-band crop is used"
        ),
        "source": {
            "fem": str(fem_path.resolve()),
            "fem_sha256": file_sha256(fem_path),
            "mat": str(mat_path.resolve()),
            "mat_sha256": file_sha256(mat_path),
            "ans_mesh": str(ans_path.resolve()),
            "ans_mesh_sha256": file_sha256(ans_path),
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_supersampled_polar360_lookup(
    fem_path: Path,
    mat_path: Path,
    ans_path: Path,
    image_size: int,
    supersample: int,
    output_path: Path,
) -> dict[str, Any]:
    """Unfold the 90-degree FEM geometry into a full-circle polar semantic grid.

    The source model has rotational/anti-periodic quadrant boundaries. Material
    geometry is queried in the source quadrant at ``theta % 90``. Each output
    pixel retains a regular subpixel table so changing genes can be rendered as
    material area fractions instead of a lossy centre sample.
    """
    if image_size != 224 or supersample < 2:
        raise ValueError("polar360 uses a 224 grid and at least 2x2 supersampling")
    fem_text = fem_path.read_text(encoding="utf-8", errors="strict")
    ans_text = ans_path.read_text(encoding="utf-8", errors="replace")
    labels = parse_labels(fem_text)
    ans_labels = parse_labels(ans_text)
    label_xy = np.asarray([[row["x_mm"], row["y_mm"]] for row in labels])
    ans_xy = np.asarray([[row["x_mm"], row["y_mm"]] for row in ans_labels])
    if len(labels) != len(ans_labels) or not np.allclose(label_xy, ans_xy, atol=1e-10, rtol=0):
        raise AssertionError("Solved mesh does not preserve the reference FEM label ordering")

    positions = np.asarray(loadmat(mat_path, variable_names=["MaterialPosition"], squeeze_me=True)["MaterialPosition"])
    mapping = match_material_positions(fem_text, positions)
    label_to_gene = np.full(len(labels), -1, dtype=np.int16)
    label_to_replica = np.full(len(labels), -1, dtype=np.int8)
    polarity_by_label = np.zeros(len(labels), dtype=np.int8)
    design_labels: list[int] = []
    for row in mapping:
        label_index = int(row["label_index_1based"]) - 1
        label_to_gene[label_index] = int(row["gene_index_1based"]) - 1
        label_to_replica[label_index] = int(row["copy_index_1based"]) - 1
        radial_angle = math.degrees(math.atan2(float(row["y_mm"]), float(row["x_mm"])))
        difference = math.radians(float(row["magnetization_deg"]) - radial_angle)
        polarity_by_label[label_index] = 1 if math.cos(difference) >= 0 else -1
        design_labels.append(label_index)
    if len(set(design_labels)) != 480:
        raise AssertionError("Expected 480 unique design labels")

    blocks = material_blocks(fem_text)
    fixed_class_by_label = np.asarray([_fixed_class_for_label(label, blocks) for label in labels], dtype=np.uint8)
    nodes, triangles, element_label = parse_ans_mesh(ans_path)
    finder = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles).get_trifinder()
    outer_radius = float(np.hypot(nodes[:, 0], nodes[:, 1]).max())
    sample_count = supersample * supersample
    shape = (image_size, image_size, sample_count)
    subpixel_gene_id = np.full(shape, -1, dtype=np.int16)
    subpixel_replica_id = np.full(shape, -1, dtype=np.int8)
    subpixel_fixed_material_map = np.zeros(shape, dtype=np.uint8)
    subpixel_geometry_mask = np.zeros(shape, dtype=np.uint8)
    subpixel_magnet_polarity_map = np.zeros(shape, dtype=np.int8)
    pixel_block_label = np.full(shape, -1, dtype=np.int16)

    rows = np.arange(image_size, dtype=np.float64)[:, None]
    columns = np.arange(image_size, dtype=np.float64)[None, :]
    sample_index = 0
    for radial_subpixel in range(supersample):
        radius = (rows + (radial_subpixel + 0.5) / supersample) * outer_radius / image_size
        for angular_subpixel in range(supersample):
            theta = (columns + (angular_subpixel + 0.5) / supersample) * 360.0 / image_size
            local_theta = np.deg2rad(np.mod(theta, 90.0))
            x_grid = radius * np.cos(local_theta)
            y_grid = radius * np.sin(local_theta)
            triangle_image = finder(x_grid, y_grid).astype(np.int32)
            valid = triangle_image >= 0
            labels_at_sample = np.full((image_size, image_size), -1, dtype=np.int16)
            labels_at_sample[valid] = element_label[triangle_image[valid]]
            genes = np.full((image_size, image_size), -1, dtype=np.int16)
            replicas = np.full((image_size, image_size), -1, dtype=np.int8)
            genes[valid] = label_to_gene[labels_at_sample[valid]]
            replicas[valid] = label_to_replica[labels_at_sample[valid]]
            design = genes >= 0
            fixed = valid & ~design
            fixed_map = np.zeros((image_size, image_size), dtype=np.uint8)
            fixed_map[fixed] = fixed_class_by_label[labels_at_sample[fixed]]
            polarity = np.zeros((image_size, image_size), dtype=np.int8)
            polarity[design] = polarity_by_label[labels_at_sample[design]]
            subpixel_gene_id[..., sample_index] = genes
            subpixel_replica_id[..., sample_index] = replicas
            subpixel_fixed_material_map[..., sample_index] = fixed_map
            subpixel_geometry_mask[..., sample_index] = valid
            subpixel_magnet_polarity_map[..., sample_index] = polarity
            pixel_block_label[..., sample_index] = labels_at_sample
            sample_index += 1

    represented = np.unique(subpixel_gene_id[subpixel_gene_id >= 0])
    if not np.array_equal(represented, np.arange(120)):
        missing = np.setdiff1d(np.arange(120), represented).tolist()
        raise AssertionError(f"Supersampled polar360 raster lost gene regions: {missing}")
    theta_axis = (np.arange(image_size, dtype=np.float64) + 0.5) * 360.0 / image_size
    radius_axis = (np.arange(image_size, dtype=np.float64) + 0.5) * outer_radius / image_size
    quadrant_id = np.floor(theta_axis / 90.0).astype(np.int8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        subpixel_gene_id=subpixel_gene_id,
        subpixel_replica_id=subpixel_replica_id,
        subpixel_fixed_material_map=subpixel_fixed_material_map,
        subpixel_geometry_mask=subpixel_geometry_mask,
        subpixel_magnet_polarity_map=subpixel_magnet_polarity_map,
        subpixel_block_label=pixel_block_label,
        theta_axis_deg=theta_axis,
        radius_axis_mm=radius_axis,
        quadrant_id=quadrant_id,
        label_to_gene=label_to_gene,
        label_to_replica=label_to_replica,
        coordinate_system=np.asarray("polar360"),
        supersample=np.int16(supersample),
    )
    design_samples = subpixel_gene_id >= 0
    samples_per_gene = np.bincount(subpixel_gene_id[design_samples], minlength=120)
    summary = {
        "status": "ready",
        "lookup_path": str(output_path.resolve()),
        "coordinate_system": "polar360",
        "image_shape": [image_size, image_size],
        "coverage": "complete 0-360 degree motor reconstructed from the audited 0-90 degree FEM quadrant",
        "coordinate_bounds": {"theta_deg": [0.0, 360.0], "radius_mm": [0.0, outer_radius]},
        "angular_pixel_width_deg": 360.0 / image_size,
        "supersampling": [supersample, supersample],
        "subpixel_angular_width_deg": 360.0 / image_size / supersample,
        "source_quadrant_mapping": "theta_query = theta_full mod 90 degrees",
        "symmetry_note": "material geometry is rotationally repeated; the FEM source uses anti-periodic boundary pairs",
        "fractional_channels": True,
        "mapped_gene_count": 120,
        "physical_gene_copies_in_full_circle": 16,
        "minimum_subpixel_samples_per_gene": int(samples_per_gene.min()),
        "maximum_subpixel_samples_per_gene": int(samples_per_gene.max()),
        "raw_centre_sampling_warning": "224 angular pixels are coarser than one angular gene cell; 4x4 area sampling is used to reduce aliasing",
        "source": {
            "fem": str(fem_path.resolve()),
            "fem_sha256": file_sha256(fem_path),
            "mat": str(mat_path.resolve()),
            "mat_sha256": file_sha256(mat_path),
            "ans_mesh": str(ans_path.resolve()),
            "ans_mesh_sha256": file_sha256(ans_path),
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_supersampled_xy360_lookup(
    fem_path: Path,
    mat_path: Path,
    ans_path: Path,
    image_size: int,
    supersample: int,
    output_path: Path,
) -> dict[str, Any]:
    """Rotate the FEM quadrant into a full circular x-y semantic image."""
    if image_size != 224 or supersample < 2:
        raise ValueError("xy360 uses a 224 grid and supersample >= 2")
    fem_text = fem_path.read_text(encoding="utf-8", errors="strict")
    ans_text = ans_path.read_text(encoding="utf-8", errors="replace")
    labels = parse_labels(fem_text)
    ans_labels = parse_labels(ans_text)
    label_xy = np.asarray([[row["x_mm"], row["y_mm"]] for row in labels])
    ans_xy = np.asarray([[row["x_mm"], row["y_mm"]] for row in ans_labels])
    if len(labels) != len(ans_labels) or not np.allclose(label_xy, ans_xy, atol=1e-10, rtol=0):
        raise AssertionError("Solved mesh does not preserve the reference FEM label ordering")

    positions = np.asarray(loadmat(mat_path, variable_names=["MaterialPosition"], squeeze_me=True)["MaterialPosition"])
    mapping = match_material_positions(fem_text, positions)
    label_to_gene = np.full(len(labels), -1, dtype=np.int16)
    label_to_replica = np.full(len(labels), -1, dtype=np.int8)
    polarity_by_label = np.zeros(len(labels), dtype=np.int8)
    for row in mapping:
        label_index = int(row["label_index_1based"]) - 1
        label_to_gene[label_index] = int(row["gene_index_1based"]) - 1
        label_to_replica[label_index] = int(row["copy_index_1based"]) - 1
        radial_angle = math.degrees(math.atan2(float(row["y_mm"]), float(row["x_mm"])))
        difference = math.radians(float(row["magnetization_deg"]) - radial_angle)
        polarity_by_label[label_index] = 1 if math.cos(difference) >= 0 else -1
    if not np.all(np.bincount(label_to_gene[label_to_gene >= 0], minlength=120) == 4):
        raise AssertionError("Each of 120 genes must map to four source-quadrant regions")

    blocks = material_blocks(fem_text)
    fixed_class_by_label = np.asarray([_fixed_class_for_label(label, blocks) for label in labels], dtype=np.uint8)
    nodes, triangles, element_label = parse_ans_mesh(ans_path)
    finder = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles).get_trifinder()
    outer_radius = float(np.hypot(nodes[:, 0], nodes[:, 1]).max())
    sample_count = supersample * supersample
    shape = (image_size, image_size, sample_count)
    subpixel_gene_id = np.full(shape, -1, dtype=np.int16)
    subpixel_replica_id = np.full(shape, -1, dtype=np.int8)
    subpixel_fixed_material_map = np.zeros(shape, dtype=np.uint8)
    subpixel_geometry_mask = np.zeros(shape, dtype=np.uint8)
    subpixel_magnet_polarity_map = np.zeros(shape, dtype=np.int8)
    subpixel_block_label = np.full(shape, -1, dtype=np.int16)

    sample_index = 0
    for y_subpixel in range(supersample):
        y_axis = -outer_radius + (np.arange(image_size, dtype=np.float64) + (y_subpixel + 0.5) / supersample) * 2.0 * outer_radius / image_size
        for x_subpixel in range(supersample):
            x_axis = -outer_radius + (np.arange(image_size, dtype=np.float64) + (x_subpixel + 0.5) / supersample) * 2.0 * outer_radius / image_size
            x_grid, y_grid = np.meshgrid(x_axis, y_axis)
            radius = np.hypot(x_grid, y_grid)
            theta = np.mod(np.degrees(np.arctan2(y_grid, x_grid)), 360.0)
            local_theta = np.deg2rad(np.mod(theta, 90.0))
            query_x = radius * np.cos(local_theta)
            query_y = radius * np.sin(local_theta)
            triangle_image = finder(query_x, query_y).astype(np.int32)
            valid = triangle_image >= 0
            labels_at_sample = np.full((image_size, image_size), -1, dtype=np.int16)
            labels_at_sample[valid] = element_label[triangle_image[valid]]
            genes = np.full((image_size, image_size), -1, dtype=np.int16)
            replicas = np.full((image_size, image_size), -1, dtype=np.int8)
            genes[valid] = label_to_gene[labels_at_sample[valid]]
            replicas[valid] = label_to_replica[labels_at_sample[valid]]
            design = genes >= 0
            fixed = valid & ~design
            fixed_map = np.zeros((image_size, image_size), dtype=np.uint8)
            fixed_map[fixed] = fixed_class_by_label[labels_at_sample[fixed]]
            polarity = np.zeros((image_size, image_size), dtype=np.int8)
            polarity[design] = polarity_by_label[labels_at_sample[design]]
            subpixel_gene_id[..., sample_index] = genes
            subpixel_replica_id[..., sample_index] = replicas
            subpixel_fixed_material_map[..., sample_index] = fixed_map
            subpixel_geometry_mask[..., sample_index] = valid
            subpixel_magnet_polarity_map[..., sample_index] = polarity
            subpixel_block_label[..., sample_index] = labels_at_sample
            sample_index += 1

    represented = np.unique(subpixel_gene_id[subpixel_gene_id >= 0])
    if not np.array_equal(represented, np.arange(120)):
        missing = np.setdiff1d(np.arange(120), represented).tolist()
        raise AssertionError(f"Supersampled xy360 raster lost gene regions: {missing}")
    axis = -outer_radius + (np.arange(image_size, dtype=np.float64) + 0.5) * 2.0 * outer_radius / image_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        subpixel_gene_id=subpixel_gene_id,
        subpixel_replica_id=subpixel_replica_id,
        subpixel_fixed_material_map=subpixel_fixed_material_map,
        subpixel_geometry_mask=subpixel_geometry_mask,
        subpixel_magnet_polarity_map=subpixel_magnet_polarity_map,
        subpixel_block_label=subpixel_block_label,
        x_axis_mm=axis,
        y_axis_mm=axis,
        label_to_gene=label_to_gene,
        label_to_replica=label_to_replica,
        coordinate_system=np.asarray("xy360"),
        supersample=np.int16(supersample),
    )
    design_samples = subpixel_gene_id >= 0
    samples_per_gene = np.bincount(subpixel_gene_id[design_samples], minlength=120)
    summary = {
        "status": "ready",
        "lookup_path": str(output_path.resolve()),
        "coordinate_system": "xy360",
        "image_shape": [image_size, image_size],
        "coverage": "complete circular motor reconstructed by rotating the audited 0-90 degree FEM quadrant",
        "coordinate_bounds_mm": {"x": [-outer_radius, outer_radius], "y": [-outer_radius, outer_radius]},
        "pixel_width_mm": 2.0 * outer_radius / image_size,
        "supersampling": [supersample, supersample],
        "subpixel_width_mm": 2.0 * outer_radius / image_size / supersample,
        "source_quadrant_mapping": "theta_query = atan2(y,x) mod 90 degrees; radius is unchanged",
        "symmetry_note": "material geometry is rotationally repeated; the FEM source uses anti-periodic boundary pairs",
        "fractional_channels": True,
        "mapped_gene_count": 120,
        "physical_gene_copies_in_full_circle": 16,
        "minimum_subpixel_samples_per_gene": int(samples_per_gene.min()),
        "maximum_subpixel_samples_per_gene": int(samples_per_gene.max()),
        "source": {
            "fem": str(fem_path.resolve()),
            "fem_sha256": file_sha256(fem_path),
            "mat": str(mat_path.resolve()),
            "mat_sha256": file_sha256(mat_path),
            "ans_mesh": str(ans_path.resolve()),
            "ans_mesh_sha256": file_sha256(ans_path),
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
