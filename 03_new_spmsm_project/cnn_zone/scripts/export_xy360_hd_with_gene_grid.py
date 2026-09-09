"""Export a high-resolution full-motor FEM audit with gene-cell guide lines."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
CNN_ROOT = ROOT / "cnn_zone"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))
from femm_zone.scripts.spmsm_mapping import (  # noqa: E402
    match_material_positions,
    material_blocks,
    parse_labels,
)
from src.fem_mesh import _fixed_class_for_label, parse_ans_mesh  # noqa: E402


FEM = ROOT / "femm_zone" / "models" / "SPMSM_discrete.fem"
MAT = ROOT / "data_zone" / "raw" / "workspace_200.mat"
ANS = ROOT / "femm_zone" / "results" / "history_replay_validation_minangle25" / "final_high_tavg" / "angle_0" / "model.ans"
DATASET = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
OUTPUT = ROOT / "reports" / "spmsm_inputs" / "xy360_hd_3072_all_regions_and_gene_grid.png"
RESOLUTION = 3072
COLORS = np.asarray(
    [
        [186, 222, 250],  # design air
        [179, 15, 56],    # PM inward
        [250, 56, 61],    # PM outward
        [209, 232, 250],  # fixed air
        [128, 128, 128],  # fixed iron
        [237, 145, 33],   # winding
        [153, 89, 184],   # other
    ],
    dtype=np.uint8,
)


def interval_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.unique(np.sort(np.asarray(centres, dtype=np.float64)))
    middle = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(([centres[0] - (middle[0] - centres[0])], middle, [centres[-1] + (centres[-1] - middle[-1])]))


def main() -> None:
    fem_text = FEM.read_text(encoding="utf-8")
    labels = parse_labels(fem_text)
    blocks = material_blocks(fem_text)
    positions = loadmat(MAT, variable_names=["MaterialPosition"], squeeze_me=True)["MaterialPosition"]
    mapping = match_material_positions(fem_text, positions)
    label_to_gene = np.full(len(labels), -1, dtype=np.int16)
    polarity = np.zeros(len(labels), dtype=np.int8)
    for row in mapping:
        label_index = int(row["label_index_1based"]) - 1
        label_to_gene[label_index] = int(row["gene_index_1based"]) - 1
        radial_angle = math.degrees(math.atan2(float(row["y_mm"]), float(row["x_mm"])))
        polarity[label_index] = 1 if math.cos(math.radians(float(row["magnetization_deg"]) - radial_angle)) >= 0 else -1
    fixed_class = np.asarray([_fixed_class_for_label(label, blocks) for label in labels], dtype=np.uint8)

    bits_all = np.load(DATASET / "topology_bits.npy")
    targets = np.load(DATASET / "targets_tavg_delta.npy")
    sample = int(np.argsort(targets[:, 0])[len(targets) // 2])
    bits = bits_all[sample].reshape(-1)
    nodes, triangles, element_label = parse_ans_mesh(ANS)
    finder = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles).get_trifinder()
    outer_radius = float(np.hypot(nodes[:, 0], nodes[:, 1]).max())
    axis = -outer_radius + (np.arange(RESOLUTION, dtype=np.float64) + 0.5) * 2.0 * outer_radius / RESOLUTION
    rgb = np.full((RESOLUTION, RESOLUTION, 3), 255, dtype=np.uint8)
    region_map = np.full((RESOLUTION, RESOLUTION), -1, dtype=np.int32)

    chunk = 192
    x_grid_template = axis[None, :]
    for start in range(0, RESOLUTION, chunk):
        stop = min(start + chunk, RESOLUTION)
        y_grid = axis[start:stop, None]
        x_grid = np.broadcast_to(x_grid_template, (stop - start, RESOLUTION))
        y_grid_full = np.broadcast_to(y_grid, (stop - start, RESOLUTION))
        radius = np.hypot(x_grid, y_grid_full)
        theta = np.mod(np.degrees(np.arctan2(y_grid_full, x_grid)), 360.0)
        quadrant = np.floor(theta / 90.0).astype(np.int32)
        local_theta = np.deg2rad(np.mod(theta, 90.0))
        query_x = radius * np.cos(local_theta)
        query_y = radius * np.sin(local_theta)
        triangle = finder(query_x, query_y).astype(np.int32)
        valid = triangle >= 0
        label_image = np.full(triangle.shape, -1, dtype=np.int16)
        label_image[valid] = element_label[triangle[valid]]
        region_map[start:stop][valid] = label_image[valid] + quadrant[valid] * len(labels)
        gene = np.full(triangle.shape, -1, dtype=np.int16)
        gene[valid] = label_to_gene[label_image[valid]]
        design = gene >= 0
        safe_gene = np.maximum(gene, 0)
        state = bits[safe_gene]
        block = rgb[start:stop]
        block[design & (state == 0)] = COLORS[0]
        block[design & (state == 1) & (polarity[label_image] < 0)] = COLORS[1]
        block[design & (state == 1) & (polarity[label_image] > 0)] = COLORS[2]
        for fixed_value, color_index in ((1, 3), (2, 4), (3, 5), (4, 6)):
            mask = valid & ~design & (fixed_class[label_image] == fixed_value)
            block[mask] = COLORS[color_index]

    # Outline every real FEM block-label region, but not individual mesh triangles.
    region_boundary = np.zeros((RESOLUTION, RESOLUTION), dtype=bool)
    region_boundary[:, 1:] |= region_map[:, 1:] != region_map[:, :-1]
    region_boundary[1:, :] |= region_map[1:, :] != region_map[:-1, :]
    region_boundary[:-1, :] |= region_boundary[1:, :]
    region_boundary[:, :-1] |= region_boundary[:, 1:]
    rgb[region_boundary] = np.asarray([45, 45, 45], dtype=np.uint8)

    radial_edges = interval_edges(np.asarray([row["radius_mm"] for row in mapping]))
    copy_angle_edges = {}
    for copy_index in range(1, 5):
        centres = [row["polar_angle_deg"] for row in mapping if int(row["copy_index_1based"]) == copy_index]
        copy_angle_edges[copy_index] = interval_edges(np.asarray(centres))

    dpi = 256
    fig = plt.figure(figsize=(RESOLUTION / dpi, RESOLUTION / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin="lower", extent=(-outer_radius, outer_radius, -outer_radius, outer_radius), interpolation="nearest")
    guide_color = "#202020"
    for rotation in (0.0, 90.0, 180.0, 270.0):
        for angle_edges in copy_angle_edges.values():
            full_edges = angle_edges + rotation
            for angle in full_edges:
                radians = math.radians(float(angle))
                ax.plot(
                    [radial_edges[0] * math.cos(radians), radial_edges[-1] * math.cos(radians)],
                    [radial_edges[0] * math.sin(radians), radial_edges[-1] * math.sin(radians)],
                    color=guide_color,
                    linewidth=0.42,
                    alpha=0.95,
                    solid_capstyle="butt",
                )
            theta = np.deg2rad(np.linspace(full_edges[0], full_edges[-1], 240))
            for radius in radial_edges:
                ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=guide_color, linewidth=0.42, alpha=0.95)
    ax.set_xlim(-outer_radius, outer_radius)
    ax.set_ylim(-outer_radius, outer_radius)
    ax.set_aspect("equal")
    ax.axis("off")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=dpi, pad_inches=0)
    plt.close(fig)
    metadata = {
        "purpose": "human geometry audit only; not a model input",
        "resolution": [RESOLUTION, RESOLUTION],
        "source_model_input_resolution": [224, 224],
        "sample_index": sample,
        "tavg": float(targets[sample, 0]),
        "delta_t": float(targets[sample, 1]),
        "fixed_region_outlines": "all physical FEM block-label boundaries; triangle mesh intentionally omitted",
        "guide_lines": "six radial layers x twenty angular cells for every physical design copy",
        "pixel_width_mm": 2.0 * outer_radius / RESOLUTION,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
