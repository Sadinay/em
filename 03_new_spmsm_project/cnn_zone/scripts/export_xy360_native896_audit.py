"""Reconstruct the stored 4x4 FEM subpixels as a true 896x896 geometry audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
LOOKUP = ROOT / "cnn_zone" / "outputs" / "lookups" / "spmsm_xy360_224_full_motor_ss4.npz"
OUTPUT = ROOT / "reports" / "spmsm_inputs" / "xy360_fem_geometry_native896_audit.png"
COLORS = np.asarray(
    [
        [0.73, 0.87, 0.98],
        [0.70, 0.06, 0.22],
        [0.98, 0.22, 0.24],
        [0.82, 0.91, 0.98],
        [0.50, 0.50, 0.50],
        [0.93, 0.57, 0.13],
        [0.60, 0.35, 0.72],
    ],
    dtype=np.float32,
)


def main() -> None:
    bits_all = np.load(DATASET / "topology_bits.npy")
    targets = np.load(DATASET / "targets_tavg_delta.npy")
    sample = int(np.argsort(targets[:, 0])[len(targets) // 2])
    bits = bits_all[sample].reshape(-1)
    data = np.load(LOOKUP)
    genes = data["subpixel_gene_id"]
    fixed = data["subpixel_fixed_material_map"]
    polarity = data["subpixel_magnet_polarity_map"]
    geometry = data["subpixel_geometry_mask"].astype(bool)
    supersample = int(data["supersample"])
    high = np.ones((224 * supersample, 224 * supersample, 3), dtype=np.float32)
    for sy in range(supersample):
        for sx in range(supersample):
            index = sy * supersample + sx
            gene = genes[..., index]
            design = gene >= 0
            safe_gene = np.maximum(gene, 0)
            state = bits[safe_gene]
            class_id = np.full((224, 224), -1, dtype=np.int8)
            class_id[design & (state == 0)] = 0
            class_id[design & (state == 1) & (polarity[..., index] < 0)] = 1
            class_id[design & (state == 1) & (polarity[..., index] > 0)] = 2
            class_id[(fixed[..., index] == 1)] = 3
            class_id[(fixed[..., index] == 2)] = 4
            class_id[(fixed[..., index] == 3)] = 5
            class_id[(fixed[..., index] == 4)] = 6
            rgb = np.ones((224, 224, 3), dtype=np.float32)
            valid_class = class_id >= 0
            rgb[valid_class] = COLORS[class_id[valid_class]]
            rgb[~geometry[..., index]] = 1.0
            high[sy::supersample, sx::supersample] = rgb
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(OUTPUT, high, origin="lower")
    saved = plt.imread(OUTPUT)
    if saved.shape[:2] != (896, 896):
        raise AssertionError(f"Expected 896x896 audit, got {saved.shape}")
    print(f"{OUTPUT} | sample={sample} Tavg={targets[sample,0]:.6f} DeltaT={targets[sample,1]:.6f}")


if __name__ == "__main__":
    main()
