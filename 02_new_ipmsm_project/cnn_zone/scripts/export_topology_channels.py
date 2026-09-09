"""Export one verified 10x10 topology and its four one-hot channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data_zone" / "processed" / "current_generation.npz"
DEFAULT_OUTPUT = ROOT / "cnn_zone" / "outputs" / "topology_channel_example"


def export_sample(dataset_path: Path, output_dir: Path, sample_index: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset_path) as data:
        fitness = np.asarray(data["fitness"], dtype=np.float64)
        if sample_index is None:
            sample_index = int(np.argmax(fitness))
        if not 0 <= sample_index < len(fitness):
            raise IndexError(f"sample_index must be in [0, {len(fitness) - 1}]")

        grid = np.asarray(data["material_grid_corrected"][sample_index], dtype=np.uint8)
        vector = np.asarray(data["material_vector_corrected"][sample_index], dtype=np.uint8)
        bits = np.asarray(data["genome_bits_corrected"][sample_index], dtype=np.uint8)
        targets = {
            "t_avg_nm": float(data["t_avg_nm"][sample_index]),
            "delta_t_nm": float(data["delta_t_nm"][sample_index]),
            "fitness": float(data["fitness"][sample_index]),
            "volume_pm_cells": int(data["volume_pm_cells"][sample_index]),
        }

    # one_hot[k, r, a] is 1 exactly where material_grid[r, a] == k.
    one_hot = np.stack([(grid == code).astype(np.uint8) for code in range(4)], axis=0)
    if not np.all(one_hot.sum(axis=0) == 1):
        raise AssertionError("Each topology cell must belong to exactly one channel")
    decoded = 2 * bits[0::2] + bits[1::2]
    if not np.array_equal(decoded, vector):
        raise AssertionError("Corrected 200-bit chromosome does not decode to the material vector")
    if not np.array_equal(vector.reshape(10, 10).T, grid):
        raise AssertionError("Material vector and radial-by-angular grid are inconsistent")

    np.savez_compressed(
        output_dir / "topology_channels.npz",
        sample_index_0based=np.int32(sample_index),
        genome_bits_corrected=bits,
        material_vector_corrected=vector,
        material_grid=grid,
        one_hot_channels=one_hot,
        t_avg_nm=np.float64(targets["t_avg_nm"]),
        delta_t_nm=np.float64(targets["delta_t_nm"]),
        fitness=np.float64(targets["fitness"]),
    )

    payload = {
        "source_dataset": str(dataset_path.resolve()),
        "selection": "highest fitness in current generation" if sample_index == int(np.argmax(fitness)) else "explicit sample index",
        "sample_index_0based": sample_index,
        "axis_definition": {
            "material_grid": "[radial_index, angular_index]",
            "one_hot_channels": "[material_code, radial_index, angular_index]",
        },
        "channel_definition": {
            "channel_0": "1 where material code is 0",
            "channel_1": "1 where material code is 1",
            "channel_2": "1 where material code is 2",
            "channel_3": "1 where material code is 3",
        },
        "genome_bits_corrected": bits.tolist(),
        "material_vector_corrected": vector.tolist(),
        "material_grid_10x10": grid.tolist(),
        "one_hot_channels_4x10x10": one_hot.tolist(),
        "cell_counts_by_code": {str(code): int(one_hot[code].sum()) for code in range(4)},
        "targets": targets,
    }
    with (output_dir / "topology_channels.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)

    np.savetxt(output_dir / "material_grid_10x10.csv", grid, fmt="%d", delimiter=",")
    for code in range(4):
        np.savetxt(output_dir / f"channel_{code}_10x10.csv", one_hot[code], fmt="%d", delimiter=",")

    colors = ["#dceeff", "#e84d4d", "#727d89", "#f1b52c"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.4), constrained_layout=True)
    axes[0].imshow(grid, origin="lower", cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5)
    axes[0].set_title("Material code grid")
    for row in range(10):
        for col in range(10):
            code = int(grid[row, col])
            axes[0].text(col, row, str(code), ha="center", va="center", fontsize=8,
                         color="white" if code in (1, 2) else "black")

    channel_cmaps = [
        ListedColormap(["#ffffff", colors[code]]) for code in range(4)
    ]
    for code in range(4):
        axes[code + 1].imshow(one_hot[code], origin="lower", cmap=channel_cmaps[code], vmin=0, vmax=1)
        axes[code + 1].set_title(f"Channel {code}\n{int(one_hot[code].sum())} active cells")
        for row in range(10):
            for col in range(10):
                axes[code + 1].text(col, row, str(int(one_hot[code, row, col])),
                                    ha="center", va="center", fontsize=7,
                                    color="white" if one_hot[code, row, col] else "#999999")

    for axis in axes:
        axis.set_xlabel("Angular index")
        axis.set_ylabel("Radial index")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
    fig.suptitle(
        f"Sample {sample_index}: 10x10 categorical topology -> four one-hot channels",
        fontsize=15,
    )
    fig.savefig(output_dir / "topology_channel_separation.png", dpi=180)
    plt.close(fig)

    print(f"Sample index: {sample_index}")
    print(f"Grid shape: {grid.shape}; one-hot shape: {one_hot.shape}")
    print(f"Cell counts: {[int(one_hot[k].sum()) for k in range(4)]}")
    print(f"Targets: {targets}")
    print(f"Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-index", type=int, default=None)
    args = parser.parse_args()
    export_sample(args.dataset, args.output_dir, args.sample_index)


if __name__ == "__main__":
    main()
