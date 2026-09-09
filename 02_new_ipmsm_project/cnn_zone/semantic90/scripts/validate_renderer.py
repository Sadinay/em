"""Create audit figures and three FEM-vs-memory topology comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from scipy.io import loadmat


PROJECT = Path(__file__).resolve().parents[3]
SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "femm_zone" / "scripts"))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.topology_renderer import FEMSemanticRenderer  # noqa: E402
from femm_zone.scripts.analyze_ipmsm_structure import match_design_cells, parse_fem  # noqa: E402
from femm_zone.scripts.build_femm_topology import (  # noqa: E402
    CODE_TO_CLASS,
    build_topology,
    expected_pm_magnetization_deg,
)


DEFAULT_LOOKUP = SEMANTIC_ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
DEFAULT_DATASET = PROJECT / "data_zone" / "processed" / "ipmsm_topology_dataset" / "training_corrected_physical_three_state"
DEFAULT_MAT = PROJECT / "data_zone" / "raw" / "workspace_600.mat"
DEFAULT_FEM = PROJECT / "femm_zone" / "models" / "IPMSM.fem"
DEFAULT_OUTPUT = SEMANTIC_ROOT / "outputs" / "audit"


def _semantic_class_image(rendered: np.ndarray) -> np.ndarray:
    result = np.zeros(rendered.shape[1:], dtype=np.uint8)
    # class 0 background; channel indices map to stable display classes 1..7.
    for channel in range(7, -1, -1):
        result[rendered[channel] > 0.5] = channel + 1
    return result


def _save_semantic_figure(path: Path, rendered: np.ndarray, title: str) -> None:
    class_image = _semantic_class_image(rendered)
    colors = ["#ffffff", "#a7d8f0", "#d62728", "#b51f8c", "#59636e", "#e5f3ff", "#858585", "#d98c36", "#f7f7f7"]
    fig, ax = plt.subplots(figsize=(6.3, 6.0), constrained_layout=True)
    ax.imshow(class_image, origin="lower", cmap=ListedColormap(colors), vmin=-0.5, vmax=8.5, interpolation="nearest")
    ax.set_title(title); ax.set_xlabel("x pixel / physical x"); ax.set_ylabel("y pixel / physical y")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _select_audit_indices(targets: np.ndarray) -> list[int]:
    tavg, delta = targets[:, 0], targets[:, 1]
    selected: list[int] = []
    for values in (tavg, delta):
        order = np.argsort(values)
        selected.extend(order[:2].tolist())
        selected.extend(order[-2:].tolist())
    selected.extend(np.argsort(np.abs(tavg - np.median(tavg)))[:2].tolist())
    selected.extend(np.argsort(np.abs(delta - np.median(delta)))[:2].tolist())
    return list(dict.fromkeys(selected))[:12]


def validate(lookup: Path, cleaned_dir: Path, mat_path: Path, fem_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    renderer = FEMSemanticRenderer(lookup)
    grids = np.load(cleaned_dir / "topology_codes.npy", mmap_mode="r")
    targets = np.load(cleaned_dir / "targets_tavg_delta.npy", mmap_mode="r")
    codes = np.asarray(grids).swapaxes(1, 2).reshape(len(grids), 100)
    selected = _select_audit_indices(targets)
    audit_dir = output / "semantic_examples"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        batch = renderer(torch.from_numpy(codes[selected].astype(np.int64))).numpy()
        repeat = renderer(torch.from_numpy(codes[selected].astype(np.int64))).numpy()
    if not np.array_equal(batch, repeat):
        raise AssertionError("Repeated rendering is not pixel-deterministic")
    for number, (index, image) in enumerate(zip(selected, batch), start=1):
        _save_semantic_figure(
            audit_dir / f"semantic_{number:02d}.png", image,
            f"sample={index}, Tavg={targets[index,0]:.5f}, DeltaT={targets[index,1]:.5f}",
        )

    # Three actual historical corrected genes: low, median and high Tavg.
    rows: list[dict[str, str]] = []
    with (cleaned_dir / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    tavg = targets[:, 0]
    comparison_indices = [int(np.argmin(tavg)), int(np.argsort(tavg)[len(tavg) // 2]), int(np.argmax(tavg))]
    mat = loadmat(mat_path, variable_names=["population_all", "MaterialPosition"], squeeze_me=True)
    history = np.asarray(mat["population_all"], dtype=np.uint8)
    comparison_results = []
    for number, dataset_index in enumerate(comparison_indices, start=1):
        generation = int(metadata[dataset_index]["representative_generation"])
        individual = int(metadata[dataset_index]["representative_individual"])
        bits = history[individual, :, generation]
        material_codes = 2 * bits[0::2] + bits[1::2]
        comparison_dir = output / "fem_memory_comparison" / f"sample_{number:02d}"
        generated_model = build_topology(
            bits, fem_path, mat_path, comparison_dir,
            provenance={"kind": "semantic_renderer_validation", "generation": generation, "individual": individual},
        )
        properties, labels, _ = parse_fem(comparison_dir / "model.fem")
        cells = match_design_cells(np.asarray(mat["MaterialPosition"]), properties, labels)
        mismatch = 0
        pm_direction_error = 0.0
        for gene_index, cell in enumerate(cells):
            expected_class = ("air", "permanent_magnet", "iron")[int(CODE_TO_CLASS[material_codes[gene_index]])]
            for copy in cell["copies"]:
                if copy["physical_class"] != expected_class:
                    mismatch += 1
                if expected_class == "permanent_magnet":
                    expected_direction = expected_pm_magnetization_deg(copy["x_mm"], copy["y_mm"])
                    error = abs((copy["magnetization_deg"] - expected_direction + 180.0) % 360.0 - 180.0)
                    pm_direction_error = max(pm_direction_error, error)
        rendered = renderer(torch.from_numpy(material_codes.astype(np.int64)[None, :])).squeeze(0).numpy()
        _save_semantic_figure(
            comparison_dir / "memory_semantic.png", rendered,
            f"FEM-memory comparison gen={generation}, individual={individual}",
        )
        comparison_results.append(
            {
                "dataset_index": dataset_index,
                "generation": generation,
                "individual": individual,
                "fem_design_label_mismatches": mismatch,
                "maximum_pm_direction_error_deg": pm_direction_error,
                "generated_fem": str(generated_model.resolve()),
            }
        )
        if mismatch or pm_direction_error > 1e-8:
            raise AssertionError("Generated FEM and memory renderer disagree")

    # CPU render throughput; this measures renderer only, not CNN training.
    benchmark_codes = torch.from_numpy(codes[:1024].astype(np.int64))
    with torch.inference_mode():
        renderer(benchmark_codes[:8])
        started = time.perf_counter()
        renderer(benchmark_codes)
        elapsed = time.perf_counter() - started
    result = {
        "lookup": str(lookup.resolve()),
        "audit_images": len(selected),
        "repeat_render_pixel_exact": True,
        "comparison_samples": comparison_results,
        "all_three_fem_memory_comparisons_pass": True,
        "renderer_channels": list(renderer.channel_names),
        "renderer_only_cpu_benchmark": {
            "samples": 1024,
            "seconds": elapsed,
            "samples_per_second": 1024 / elapsed,
        },
        "no_training_performed": True,
        "no_bulk_images_written": True,
    }
    (output / "renderer_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--fem", type=Path, default=DEFAULT_FEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = validate(args.lookup, args.dataset, args.mat, args.fem, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
