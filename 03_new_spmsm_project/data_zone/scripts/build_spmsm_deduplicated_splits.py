"""Build a deduplicated, leakage-safe SPMSM dataset and Tavg-stratified splits.

The historical FEMM labels belong to ``population_all``. Exact repeated
120-bit corrected chromosomes are collapsed. Groups whose repeated Tavg or
DeltaT labels disagree are quarantined rather than averaged into supervision.
The clean catalog is split 80/10/10 using seven fixed Tavg performance bands,
while raw-to-corrected repair-related topologies remain in the same split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAT = ROOT / "data_zone" / "raw" / "workspace_200.mat"
DEFAULT_EXPORT = ROOT / "data_zone" / "exports" / "deduplicated_femm_results"
DEFAULT_DATASET = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
DEFAULT_SPLITS = ROOT / "cnn_zone" / "outputs" / "splits"
DEFAULT_REPORTS = ROOT / "reports" / "spmsm_topology_dataset"

SEED = 20260902
ATOL = 1e-6
SPLIT_NAMES = np.asarray(["train", "validation", "test"])
SPLIT_RATIOS = np.asarray([0.80, 0.10, 0.10], dtype=np.float64)
TAVG_EDGES = np.asarray([-np.inf, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, np.inf])
TAVG_LABELS = np.asarray(
    ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.25", "3.25-3.5", "3.5-3.75", ">=3.75"]
)


@dataclass
class Aggregate:
    bits: np.ndarray
    first: np.ndarray
    inverse: np.ndarray
    count: np.ndarray
    generation_first: np.ndarray
    generation_last: np.ndarray
    distinct_generation_count: np.ndarray
    tavg_min: np.ndarray
    tavg_median: np.ndarray
    tavg_max: np.ndarray
    delta_min: np.ndarray
    delta_median: np.ndarray
    delta_max: np.ndarray
    fitness_median: np.ndarray
    volume_pm_median: np.ndarray
    conflict: np.ndarray


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = int(self.parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def row_keys(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def row_hashes(bits: np.ndarray) -> np.ndarray:
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), axis=1, bitorder="big")
    return np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in packed], dtype="U64")


def group_stat(values: np.ndarray, inverse: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(inverse, kind="stable")
    sorted_group = inverse[order]
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]
    ends = np.r_[starts[1:], len(order)]
    minimum = np.minimum.reduceat(sorted_values, starts)
    maximum = np.maximum.reduceat(sorted_values, starts)
    median = np.fromiter(
        (np.median(sorted_values[start:end]) for start, end in zip(starts, ends)),
        dtype=np.float64,
        count=group_count,
    )
    return minimum, median, maximum


def aggregate(
    bits: np.ndarray,
    generation: np.ndarray,
    tavg: np.ndarray,
    delta: np.ndarray,
    fitness: np.ndarray,
    volume_pm: np.ndarray,
    atol: float,
) -> Aggregate:
    unique, first, inverse, count = np.unique(
        np.asarray(bits, dtype=np.uint8),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    group_count = len(unique)
    order = np.argsort(inverse, kind="stable")
    sorted_group = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]
    sorted_generation = generation[order]
    generation_first = np.minimum.reduceat(sorted_generation, starts)
    generation_last = np.maximum.reduceat(sorted_generation, starts)
    pairs = np.unique(np.column_stack((inverse, generation)), axis=0)
    distinct_generation_count = np.bincount(pairs[:, 0], minlength=group_count)

    tmin, tmed, tmax = group_stat(tavg, inverse, group_count)
    dmin, dmed, dmax = group_stat(delta, inverse, group_count)
    _, fmed, _ = group_stat(fitness, inverse, group_count)
    _, vmed, _ = group_stat(volume_pm, inverse, group_count)
    conflict = (
        ~np.isfinite(tmin)
        | ~np.isfinite(tmax)
        | ~np.isfinite(dmin)
        | ~np.isfinite(dmax)
        | ((tmax - tmin) > atol)
        | ((dmax - dmin) > atol)
    )
    return Aggregate(
        unique,
        first,
        inverse,
        count,
        generation_first,
        generation_last,
        distinct_generation_count,
        tmin,
        tmed,
        tmax,
        dmin,
        dmed,
        dmax,
        fmed,
        vmed,
        conflict,
    )


def repair_components(raw_bits: np.ndarray, corrected_bits: np.ndarray, target_bits: np.ndarray) -> np.ndarray:
    """Keep topologies connected by any historical raw->corrected repair edge together."""
    nodes, inverse = np.unique(np.vstack((raw_bits, corrected_bits)), axis=0, return_inverse=True)
    record_count = len(raw_bits)
    raw_node = inverse[:record_count]
    corrected_node = inverse[record_count:]
    union_find = UnionFind(len(nodes))
    changed = raw_node != corrected_node
    for left, right in zip(raw_node[changed], corrected_node[changed], strict=True):
        union_find.union(int(left), int(right))
    roots = np.asarray([union_find.find(index) for index in range(len(nodes))], dtype=np.int32)
    node_position = np.searchsorted(row_keys(nodes), row_keys(target_bits))
    target_roots = roots[node_position]
    _, compact = np.unique(target_roots, return_inverse=True)
    return compact.astype(np.int32)


def assign_tavg_bands(values: np.ndarray) -> np.ndarray:
    return np.searchsorted(TAVG_EDGES[1:-1], values, side="right").astype(np.int8)


def grouped_stratified_split(bands: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    members: list[list[int]] = [[] for _ in unique_groups]
    for sample_index, group_index in enumerate(group_inverse):
        members[int(group_index)].append(sample_index)

    band_count = len(TAVG_LABELS)
    totals = np.bincount(bands, minlength=band_count)
    target = SPLIT_RATIOS[:, None] * totals[None, :]
    target_total = SPLIT_RATIOS * len(bands)
    current = np.zeros_like(target)
    current_total = np.zeros(3, dtype=np.float64)
    assignment = np.full(len(bands), -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(3)
    rarity = 1.0 / np.maximum(totals, 1)
    order = rng.permutation(len(members)).tolist()
    order.sort(
        key=lambda group_index: (
            -max(rarity[bands[np.asarray(members[group_index], dtype=np.int64)]]),
            -len(members[group_index]),
        )
    )

    for group_index in order:
        indices = np.asarray(members[group_index], dtype=np.int64)
        contribution = np.bincount(bands[indices], minlength=band_count)
        scores = np.empty(3, dtype=np.float64)
        for split_index in range(3):
            band_need = (target[split_index] - current[split_index]) / np.maximum(target[split_index], 1.0)
            total_need = (target_total[split_index] - current_total[split_index]) / max(target_total[split_index], 1.0)
            scores[split_index] = float(contribution @ band_need + 0.15 * len(indices) * total_need)
        selected = sorted(
            range(3),
            key=lambda split_index: (
                -scores[split_index],
                int(np.where(tie_order == split_index)[0][0]),
            ),
        )[0]
        assignment[indices] = selected
        current[selected] += contribution
        current_total[selected] += len(indices)

    if np.any(assignment < 0):
        raise AssertionError("Some samples were not assigned to a split")
    return assignment


def vectors_to_grids(bits: np.ndarray) -> np.ndarray:
    """Convert angular-major [N,120] vectors to [N,6 radial,20 angular]."""
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != 120:
        raise ValueError(f"Expected [N,120] binary genes, got {values.shape}")
    return values.reshape(-1, 20, 6).swapaxes(1, 2)


def write_csv(path: Path, headers: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def split_band_counts(bands: np.ndarray, split: np.ndarray) -> np.ndarray:
    result = np.zeros((len(TAVG_LABELS), 3), dtype=np.int64)
    for band_index in range(len(TAVG_LABELS)):
        for split_index in range(3):
            result[band_index, split_index] = np.count_nonzero((bands == band_index) & (split == split_index))
    return result


def plot_distribution(path: Path, tavg: np.ndarray, bands: np.ndarray, split: np.ndarray) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    colors = ("#2563b8", "#e67e22", "#2ca02c")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    histogram_edges = np.linspace(float(tavg.min()), float(tavg.max()), 75)
    for split_index, (name, color) in enumerate(zip(SPLIT_NAMES, colors, strict=True)):
        axes[0].hist(
            tavg[split == split_index],
            bins=histogram_edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{name} ({np.count_nonzero(split == split_index):,})",
        )
    for edge in TAVG_EDGES[1:-1]:
        axes[0].axvline(edge, color="0.5", linestyle="--", linewidth=0.9)
    axes[0].set_title("SPMSM去重数据：训练/验证/测试平均转矩分布")
    axes[0].set_xlabel("平均转矩 Tavg (N·m)")
    axes[0].set_ylabel("概率密度")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    counts = split_band_counts(bands, split)
    positions = np.arange(len(TAVG_LABELS))
    width = 0.25
    for split_index, (name, color) in enumerate(zip(SPLIT_NAMES, colors, strict=True)):
        bars = axes[1].bar(positions + (split_index - 1) * width, counts[:, split_index], width, color=color, label=name)
        axes[1].bar_label(bars, fontsize=8, rotation=90, padding=2)
    axes[1].set_xticks(positions, TAVG_LABELS)
    axes[1].set_xlabel("Tavg分布带 (N·m)")
    axes[1].set_ylabel("唯一基因数量")
    axes[1].set_title("各平均转矩分布带中的80/10/10划分")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build(
    mat_path: Path,
    export_dir: Path,
    dataset_dir: Path,
    split_dir: Path,
    reports_dir: Path,
    seed: int,
    atol: float,
) -> dict:
    data = loadmat(
        mat_path,
        variable_names=[
            "population_all",
            "population_noChange_all",
            "Tavg_all",
            "DeltaT_all",
            "Fitvalue_all",
            "VolumePM_all",
        ],
        squeeze_me=True,
    )
    corrected_history = np.asarray(data["population_all"], dtype=np.uint8)
    raw_history = np.asarray(data["population_noChange_all"], dtype=np.uint8)
    if corrected_history.shape != (612, 120, 201) or raw_history.shape != corrected_history.shape:
        raise AssertionError(f"Unexpected population shape: {corrected_history.shape}")
    if np.any((corrected_history != 0) & (corrected_history != 1)):
        raise AssertionError("population_all is not binary")

    state_count = corrected_history.shape[2]
    population_size = corrected_history.shape[0]
    generation = np.repeat(np.arange(state_count, dtype=np.int16), population_size)
    individual = np.tile(np.arange(population_size, dtype=np.int16), state_count)
    corrected_bits = corrected_history.transpose(2, 0, 1).reshape(-1, 120)
    raw_bits = raw_history.transpose(2, 0, 1).reshape(-1, 120)
    tavg = np.asarray(data["Tavg_all"], dtype=np.float64).T.reshape(-1)
    delta = np.asarray(data["DeltaT_all"], dtype=np.float64).T.reshape(-1)
    fitness = np.asarray(data["Fitvalue_all"], dtype=np.float64).T.reshape(-1)
    volume_pm = np.asarray(data["VolumePM_all"], dtype=np.float64).T.reshape(-1)
    if not (len(corrected_bits) == len(tavg) == len(delta) == len(fitness) == len(volume_pm)):
        raise AssertionError("History arrays are not aligned")
    history_pm_mismatch = np.sum(corrected_bits, axis=1) != volume_pm
    if np.any(history_pm_mismatch & (generation != 0)):
        raise AssertionError("PM-count/VolumePM mismatch exists outside the initial state")

    aggregated = aggregate(corrected_bits, generation, tavg, delta, fitness, volume_pm, atol)
    hashes = row_hashes(aggregated.bits)
    pm_count = np.sum(aggregated.bits, axis=1)

    export_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        export_dir / "deduplicated_topology_results.npz",
        chromosome_bits=aggregated.bits,
        occurrence_count=aggregated.count,
        generation_first=aggregated.generation_first,
        generation_last=aggregated.generation_last,
        tavg_median=aggregated.tavg_median,
        tavg_min=aggregated.tavg_min,
        tavg_max=aggregated.tavg_max,
        delta_t_median=aggregated.delta_median,
        delta_t_min=aggregated.delta_min,
        delta_t_max=aggregated.delta_max,
        label_conflict=aggregated.conflict,
    )
    write_csv(
        export_dir / "deduplicated_topology_results.csv",
        [
            "deduplicated_index",
            "topology_hash",
            "gene_hex",
            "occurrence_count",
            "generation_first",
            "generation_last",
            "distinct_generation_count",
            "pm_cell_count",
            "tavg_median_nm",
            "tavg_min_nm",
            "tavg_max_nm",
            "delta_t_median",
            "delta_t_min",
            "delta_t_max",
            "label_conflict",
        ],
        (
            (
                index,
                hashes[index],
                np.packbits(aggregated.bits[index], bitorder="big").tobytes().hex(),
                int(aggregated.count[index]),
                int(aggregated.generation_first[index]),
                int(aggregated.generation_last[index]),
                int(aggregated.distinct_generation_count[index]),
                int(pm_count[index]),
                float(aggregated.tavg_median[index]),
                float(aggregated.tavg_min[index]),
                float(aggregated.tavg_max[index]),
                float(aggregated.delta_median[index]),
                float(aggregated.delta_min[index]),
                float(aggregated.delta_max[index]),
                bool(aggregated.conflict[index]),
            )
            for index in range(len(aggregated.bits))
        ),
    )

    clean_source = np.flatnonzero(~aggregated.conflict)
    clean_bits = aggregated.bits[clean_source]
    clean_tavg = aggregated.tavg_median[clean_source]
    clean_delta = aggregated.delta_median[clean_source]
    clean_fitness = aggregated.fitness_median[clean_source]
    clean_volume = aggregated.volume_pm_median[clean_source]
    clean_hashes = hashes[clean_source]
    clean_groups = repair_components(raw_bits, corrected_bits, clean_bits)
    clean_bands = assign_tavg_bands(clean_tavg)
    split = grouped_stratified_split(clean_bands, clean_groups, seed)
    grids = vectors_to_grids(clean_bits)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "topology_bits.npy", grids)
    np.save(dataset_dir / "targets_tavg_delta.npy", np.column_stack((clean_tavg, clean_delta)))
    np.save(dataset_dir / "split_codes.npy", split)
    train = np.flatnonzero(split == 0)
    validation = np.flatnonzero(split == 1)
    test = np.flatnonzero(split == 2)
    split_path = split_dir / "scheme_a_tavg_bands_80_10_10.npz"
    np.savez_compressed(
        split_path,
        split_code=split,
        train=train,
        validation=validation,
        test=test,
        tavg_edges=TAVG_EDGES,
        tavg_labels=TAVG_LABELS,
        seed=seed,
    )
    np.savez_compressed(
        dataset_dir / "split_indices.npz",
        train=train,
        validation=validation,
        test=test,
    )

    representative = aggregated.first[clean_source]
    clean_pm_mismatch = np.sum(clean_bits, axis=1) != clean_volume
    write_csv(
        dataset_dir / "split_manifest.csv",
        [
            "sample_index",
            "topology_hash",
            "repair_component_id",
            "split",
            "tavg_band",
            "representative_generation",
            "representative_individual",
            "generation_first",
            "generation_last",
            "occurrence_count",
            "pm_cell_count",
            "tavg_nm",
            "delta_t",
            "fitness_median",
            "volume_pm_median",
        ],
        (
            (
                index,
                clean_hashes[index],
                int(clean_groups[index]),
                SPLIT_NAMES[split[index]],
                TAVG_LABELS[clean_bands[index]],
                int(generation[representative[index]]),
                int(individual[representative[index]]),
                int(aggregated.generation_first[source]),
                int(aggregated.generation_last[source]),
                int(aggregated.count[source]),
                int(np.sum(clean_bits[index])),
                float(clean_tavg[index]),
                float(clean_delta[index]),
                float(clean_fitness[index]),
                float(clean_volume[index]),
            )
            for index, source in enumerate(clean_source)
        ),
    )

    counts = split_band_counts(clean_bands, split)
    write_csv(
        reports_dir / "tavg_band_split_distribution.csv",
        [
            "tavg_band",
            "lower_edge_nm",
            "upper_edge_nm",
            "total",
            "train",
            "validation",
            "test",
            "train_percent",
            "validation_percent",
            "test_percent",
        ],
        (
            (
                TAVG_LABELS[band],
                TAVG_EDGES[band],
                TAVG_EDGES[band + 1],
                int(counts[band].sum()),
                int(counts[band, 0]),
                int(counts[band, 1]),
                int(counts[band, 2]),
                float(100 * counts[band, 0] / counts[band].sum()),
                float(100 * counts[band, 1] / counts[band].sum()),
                float(100 * counts[band, 2] / counts[band].sum()),
            )
            for band in range(len(TAVG_LABELS))
        ),
    )
    plot_distribution(reports_dir / "tavg_band_split_distribution.png", clean_tavg, clean_bands, split)

    component_crossing = sum(
        len(np.unique(split[clean_groups == component])) > 1 for component in np.unique(clean_groups)
    )
    summary = {
        "status": "deduplicated_and_split_ready_no_training_started",
        "source_mat": str(mat_path.resolve()),
        "deduplication": {
            "key": "exact corrected 120-bit chromosome from population_all",
            "historical_records": int(len(corrected_bits)),
            "unique_chromosomes": int(len(aggregated.bits)),
            "removed_duplicate_records": int(len(corrected_bits) - len(aggregated.bits)),
            "duplicate_groups": int(np.count_nonzero(aggregated.count > 1)),
            "maximum_occurrence_count": int(aggregated.count.max()),
            "quarantined_label_conflicts": int(np.count_nonzero(aggregated.conflict)),
            "clean_unique_chromosomes": int(len(clean_bits)),
            "label_conflict_atol": atol,
            "repeated_label_policy": "median only when repeated labels agree within tolerance; otherwise quarantine",
        },
        "gene": {
            "bits": 120,
            "grid": "6 radial x 20 angular",
            "vector_order": "gene_index = angular_index * 6 + radial_index",
            "semantics": {"0": "Air", "1": "N38 permanent magnet"},
        },
        "tavg_bands": [
            {
                "label": str(TAVG_LABELS[index]),
                "lower_edge_nm": finite_or_none(TAVG_EDGES[index]),
                "upper_edge_nm": finite_or_none(TAVG_EDGES[index + 1]),
                "total": int(counts[index].sum()),
                "train": int(counts[index, 0]),
                "validation": int(counts[index, 1]),
                "test": int(counts[index, 2]),
            }
            for index in range(len(TAVG_LABELS))
        ],
        "split": {
            "method": "repair-component grouped stratification using Tavg bands only",
            "ratios_requested": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "seed": seed,
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "train_ratio": float(len(train) / len(clean_bits)),
            "validation_ratio": float(len(validation) / len(clean_bits)),
            "test_ratio": float(len(test) / len(clean_bits)),
            "topology_hash_overlap": 0,
            "repair_components_crossing_splits": int(component_crossing),
        },
        "targets": {
            "columns": ["Tavg", "DeltaT"],
            "split_stratification_uses": ["Tavg"],
            "delta_t_note": "retained as a label but excluded from split stratification until its historical definition is fully reproduced",
            "tavg_min_nm": float(clean_tavg.min()),
            "tavg_median_nm": float(np.median(clean_tavg)),
            "tavg_max_nm": float(clean_tavg.max()),
            "delta_t_min": float(clean_delta.min()),
            "delta_t_median": float(np.median(clean_delta)),
            "delta_t_max": float(clean_delta.max()),
        },
        "integrity": {
            "train_validation_overlap": int(len(np.intersect1d(train, validation))),
            "train_test_overlap": int(len(np.intersect1d(train, test))),
            "validation_test_overlap": int(len(np.intersect1d(validation, test))),
            "all_clean_hashes_unique": bool(len(np.unique(clean_hashes)) == len(clean_hashes)),
            "pm_count_matches_volume_pm": int(np.count_nonzero(~clean_pm_mismatch)),
            "pm_count_volume_pm_mismatches": int(np.count_nonzero(clean_pm_mismatch)),
            "history_pm_count_volume_pm_mismatches": int(np.count_nonzero(history_pm_mismatch)),
            "history_pm_count_mismatches_all_initial_state": bool(
                np.all(generation[history_pm_mismatch] == 0)
            ),
            "pm_count_mismatch_policy": (
                "retained as the same initial-state source audit exception used in project 02; "
                "Tavg/DeltaT are aligned to population_all and the mismatch does not occur after state 0"
            ),
            "clean_sample_count": int(len(clean_bits)),
        },
        "artifacts": {
            "deduplicated_npz": str((export_dir / "deduplicated_topology_results.npz").resolve()),
            "deduplicated_csv": str((export_dir / "deduplicated_topology_results.csv").resolve()),
            "dataset": str(dataset_dir.resolve()),
            "split_npz": str(split_path.resolve()),
            "band_csv": str((reports_dir / "tavg_band_split_distribution.csv").resolve()),
            "band_figure": str((reports_dir / "tavg_band_split_distribution.png").resolve()),
        },
        "index_hashes_sha256": {
            "train": index_hash(train),
            "validation": index_hash(validation),
            "test": index_hash(test),
        },
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "deduplication_summary.json").write_text(
        json.dumps(summary["deduplication"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# SPMSM去重与平均转矩分带划分",
        "",
        "本阶段未启动训练。历史MAT保持只读。",
        "",
        f"- 历史记录：{len(corrected_bits):,}",
        f"- 精确基因去重后：{len(aggregated.bits):,}",
        f"- 移除重复记录：{len(corrected_bits) - len(aggregated.bits):,}",
        f"- 标签冲突隔离：{np.count_nonzero(aggregated.conflict):,}",
        f"- 最终可监督学习：{len(clean_bits):,}",
        f"- 训练/验证/测试：{len(train):,}/{len(validation):,}/{len(test):,}",
        "- 划分依据：仅使用Tavg固定物理分布带；DeltaT保留为标签但不参与当前分层。",
        "- 防泄漏：相同基因先去重，raw→corrected修复关联组不得跨集合。",
        "",
        "| Tavg区间 (N·m) | 总数 | 训练 | 验证 | 测试 |",
        "|---|---:|---:|---:|---:|",
    ]
    for band, label in enumerate(TAVG_LABELS):
        report_lines.append(
            f"| {label} | {counts[band].sum():,} | {counts[band,0]:,} | {counts[band,1]:,} | {counts[band,2]:,} |"
        )
    report_lines.extend(
        [
            "",
            "## 训练入口",
            "",
            f"- 数据：`{dataset_dir}`",
            f"- 固定索引：`{split_path}`",
            "- `topology_bits.npy`形状为`[N,6,20]`。",
            "- `targets_tavg_delta.npy`两列依次是`Tavg`和历史`DeltaT`。",
        ]
    )
    (reports_dir / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--label-atol", type=float, default=ATOL)
    args = parser.parse_args()
    summary = build(
        args.mat,
        args.export_dir,
        args.dataset_dir,
        args.split_dir,
        args.reports_dir,
        args.seed,
        args.label_atol,
    )
    print(json.dumps({"deduplication": summary["deduplication"], "split": summary["split"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
