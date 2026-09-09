"""Build an audited, leakage-safe CNN dataset from workspace_600.mat.

The historical FEMM targets belong to ``population_all`` (the topology after
the MATLAB structure repair), not to ``population_noChange_all``.  The latter
is retained for repair auditing and grouping, but raw-only topologies are never
given the corrected topology's FEMM label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAT = ROOT / "data_zone" / "raw" / "workspace_600.mat"
DEFAULT_OUTPUT = ROOT / "data_zone" / "processed" / "ipmsm_topology_dataset"
DEFAULT_REPORTS = ROOT / "reports" / "ipmsm_topology_dataset"
ATOL = 1e-6
SEED = 20260821
SPLIT_NAMES = np.asarray(["train", "validation", "test"])
SPLIT_RATIOS = np.asarray([0.70, 0.15, 0.15], dtype=np.float64)


def decode_bit_pairs(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] != 200:
        raise ValueError(f"Expected 200 bits, got {bits.shape}")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Chromosomes must contain only 0 and 1")
    return 2 * bits[..., 0::2] + bits[..., 1::2]


def encode_material_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] != 100:
        raise ValueError(f"Expected 100 material cells, got {codes.shape}")
    if np.any(codes > 3):
        raise ValueError("Material codes must be in {0,1,2,3}")
    bits = np.empty((*codes.shape[:-1], 200), dtype=np.uint8)
    bits[..., 0::2] = codes >> 1
    bits[..., 1::2] = codes & 1
    return bits


def vectors_to_grids(codes: np.ndarray) -> np.ndarray:
    """Convert angular-major vectors to grids with axes [radius, angle]."""
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] != 100:
        raise ValueError(f"Expected 100 material cells, got {codes.shape}")
    return codes.reshape(*codes.shape[:-1], 10, 10).swapaxes(-2, -1)


def grids_to_vectors(grids: np.ndarray) -> np.ndarray:
    grids = np.asarray(grids, dtype=np.uint8)
    if grids.shape[-2:] != (10, 10):
        raise ValueError(f"Expected [...,10,10], got {grids.shape}")
    return grids.swapaxes(-2, -1).reshape(*grids.shape[:-2], 100)


def collapse_physical_materials(codes: np.ndarray) -> np.ndarray:
    """Map genetic states to verified FEMM classes: 0 Air, 1 PM, 2/3 Iron."""
    result = np.asarray(codes, dtype=np.uint8).copy()
    result[result == 3] = 2
    return result


def topology_hash(codes: np.ndarray) -> str:
    vector = np.ascontiguousarray(codes, dtype=np.uint8).reshape(-1)
    if vector.size != 100:
        raise ValueError("A topology hash requires exactly 100 material cells")
    return hashlib.sha256(vector.tobytes()).hexdigest()


def _hash_rows(codes: np.ndarray) -> np.ndarray:
    return np.asarray([topology_hash(row) for row in codes], dtype="U64")


def _void_rows(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    return array.view(np.dtype((np.void, array.dtype.itemsize * array.shape[1]))).ravel()


@dataclass
class Aggregate:
    codes: np.ndarray
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
    fitness_min: np.ndarray
    fitness_median: np.ndarray
    fitness_max: np.ndarray
    conflict: np.ndarray


def _group_stats(values: np.ndarray, inverse: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def aggregate_topologies(
    codes: np.ndarray,
    generation: np.ndarray,
    tavg: np.ndarray,
    delta: np.ndarray,
    fitness: np.ndarray,
    atol: float = ATOL,
) -> Aggregate:
    unique, first, inverse, count = np.unique(
        np.asarray(codes, dtype=np.uint8),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    group_count = len(unique)
    order = np.argsort(inverse, kind="stable")
    sorted_group = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]
    gsorted = generation[order]
    generation_first = np.minimum.reduceat(gsorted, starts)
    generation_last = np.maximum.reduceat(gsorted, starts)
    pairs = np.unique(np.column_stack((inverse, generation)), axis=0)
    distinct_generation_count = np.bincount(pairs[:, 0], minlength=group_count)
    tmin, tmed, tmax = _group_stats(tavg, inverse, group_count)
    dmin, dmed, dmax = _group_stats(delta, inverse, group_count)
    fmin, fmed, fmax = _group_stats(fitness, inverse, group_count)
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
        fmin,
        fmed,
        fmax,
        conflict,
    )


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def repair_components(raw_physical: np.ndarray, corrected_physical: np.ndarray, target_codes: np.ndarray) -> np.ndarray:
    """Group corrected topologies connected by any historical raw->repair edge."""
    combined = np.vstack((raw_physical, corrected_physical))
    nodes, inverse = np.unique(combined, axis=0, return_inverse=True)
    record_count = len(raw_physical)
    raw_node, corrected_node = inverse[:record_count], inverse[record_count:]
    union_find = UnionFind(len(nodes))
    for left, right in zip(raw_node, corrected_node):
        union_find.union(int(left), int(right))
    roots = np.asarray([union_find.find(index) for index in range(len(nodes))], dtype=np.int32)
    node_keys = _void_rows(nodes)
    target_node = np.searchsorted(node_keys, _void_rows(target_codes))
    target_roots = roots[target_node]
    _, compact = np.unique(target_roots, return_inverse=True)
    return compact.astype(np.int32)


def _delta_quantile_bins(tavg_bin: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    result = np.zeros(len(delta), dtype=np.uint8)
    edges: dict[str, list[float]] = {}
    for band in range(7):
        mask = tavg_bin == band
        quartiles = np.quantile(delta[mask], [0.25, 0.50, 0.75])
        result[mask] = np.searchsorted(quartiles, delta[mask], side="right")
        edges[str(band)] = [float(value) for value in quartiles]
    return result, edges


def stratified_group_split(
    tavg: np.ndarray,
    delta: np.ndarray,
    group: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    tavg_edges = np.asarray([-np.inf, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf])
    tavg_bin = np.digitize(tavg, tavg_edges[1:-1], right=False).astype(np.uint8)
    delta_bin, quartiles = _delta_quantile_bins(tavg_bin, delta)
    stratum = (4 * tavg_bin + delta_bin).astype(np.int16)
    strata_count = 28
    target = SPLIT_RATIOS[:, None] * np.bincount(stratum, minlength=strata_count)[None, :]
    target_total = SPLIT_RATIOS * len(stratum)
    current = np.zeros((3, strata_count), dtype=np.float64)
    current_total = np.zeros(3, dtype=np.float64)
    split = np.full(len(stratum), -1, dtype=np.int8)
    unique_groups, group_inverse = np.unique(group, return_inverse=True)
    members = [[] for _ in range(len(unique_groups))]
    for sample_index, group_index in enumerate(group_inverse):
        members[int(group_index)].append(sample_index)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(members))
    rarity = 1.0 / np.maximum(np.bincount(stratum, minlength=strata_count), 1)
    order = sorted(
        order.tolist(),
        key=lambda idx: (
            -max(rarity[stratum[np.asarray(members[idx], dtype=np.int64)]]),
            -len(members[idx]),
        ),
    )
    split_tie_order = rng.permutation(3)
    for group_index in order:
        sample_indices = np.asarray(members[group_index], dtype=np.int64)
        group_counts = np.bincount(stratum[sample_indices], minlength=strata_count)
        scores = np.empty(3, dtype=np.float64)
        for split_index in range(3):
            stratum_need = (target[split_index] - current[split_index]) / np.maximum(target[split_index], 1.0)
            total_need = (target_total[split_index] - current_total[split_index]) / max(target_total[split_index], 1.0)
            scores[split_index] = float(group_counts @ stratum_need + 0.15 * len(sample_indices) * total_need)
        ranked = sorted(range(3), key=lambda s: (-scores[s], int(np.where(split_tie_order == s)[0][0])))
        selected = ranked[0]
        split[sample_indices] = selected
        current[selected] += group_counts
        current_total[selected] += len(sample_indices)
    if np.any(split < 0):
        raise AssertionError("Some samples were not assigned to a split")
    return split, tavg_bin, delta_bin, quartiles


def _write_csv(path: Path, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _write_aggregate(directory: Path, aggregate: Aggregate, source_type: str, label_semantics: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    grids = vectors_to_grids(aggregate.codes)
    np.save(directory / "topology_codes.npy", grids)
    hashes = _hash_rows(aggregate.codes)
    headers = [
        "sample_index", "topology_hash", "source_type", "label_semantics", "occurrence_count",
        "distinct_generation_count", "generation_first", "generation_last", "t_avg_nm_median",
        "t_avg_nm_min", "t_avg_nm_max", "delta_t_nm_median", "delta_t_nm_min", "delta_t_nm_max",
        "fitness_median", "fitness_min", "fitness_max", "is_label_conflict",
    ]
    rows = (
        (
            index, hashes[index], source_type, label_semantics, int(aggregate.count[index]),
            int(aggregate.distinct_generation_count[index]), int(aggregate.generation_first[index]),
            int(aggregate.generation_last[index]), float(aggregate.tavg_median[index]),
            float(aggregate.tavg_min[index]), float(aggregate.tavg_max[index]),
            float(aggregate.delta_median[index]), float(aggregate.delta_min[index]),
            float(aggregate.delta_max[index]), float(aggregate.fitness_median[index]),
            float(aggregate.fitness_min[index]), float(aggregate.fitness_max[index]),
            bool(aggregate.conflict[index]),
        )
        for index in range(len(aggregate.codes))
    )
    _write_csv(directory / "metadata.csv", headers, rows)
    conflict_indices = np.flatnonzero(aggregate.conflict)
    _write_csv(
        directory / "quarantine_label_conflicts.csv",
        ["sample_index", "topology_hash", "t_avg_range", "delta_t_range", "occurrence_count"],
        (
            (
                int(index), hashes[index], float(aggregate.tavg_max[index] - aggregate.tavg_min[index]),
                float(aggregate.delta_max[index] - aggregate.delta_min[index]), int(aggregate.count[index]),
            )
            for index in conflict_indices
        ),
    )


def _write_combined_audit(directory: Path, raw: Aggregate, corrected: Aggregate) -> dict[str, int]:
    directory.mkdir(parents=True, exist_ok=True)
    union_codes = np.unique(np.vstack((raw.codes, corrected.codes)), axis=0)
    np.save(directory / "topology_codes.npy", vectors_to_grids(union_codes))
    union_keys = _void_rows(union_codes)
    raw_pos = np.searchsorted(union_keys, _void_rows(raw.codes))
    corrected_pos = np.searchsorted(union_keys, _void_rows(corrected.codes))
    has_raw = np.zeros(len(union_codes), dtype=bool)
    has_corrected = np.zeros(len(union_codes), dtype=bool)
    corrected_conflict = np.zeros(len(union_codes), dtype=bool)
    has_raw[raw_pos] = True
    has_corrected[corrected_pos] = True
    corrected_conflict[corrected_pos] = corrected.conflict
    hashes = _hash_rows(union_codes)
    _write_csv(
        directory / "metadata.csv",
        ["sample_index", "topology_hash", "present_as_raw", "present_as_corrected", "corrected_label_conflict", "supervised_eligible"],
        (
            (
                index, hashes[index], bool(has_raw[index]), bool(has_corrected[index]),
                bool(corrected_conflict[index]), bool(has_corrected[index] and not corrected_conflict[index]),
            )
            for index in range(len(union_codes))
        ),
    )
    return {
        "unique_union": int(len(union_codes)),
        "intersection": int(np.sum(has_raw & has_corrected)),
        "raw_only": int(np.sum(has_raw & ~has_corrected)),
        "corrected_only": int(np.sum(~has_raw & has_corrected)),
        "supervised_eligible_from_corrected": int(np.sum(has_corrected & ~corrected_conflict)),
    }


def _validate_material_position(position: np.ndarray) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (100, 8):
        raise AssertionError(f"MaterialPosition must be 100x8, got {position.shape}")
    x, y = position[:, 6], position[:, 7]
    angle = np.degrees(np.arctan2(y, x)).reshape(10, 10)
    radius = np.hypot(x, y).reshape(10, 10)
    expected_angles = 1.125 + 2.25 * np.arange(10)
    if not np.allclose(angle, expected_angles[:, None], atol=1e-10):
        raise AssertionError("MaterialPosition does not support angular-major vector ordering")
    if not np.all(np.diff(radius, axis=1) > 0):
        raise AssertionError("Radius is not increasing within each ten-cell angular block")
    return {
        "copy_used_1based": 4,
        "angle_centres_deg": expected_angles.tolist(),
        "angle_step_deg": 2.25,
        "radius_centres_mm": radius[0].tolist(),
        "verified_vector_index_formula": "index = angular_index * 10 + radial_index",
        "exported_grid_axes": "[radial_index, angular_index]",
    }


def _plot_examples(directory: Path, grids: np.ndarray, tavg: np.ndarray, delta: np.ndarray, generation: np.ndarray, split: np.ndarray, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    candidates: list[int] = []
    for order in (np.argsort(tavg), np.argsort(delta)):
        candidates.extend(order[:3].tolist())
        candidates.extend(order[-3:].tolist())
    median_order = np.argsort(np.abs(tavg - np.median(tavg)))
    candidates.extend(median_order[:4].tolist())
    candidates.extend(rng.choice(len(grids), size=12, replace=False).tolist())
    chosen = list(dict.fromkeys(candidates))[:20]
    colors = ListedColormap(["#dceeff", "#e84d4d", "#727d89"])
    for number, index in enumerate(chosen, start=1):
        fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
        ax.imshow(grids[index], origin="lower", cmap=colors, vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True); ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.set_xlabel("Angular index"); ax.set_ylabel("Radial index")
        ax.set_title(
            f"sample={index}, gen={generation[index]}, split={SPLIT_NAMES[split[index]]}\n"
            f"Tavg={tavg[index]:.5f} N·m, DeltaT={delta[index]:.5f} N·m"
        )
        fig.savefig(directory / f"topology_{number:02d}.png", dpi=150)
        plt.close(fig)
    base = grids[chosen[0]]
    expanded = np.concatenate((base, np.flip(base, axis=1), base, np.flip(base, axis=1)), axis=1)
    fig, ax = plt.subplots(figsize=(12, 3.6), constrained_layout=True)
    ax.imshow(expanded, origin="lower", cmap=colors, vmin=-0.5, vmax=2.5, interpolation="nearest", aspect="auto")
    ax.set_xlabel("Angular index in diagnostic 90° expansion"); ax.set_ylabel("Radial index")
    ax.set_title("Diagnostic material-only mirror expansion: base | mirror | base | mirror")
    fig.savefig(directory / "diagnostic_90deg_material_expansion.png", dpi=170)
    plt.close(fig)


def _plot_dataset_distributions(
    directory: Path,
    grids: np.ndarray,
    tavg: np.ndarray,
    delta: np.ndarray,
    split: np.ndarray,
    component: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for values, name, xlabel, color in (
        (tavg, "tavg_distribution", "Historical Tavg (N·m)", "#3969ac"),
        (delta, "delta_t_distribution", "Historical DeltaT (N·m)", "#e73f74"),
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.7), constrained_layout=True)
        ax.hist(values, bins=100, color=color, alpha=0.84, edgecolor="white", linewidth=0.3)
        for quantile in (0.1, 0.5, 0.9):
            value = float(np.quantile(values, quantile))
            ax.axvline(value, linestyle="--", linewidth=1.5, label=f"P{int(quantile*100)}={value:.4f}")
        ax.set_xlabel(xlabel); ax.set_ylabel("Number of unique physical topologies")
        ax.set_title(f"Clean supervised dataset: {xlabel} distribution")
        ax.grid(axis="y", alpha=0.2); ax.legend()
        fig.savefig(directory / f"{name}.png", dpi=180)
        plt.close(fig)
    cell_counts = np.bincount(grids.reshape(-1), minlength=3)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    bars = ax.bar(["Air", "PM", "Iron"], cell_counts, color=["#9ecae1", "#e84d4d", "#727d89"])
    ax.bar_label(bars, labels=[f"{value:,}" for value in cell_counts])
    ax.set_ylabel("Cells across clean unique topologies")
    ax.set_title("Physical material-class distribution")
    fig.savefig(directory / "physical_material_class_distribution.png", dpi=180)
    plt.close(fig)
    _, component_sizes = np.unique(component, return_counts=True)
    size_values, size_frequencies = np.unique(component_sizes, return_counts=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    ax.bar(size_values.astype(str), size_frequencies, color="#11a579")
    ax.set_xlabel("Corrected physical topologies per repair-related group")
    ax.set_ylabel("Number of groups")
    ax.set_title("Raw→corrected repair-group size distribution")
    fig.savefig(directory / "repair_group_size_distribution.png", dpi=180)
    plt.close(fig)


def _write_reports(reports: Path, summary: dict[str, Any]) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    counts = summary["counts"]
    position = summary["material_position_validation"]
    split = summary["split"]
    (reports / "mat_audit.md").write_text(
        "# workspace_600.mat audit\n\n"
        f"- Historical records: {counts['historical_records']:,} = 601 states × 504 individuals.\n"
        "- Array order: MATLAB `[individual, bit, state]`; exported order is state-major.\n"
        "- `population_noChange_all` is the pre-repair gene; `population_all` is the corrected FEMM input.\n"
        "- Final `population == population_all[:,:,600]`; scalar final arrays equal history column 600.\n"
        "- `Material` decodes from pre-repair genes; `Material_change` decodes from corrected genes.\n"
        f"- VolumePM mismatch: {counts['initial_state_volume_pm_mismatch_records']} records, all confined to state 0; states 1..600 align exactly.\n"
        "- Historical MAT contains scalar Tavg/DeltaT, not angle-by-angle torque curves.\n",
        encoding="utf-8",
    )
    (reports / "conversion_audit.md").write_text(
        "# Conversion audit\n\n"
        "1. Every 200-bit chromosome decodes stably to 100 four-state cells.\n"
        "2. Adjacent pairs are MSB-first: `code = 2*bit[2i] + bit[2i+1]`.\n"
        "3. Exported 10×10 axes are row=radius and column=angle.\n"
        f"4. MaterialPosition verifies centres {position['angle_centres_deg']} degrees.\n"
        "5. Verified physical mapping is code 0=Air, code 1=N38 PM, codes 2/3=Pure Iron.\n"
        "6. The default CNN dataset collapses 2/3 and dynamically returns three one-hot channels.\n"
        "7. Four-state files remain available for traceability and an ablation experiment.\n"
        "8. Raw-only genes are not assigned corrected FEMM labels; doing so would create false supervision.\n"
        f"9. Default usable physical samples: {counts['training_physical_clean']:,}.\n"
        "10. No resize, interpolation, PNG storage, or FEMM rerun was performed.\n",
        encoding="utf-8",
    )
    (reports / "deduplication_report.md").write_text(
        "# Deduplication and label-conflict report\n\n"
        f"- Raw four-state unique: {counts['raw_unique_four_state']:,}; conflicts: {counts['raw_conflict_four_state']:,}.\n"
        f"- Corrected four-state unique: {counts['corrected_unique_four_state']:,}; conflicts: {counts['corrected_conflict_four_state']:,}.\n"
        f"- Raw/corrected exact union: {counts['combined_unique_four_state']:,}; intersection: {counts['combined_intersection_four_state']:,}.\n"
        f"- Corrected three-class physical unique: {counts['corrected_unique_physical']:,}; conflicts: {counts['corrected_conflict_physical']:,}.\n"
        f"- Clean supervised physical dataset: {counts['training_physical_clean']:,}.\n"
        "- Conflict tolerance: absolute Tavg and DeltaT range must each be <= 1e-6.\n"
        "- Fitvalue disagreement is reported but does not quarantine a physical label.\n"
        "- Repeated consistent labels are represented by their group median.\n",
        encoding="utf-8",
    )
    (reports / "split_report.md").write_text(
        "# Fixed split report\n\n"
        f"- Seed: {summary['random_seed']}.\n"
        f"- Train: {split['train']:,} ({split['train_ratio']:.4%}).\n"
        f"- Validation: {split['validation']:,} ({split['validation_ratio']:.4%}).\n"
        f"- Test: {split['test']:,} ({split['test_ratio']:.4%}).\n"
        "- Stratification: seven fixed Tavg bands, then DeltaT quartiles inside each band.\n"
        "- Grouping: connected components of historical raw→corrected physical-topology repair edges.\n"
        f"- Repair groups crossing splits: {split['repair_groups_crossing_splits']}.\n"
        f"- Topology hashes crossing splits: {split['topology_hashes_crossing_splits']}.\n"
        "- The manifest is fixed and must be reused by every model.\n",
        encoding="utf-8",
    )
    target = summary["target_audit"]
    material = summary["material_class_counts"]
    family = summary["repair_group_audit"]
    (reports / "data_audit.md").write_text(
        "# CNN data audit\n\n"
        f"- Samples after physical deduplication and conflict quarantine: {counts['training_physical_clean']:,}.\n"
        "- Input shape on disk: `[N,10,10]`; dynamic model input: `[B,3,10,10]`.\n"
        "- Input classes: Air, N38 permanent magnet, Pure Iron.\n"
        "- Missing or non-finite targets: 0.\n"
        "- Duplicate physical topologies in the training catalog: 0.\n"
        f"- Tavg min/median/mean/max: {target['tavg_min']:.6f} / {target['tavg_median']:.6f} / {target['tavg_mean']:.6f} / {target['tavg_max']:.6f}.\n"
        f"- DeltaT min/median/mean/max: {target['delta_min']:.6f} / {target['delta_median']:.6f} / {target['delta_mean']:.6f} / {target['delta_max']:.6f}.\n"
        f"- Material cells Air/PM/Iron: {material['Air']:,} / {material['PM']:,} / {material['Iron']:,}.\n"
        f"- Repair-related topology groups: {family['group_count']:,}; maximum group size: {family['maximum_group_size']}.\n"
        "- Performance stratification uses seven Tavg bands and DeltaT quartiles inside each band.\n"
        "- Figures include target histograms, class counts, repair-group sizes, 20 topology examples and a diagnostic mirror expansion.\n",
        encoding="utf-8",
    )


def build(mat_path: Path, output: Path, reports: Path, seed: int = SEED, atol: float = ATOL) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    names = [
        "population_all", "population_noChange_all", "population", "Material", "Material_change",
        "MaterialPosition", "Tavg_all", "DeltaT_all", "Fitvalue_all", "VolumePM_all",
        "Tavg", "DeltaT", "Fitvalue", "VolumePM", "Generation", "popsize",
    ]
    data = loadmat(mat_path, variable_names=names, squeeze_me=True, struct_as_record=False)
    corrected_bits_history = np.asarray(data["population_all"], dtype=np.uint8)
    raw_bits_history = np.asarray(data["population_noChange_all"], dtype=np.uint8)
    if corrected_bits_history.shape != (504, 200, 601) or raw_bits_history.shape != (504, 200, 601):
        raise AssertionError("Unexpected chromosome history shape")
    if not np.array_equal(np.asarray(data["population"], dtype=np.uint8), corrected_bits_history[:, :, -1]):
        raise AssertionError("Final population is not corrected history state 600")
    if not np.array_equal(decode_bit_pairs(raw_bits_history[:, :, -1]), np.asarray(data["Material"], dtype=np.uint8)):
        raise AssertionError("Material does not align with final raw chromosome")
    if not np.array_equal(decode_bit_pairs(corrected_bits_history[:, :, -1]), np.asarray(data["Material_change"], dtype=np.uint8)):
        raise AssertionError("Material_change does not align with final corrected chromosome")
    for current_name, history_name in (("Tavg", "Tavg_all"), ("DeltaT", "DeltaT_all"), ("Fitvalue", "Fitvalue_all"), ("VolumePM", "VolumePM_all")):
        if not np.array_equal(np.asarray(data[current_name]), np.asarray(data[history_name])[:, -1]):
            raise AssertionError(f"{current_name} does not align with final history column")

    state_count, population_size = 601, 504
    generation = np.repeat(np.arange(state_count, dtype=np.int32), population_size)
    individual = np.tile(np.arange(population_size, dtype=np.int16), state_count)
    raw_bits = raw_bits_history.transpose(2, 0, 1).reshape(-1, 200)
    corrected_bits = corrected_bits_history.transpose(2, 0, 1).reshape(-1, 200)
    raw_four = decode_bit_pairs(raw_bits)
    corrected_four = decode_bit_pairs(corrected_bits)
    raw_physical = collapse_physical_materials(raw_four)
    corrected_physical = collapse_physical_materials(corrected_four)
    tavg = np.asarray(data["Tavg_all"], dtype=np.float64).T.reshape(-1)
    delta = np.asarray(data["DeltaT_all"], dtype=np.float64).T.reshape(-1)
    fitness = np.asarray(data["Fitvalue_all"], dtype=np.float64).T.reshape(-1)
    volume_pm = np.asarray(data["VolumePM_all"], dtype=np.float64).T.reshape(-1)
    pm_count_difference = np.sum(corrected_four == 1, axis=1) - volume_pm
    pm_count_mismatch = np.flatnonzero(pm_count_difference)
    # The discrepancy is confined to 445 records in state 0.  Every saved GA
    # generation (1..600), including the final arrays, aligns exactly.  Keep
    # this as a source-data audit finding rather than discarding the records.
    if len(pm_count_mismatch) and np.any(generation[pm_count_mismatch] != 0):
        raise AssertionError("VolumePM/code-1 mismatch exists outside the initial state")

    raw_aggregate = aggregate_topologies(raw_four, generation, tavg, delta, fitness, atol)
    corrected_aggregate = aggregate_topologies(corrected_four, generation, tavg, delta, fitness, atol)
    physical_aggregate = aggregate_topologies(corrected_physical, generation, tavg, delta, fitness, atol)
    _write_aggregate(output / "raw_four_state", raw_aggregate, "pre_repair", "associated corrected-FEMM result; not direct supervision")
    _write_aggregate(output / "corrected_four_state", corrected_aggregate, "post_repair", "direct historical FEMM result")
    _write_aggregate(output / "corrected_physical_three_state_all", physical_aggregate, "post_repair_physical", "direct historical FEMM result")
    combined = _write_combined_audit(output / "combined_audit_four_state", raw_aggregate, corrected_aggregate)

    component_all = repair_components(raw_physical, corrected_physical, physical_aggregate.codes)
    clean_mask = ~physical_aggregate.conflict
    clean_source_indices = np.flatnonzero(clean_mask)
    clean_codes = physical_aggregate.codes[clean_mask]
    clean_grids = vectors_to_grids(clean_codes)
    clean_tavg = physical_aggregate.tavg_median[clean_mask]
    clean_delta = physical_aggregate.delta_median[clean_mask]
    clean_fitness = physical_aggregate.fitness_median[clean_mask]
    clean_component = component_all[clean_mask]
    split_code, tavg_bin, delta_bin, quartiles = stratified_group_split(clean_tavg, clean_delta, clean_component, seed)
    training = output / "training_corrected_physical_three_state"
    training.mkdir(parents=True, exist_ok=True)
    np.save(training / "topology_codes.npy", clean_grids)
    np.save(training / "targets_tavg_delta.npy", np.column_stack((clean_tavg, clean_delta)))
    np.save(training / "split_codes.npy", split_code)
    np.savez_compressed(
        training / "split_indices.npz",
        train=np.flatnonzero(split_code == 0),
        validation=np.flatnonzero(split_code == 1),
        test=np.flatnonzero(split_code == 2),
    )
    physical_hashes = _hash_rows(clean_codes)
    representative_record = physical_aggregate.first[clean_mask]
    representative_generation = generation[representative_record]
    representative_individual = individual[representative_record]
    repair_cell_count = np.count_nonzero(raw_four != corrected_four, axis=1)
    order = np.argsort(physical_aggregate.inverse, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(physical_aggregate.inverse[order])) + 1]
    repair_max_all = np.maximum.reduceat(repair_cell_count[order], starts)
    repair_changed_count_all = np.add.reduceat((repair_cell_count[order] > 0).astype(np.int32), starts)
    exact_to_physical = physical_aggregate.inverse[corrected_aggregate.first]
    four_state_variant_count_all = np.bincount(exact_to_physical, minlength=len(physical_aggregate.codes))
    metadata_headers = [
        "sample_index", "physical_topology_hash", "split", "repair_component_id", "tavg_band",
        "delta_t_quartile_within_tavg_band", "representative_generation", "representative_individual",
        "generation_first", "generation_last", "occurrence_count", "distinct_generation_count", "four_state_variant_count",
        "records_changed_by_repair", "maximum_repair_cell_count", "t_avg_nm", "delta_t_nm",
        "fitness_median", "t_avg_range", "delta_t_range",
    ]
    metadata_rows = (
        (
            index, physical_hashes[index], SPLIT_NAMES[split_code[index]], int(clean_component[index]),
            int(tavg_bin[index]) + 1, int(delta_bin[index]) + 1, int(representative_generation[index]),
            int(representative_individual[index]), int(physical_aggregate.generation_first[source]),
            int(physical_aggregate.generation_last[source]), int(physical_aggregate.count[source]),
            int(physical_aggregate.distinct_generation_count[source]), int(four_state_variant_count_all[source]),
            int(repair_changed_count_all[source]), int(repair_max_all[source]), float(clean_tavg[index]),
            float(clean_delta[index]), float(clean_fitness[index]),
            float(physical_aggregate.tavg_max[source] - physical_aggregate.tavg_min[source]),
            float(physical_aggregate.delta_max[source] - physical_aggregate.delta_min[source]),
        )
        for index, source in enumerate(clean_source_indices)
    )
    _write_csv(training / "metadata.csv", metadata_headers, metadata_rows)
    _write_csv(
        training / "split_manifest.csv",
        ["sample_index", "physical_topology_hash", "repair_component_id", "split", "tavg_band", "delta_t_quartile"],
        (
            (index, physical_hashes[index], int(clean_component[index]), SPLIT_NAMES[split_code[index]], int(tavg_bin[index]) + 1, int(delta_bin[index]) + 1)
            for index in range(len(clean_codes))
        ),
    )
    np.savez_compressed(
        output / "repair_audit.npz",
        generation=generation,
        individual=individual,
        repair_changed=(repair_cell_count > 0),
        repair_bit_count=np.count_nonzero(raw_bits != corrected_bits, axis=1).astype(np.uint8),
        repair_cell_count=repair_cell_count.astype(np.uint8),
    )

    position_validation = _validate_material_position(np.asarray(data["MaterialPosition"], dtype=np.float64))
    _plot_examples(
        reports / "figures", clean_grids, clean_tavg, clean_delta,
        representative_generation, split_code, seed,
    )
    _plot_dataset_distributions(reports / "figures", clean_grids, clean_tavg, clean_delta, split_code, clean_component)
    component_crossing = 0
    for component in np.unique(clean_component):
        if len(np.unique(split_code[clean_component == component])) > 1:
            component_crossing += 1
    split_counts = np.bincount(split_code, minlength=3)
    counts = {
        "historical_records": int(len(tavg)),
        "raw_unique_four_state": int(len(raw_aggregate.codes)),
        "raw_conflict_four_state": int(np.sum(raw_aggregate.conflict)),
        "raw_clean_four_state": int(np.sum(~raw_aggregate.conflict)),
        "corrected_unique_four_state": int(len(corrected_aggregate.codes)),
        "corrected_conflict_four_state": int(np.sum(corrected_aggregate.conflict)),
        "corrected_clean_four_state": int(np.sum(~corrected_aggregate.conflict)),
        "combined_unique_four_state": int(combined["unique_union"]),
        "combined_intersection_four_state": int(combined["intersection"]),
        "corrected_unique_physical": int(len(physical_aggregate.codes)),
        "corrected_conflict_physical": int(np.sum(physical_aggregate.conflict)),
        "training_physical_clean": int(len(clean_codes)),
        "history_records_changed_by_repair": int(np.sum(repair_cell_count > 0)),
        "initial_state_volume_pm_mismatch_records": int(len(pm_count_mismatch)),
    }
    summary = {
        "source_mat": str(mat_path.resolve()),
        "random_seed": seed,
        "label_conflict_atol": atol,
        "material_mapping": {"0": "Air", "1": "N38 permanent magnet", "2": "Pure Iron", "3": "Pure Iron"},
        "default_training_encoding": {"classes": 3, "mapping": {"0": "Air", "1": "PM", "2": "Iron"}},
        "counts": counts,
        "combined_audit": combined,
        "material_position_validation": position_validation,
        "delta_t_quartiles_by_tavg_band": quartiles,
        "split": {
            "train": int(split_counts[0]), "validation": int(split_counts[1]), "test": int(split_counts[2]),
            "train_ratio": float(split_counts[0] / len(split_code)),
            "validation_ratio": float(split_counts[1] / len(split_code)),
            "test_ratio": float(split_counts[2] / len(split_code)),
            "repair_groups_crossing_splits": int(component_crossing),
            "topology_hashes_crossing_splits": 0,
        },
        "target_audit": {
            "tavg_min": float(np.min(clean_tavg)), "tavg_median": float(np.median(clean_tavg)),
            "tavg_mean": float(np.mean(clean_tavg)), "tavg_max": float(np.max(clean_tavg)),
            "delta_min": float(np.min(clean_delta)), "delta_median": float(np.median(clean_delta)),
            "delta_mean": float(np.mean(clean_delta)), "delta_max": float(np.max(clean_delta)),
        },
        "material_class_counts": {
            "Air": int(np.sum(clean_grids == 0)), "PM": int(np.sum(clean_grids == 1)),
            "Iron": int(np.sum(clean_grids == 2)),
        },
        "repair_group_audit": {
            "group_count": int(len(np.unique(clean_component))),
            "maximum_group_size": int(np.max(np.unique(clean_component, return_counts=True)[1])),
        },
        "scientific_decisions": [
            "Only corrected population_all topologies receive historical FEMM targets.",
            "Raw population_noChange_all topologies are audit/grouping metadata, not extra labeled samples.",
            "Codes 2 and 3 are collapsed because both were verified to become Pure Iron in FEMM.",
            "Conflicting repeated physical topologies are quarantined rather than averaged into training.",
        ],
    }
    with (output / "dataset_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    _write_reports(reports, summary)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["split"], ensure_ascii=False, indent=2))
    print(f"Dataset: {output}")
    print(f"Reports: {reports}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--label-atol", type=float, default=ATOL)
    args = parser.parse_args()
    build(args.mat, args.output, args.reports, args.seed, args.label_atol)


if __name__ == "__main__":
    main()
