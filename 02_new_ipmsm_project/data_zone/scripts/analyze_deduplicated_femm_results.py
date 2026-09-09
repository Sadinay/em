"""Deduplicate historical chromosomes and plot their stored FEMM-result distribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAT = ROOT / "data_zone" / "raw" / "workspace_600.mat"
DEFAULT_OUTPUT = ROOT / "data_zone" / "exports" / "deduplicated_femm_results"
DEFAULT_REPORTS = ROOT / "reports"


def grouped_statistics(values: np.ndarray, inverse: np.ndarray, unique_count: int) -> dict[str, np.ndarray]:
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
        count=unique_count,
    )
    return {"min": minimum, "median": median, "max": maximum, "range": maximum - minimum}


def plot_distribution(
    values: np.ndarray,
    title: str,
    xlabel: str,
    color: str,
    output_path: Path,
) -> dict[str, float]:
    quantile_levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    quantiles = np.quantile(values, quantile_levels)
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    ax.hist(values, bins=90, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
    line_colors = ["#7f3c8d", "#11a579", "#3969ac", "#e73f74", "#f2b701"]
    for level, value, line_color in zip(quantile_levels, quantiles, line_colors):
        ax.axvline(value, color=line_color, linewidth=1.8, linestyle="--",
                   label=f"P{int(level * 100)} = {value:.4f}")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of unique topologies per bin")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=True, ncol=2)
    ax.text(
        0.015,
        0.97,
        f"N = {len(values):,} unique chromosomes\n"
        f"min = {values.min():.4f}\nmean = {values.mean():.4f}\nmax = {values.max():.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#aaaaaa"},
    )
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        **{f"p{int(level * 100)}": float(value) for level, value in zip(quantile_levels, quantiles)},
    }


def analyze(mat_path: Path, output_dir: Path, reports_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # MATLAB arrays are [population, bit, state].  Flatten generation-major so
    # the state/population indices remain easy to reconstruct.
    # Historical FEMM targets belong to the structure-corrected population_all.
    # population_noChange_all is the pre-repair proposal and must not be used as
    # though it were the directly evaluated topology.
    bits = np.asarray(data["population_all"], dtype=np.uint8).transpose(2, 0, 1)
    state_count, population_size, bit_count = bits.shape
    flat_bits = bits.reshape(-1, bit_count)
    packed = np.packbits(flat_bits, axis=1)
    unique_packed, first_indices, inverse, counts = np.unique(
        packed, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    unique_count = len(unique_packed)

    tavg = np.asarray(data["Tavg_all"], dtype=np.float64).T.reshape(-1)
    delta_t = np.asarray(data["DeltaT_all"], dtype=np.float64).T.reshape(-1)
    fitness = np.asarray(data["Fitvalue_all"], dtype=np.float64).T.reshape(-1)
    tavg_stats = grouped_statistics(tavg, inverse, unique_count)
    delta_stats = grouped_statistics(delta_t, inverse, unique_count)
    fitness_stats = grouped_statistics(fitness, inverse, unique_count)

    state_indices = np.repeat(np.arange(state_count, dtype=np.int32), population_size)
    population_indices = np.tile(np.arange(population_size, dtype=np.int32), state_count)
    order = np.argsort(inverse, kind="stable")
    sorted_group = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]
    first_state = np.minimum.reduceat(state_indices[order], starts)
    last_state = np.maximum.reduceat(state_indices[order], starts)

    np.savez_compressed(
        output_dir / "deduplicated_topology_results.npz",
        chromosome_packed=unique_packed,
        representative_bits=flat_bits[first_indices],
        repeat_count=counts.astype(np.int32),
        first_state_index_0based=first_state,
        last_state_index_0based=last_state,
        t_avg_nm_median=tavg_stats["median"],
        t_avg_nm_min=tavg_stats["min"],
        t_avg_nm_max=tavg_stats["max"],
        delta_t_median=delta_stats["median"],
        delta_t_min=delta_stats["min"],
        delta_t_max=delta_stats["max"],
        fitness_median=fitness_stats["median"],
        fitness_min=fitness_stats["min"],
        fitness_max=fitness_stats["max"],
    )

    csv_path = output_dir / "deduplicated_topology_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "topology_key_hex",
                "repeat_count",
                "first_state_index_0based",
                "last_state_index_0based",
                "t_avg_nm_median",
                "t_avg_nm_min",
                "t_avg_nm_max",
                "delta_t_median",
                "delta_t_min",
                "delta_t_max",
                "fitness_median",
                "fitness_min",
                "fitness_max",
            ]
        )
        for index in range(unique_count):
            writer.writerow(
                [
                    unique_packed[index].tobytes().hex(),
                    int(counts[index]),
                    int(first_state[index]),
                    int(last_state[index]),
                    float(tavg_stats["median"][index]),
                    float(tavg_stats["min"][index]),
                    float(tavg_stats["max"][index]),
                    float(delta_stats["median"][index]),
                    float(delta_stats["min"][index]),
                    float(delta_stats["max"][index]),
                    float(fitness_stats["median"][index]),
                    float(fitness_stats["min"][index]),
                    float(fitness_stats["max"][index]),
                ]
            )

    tavg_summary = plot_distribution(
        tavg_stats["median"],
        "Mean torque distribution after chromosome deduplication",
        "Stored Tavg (N·m), median across repeated records",
        "#3969ac",
        reports_dir / "deduplicated_tavg_distribution.png",
    )
    delta_summary = plot_distribution(
        delta_stats["median"],
        "DeltaT distribution after chromosome deduplication",
        "Stored DeltaT, median across repeated records",
        "#e73f74",
        reports_dir / "deduplicated_delta_t_distribution.png",
    )

    disagreement_tolerance = 1e-10
    summary = {
        "deduplication_key": "exact corrected 200-bit chromosome (population_all)",
        "representative_result": "median of all records sharing the same chromosome",
        "total_historical_records": int(len(flat_bits)),
        "unique_chromosomes": int(unique_count),
        "removed_duplicate_records": int(len(flat_bits) - unique_count),
        "single_occurrence_topologies": int(np.sum(counts == 1)),
        "maximum_repeat_count": int(counts.max()),
        "groups_with_tavg_disagreement": int(np.sum(tavg_stats["range"] > disagreement_tolerance)),
        "groups_with_delta_t_disagreement": int(np.sum(delta_stats["range"] > disagreement_tolerance)),
        "groups_with_fitness_disagreement": int(np.sum(fitness_stats["range"] > disagreement_tolerance)),
        "tavg_median_distribution": tavg_summary,
        "delta_t_median_distribution": delta_summary,
        "notes": [
            "The MAT file stores scalar results, not the original angle-by-angle torque curves.",
            "A repeated chromosome may be copied between GA generations rather than rerun in FEMM.",
            "Quantile lines are descriptive aids, not finalized training-band boundaries.",
        ],
    }
    with (output_dir / "deduplication_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(f"Historical records: {len(flat_bits):,}")
    print(f"Unique corrected chromosomes: {unique_count:,}")
    print(f"Removed duplicate records: {len(flat_bits) - unique_count:,}")
    print(f"Tavg quantiles: {tavg_summary}")
    print(f"DeltaT quantiles: {delta_summary}")
    print(f"Outputs: {output_dir} and {reports_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    analyze(args.mat, args.output_dir, args.reports_dir)


if __name__ == "__main__":
    main()
