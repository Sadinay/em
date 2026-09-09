"""Overlay the fixed CNN training subset on the historical DeltaT distribution.

The legacy histogram contains exact corrected four-state chromosomes.  CNN
training uses the conflict-free, physically deduplicated three-state catalog,
so the figure also shows that eligible catalog as an outline and computes
selection percentages against it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEDUPLICATED = (
    ROOT
    / "data_zone"
    / "exports"
    / "deduplicated_femm_results"
    / "deduplicated_topology_results.npz"
)
DEFAULT_DATASET = (
    ROOT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)
DEFAULT_SPLIT = ROOT / "reports" / "test" / "fixed_scheme_a_subsample_11700_1500_1500.npz"
DEFAULT_FIGURE = ROOT / "reports" / "deduplicated_delta_t_distribution_with_training_selection.png"
DEFAULT_TABLE = ROOT / "reports" / "deduplicated_delta_t_training_selection_bands.csv"
DEFAULT_SUMMARY = ROOT / "reports" / "deduplicated_delta_t_training_selection_summary.json"


BAND_EDGES = np.asarray([-np.inf, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf], dtype=np.float64)
BAND_LABELS = ("<0.50", "0.50-0.75", "0.75-1.00", "1.00-1.50", "1.50-2.00", ">=2.00")


def band_counts(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.count_nonzero((values >= BAND_EDGES[index]) & (values < BAND_EDGES[index + 1]))
            for index in range(len(BAND_LABELS))
        ],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deduplicated", type=Path, default=DEFAULT_DEDUPLICATED)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    with np.load(args.deduplicated) as archive:
        legacy_delta = np.asarray(archive["delta_t_median"], dtype=np.float64)
    targets = np.load(args.dataset / "targets_tavg_delta.npy", mmap_mode="r")
    eligible_delta = np.asarray(targets[:, 1], dtype=np.float64)
    with np.load(args.split) as split:
        train_indices = np.asarray(split["train"], dtype=np.int64)
    if len(train_indices) != 11_700:
        raise RuntimeError(f"Expected 11,700 training indices, found {len(train_indices):,}")
    if train_indices.min() < 0 or train_indices.max() >= len(eligible_delta):
        raise RuntimeError("Training split contains an out-of-range sample index")
    selected_delta = eligible_delta[train_indices]

    legacy_by_band = band_counts(legacy_delta)
    eligible_by_band = band_counts(eligible_delta)
    selected_by_band = band_counts(selected_delta)
    selection_percent = 100.0 * selected_by_band / eligible_by_band
    legacy_comparison_percent = 100.0 * selected_by_band / legacy_by_band
    overall_selection_percent = 100.0 * len(selected_delta) / len(eligible_delta)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    histogram_edges = np.histogram_bin_edges(legacy_delta, bins=90)
    legacy_hist, _ = np.histogram(legacy_delta, bins=histogram_edges)
    eligible_hist, _ = np.histogram(eligible_delta, bins=histogram_edges)
    selected_hist, _ = np.histogram(selected_delta, bins=histogram_edges)
    centers = 0.5 * (histogram_edges[:-1] + histogram_edges[1:])
    widths = np.diff(histogram_edges)

    fig = plt.figure(figsize=(14.2, 9.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(3.1, 1.45))
    ax = fig.add_subplot(grid[0])
    ax.bar(
        centers,
        legacy_hist,
        width=widths,
        color="#e73f74",
        alpha=0.68,
        edgecolor="white",
        linewidth=0.28,
        label=f"Legacy exact corrected chromosomes (N={len(legacy_delta):,})",
        zorder=1,
    )
    ax.stairs(
        eligible_hist,
        histogram_edges,
        color="#444444",
        linewidth=1.15,
        label=f"CNN-eligible physical catalog (N={len(eligible_delta):,})",
        zorder=2,
    )
    ax.bar(
        centers,
        selected_hist,
        width=0.68 * widths,
        color="#2563b8",
        alpha=0.90,
        edgecolor="#174a8b",
        linewidth=0.20,
        label=f"Fixed CNN training subset (N={len(selected_delta):,})",
        zorder=3,
    )

    quantile_levels = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
    quantiles = np.quantile(legacy_delta, quantile_levels)
    line_colors = ["#7f3c8d", "#11a579", "#3969ac", "#e73f74", "#f2b701"]
    for level, value, color in zip(quantile_levels, quantiles, line_colors):
        ax.axvline(
            value,
            color=color,
            linewidth=1.35,
            linestyle="--",
            label=f"P{int(level * 100)}={value:.4f}",
            zorder=4,
        )
    ax.set_title("DeltaT distribution with the fixed CNN training subset", fontsize=15)
    ax.set_xlabel("Stored DeltaT (N m), median across repeated records")
    ax.set_ylabel("Number of unique topologies per 90-bin interval")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=True, ncol=2, fontsize=9, loc="upper right")
    ax.text(
        0.015,
        0.965,
        "Pink = original four-state deduplication figure\n"
        "Gray outline = actual clean CNN-eligible catalog\n"
        "Blue = the 11,700 samples used for model fitting",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.90, "edgecolor": "#aaaaaa"},
    )

    ratio_ax = fig.add_subplot(grid[1])
    positions = np.arange(len(BAND_LABELS))
    bars = ratio_ax.bar(positions, selection_percent, width=0.66, color="#2563b8", alpha=0.90)
    ratio_ax.axhline(
        overall_selection_percent,
        color="#444444",
        linestyle="--",
        linewidth=1.25,
        label=f"Overall training share = {overall_selection_percent:.3f}%",
    )
    ratio_ax.set_xticks(positions, BAND_LABELS)
    ratio_ax.set_xlabel("DeltaT performance band (N m)")
    ratio_ax.set_ylabel("Selected / eligible in band (%)")
    ratio_ax.set_title("Concrete counts and within-band training share")
    ratio_ax.set_ylim(0.0, max(10.0, float(selection_percent.max()) * 1.28))
    ratio_ax.grid(axis="y", alpha=0.20)
    ratio_ax.legend(loc="upper left", frameon=True, fontsize=9)
    for bar, selected, eligible, percent in zip(
        bars, selected_by_band, eligible_by_band, selection_percent
    ):
        ratio_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.13,
            f"{selected:,} / {eligible:,}\n{percent:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(args.figure, dpi=190)
    plt.close(fig)

    with args.table.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "delta_t_band_nm",
                "legacy_exact_four_state_count",
                "cnn_eligible_physical_count",
                "selected_training_count",
                "selected_over_eligible_percent",
                "selected_over_legacy_percent",
            )
        )
        for row in zip(
            BAND_LABELS,
            legacy_by_band,
            eligible_by_band,
            selected_by_band,
            selection_percent,
            legacy_comparison_percent,
        ):
            writer.writerow((row[0], int(row[1]), int(row[2]), int(row[3]), float(row[4]), float(row[5])))

    summary = {
        "legacy_exact_corrected_four_state_count": int(len(legacy_delta)),
        "cnn_eligible_clean_physical_count": int(len(eligible_delta)),
        "fixed_training_count": int(len(selected_delta)),
        "overall_selected_over_eligible_percent": float(overall_selection_percent),
        "histogram_bins": 90,
        "band_definition": list(BAND_LABELS),
        "bands": [
            {
                "delta_t_band_nm": label,
                "legacy_exact_four_state_count": int(legacy),
                "cnn_eligible_physical_count": int(eligible),
                "selected_training_count": int(selected),
                "selected_over_eligible_percent": float(percent),
                "selected_over_legacy_percent": float(legacy_percent),
            }
            for label, legacy, eligible, selected, percent, legacy_percent in zip(
                BAND_LABELS,
                legacy_by_band,
                eligible_by_band,
                selected_by_band,
                selection_percent,
                legacy_comparison_percent,
            )
        ],
        "note": (
            "The fixed short-test subset was selected by joint rank-based Tavg x DeltaT strata. "
            "These six DeltaT bands are a readable marginal audit, not the sole sampling rule."
        ),
        "sources": {
            "legacy_distribution": str(args.deduplicated.resolve()),
            "eligible_dataset": str(args.dataset.resolve()),
            "fixed_split": str(args.split.resolve()),
        },
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Figure: {args.figure.resolve()}")
    print(f"Table: {args.table.resolve()}")


if __name__ == "__main__":
    main()
