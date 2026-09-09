"""Plot FEMM error against four-architecture V3 model disagreement."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_v3_disagreement_1000 import MODELS, REPORT_ROOT, load_predictions, rankdata


PROJECT = Path(__file__).resolve().parents[3]
ANALYSIS = (
    PROJECT
    / "outputs"
    / "v3_model_disagreement_1000_20260827"
    / "v3_four_model_disagreement_1000_analysis.json"
)
OUTPUT = REPORT_ROOT / "femm_error_vs_model_disagreement_v3.png"


def correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    if rank:
        left, right = rankdata(left), rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def quantile_median_line(x: np.ndarray, y: np.ndarray, bin_count: int = 20) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    groups = np.array_split(order, bin_count)
    return (
        np.asarray([np.median(x[group]) for group in groups]),
        np.asarray([np.median(y[group]) for group in groups]),
    )


def main() -> None:
    indices, actual, by_model = load_predictions()
    scaler = json.loads((REPORT_ROOT / "target_scaler.json").read_text(encoding="utf-8"))
    target_std = np.asarray(scaler["std"], dtype=np.float64)
    stacked = np.stack([by_model[name] for name in MODELS], axis=1)
    consensus = stacked.mean(axis=1)
    disagreement = stacked.std(axis=1, ddof=1)
    absolute_error = np.abs(consensus - actual)
    joint_error = np.sqrt(np.mean(((consensus - actual) / target_std) ** 2, axis=1))
    joint_disagreement = np.sqrt(np.mean((disagreement / target_std) ** 2, axis=1))

    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    group_by_index = {
        int(sample["sample_index"]): sample["selection_group"]
        for sample in analysis["samples"]
    }
    groups = np.asarray([group_by_index.get(int(index), "other") for index in indices])

    panels = (
        (
            absolute_error[:, 0],
            disagreement[:, 0],
            "Mean torque Tavg",
            "Absolute ensemble error vs FEMM (N m)",
            "Four-model disagreement sigma (N m)",
        ),
        (
            absolute_error[:, 1],
            disagreement[:, 1],
            "Torque ripple DeltaT",
            "Absolute ensemble error vs FEMM (N m)",
            "Four-model disagreement sigma (N m)",
        ),
        (
            joint_error,
            joint_disagreement,
            "Two-target standardized joint metric",
            "Joint standardized error vs FEMM",
            "Joint standardized disagreement",
        ),
    )

    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)
    styles = {
        "other": dict(s=4, alpha=0.09, color="#7f8c8d", linewidths=0, label="Other full-test genes"),
        "random_control": dict(s=11, alpha=0.55, color="#1f77b4", linewidths=0, label="Random control (500)"),
        "high_error": dict(s=13, alpha=0.58, color="#ff7f0e", linewidths=0, label="High-error selection (500)"),
    }
    for axis, (x, y, title, xlabel, ylabel) in zip(axes, panels):
        for group in ("other", "random_control", "high_error"):
            mask = groups == group
            axis.scatter(x[mask], y[mask], **styles[group])
        line_x, line_y = quantile_median_line(x, y)
        axis.plot(
            line_x,
            line_y,
            color="#202020",
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            label="Population median by error quantile",
            zorder=5,
        )
        pearson = correlation(x, y)
        spearman = correlation(x, y, rank=True)
        axis.text(
            0.98,
            0.97,
            f"Full test n={len(x):,}\nPearson r={pearson:.3f}\nSpearman rho={spearman:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.88, boxstyle="round,pad=0.35"),
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    figure.suptitle(
        "V3 four-model disagreement versus error to FEMM truth\n"
        "Each architecture prediction is the mean of its two V3 seeds; sigma uses M=4 and ddof=1"
    )
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
