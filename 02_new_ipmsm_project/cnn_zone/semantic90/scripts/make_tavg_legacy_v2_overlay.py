"""Overlay legacy (red) and available v2 (blue) Tavg test predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from make_v2_interim_plots import completed_predictions, metrics


PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT / "reports" / "v2_test_11700" / "interim"
MODELS = (
    (
        "logical10/mini_inception_v2",
        "10x10 Mini-Inception",
        PROJECT / "cnn_zone/models/test_11700/logical10/mini_inception/test_predictions.csv",
    ),
    (
        "logical10/resnet20_v2",
        "10x10 ResNet20",
        PROJECT / "cnn_zone/models/test_11700/logical10/resnet20/test_predictions.csv",
    ),
    (
        "logical10/small_cnn_v2",
        "10x10 SmallCNN",
        PROJECT / "cnn_zone/models/test_11700/logical10/small_cnn/test_predictions.csv",
    ),
    (
        "semantic224/mini_inception_v2",
        "224 Mini-Inception",
        PROJECT / "cnn_zone/models/test_11700/semantic224/mini_inception/test_predictions.csv",
    ),
    (
        "semantic224/resnet18_v2",
        "224 ResNet18",
        PROJECT / "cnn_zone/models/test_11700/semantic224/resnet18/test_predictions.csv",
    ),
    (
        "semantic224/vgg16_v2",
        "224 VGG16",
        PROJECT / "cnn_zone/models/test_11700/semantic224/vgg16/test_predictions.csv",
    ),
)


def read_legacy(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    return (
        np.asarray(data["sample_index"], dtype=np.int64),
        np.asarray(data["actual_tavg_nm"], dtype=np.float64),
        np.asarray(data["predicted_tavg_nm"], dtype=np.float64),
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    for axis, (model_id, title, legacy_path) in zip(axes.flat, MODELS):
        legacy_indices, legacy_actual, legacy_prediction = read_legacy(legacy_path)
        legacy_metrics = metrics(legacy_actual, legacy_prediction)
        current = completed_predictions(
            model_id, "actual_tavg_nm", "predicted_tavg_nm"
        )
        axis.scatter(
            legacy_actual,
            legacy_prediction,
            s=8,
            color="#d62728",
            alpha=0.24,
            linewidths=0,
            label="Legacy",
            zorder=2,
        )
        all_values = [legacy_actual, legacy_prediction]
        if current is not None:
            if not np.array_equal(legacy_indices, current["indices"]):
                raise RuntimeError(f"Legacy/v2 test sample order differs for {model_id}")
            if not np.allclose(legacy_actual, current["actual"], rtol=0.0, atol=1e-7):
                raise RuntimeError(f"Legacy/v2 actual Tavg differs for {model_id}")
            current_metrics = metrics(current["actual"], current["predicted"])
            axis.scatter(
                current["actual"],
                current["predicted"],
                s=7,
                color="#1565c0",
                alpha=0.34,
                linewidths=0,
                label="V2",
                zorder=3,
            )
            all_values.append(current["predicted"])
            seed_count = len(current["seeds"])
            current_label = "3-seed mean" if seed_count == 3 else f"interim {seed_count}/3 seed"
            axis.text(
                0.97,
                0.03,
                f"V2 blue ({current_label})\nMAE={current_metrics['mae']:.4f}\nR2={current_metrics['r2']:.4f}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.7,
                color="#0d47a1",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.84, "edgecolor": "#90caf9"},
            )
            subtitle = "V2 3/3 complete" if seed_count == 3 else f"V2 interim {seed_count}/3"
        else:
            axis.text(
                0.97,
                0.03,
                "V2 blue: PENDING 0/3",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#0d47a1",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.84, "edgecolor": "#90caf9"},
            )
            subtitle = "V2 pending"
        low = float(min(values.min() for values in all_values))
        high = float(max(values.max() for values in all_values))
        padding = 0.03 * (high - low)
        axis.plot(
            (low - padding, high + padding),
            (low - padding, high + padding),
            "--",
            color="#333333",
            linewidth=1.1,
            zorder=1,
        )
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{title}\n{subtitle}")
        axis.set_xlabel("Actual mean torque Tavg (N m)")
        axis.set_ylabel("Predicted mean torque Tavg (N m)")
        axis.grid(alpha=0.18)
        axis.text(
            0.03,
            0.97,
            f"Legacy red\nMAE={legacy_metrics['mae']:.4f}\nR2={legacy_metrics['r2']:.4f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            color="#9b1c1c",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.84, "edgecolor": "#ef9a9a"},
        )
    legend = (
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markersize=7, label="Legacy short test (red)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1565c0", markersize=7, label="V2 available result (blue)"),
        Line2D([0], [0], linestyle="--", color="#333333", label="Ideal prediction"),
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=3,
        frameon=True,
    )
    figure.suptitle(
        "Mean torque Tavg: legacy versus V2 on the identical fixed test set\n"
        "Red = previous short test; blue = current best-checkpoint prediction",
        fontsize=15,
    )
    output = OUTPUT / "interim_tavg_prediction_legacy_red_v2_blue.png"
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__":
    main()
