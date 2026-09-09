"""Create six-panel signed absolute-error plots from the fixed test predictions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "test_11700"
REPORT_ROOT = PROJECT / "reports" / "test"
MODELS = (
    ("logical10", "mini_inception", "10×10 Mini-Inception"),
    ("logical10", "resnet20", "10×10 ResNet20"),
    ("logical10", "small_cnn", "10×10 SmallCNN"),
    ("semantic224", "mini_inception", "224×224 Mini-Inception"),
    ("semantic224", "resnet18", "224×224 ResNet18"),
    ("semantic224", "vgg16", "224×224 VGG16"),
)


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = np.asarray(
        [[float(row["actual_tavg_nm"]), float(row["actual_delta_t_nm"])] for row in rows]
    )
    predicted = np.asarray(
        [[float(row["predicted_tavg_nm"]), float(row["predicted_delta_t_nm"])] for row in rows]
    )
    return actual, predicted


def create_figure(target_index: int, output_name: str) -> None:
    records = []
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    for family, model, label in MODELS:
        actual, predicted = read_predictions(MODEL_ROOT / family / model / "test_predictions.csv")
        x = actual[:, target_index]
        signed_error = predicted[:, target_index] - x
        x_min, x_max = min(x_min, float(x.min())), max(x_max, float(x.max()))
        y_min, y_max = min(y_min, float(signed_error.min())), max(y_max, float(signed_error.max()))
        records.append((label, x, signed_error))

    x_padding = max(0.01, 0.025 * (x_max - x_min))
    y_padding = max(0.01, 0.045 * (y_max - y_min))
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), sharex=True, sharey=True)
    for ax, (label, x, signed_error) in zip(axes.flat, records):
        positive = signed_error >= 0
        ax.axhspan(-0.05, 0.05, color="#48a868", alpha=0.10, zorder=0, label="±0.05 N m band")
        ax.scatter(
            x[~positive], signed_error[~positive], s=5, alpha=0.42, color="#2878b5",
            edgecolors="none", label="Under-prediction",
        )
        ax.scatter(
            x[positive], signed_error[positive], s=5, alpha=0.42, color="#df5f2a",
            edgecolors="none", label="Over-prediction",
        )
        ax.axhline(0, color="black", linewidth=1.0)
        bias = float(np.mean(signed_error))
        mae = float(np.mean(np.abs(signed_error)))
        ax.set_title(f"{label}\nBias={bias:+.4f} N m, MAE={mae:.4f} N m", fontsize=11)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.grid(alpha=0.20)

    target_label = "actual mean torque Tavg (N m)" if target_index == 0 else "actual torque ripple DeltaT (N m)"
    title = "Mean torque: signed prediction error" if target_index == 0 else "Torque ripple: signed prediction error"
    for ax in axes[1, :]:
        ax.set_xlabel(target_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("prediction - actual (N m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945), frameon=True)
    fig.suptitle(title, fontsize=17, y=0.995)
    fig.text(
        0.5,
        0.014,
        "Signed error = prediction - actual. All 1,500 independent test samples are shown in every panel.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.84, hspace=0.30, wspace=0.06)
    fig.savefig(REPORT_ROOT / output_name, dpi=190)
    plt.close(fig)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    create_figure(0, output_name="mean_torque_signed_absolute_error_six_models.png")
    create_figure(1, output_name="torque_ripple_signed_absolute_error_six_models.png")
    print(REPORT_ROOT / "mean_torque_signed_absolute_error_six_models.png")
    print(REPORT_ROOT / "torque_ripple_signed_absolute_error_six_models.png")


if __name__ == "__main__":
    main()
