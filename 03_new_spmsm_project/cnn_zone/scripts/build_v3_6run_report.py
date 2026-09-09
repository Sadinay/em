"""Build the audited comparison report for the completed six-run V3 matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "cnn_zone" / "models" / "v3_40000_6runs"
OUTPUT = ROOT / "reports" / "V3" / "03_完整审核资料" / "训练过程记录" / "result_report"
RUNS = (
    ("Logical Small CNN", "logical6x20/small_cnn_v2"),
    ("Logical Mini-Inception", "logical6x20/mini_inception_v2"),
    ("XY 90 VGG16", "xy90_224/vgg16_v2"),
    ("Polar 90 VGG16", "polar90_224/vgg16_v2"),
    ("XY 360 VGG16", "xy360_224/vgg16_v2"),
    ("Polar 360 VGG16", "polar360_224/vgg16_v2"),
)
SEED = 20260903


def run_dir(run_id: str) -> Path:
    return MODEL_ROOT / run_id / f"seed_{SEED}"


def pct_change(new: float, reference: float) -> float:
    return 100.0 * (new / reference - 1.0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    histories = {}
    predictions = {}
    for label, run_id in RUNS:
        directory = run_dir(run_id)
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        prediction = np.genfromtxt(directory / "test_predictions.csv", delimiter=",", names=True)
        if result.get("status") != "complete" or len(prediction) != 6483:
            raise RuntimeError(f"Incomplete result: {run_id}")
        tavg = result["test_metrics"]["tavg"]
        delta = result["test_metrics"]["delta_t"]
        rows.append({
            "model": label,
            "run_id": run_id,
            "seed": result["seed"],
            "epochs_completed": result["epochs_completed"],
            "best_epoch": result["best_epoch"],
            "validation_standardized_mse": result["best_validation_standardized_mse"],
            "test_standardized_mse": result["test_standardized_mse"],
            "tavg_mae_nm": tavg["mae"],
            "tavg_rmse_nm": tavg["rmse"],
            "tavg_r2": tavg["r2"],
            "tavg_bias_nm": tavg["bias"],
            "tavg_abs_error_p95_nm": tavg["absolute_error_p95"],
            "delta_t_mae_nm": delta["mae"],
            "delta_t_rmse_nm": delta["rmse"],
            "delta_t_r2": delta["r2"],
            "delta_t_bias_nm": delta["bias"],
            "delta_t_abs_error_p95_nm": delta["absolute_error_p95"],
            "elapsed_seconds": result["elapsed_seconds"],
            "test_samples": len(prediction),
        })
        histories[label] = history
        predictions[label] = prediction

    rows.sort(key=lambda item: item["test_standardized_mse"])
    for rank, row in enumerate(rows, start=1):
        row["rank_by_test_standardized_mse"] = rank

    columns = [
        "rank_by_test_standardized_mse", "model", "run_id", "seed", "epochs_completed", "best_epoch",
        "validation_standardized_mse", "test_standardized_mse", "tavg_mae_nm", "tavg_rmse_nm", "tavg_r2",
        "tavg_bias_nm", "tavg_abs_error_p95_nm", "delta_t_mae_nm", "delta_t_rmse_nm", "delta_t_r2",
        "delta_t_bias_nm", "delta_t_abs_error_p95_nm", "elapsed_seconds", "test_samples",
    ]
    with (OUTPUT / "metrics_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["model"] for row in rows]
    short = [label.replace("Logical ", "L-").replace(" VGG16", "") for label in labels]
    x = np.arange(len(rows))
    colors = ["#2f5597" if "Logical" in label else "#ed7d31" for label in labels]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].bar(x, [row["tavg_mae_nm"] for row in rows], color=colors)
    axes[0, 0].set_title("Mean torque test MAE (lower is better)")
    axes[0, 0].set_ylabel("MAE (N m)")
    axes[0, 1].bar(x, [row["delta_t_mae_nm"] for row in rows], color=colors)
    axes[0, 1].set_title("Torque ripple test MAE (lower is better)")
    axes[0, 1].set_ylabel("MAE (N m)")
    width = 0.38
    axes[1, 0].bar(x - width / 2, [row["tavg_r2"] for row in rows], width, label="Tavg")
    axes[1, 0].bar(x + width / 2, [row["delta_t_r2"] for row in rows], width, label="DeltaT")
    axes[1, 0].set_title("Test R2 (higher is better)")
    axes[1, 0].set_ylim(0.93, 1.001)
    axes[1, 0].legend()
    axes[1, 1].bar(x, [row["elapsed_seconds"] / 3600 for row in rows], color=colors)
    axes[1, 1].set_title("Training elapsed time")
    axes[1, 1].set_ylabel("Hours")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x, short, rotation=22, ha="right")
    fig.suptitle("SPMSM V3: six-model comparison on the same 6,483-sample test set", fontsize=16)
    fig.savefig(OUTPUT / "model_comparison.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for label, history in histories.items():
        epochs = [item["epoch"] for item in history]
        values = [item["validation_standardized_mse"] for item in history]
        axis.plot(epochs, values, linewidth=1.8, label=label)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation standardized MSE (log scale)")
    axis.set_title("Validation learning curves")
    axis.grid(alpha=0.25, which="both")
    axis.legend(ncol=2)
    fig.savefig(OUTPUT / "validation_learning_curves.png", dpi=220)
    plt.close(fig)

    best = rows[0]
    pred = predictions[best["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    for axis, actual_name, predicted_name, title in (
        (axes[0], "actual_tavg", "predicted_tavg", "Mean torque Tavg"),
        (axes[1], "actual_delta_t", "predicted_delta_t", "Torque ripple DeltaT"),
    ):
        actual = pred[actual_name]
        predicted = pred[predicted_name]
        lower = min(float(actual.min()), float(predicted.min()))
        upper = max(float(actual.max()), float(predicted.max()))
        graphic = axis.hexbin(actual, predicted, gridsize=65, mincnt=1, bins="log", cmap="viridis")
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1.2, label="Ideal")
        axis.set_xlabel("FEMM actual (N m)")
        axis.set_ylabel("Prediction (N m)")
        axis.set_title(title)
        axis.grid(alpha=0.18)
        axis.legend()
        fig.colorbar(graphic, ax=axis, label="log density")
    fig.suptitle(f"Best model parity: {best['model']} (test n=6,483)", fontsize=15)
    fig.savefig(OUTPUT / "best_model_parity.png", dpi=220)
    plt.close(fig)

    by_name = {row["model"]: row for row in rows}
    polar90 = by_name["Polar 90 VGG16"]
    xy90 = by_name["XY 90 VGG16"]
    xy360 = by_name["XY 360 VGG16"]
    polar360 = by_name["Polar 360 VGG16"]
    logical = by_name["Logical Small CNN"]
    total_hours = sum(row["elapsed_seconds"] for row in rows) / 3600

    table_lines = [
        "| Rank | Model | Best epoch | Test std. MSE | Tavg MAE | Tavg R2 | DeltaT MAE | DeltaT R2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['rank_by_test_standardized_mse']} | {row['model']} | {row['best_epoch']} | "
            f"{row['test_standardized_mse']:.6f} | {row['tavg_mae_nm']:.6f} | {row['tavg_r2']:.6f} | "
            f"{row['delta_t_mae_nm']:.6f} | {row['delta_t_r2']:.6f} |"
        )
    report = f"""# SPMSM V3 six-model test report

All six runs completed successfully using the same audited split: 40,000 training, 6,483 validation and 6,483 held-out test genes. Results below are from seed `{SEED}`. The test split was accessed only after validation selected the best checkpoint.

## Ranking

{chr(10).join(table_lines)}

Units for both MAE columns are N m.

## Main findings

1. **Polar 90 VGG16 is the best overall model.** It has the lowest test standardized MSE (`{polar90['test_standardized_mse']:.6f}`), Tavg MAE (`{polar90['tavg_mae_nm']:.6f}` N m), and DeltaT MAE (`{polar90['delta_t_mae_nm']:.6f}` N m).
2. Against XY 90 VGG16, Polar 90 reduces Tavg MAE by `{-pct_change(polar90['tavg_mae_nm'], xy90['tavg_mae_nm']):.1f}%` and DeltaT MAE by `{-pct_change(polar90['delta_t_mae_nm'], xy90['delta_t_mae_nm']):.1f}%`.
3. Against the best logical model, Polar 90 reduces Tavg MAE by `{-pct_change(polar90['tavg_mae_nm'], logical['tavg_mae_nm']):.1f}%` and DeltaT MAE by `{-pct_change(polar90['delta_t_mae_nm'], logical['delta_t_mae_nm']):.1f}%`.
4. **The 360-degree representation did not improve accuracy at 224x224.** XY 360 is slightly worse than XY 90 (Tavg MAE `{pct_change(xy360['tavg_mae_nm'], xy90['tavg_mae_nm']):+.1f}%`, DeltaT MAE `{pct_change(xy360['delta_t_mae_nm'], xy90['delta_t_mae_nm']):+.1f}%`). Polar 360 is substantially worse than Polar 90 (Tavg MAE `{pct_change(polar360['tavg_mae_nm'], polar90['tavg_mae_nm']):+.1f}%`, DeltaT MAE `{pct_change(polar360['delta_t_mae_nm'], polar90['delta_t_mae_nm']):+.1f}%`).
5. The likely explanation is effective resolution: at fixed 224 angular columns, Polar 90 has about `0.402` degree/column, whereas Polar 360 has about `1.607` degree/column. The 360-degree representation spends resolution on repeated motor sectors rather than giving the 90-degree design region more detail.
6. XY 360 reached its best validation score at epoch 40, the configured maximum. Therefore the present comparison proves it did not outperform under the common 40-epoch budget; it does not prove that further training could never improve it.

## Audit notes

- Total recorded training time: `{total_hours:.2f}` hours.
- Only one seed was retained by design, so these results compare the selected runs but do not quantify random-seed variance.
- Test labels come from FEMM and are identical across all six models.
- Both targets are predicted simultaneously by every model.

## Figures and data

- `model_comparison.png`: accuracy and time across all six models.
- `validation_learning_curves.png`: validation standardized MSE by epoch.
- `best_model_parity.png`: FEMM actual versus Polar 90 predictions.
- `metrics_summary.csv`: machine-readable full metric table.
"""
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Report written to {OUTPUT}")


if __name__ == "__main__":
    main()
