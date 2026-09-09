"""Add the supplementary logical 6x20 ResNet20 run to the visible V3 report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v3_40000_6runs"
FIGURE_ROOT = PROJECT / "reports" / "V3" / "01_结果图"
DATA_ROOT = PROJECT / "reports" / "V3" / "02_汇总数据"
SEED = 20260903
RUNS = (
    ("6x20 SmallCNN", "logical6x20/small_cnn_v2"),
    ("6x20 Mini-Inception", "logical6x20/mini_inception_v2"),
    ("6x20 ResNet20", "logical6x20/resnet20_v2"),
    ("XY90 VGG16", "xy90_224/vgg16_v2"),
    ("Polar90 VGG16", "polar90_224/vgg16_v2"),
    ("XY360 VGG16", "xy360_224/vgg16_v2"),
    ("Polar360 VGG16", "polar360_224/vgg16_v2"),
)


def load_prediction(run_id: str) -> np.ndarray:
    path = MODEL_ROOT / run_id / f"seed_{SEED}" / "test_predictions.csv"
    return np.genfromtxt(path, delimiter=",", names=True)


def high_tail_record(prediction: np.ndarray) -> dict:
    actual = np.asarray(prediction["actual_tavg"])
    predicted = np.asarray(prediction["predicted_tavg"])
    mask = actual >= 3.5
    slope, intercept = np.polyfit(actual[mask], predicted[mask], 1)
    return {
        "threshold_actual_tavg": 3.5,
        "sample_count": int(mask.sum()),
        "actual_range": [float(actual[mask].min()), float(actual[mask].max())],
        "predicted_range": [float(predicted[mask].min()), float(predicted[mask].max())],
        "mae": float(np.mean(np.abs(predicted[mask] - actual[mask]))),
        "bias": float(np.mean(predicted[mask] - actual[mask])),
        "linear_slope_predicted_vs_actual": float(slope),
        "linear_intercept": float(intercept),
    }


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for label, run_id in RUNS:
        directory = MODEL_ROOT / run_id / f"seed_{SEED}"
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        results.append({"label": label, "run_id": run_id, **result})

    labels = [row["label"] for row in results]
    x = np.arange(len(labels))
    colors = ["#4472c4"] * 3 + ["#ed7d31", "#70ad47", "#ffc000", "#a5a5a5"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    for axis, target, title in (
        (axes[0], "tavg", "Mean torque Tavg test MAE"),
        (axes[1], "delta_t", "Historical DeltaT-label test MAE"),
    ):
        values = [row["test_metrics"][target]["mae"] for row in results]
        bars = axis.bar(x, values, color=colors)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("MAE")
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("SPMSM V3: seven models on the same frozen 6,483-sample test set", fontsize=15)
    fig.savefig(FIGURE_ROOT / "09_七模型测试误差对比.png", dpi=220)
    plt.close(fig)

    run_id = "logical6x20/resnet20_v2"
    directory = MODEL_ROOT / run_id / f"seed_{SEED}"
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    prediction = load_prediction(run_id)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.1), constrained_layout=True)
    epochs = [item["epoch"] for item in history]
    axes[0].plot(epochs, [item["train_standardized_mse"] for item in history], "--", label="training")
    axes[0].plot(epochs, [item["validation_standardized_mse"] for item in history], label="validation")
    axes[0].axvline(result["best_epoch"], color="black", linestyle=":", label=f"best={result['best_epoch']}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Standardized two-target MSE")
    axes[0].set_title("Training and validation")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2, which="both")

    for axis, actual_name, predicted_name, title in (
        (axes[1], "actual_tavg", "predicted_tavg", "Mean torque Tavg"),
        (axes[2], "actual_delta_t", "predicted_delta_t", "Historical DeltaT label"),
    ):
        actual = np.asarray(prediction[actual_name])
        predicted = np.asarray(prediction[predicted_name])
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        axis.scatter(actual, predicted, s=3, alpha=0.22, linewidths=0)
        axis.plot((low, high), (low, high), "r--", linewidth=1)
        axis.set_xlabel("FEMM/historical actual")
        axis.set_ylabel("Prediction")
        axis.set_title(title)
        axis.grid(alpha=0.18)
    fig.suptitle("6x20 ResNet20 V2 adapted to project 03 (test n=6,483)", fontsize=15)
    fig.savefig(FIGURE_ROOT / "10_6x20_ResNet20训练与预测.png", dpi=220)
    plt.close(fig)

    mini_tail = high_tail_record(load_prediction("logical6x20/mini_inception_v2"))
    resnet_tail = high_tail_record(prediction)
    supplement = {
        "status": "complete",
        "seed": SEED,
        "split": {"train": 40000, "validation": 6483, "test": 6483},
        "resnet20_result": result,
        "high_tavg_tail_comparison": {
            "mini_inception": mini_tail,
            "resnet20": resnet_tail,
        },
        "delta_t_warning": "DeltaT remains the historical dataset label; its exact physical replay definition is unresolved.",
    }
    (DATA_ROOT / "13_ResNet20补充结果.json").write_text(
        json.dumps(supplement, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (DATA_ROOT / "14_七模型测试指标.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "run_id", "best_epoch", "test_standardized_mse", "tavg_mae", "tavg_r2", "delta_t_mae", "delta_t_r2"))
        for row in results:
            writer.writerow((
                row["label"], row["run_id"], row["best_epoch"], row["test_standardized_mse"],
                row["test_metrics"]["tavg"]["mae"], row["test_metrics"]["tavg"]["r2"],
                row["test_metrics"]["delta_t"]["mae"], row["test_metrics"]["delta_t"]["r2"],
            ))

    print(FIGURE_ROOT / "09_七模型测试误差对比.png")
    print(FIGURE_ROOT / "10_6x20_ResNet20训练与预测.png")
    print(DATA_ROOT / "13_ResNet20补充结果.json")


if __name__ == "__main__":
    main()
