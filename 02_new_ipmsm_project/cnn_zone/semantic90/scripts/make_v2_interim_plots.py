"""Create six-panel plots that remain usable before and after all V2 runs finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v2_test_11700"
OUTPUT = PROJECT / "reports" / "v2_test_11700" / "interim"
MODELS = (
    ("logical10/mini_inception_v2", "10x10 Mini-Inception V2"),
    ("logical10/resnet20_v2", "10x10 ResNet20 V2"),
    ("logical10/small_cnn_v2", "10x10 SmallCNN V2"),
    ("semantic224/mini_inception_v2", "224 Mini-Inception V2"),
    ("semantic224/resnet18_v2", "224 ResNet18 V2"),
    ("semantic224/vgg16_v2", "224 VGG16 V2"),
)
TARGETS = (
    ("tavg", "actual_tavg_nm", "predicted_tavg_nm", "Mean torque Tavg"),
    ("delta_t", "actual_delta_t_nm", "predicted_delta_t_nm", "Torque ripple DeltaT"),
)


def read_prediction(path: Path, actual_column: str, predicted_column: str):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    return (
        np.asarray(data["sample_index"], dtype=np.int64),
        np.asarray(data[actual_column], dtype=np.float64),
        np.asarray(data[predicted_column], dtype=np.float64),
    )


def completed_predictions(model_id: str, actual_column: str, predicted_column: str):
    family, name = model_id.split("/")
    completed = []
    for result_path in sorted((MODEL_ROOT / family / name).glob("seed_*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        prediction_path = result_path.parent / "test_predictions.csv"
        indices, actual, predicted = read_prediction(
            prediction_path, actual_column, predicted_column
        )
        completed.append((result["seed"], indices, actual, predicted))
    if not completed:
        return None
    reference_indices, reference_actual = completed[0][1], completed[0][2]
    for _, indices, actual, _ in completed[1:]:
        if not np.array_equal(reference_indices, indices):
            raise RuntimeError(f"Test sample order differs between seeds for {model_id}")
        if not np.allclose(reference_actual, actual, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"Test targets differ between seeds for {model_id}")
    mean_prediction = np.mean(np.stack([item[3] for item in completed]), axis=0)
    return {
        "seeds": [int(item[0]) for item in completed],
        "indices": reference_indices,
        "actual": reference_actual,
        "predicted": mean_prediction,
    }


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / max(float(denominator), 1e-12)),
        "bias": float(np.mean(error)),
    }


def empty_panel(axis: plt.Axes, title: str) -> None:
    axis.set_title(title)
    axis.text(
        0.5,
        0.53,
        "PENDING",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=17,
        color="#777777",
        weight="bold",
    )
    axis.text(
        0.5,
        0.42,
        "0/3 completed seeds",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#777777",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linestyle("--")


def make_plot(target_key: str, actual_column: str, predicted_column: str, label: str, kind: str):
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    records = []
    for axis, (model_id, title) in zip(axes.flat, MODELS):
        aggregate = completed_predictions(model_id, actual_column, predicted_column)
        if aggregate is None:
            empty_panel(axis, title)
            records.append({"model_id": model_id, "completed_seeds": 0, "metrics": None})
            continue
        actual = aggregate["actual"]
        predicted = aggregate["predicted"]
        result_metrics = metrics(actual, predicted)
        completed_count = len(aggregate["seeds"])
        completeness = (
            "FINAL 3-SEED MEAN" if completed_count == 3 else f"INTERIM {completed_count}/3 SEED"
        )
        if kind == "prediction":
            axis.scatter(actual, predicted, s=6, alpha=0.32, linewidths=0, color="#1769aa")
            low = float(min(actual.min(), predicted.min()))
            high = float(max(actual.max(), predicted.max()))
            axis.plot((low, high), (low, high), "--", color="#d32f2f", linewidth=1.2)
            axis.set_ylabel("Predicted (N m)")
            annotation = (
                f"MAE={result_metrics['mae']:.4f} N m\n"
                f"RMSE={result_metrics['rmse']:.4f} N m\nR2={result_metrics['r2']:.4f}"
            )
        else:
            residual = predicted - actual
            axis.scatter(actual, residual, s=6, alpha=0.32, linewidths=0, color="#00796b")
            axis.axhline(0.0, linestyle="--", color="#d32f2f", linewidth=1.2)
            axis.set_ylabel("Signed error: prediction - actual (N m)")
            annotation = (
                f"MAE={result_metrics['mae']:.4f} N m\n"
                f"Bias={result_metrics['bias']:+.4f} N m"
            )
        axis.set_title(f"{title}\n{completeness}")
        axis.set_xlabel(f"Actual {label} (N m)")
        axis.grid(alpha=0.18)
        axis.text(
            0.03,
            0.97,
            annotation,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.86, "edgecolor": "#bbbbbb"},
        )
        records.append(
            {
                "model_id": model_id,
                "completed_seeds": completed_count,
                "seeds": aggregate["seeds"],
                "prediction_aggregation": "arithmetic mean across completed seed predictions",
                "metrics": result_metrics,
            }
        )
    description = "Actual versus prediction" if kind == "prediction" else "Signed residual versus actual"
    all_complete = all(record["completed_seeds"] == 3 for record in records)
    result_state = "final three-seed" if all_complete else "interim"
    figure.suptitle(
        f"V2 {result_state} test results: {description} - {label}\n"
        "Every prediction comes from its run's best validation checkpoint",
        fontsize=15,
    )
    path = OUTPUT / f"interim_{target_key}_{kind}_six_models.png"
    figure.savefig(path, dpi=190)
    plt.close(figure)
    return records, path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "building",
        "warning": None,
        "figures": [],
        "targets": {},
    }
    for target_key, actual_column, predicted_column, label in TARGETS:
        for kind in ("prediction", "residual"):
            records, path = make_plot(
                target_key, actual_column, predicted_column, label, kind
            )
            manifest["figures"].append(str(path.resolve()))
            manifest["targets"].setdefault(target_key, records)
    all_complete = all(
        record["completed_seeds"] == 3
        for records in manifest["targets"].values()
        for record in records
    )
    manifest["status"] = "complete" if all_complete else "interim"
    manifest["warning"] = (
        None
        if all_complete
        else "Incomplete models are blank; partial-seed results are not final comparisons."
    )
    (OUTPUT / "interim_metrics.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "README.md").open("w", encoding="utf-8") as handle:
        if all_complete:
            handle.write(
                "# V2 complete three-seed plots\n\n"
                "All six models have completed three seeds. Every panel uses the arithmetic mean of test "
                "predictions produced by the three best-validation checkpoints. The historical `interim_` "
                "filenames are retained so existing links continue to work.\n"
            )
        else:
            handle.write(
                "# V2 interim plots\n\n"
                "These plots use test predictions made from each completed run's best validation checkpoint. "
                "Models with three seeds use the arithmetic mean prediction. Partial models are explicitly marked; "
                "models without a completed seed are blank. Do not use partial panels as the final six-model ranking.\n"
            )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
