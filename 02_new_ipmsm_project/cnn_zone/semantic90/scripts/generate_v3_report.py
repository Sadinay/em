"""Aggregate the completed V3 runs and create full-test result figures."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v3_30000"
REPORT_ROOT = PROJECT / "reports" / "v3_30000"
V2_AGGREGATE = PROJECT / "reports" / "v2_test_11700" / "aggregate_metrics.json"
MODEL_IDS = (
    "logical10/mini_inception_v2",
    "logical10/resnet20_v2",
    "logical10/small_cnn_v2",
    "semantic224/vgg16_v2",
)
EXPECTED_SEEDS = (20260823, 20260824)
MODEL_LABELS = {
    "logical10/mini_inception_v2": "10x10 Mini-Inception V3",
    "logical10/resnet20_v2": "10x10 ResNet20 V3",
    "logical10/small_cnn_v2": "10x10 SmallCNN V3",
    "semantic224/vgg16_v2": "224x224 VGG16 V3",
}
TARGETS = (
    ("tavg", "Mean torque Tavg", 0),
    ("delta_t", "Torque ripple DeltaT", 1),
)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def load_runs() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for model_id in MODEL_IDS:
        family, name = model_id.split("/")
        for seed in EXPECTED_SEEDS:
            directory = MODEL_ROOT / family / name / f"seed_{seed}"
            paths = {
                "core_test": directory / "result.json",
                "full_validation": directory / "full_validation_result.json",
                "full_test": directory / "full_test_result.json",
            }
            if any(not path.exists() for path in paths.values()):
                missing = [str(path) for path in paths.values() if not path.exists()]
                raise RuntimeError(f"V3 run is incomplete; missing {missing}")
            records = {
                key: json.loads(path.read_text(encoding="utf-8"))
                for key, path in paths.items()
            }
            if any(record.get("status") != "complete" for record in records.values()):
                raise RuntimeError(f"V3 result is not complete: {directory}")
            grouped[model_id].append(
                {"seed": seed, "directory": directory, **records}
            )
    return grouped


def load_ensemble_predictions(runs: list[dict], set_name: str) -> tuple[np.ndarray, np.ndarray]:
    reference_actual = None
    reference_indices = None
    predictions = []
    for run in runs:
        data = np.genfromtxt(
            run["directory"] / f"{set_name}_predictions.csv",
            delimiter=",",
            names=True,
            encoding="utf-8",
        )
        indices = np.asarray(data["sample_index"], dtype=np.int64)
        actual = np.column_stack((data["actual_tavg_nm"], data["actual_delta_t_nm"]))
        predicted = np.column_stack(
            (data["predicted_tavg_nm"], data["predicted_delta_t_nm"])
        )
        if reference_indices is None:
            reference_indices, reference_actual = indices, actual
        elif not np.array_equal(indices, reference_indices) or not np.allclose(
            actual, reference_actual, rtol=0.0, atol=1e-7
        ):
            raise RuntimeError("Prediction rows differ between V3 seeds")
        predictions.append(predicted)
    assert reference_actual is not None
    return reference_actual, np.mean(np.stack(predictions), axis=0)


def aggregate(grouped: dict[str, list[dict]]) -> dict:
    summary = {"seed_count": len(EXPECTED_SEEDS), "models": {}}
    for model_id, runs in grouped.items():
        model_record = {"seeds": [run["seed"] for run in runs], "sets": {}}
        for set_name in ("core_test", "full_validation", "full_test"):
            set_record = {"targets": {}}
            for target_name, _, _ in TARGETS:
                metrics = runs[0][set_name]["metrics"]["overall"][target_name]
                set_record["targets"][target_name] = {
                    metric_name: dict(
                        zip(
                            ("mean", "std"),
                            mean_std(
                                [
                                    run[set_name]["metrics"]["overall"][target_name][metric_name]
                                    for run in runs
                                ]
                            ),
                        )
                    )
                    for metric_name in metrics
                }
            model_record["sets"][set_name] = set_record
        model_record["training"] = {
            "best_epoch": dict(
                zip(
                    ("mean", "std"),
                    mean_std([float(run["core_test"]["best_epoch"]) for run in runs]),
                )
            ),
            "elapsed_seconds": dict(
                zip(
                    ("mean", "std"),
                    mean_std([float(run["core_test"]["elapsed_seconds"]) for run in runs]),
                )
            ),
        }
        summary["models"][model_id] = model_record
    return summary


def write_metrics_csv(summary: dict) -> None:
    with (REPORT_ROOT / "v3_metrics_mean_std.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("model_id", "evaluation_set", "target", "metric", "mean", "std"))
        for model_id, model in summary["models"].items():
            for set_name, set_record in model["sets"].items():
                for target_name, metrics in set_record["targets"].items():
                    for metric_name, values in metrics.items():
                        writer.writerow(
                            (model_id, set_name, target_name, metric_name, values["mean"], values["std"])
                        )


def plot_learning_curves(grouped: dict[str, list[dict]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    for axis, model_id in zip(axes.flat, MODEL_IDS):
        for run in grouped[model_id]:
            history = json.loads(
                (run["directory"] / "history.json").read_text(encoding="utf-8")
            )
            epochs = [record["epoch"] for record in history]
            axis.plot(
                epochs,
                [record["training_standardized_mse"] for record in history],
                "--",
                linewidth=1.1,
                alpha=0.65,
            )
            axis.plot(
                epochs,
                [record["validation_standardized_mse"] for record in history],
                linewidth=1.35,
                label=f"validation seed {run['seed']}",
            )
        axis.set_title(MODEL_LABELS[model_id])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Standardized two-target MSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("V3 training (dashed) and 5,000-sample validation (solid)")
    figure.savefig(REPORT_ROOT / "training_validation_curves_four_v3.png", dpi=190)
    plt.close(figure)


def plot_full_test_predictions_and_residuals(grouped: dict[str, list[dict]]) -> None:
    for target_name, target_label, target_index in TARGETS:
        for kind in ("prediction", "residual"):
            figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
            for axis, model_id in zip(axes.flat, MODEL_IDS):
                actual, predicted = load_ensemble_predictions(grouped[model_id], "full_test")
                truth = actual[:, target_index]
                estimate = predicted[:, target_index]
                values = estimate if kind == "prediction" else estimate - truth
                axis.scatter(truth, values, s=3, alpha=0.22, linewidths=0)
                if kind == "prediction":
                    low = float(min(truth.min(), values.min()))
                    high = float(max(truth.max(), values.max()))
                    axis.plot((low, high), (low, high), "r--", linewidth=1)
                    axis.set_ylabel("Two-seed mean prediction (N m)")
                else:
                    axis.axhline(0.0, color="red", linestyle="--", linewidth=1)
                    axis.set_ylabel("Signed error (N m)")
                axis.set_title(MODEL_LABELS[model_id])
                axis.set_xlabel(f"Actual {target_label} (N m)")
                axis.grid(alpha=0.18)
            figure.suptitle(
                f"V3 complete Scheme-A test set (n=14,655): "
                f"{target_label}, two-seed mean {kind}"
            )
            figure.savefig(REPORT_ROOT / f"{target_name}_{kind}_four_v3.png", dpi=190)
            plt.close(figure)


def plot_v2_comparison(summary: dict, v2_summary: dict) -> None:
    """Compare V2 and V3 only on the shared frozen 1,500-sample test set."""
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(MODEL_IDS))
    width = 0.36
    labels = [MODEL_LABELS[model_id].replace(" V3", "") for model_id in MODEL_IDS]
    for axis, (target_name, target_label, _) in zip(axes, TARGETS):
        v2_values = [
            v2_summary["models"][model_id]["overall"][target_name]["mae"]["mean"]
            for model_id in MODEL_IDS
        ]
        v2_errors = [
            v2_summary["models"][model_id]["overall"][target_name]["mae"]["std"]
            for model_id in MODEL_IDS
        ]
        v3_values = [
            summary["models"][model_id]["sets"]["core_test"]["targets"][target_name]["mae"]["mean"]
            for model_id in MODEL_IDS
        ]
        v3_errors = [
            summary["models"][model_id]["sets"]["core_test"]["targets"][target_name]["mae"]["std"]
            for model_id in MODEL_IDS
        ]
        axis.bar(
            x - width / 2,
            v2_values,
            width,
            yerr=v2_errors,
            capsize=3,
            label="V2 mean +/- std (3 seeds)",
        )
        axis.bar(
            x + width / 2,
            v3_values,
            width,
            yerr=v3_errors,
            capsize=3,
            label="V3 mean +/- std (2 seeds)",
        )
        axis.set_title(f"{target_label} core-test MAE")
        axis.set_ylabel("MAE (N m)")
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("V2 vs V3 on the same frozen 1,500-sample test set")
    figure.savefig(REPORT_ROOT / "v2_vs_v3_core_test_mae.png", dpi=190)
    plt.close(figure)


def write_readme(summary: dict) -> None:
    lines = [
        "# V3 30,000-sample training report",
        "",
        "This report uses two seeds for each selected model. Early stopping uses 5,000 validation samples. The 1,500 core test preserves V2 comparability; the complete Scheme-A test contains 14,655 samples.",
        "",
        "| Model | Best epoch | Full-test Tavg MAE (N m) | Full-test DeltaT MAE (N m) |",
        "|---|---:|---:|---:|",
    ]
    for model_id in MODEL_IDS:
        model = summary["models"][model_id]
        best = model["training"]["best_epoch"]
        tavg = model["sets"]["full_test"]["targets"]["tavg"]["mae"]
        delta = model["sets"]["full_test"]["targets"]["delta_t"]["mae"]
        lines.append(
            f"| {MODEL_LABELS[model_id]} | {best['mean']:.1f} +/- {best['std']:.1f} | "
            f"{tavg['mean']:.4f} +/- {tavg['std']:.4f} | "
            f"{delta['mean']:.4f} +/- {delta['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `aggregate_metrics.json` and `v3_metrics_mean_std.csv`;",
            "- `training_validation_curves_four_v3.png`;",
            "- `tavg_prediction_four_v3.png` and `tavg_residual_four_v3.png`;",
            "- `delta_t_prediction_four_v3.png` and `delta_t_residual_four_v3.png`;",
            "- `v2_vs_v3_core_test_mae.png`: V2/V3 comparison on the shared frozen 1,500-sample core test.",
            "- `femm_error_vs_model_disagreement_v3.png`: FEMM error versus four-architecture disagreement on the complete test population, with the reviewed 1,000 genes highlighted.",
        ]
    )
    (REPORT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    grouped = load_runs()
    summary = aggregate(grouped)
    (REPORT_ROOT / "aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_metrics_csv(summary)
    plot_learning_curves(grouped)
    plot_full_test_predictions_and_residuals(grouped)
    v2_summary = json.loads(V2_AGGREGATE.read_text(encoding="utf-8"))
    plot_v2_comparison(summary, v2_summary)
    write_readme(summary)
    print(f"V3 report written to {REPORT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
