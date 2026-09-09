"""Generate the project-03 V3 report using the project-02 audit specification."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
CNN_ROOT = PROJECT / "cnn_zone"
SCRIPT_ROOT = CNN_ROOT / "scripts"
for path in (CNN_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_v3_40000_6runs import DATASET_DIR, SPECS, SPLIT_PATH, load_split  # noqa: E402
from src.dataset import SPMSMGeneDataset  # noqa: E402
from src.training import build_model, build_renderer, evaluate  # noqa: E402
from audit_split_novelty import main as audit_split_novelty  # noqa: E402


MODEL_ROOT = CNN_ROOT / "models" / "v3_40000_6runs"
PACKAGE_ROOT = PROJECT / "reports" / "V3" / "03_完整审核资料" / "原始审核包"
SUMMARY_ROOT = PACKAGE_ROOT / "summary_reports"
PER_RUN_ROOT = PACKAGE_ROOT / "per_run_test_results"
DISAGREEMENT_ROOT = PACKAGE_ROOT / "disagreement_analysis"
SEED = 20260903
MODEL_IDS = (
    "logical6x20/small_cnn_v2",
    "logical6x20/mini_inception_v2",
    "xy90_224/vgg16_v2",
    "polar90_224/vgg16_v2",
    "xy360_224/vgg16_v2",
    "polar360_224/vgg16_v2",
)
MODEL_LABELS = {
    "logical6x20/small_cnn_v2": "6x20 SmallCNN V3",
    "logical6x20/mini_inception_v2": "6x20 Mini-Inception V3",
    "xy90_224/vgg16_v2": "XY90 224x224 VGG16 V3",
    "polar90_224/vgg16_v2": "Polar90 224x224 VGG16 V3",
    "xy360_224/vgg16_v2": "XY360 224x224 VGG16 V3",
    "polar360_224/vgg16_v2": "Polar360 224x224 VGG16 V3",
}
TARGETS = (
    ("tavg", "Mean torque Tavg", "actual_tavg_nm", "predicted_tavg_nm"),
    ("delta_t", "Torque ripple DeltaT", "actual_delta_t_nm", "predicted_delta_t_nm"),
)
TAVG_EDGES = np.asarray([-np.inf, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, np.inf])
TAVG_LABELS = ("<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.25", "3.25-3.5", "3.5-3.75", ">=3.75")
DELTA_EDGES = np.asarray([-np.inf, 0.2, 0.3, 0.4, 0.5, 0.75, np.inf])
DELTA_LABELS = ("<0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.75", ">=0.75")


def source_dir(model_id: str) -> Path:
    return MODEL_ROOT / model_id / f"seed_{SEED}"


def metric_record(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    absolute = np.abs(error)
    denominator = max(float(np.sum((actual - actual.mean()) ** 2)), 1e-12)
    slope, intercept = np.polyfit(actual.astype(np.float64), predicted.astype(np.float64), 1)
    actual_range = max(float(actual.max() - actual.min()), 1e-12)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / denominator),
        "normalized_mae_by_actual_range": float(absolute.mean() / actual_range),
        "bias": float(error.mean()),
        "calibration_slope_predicted_vs_actual": float(slope),
        "calibration_intercept_nm": float(intercept),
        "absolute_error_p50": float(np.percentile(absolute, 50)),
        "absolute_error_p90": float(np.percentile(absolute, 90)),
        "absolute_error_p95": float(np.percentile(absolute, 95)),
        "fraction_absolute_error_le_0_05_nm": float(np.mean(absolute <= 0.05)),
        "fraction_absolute_error_le_0_10_nm": float(np.mean(absolute <= 0.10)),
    }


def band_records(actual: np.ndarray, predicted: np.ndarray, edges: np.ndarray, labels: tuple[str, ...]) -> list[dict]:
    output = []
    error = predicted - actual
    for index, label in enumerate(labels):
        mask = (actual >= edges[index]) & (actual < edges[index + 1])
        output.append({
            "band": label,
            "count": int(mask.sum()),
            "mae": float(np.mean(np.abs(error[mask]))) if mask.any() else None,
            "bias": float(np.mean(error[mask])) if mask.any() else None,
        })
    return output


def evaluation_result(
    set_name: str,
    indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    standardized_mse: float,
    checkpoint: Path,
    checkpoint_epoch: int,
) -> dict:
    return {
        "status": "complete",
        "evaluation_set": set_name,
        "sample_count": int(len(indices)),
        "standardized_mse": float(standardized_mse),
        "metrics": {
            "overall": {
                "tavg": metric_record(actual[:, 0], predicted[:, 0]),
                "delta_t": metric_record(actual[:, 1], predicted[:, 1]),
            },
            "bands": {
                "tavg": band_records(actual[:, 0], predicted[:, 0], TAVG_EDGES, TAVG_LABELS),
                "delta_t": band_records(actual[:, 1], predicted[:, 1], DELTA_EDGES, DELTA_LABELS),
            },
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint_epoch),
    }


def write_predictions(path: Path, indices: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> None:
    values = np.column_stack((
        indices,
        actual[:, 0], predicted[:, 0], predicted[:, 0] - actual[:, 0],
        actual[:, 1], predicted[:, 1], predicted[:, 1] - actual[:, 1],
    ))
    np.savetxt(
        path,
        values,
        delimiter=",",
        header=("sample_index,actual_tavg_nm,predicted_tavg_nm,error_tavg_nm,"
                "actual_delta_t_nm,predicted_delta_t_nm,error_delta_t_nm"),
        comments="",
    )


def load_saved_test_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    indices = np.asarray(data["sample_index"], dtype=np.int64)
    actual = np.column_stack((data["actual_tavg"], data["actual_delta_t"]))
    predicted = np.column_stack((data["predicted_tavg"], data["predicted_delta_t"]))
    return indices, actual, predicted


def prepare_per_run_results(dataset: SPMSMGeneDataset, split: dict[str, np.ndarray]) -> dict[str, dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runs: dict[str, dict] = {}
    for model_id in MODEL_IDS:
        source = source_dir(model_id)
        destination = PER_RUN_ROOT / model_id / f"seed_{SEED}"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "history.json", "result.json", "test_predictions.csv"):
            shutil.copy2(source / filename, destination / filename)

        checkpoint_path = source / "best_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        spec = SPECS[model_id]
        model = build_model(spec).to(device)
        model.load_state_dict(checkpoint["model_state"])
        renderer = build_renderer(spec, PROJECT, device)
        if spec.input_mode != "logical6x20":
            model = model.to(memory_format=torch.channels_last)
        target_mean = checkpoint["target_mean"].cpu()
        target_std = checkpoint["target_std"].cpu()

        val_loss, _, val_indices, val_actual, val_predicted = evaluate(
            model, dataset, split["validation"], spec, renderer, target_mean, target_std, device
        )
        write_predictions(destination / "full_validation_predictions.csv", val_indices, val_actual, val_predicted)
        val_result = evaluation_result(
            "full_validation", val_indices, val_actual, val_predicted, val_loss,
            checkpoint_path, checkpoint["epoch"],
        )
        (destination / "full_validation_result.json").write_text(
            json.dumps(val_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        test_indices, test_actual, test_predicted = load_saved_test_predictions(source / "test_predictions.csv")
        test_loss = float(np.mean(((test_predicted - test_actual) / target_std.numpy()) ** 2))
        write_predictions(destination / "full_test_predictions.csv", test_indices, test_actual, test_predicted)
        test_result = evaluation_result(
            "full_test", test_indices, test_actual, test_predicted, test_loss,
            checkpoint_path, checkpoint["epoch"],
        )
        (destination / "full_test_result.json").write_text(
            json.dumps(test_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        original_result = json.loads((source / "result.json").read_text(encoding="utf-8"))
        runs[model_id] = {
            "directory": destination,
            "history": json.loads((source / "history.json").read_text(encoding="utf-8")),
            "training_result": original_result,
            "full_validation": val_result,
            "full_test": test_result,
            "test_indices": test_indices,
            "test_actual": test_actual,
            "test_predicted": test_predicted,
        }
        del model, renderer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return runs


def aggregate_runs(runs: dict[str, dict]) -> dict:
    summary = {"seed_count": 1, "seeds": [SEED], "models": {}}
    for model_id, run in runs.items():
        model = {"seeds": [SEED], "sets": {}, "training": {}}
        for set_name in ("full_validation", "full_test"):
            result = run[set_name]
            set_record = {
                "standardized_mse": {"mean": result["standardized_mse"], "std": 0.0},
                "targets": {},
            }
            for target_name in ("tavg", "delta_t"):
                set_record["targets"][target_name] = {
                    name: {"mean": value, "std": 0.0}
                    for name, value in result["metrics"]["overall"][target_name].items()
                }
            model["sets"][set_name] = set_record
        training = run["training_result"]
        model["training"] = {
            "best_epoch": {"mean": float(training["best_epoch"]), "std": 0.0},
            "epochs_completed": {"mean": float(training["epochs_completed"]), "std": 0.0},
            "elapsed_seconds": {"mean": float(training["elapsed_seconds"]), "std": 0.0},
        }
        summary["models"][model_id] = model
    return summary


def write_metrics(summary: dict) -> None:
    (SUMMARY_ROOT / "aggregate_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (SUMMARY_ROOT / "v3_metrics_mean_std.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model_id", "evaluation_set", "target", "metric", "mean", "std"))
        for model_id, model in summary["models"].items():
            for set_name, set_record in model["sets"].items():
                writer.writerow((model_id, set_name, "both", "standardized_mse", set_record["standardized_mse"]["mean"], 0.0))
                for target, metrics in set_record["targets"].items():
                    for metric, values in metrics.items():
                        writer.writerow((model_id, set_name, target, metric, values["mean"], values["std"]))


def plot_learning_curves(runs: dict[str, dict]) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14.5, 12.0), constrained_layout=True)
    for axis, model_id in zip(axes.flat, MODEL_IDS):
        history = runs[model_id]["history"]
        epochs = [record["epoch"] for record in history]
        axis.plot(epochs, [record["train_standardized_mse"] for record in history], "--", linewidth=1.15, label="training")
        axis.plot(epochs, [record["validation_standardized_mse"] for record in history], linewidth=1.4, label="validation")
        axis.axvline(runs[model_id]["training_result"]["best_epoch"], color="black", linestyle=":", linewidth=1, label="best epoch")
        axis.set_title(MODEL_LABELS[model_id])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Standardized two-target MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=8)
    figure.suptitle("V3 training (dashed) and 6,483-sample validation (solid)", fontsize=15)
    figure.savefig(SUMMARY_ROOT / "training_validation_curves_six_v3.png", dpi=190)
    plt.close(figure)


def plot_predictions_and_residuals(runs: dict[str, dict]) -> None:
    for target_index, (target_name, target_label, _, _) in enumerate(TARGETS):
        for kind in ("prediction", "residual"):
            figure, axes = plt.subplots(3, 2, figsize=(14.5, 12.0), constrained_layout=True)
            for axis, model_id in zip(axes.flat, MODEL_IDS):
                actual = runs[model_id]["test_actual"][:, target_index]
                predicted = runs[model_id]["test_predicted"][:, target_index]
                values = predicted if kind == "prediction" else predicted - actual
                axis.scatter(actual, values, s=3, alpha=0.22, linewidths=0)
                if kind == "prediction":
                    low = float(min(actual.min(), values.min()))
                    high = float(max(actual.max(), values.max()))
                    axis.plot((low, high), (low, high), "r--", linewidth=1)
                    axis.set_ylabel("Prediction (N m)")
                else:
                    axis.axhline(0.0, color="red", linestyle="--", linewidth=1)
                    axis.set_ylabel("Signed error (N m)")
                axis.set_title(MODEL_LABELS[model_id])
                axis.set_xlabel(f"Actual {target_label} (N m)")
                axis.grid(alpha=0.18)
            figure.suptitle(f"V3 frozen test set (n=6,483): {target_label}, one-seed {kind}", fontsize=15)
            figure.savefig(SUMMARY_ROOT / f"{target_name}_{kind}_six_v3.png", dpi=190)
            plt.close(figure)


def plot_model_mae(summary: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), constrained_layout=True)
    x = np.arange(len(MODEL_IDS))
    labels = [MODEL_LABELS[item].replace(" V3", "") for item in MODEL_IDS]
    for axis, (target_name, target_label, _, _) in zip(axes, TARGETS):
        values = [summary["models"][item]["sets"]["full_test"]["targets"][target_name]["mae"]["mean"] for item in MODEL_IDS]
        bars = axis.bar(x, values, color=("#4472c4", "#4472c4", "#ed7d31", "#70ad47", "#ffc000", "#a5a5a5"))
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        axis.set_title(f"{target_label} test MAE")
        axis.set_ylabel("MAE (N m)")
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("V3 six-model comparison on the same frozen test set")
    figure.savefig(SUMMARY_ROOT / "input_representation_test_mae.png", dpi=190)
    plt.close(figure)


def write_distribution_files(dataset: SPMSMGeneDataset, split: dict[str, np.ndarray]) -> None:
    targets = np.asarray(dataset.targets)
    definitions = (("tavg", 0, TAVG_EDGES, TAVG_LABELS), ("delta_t", 1, DELTA_EDGES, DELTA_LABELS))
    for name, column, edges, labels in definitions:
        with (SUMMARY_ROOT / f"{name}_band_distribution.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow((f"{name}_band_nm", "clean_unique_pool", "v3_train_count", "validation_count", "test_count", "v3_train_share_percent"))
            for index, label in enumerate(labels):
                mask = (targets[:, column] >= edges[index]) & (targets[:, column] < edges[index + 1])
                counts = [int(mask[values].sum()) for values in (split["train"], split["validation"], split["test"])]
                writer.writerow((label, int(mask.sum()), *counts, 100.0 * counts[0] / len(split["train"])))

    with (SUMMARY_ROOT / "evaluation_split_distribution.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "target", "band_nm", "sample_count", "split_percent"))
        for split_name, indices in split.items():
            for name, column, edges, labels in definitions:
                values = targets[indices, column]
                for index, label in enumerate(labels):
                    count = int(((values >= edges[index]) & (values < edges[index + 1])).sum())
                    writer.writerow((f"v3_{split_name}", name, label, count, 100.0 * count / len(indices)))

    with (SUMMARY_ROOT / "joint_tavg_delta_distribution.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tavg_band_nm", "delta_t_band_nm", "clean_unique_pool", "v3_train", "v3_validation", "v3_test", "v3_train_selected_over_pool_percent"))
        for ti, tlabel in enumerate(TAVG_LABELS):
            tmask = (targets[:, 0] >= TAVG_EDGES[ti]) & (targets[:, 0] < TAVG_EDGES[ti + 1])
            for di, dlabel in enumerate(DELTA_LABELS):
                mask = tmask & (targets[:, 1] >= DELTA_EDGES[di]) & (targets[:, 1] < DELTA_EDGES[di + 1])
                pool = int(mask.sum())
                counts = [int(mask[values].sum()) for values in (split["train"], split["validation"], split["test"])]
                writer.writerow((tlabel, dlabel, pool, *counts, 100.0 * counts[0] / pool if pool else 0.0))

    train_targets = targets[split["train"]]
    scaler = {
        "target_order": ["Tavg_Nm", "DeltaT_Nm"],
        "mean": train_targets.mean(axis=0).tolist(),
        "std": train_targets.std(axis=0).tolist(),
        "statistics_source": "only the frozen 40,000-sample V3 training set",
    }
    (SUMMARY_ROOT / "target_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    shutil.copy2(SPLIT_PATH, SUMMARY_ROOT / "fixed_v3_split_40000_6483_6483_full64804.npz")
    shutil.copy2(PROJECT / "reports" / "spmsm_topology_dataset" / "train40000_summary.json", SUMMARY_ROOT / "selection_summary.json")
    shutil.copy2(PROJECT / "reports" / "V3" / "03_完整审核资料" / "训练过程记录" / "run_summary.json", SUMMARY_ROOT / "run_summary.json")


def disagreement_analysis(runs: dict[str, dict], dataset: SPMSMGeneDataset, split: dict[str, np.ndarray]) -> None:
    reference_indices = runs[MODEL_IDS[0]]["test_indices"]
    actual = runs[MODEL_IDS[0]]["test_actual"]
    predictions = []
    for model_id in MODEL_IDS:
        if not np.array_equal(reference_indices, runs[model_id]["test_indices"]):
            raise RuntimeError("Test rows differ between models")
        predictions.append(runs[model_id]["test_predicted"])
    stack = np.stack(predictions, axis=1)
    mean_prediction = stack.mean(axis=1)
    disagreement = stack.std(axis=1, ddof=1)
    absolute_error = np.abs(mean_prediction - actual)
    train_std = np.asarray(dataset.targets[split["train"]]).std(axis=0)
    combined_error = np.sqrt(np.mean(((mean_prediction - actual) / train_std) ** 2, axis=1))
    combined_disagreement = np.sqrt(np.mean((disagreement / train_std) ** 2, axis=1))
    rng = np.random.default_rng(SEED)
    high_positions = np.argsort(combined_error)[-500:]
    remaining = np.setdiff1d(np.arange(len(reference_indices)), high_positions)
    random_positions = rng.choice(remaining, size=500, replace=False)
    selected_positions = np.concatenate((random_positions, high_positions))
    selected_types = np.asarray(["random"] * 500 + ["high_ensemble_femm_error"] * 500)

    records = []
    for position, selection_type in zip(selected_positions, selected_types):
        records.append({
            "sample_index": int(reference_indices[position]),
            "selection_type": str(selection_type),
            "actual": {"tavg_nm": float(actual[position, 0]), "delta_t_nm": float(actual[position, 1])},
            "model_predictions": {
                model_id: {"tavg_nm": float(stack[position, mi, 0]), "delta_t_nm": float(stack[position, mi, 1])}
                for mi, model_id in enumerate(MODEL_IDS)
            },
            "model_mean": {"tavg_nm": float(mean_prediction[position, 0]), "delta_t_nm": float(mean_prediction[position, 1])},
            "model_sample_std": {"tavg_nm": float(disagreement[position, 0]), "delta_t_nm": float(disagreement[position, 1])},
            "absolute_femm_error_of_model_mean": {"tavg_nm": float(absolute_error[position, 0]), "delta_t_nm": float(absolute_error[position, 1])},
            "combined_standardized_error": float(combined_error[position]),
            "combined_standardized_disagreement": float(combined_disagreement[position]),
        })
    correlation = {
        "combined_pearson": float(np.corrcoef(combined_error, combined_disagreement)[0, 1]),
        "tavg_pearson": float(np.corrcoef(absolute_error[:, 0], disagreement[:, 0])[0, 1]),
        "delta_t_pearson": float(np.corrcoef(absolute_error[:, 1], disagreement[:, 1])[0, 1]),
    }
    analysis = {
        "status": "complete",
        "model_count": len(MODEL_IDS),
        "models": list(MODEL_IDS),
        "test_population": len(reference_indices),
        "selection": {"random": 500, "high_ensemble_femm_error": 500, "overlap": 0},
        "definition": "mu is the six-model mean; sigma is sample standard deviation with denominator M-1",
        "correlations_full_test": correlation,
        "records": records,
    }
    (DISAGREEMENT_ROOT / "v3_six_model_disagreement_1000_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    for axis, column, name, correlation_key in (
        (axes[0], 0, "Tavg", "tavg_pearson"),
        (axes[1], 1, "DeltaT", "delta_t_pearson"),
    ):
        x = absolute_error[:, column]
        y = disagreement[:, column]
        axis.scatter(x, y, s=4, alpha=0.16, label="all test genes")
        axis.scatter(x[selected_positions], y[selected_positions], s=10, alpha=0.42, color="#ed7d31", label="reviewed 1,000")
        bins = np.quantile(x, np.linspace(0, 1, 21))
        centers, medians = [], []
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (x >= lower) & (x <= upper)
            if mask.any():
                centers.append(float(np.median(x[mask])))
                medians.append(float(np.median(y[mask])))
        axis.plot(centers, medians, color="black", linewidth=2, label="binned median trend")
        axis.set_xlabel(f"Absolute FEMM error of model mean, {name} (N m)")
        axis.set_ylabel(f"Six-model sample std, {name} (N m)")
        axis.set_title(f"{name}: Pearson r={correlation[correlation_key]:.3f}")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("FEMM error versus six-model disagreement on the frozen V3 test set")
    figure.savefig(SUMMARY_ROOT / "femm_error_vs_model_disagreement_v3.png", dpi=190)
    plt.close(figure)


def write_guides(summary: dict) -> None:
    best_id = min(MODEL_IDS, key=lambda item: summary["models"][item]["sets"]["full_test"]["standardized_mse"]["mean"])
    lines = [
        "# V3 40,000-sample training report", "",
        "This report follows the project-02 V3 report layout. All six models use one seed and the same frozen 6,483-sample test set.", "",
        "| Model | Best epoch | Test Tavg MAE (N m) | Test DeltaT MAE (N m) |", "|---|---:|---:|---:|",
    ]
    for model_id in MODEL_IDS:
        model = summary["models"][model_id]
        lines.append(
            f"| {MODEL_LABELS[model_id]} | {model['training']['best_epoch']['mean']:.0f} | "
            f"{model['sets']['full_test']['targets']['tavg']['mae']['mean']:.4f} | "
            f"{model['sets']['full_test']['targets']['delta_t']['mae']['mean']:.4f} |"
        )
    lines += ["", f"Best overall model: **{MODEL_LABELS[best_id]}**.", "", "Artifacts:", "",
              "- `aggregate_metrics.json` and `v3_metrics_mean_std.csv`;",
              "- `training_validation_curves_six_v3.png`;",
              "- `tavg_prediction_six_v3.png` and `tavg_residual_six_v3.png`;",
              "- `delta_t_prediction_six_v3.png` and `delta_t_residual_six_v3.png`;",
              "- `input_representation_test_mae.png`;",
              "- `femm_error_vs_model_disagreement_v3.png` and the associated 1,000-gene JSON;",
              "- `split_novelty_audit.json`: exact-overlap, Hamming-distance and nearest-neighbour baseline audit.", "",
              "There is no V2-versus-V3 plot because project 03 is a new motor/gene definition and has no same-split project-03 V2 baseline. A cross-motor comparison would not be scientifically valid."]
    (SUMMARY_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    guide = """# V3 测试结果审核包

生成日期：2026-09-03

## 实验范围

- 训练集：40,000；提前停止验证集：6,483；冻结测试集：6,483。
- 六种模型/输入组合，统一随机种子 20260903。
- 所有模型同时预测平均转矩 Tavg 和转矩波动 DeltaT，单位均为 N·m。
- 最佳检查点仅由验证集标准化双目标 MSE 决定，测试集在训练结束后评价。

## 目录

- `summary_reports/`：汇总指标、训练曲线、预测图、残差图、数据分布和模型分歧图。
- `per_run_test_results/`：每次运行的配置、历史、验证/测试统计以及逐样本预测。
- `disagreement_analysis/`：500个随机样本加500个高集成误差样本的六模型分歧数据。
- `summary_reports/split_novelty_audit.json`：训练/测试重复、最近汉明距离和最近邻基线审计。

## 审核注意事项

- 本轮按要求只保留一个随机种子，所以汇总表的标准差为0，仅表示“不可由单次运行估计”，不代表真实随机波动为0。
- `full_test_predictions.csv` 的误差定义为“预测值减FEMM真实值”。
- 03是新的SPMSM结构和6×20基因，不存在同一电机、同一测试集的V2基线，因此没有伪造V2/V3对比图。
- 审核包不包含`.pt`权重，以控制体积；全部检查点仍保留在项目模型目录。
"""
    (PACKAGE_ROOT / "AUDIT_GUIDE.md").write_text(guide, encoding="utf-8")


def main() -> None:
    for directory in (SUMMARY_ROOT, PER_RUN_ROOT, DISAGREEMENT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    dataset = SPMSMGeneDataset(DATASET_DIR)
    split = load_split(len(dataset))
    runs = prepare_per_run_results(dataset, split)
    summary = aggregate_runs(runs)
    write_metrics(summary)
    plot_learning_curves(runs)
    plot_predictions_and_residuals(runs)
    plot_model_mae(summary)
    write_distribution_files(dataset, split)
    disagreement_analysis(runs, dataset, split)
    audit_split_novelty()
    write_guides(summary)
    archive_base = PACKAGE_ROOT.parent / "V3完整审核包_可发送"
    archive = shutil.make_archive(str(archive_base), "zip", root_dir=PACKAGE_ROOT.parent, base_dir=PACKAGE_ROOT.name)
    print(f"Audit package: {PACKAGE_ROOT}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
