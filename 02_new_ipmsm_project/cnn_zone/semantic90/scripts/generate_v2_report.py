"""Aggregate the completed three-seed v2 short test and build its report artifacts."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.training_v2 import (  # noqa: E402
    V2_TRAINING_SPECS,
    build_v2_model,
)


MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v2_test_11700"
REPORT_ROOT = PROJECT / "reports" / "v2_test_11700"
LEGACY_SUMMARY = PROJECT / "reports" / "test" / "run_summary.json"
OVERFIT_SUMMARY = PROJECT / "reports" / "v2_overfit_256" / "summary.json"
TARGETS = ("tavg", "delta_t")
TARGET_LABELS = {"tavg": "Mean torque Tavg", "delta_t": "Torque ripple DeltaT"}
MODEL_LABELS = {
    "logical10/mini_inception_v2": "10 Mini-Inception V2",
    "logical10/resnet20_v2": "10 ResNet20 V2",
    "logical10/small_cnn_v2": "10 SmallCNN V2",
    "semantic224/mini_inception_v2": "224 Mini-Inception V2",
    "semantic224/resnet18_v2": "224 ResNet18 V2",
    "semantic224/vgg16_v2": "224 VGG16 V2",
}
LEGACY_IDS = {
    "logical10/mini_inception_v2": "logical10/mini_inception",
    "logical10/resnet20_v2": "logical10/resnet20",
    "logical10/small_cnn_v2": "logical10/small_cnn",
    "semantic224/mini_inception_v2": "semantic224/mini_inception",
    "semantic224/resnet18_v2": "semantic224/resnet18",
    "semantic224/vgg16_v2": "semantic224/vgg16",
}
CHANGE_SUMMARY = {
    "logical10/mini_inception_v2": "Inception blocks gain projection residuals; two independent 128-unit heads; one pool retained.",
    "logical10/resnet20_v2": "Stage 3 uses stride 1, preserving 5x5; two independent 128-unit heads.",
    "logical10/small_cnn_v2": "Simple trunk retained; DeltaT gains a dedicated 3x3 Conv128 branch; independent heads.",
    "semantic224/mini_inception_v2": "BatchNorm replaced by GroupNorm; residual Inception blocks; 4x4 spatial output and independent heads.",
    "semantic224/resnet18_v2": "Width reduced to 32/64/128/256; GroupNorm; 4x4 spatial output and independent heads.",
    "semantic224/vgg16_v2": "All 13 conv layers retained; GroupNorm; 4x4 spatial output; two independent 256-unit heads.",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def load_results() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for model_id in V2_TRAINING_SPECS:
        family, name = model_id.split("/")
        for result_path in sorted((MODEL_ROOT / family / name).glob("seed_*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "complete":
                result["_directory"] = str(result_path.parent)
                grouped[model_id].append(result)
        if len(grouped[model_id]) != 3:
            raise RuntimeError(f"Expected exactly three complete seeds for {model_id}")
    return grouped


def aggregate(grouped: dict[str, list[dict]]) -> dict:
    output = {"seed_count": 3, "models": {}}
    for model_id, results in grouped.items():
        overall = {}
        for target in TARGETS:
            metric_names = results[0]["metrics"]["overall"][target]
            overall[target] = {
                metric: dict(zip(("mean", "std"), mean_std([
                    result["metrics"]["overall"][target][metric] for result in results
                ])))
                for metric in metric_names
            }
        bands = {}
        for target in TARGETS:
            by_label = []
            for band_index, exemplar in enumerate(results[0]["metrics"]["bands"][target]):
                entries = [result["metrics"]["bands"][target][band_index] for result in results]
                if any(entry["band"] != exemplar["band"] for entry in entries):
                    raise RuntimeError("Band ordering differs between seeds")
                if len({entry["count"] for entry in entries}) != 1:
                    raise RuntimeError("Test band counts differ between seeds")
                record = {"band": exemplar["band"], "count": exemplar["count"]}
                for metric in ("mae", "bias"):
                    values = [entry[metric] for entry in entries if entry[metric] is not None]
                    record[metric] = (
                        dict(zip(("mean", "std"), mean_std(values))) if values else None
                    )
                by_label.append(record)
            bands[target] = by_label
        runtime_metrics = {}
        for key in ("elapsed_seconds", "torch_peak_allocated_mb", "best_epoch"):
            runtime_metrics[key] = dict(zip(("mean", "std"), mean_std([
                float(result[key]) for result in results
            ])))
        for key in (
            "gpu_utilization_percent_mean",
            "gpu_utilization_percent_max",
            "driver_memory_used_mb_peak",
            "power_watts_mean",
            "temperature_celsius_max",
        ):
            runtime_metrics[key] = dict(zip(("mean", "std"), mean_std([
                float(result["gpu_monitor"][key]) for result in results
            ])))
        output["models"][model_id] = {
            "seeds": [result["seed"] for result in results],
            "overall": overall,
            "bands": bands,
            "runtime": runtime_metrics,
            "parameter_count": json.loads(
                (Path(results[0]["_directory"]) / "config.json").read_text(encoding="utf-8")
            )["parameter_count"],
            "physical_batch_size": json.loads(
                (Path(results[0]["_directory"]) / "config.json").read_text(encoding="utf-8")
            )["physical_batch_size"],
            "effective_batch_size": 64,
        }
    return output


def write_metric_csvs(summary: dict) -> None:
    with (REPORT_ROOT / "overall_metrics_mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_id", "target", "metric", "mean", "std", "seeds"))
        for model_id, model in summary["models"].items():
            for target, metrics in model["overall"].items():
                for metric, values in metrics.items():
                    writer.writerow((model_id, target, metric, values["mean"], values["std"], 3))
    with (REPORT_ROOT / "band_metrics_mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_id", "target", "band", "count", "mae_mean", "mae_std", "bias_mean", "bias_std"))
        for model_id, model in summary["models"].items():
            for target, bands in model["bands"].items():
                for band in bands:
                    mae = band["mae"] or {"mean": "", "std": ""}
                    bias = band["bias"] or {"mean": "", "std": ""}
                    writer.writerow((model_id, target, band["band"], band["count"], mae["mean"], mae["std"], bias["mean"], bias["std"]))
    with (REPORT_ROOT / "runtime_gpu_mean_std.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        keys = list(next(iter(summary["models"].values()))["runtime"])
        writer.writerow(("model_id", "parameter_count", "physical_batch", "effective_batch", *[f"{key}_{suffix}" for key in keys for suffix in ("mean", "std")]))
        for model_id, model in summary["models"].items():
            row = [model_id, model["parameter_count"], model["physical_batch_size"], 64]
            for key in keys:
                row.extend((model["runtime"][key]["mean"], model["runtime"][key]["std"]))
            writer.writerow(row)


def plot_learning_curves(grouped: dict[str, list[dict]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, (model_id, results) in zip(axes.flat, grouped.items()):
        for result in results:
            history = json.loads((Path(result["_directory"]) / "history.json").read_text(encoding="utf-8"))
            epochs = [record["epoch"] for record in history]
            axis.plot(epochs, [record["training_standardized_mse"] for record in history], "--", alpha=0.45, linewidth=1)
            axis.plot(epochs, [record["validation_standardized_mse"] for record in history], alpha=0.8, linewidth=1.2, label=str(result["seed"]))
        axis.set_title(MODEL_LABELS[model_id])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Standardized MSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, title="Validation seed")
    figure.suptitle("V2 training (dashed) and validation (solid) curves")
    figure.savefig(REPORT_ROOT / "training_validation_curves_six_v2.png", dpi=180)
    plt.close(figure)


def averaged_predictions(results: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = actual = None
    predictions = []
    for result in results:
        data = np.genfromtxt(
            Path(result["_directory"]) / "test_predictions.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        current_indices = np.asarray(data["sample_index"], dtype=np.int64)
        current_actual = np.column_stack((data["actual_tavg_nm"], data["actual_delta_t_nm"]))
        current_prediction = np.column_stack((data["predicted_tavg_nm"], data["predicted_delta_t_nm"]))
        if indices is None:
            indices, actual = current_indices, current_actual
        elif not np.array_equal(indices, current_indices) or not np.allclose(actual, current_actual):
            raise RuntimeError("Test prediction rows differ between seeds")
        predictions.append(current_prediction)
    return indices, actual, np.mean(np.stack(predictions), axis=0)


def plot_predictions_and_residuals(grouped: dict[str, list[dict]]) -> None:
    for target_index, target in enumerate(TARGETS):
        for kind in ("prediction", "residual"):
            figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
            for axis, (model_id, results) in zip(axes.flat, grouped.items()):
                _, actual, predicted = averaged_predictions(results)
                x = actual[:, target_index]
                y = predicted[:, target_index] if kind == "prediction" else predicted[:, target_index] - x
                axis.scatter(x, y, s=5, alpha=0.32, linewidths=0)
                if kind == "prediction":
                    low, high = min(x.min(), y.min()), max(x.max(), y.max())
                    axis.plot((low, high), (low, high), "r--", linewidth=1)
                    axis.set_ylabel("Three-seed mean prediction (N m)")
                else:
                    axis.axhline(0.0, color="red", linestyle="--", linewidth=1)
                    axis.set_ylabel("Signed error (N m)")
                axis.set_title(MODEL_LABELS[model_id])
                axis.set_xlabel(f"Actual {TARGET_LABELS[target]} (N m)")
                axis.grid(alpha=0.18)
            figure.suptitle(f"{TARGET_LABELS[target]}: three-seed mean {kind}")
            figure.savefig(REPORT_ROOT / f"{target}_{kind}_six_v2.png", dpi=180)
            plt.close(figure)


def plot_legacy_comparison(summary: dict, legacy: dict) -> None:
    legacy_by_id = {item["model_id"]: item for item in legacy["results"]}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    x = np.arange(len(V2_TRAINING_SPECS))
    width = 0.36
    labels = [MODEL_LABELS[model_id].replace(" V2", "") for model_id in V2_TRAINING_SPECS]
    for axis, target in zip(axes, TARGETS):
        old = [legacy_by_id[LEGACY_IDS[model_id]]["test_metrics"][target]["mae"] for model_id in V2_TRAINING_SPECS]
        new = [summary["models"][model_id]["overall"][target]["mae"]["mean"] for model_id in V2_TRAINING_SPECS]
        error = [summary["models"][model_id]["overall"][target]["mae"]["std"] for model_id in V2_TRAINING_SPECS]
        axis.bar(x - width / 2, old, width, label="Legacy single seed")
        axis.bar(x + width / 2, new, width, yerr=error, capsize=3, label="V2 mean +/- std (3 seeds)")
        axis.set_title(f"{TARGET_LABELS[target]} test MAE")
        axis.set_ylabel("MAE (N m)")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(REPORT_ROOT / "legacy_vs_v2_test_mae.png", dpi=190)
    plt.close(figure)


def export_layer_shapes(summary: dict) -> None:
    rows = []
    for model_id, spec in V2_TRAINING_SPECS.items():
        model = build_v2_model(spec).eval()
        hooks = []
        calls = []
        for name, module in model.named_modules():
            if name and not list(module.children()):
                def hook(_module, _inputs, output, layer_name=name):
                    shape = list(output.shape) if isinstance(output, torch.Tensor) else str(type(output))
                    calls.append((layer_name, type(_module).__name__, shape, sum(parameter.numel() for parameter in _module.parameters(recurse=False))))
                hooks.append(module.register_forward_hook(hook))
        shape = (1, spec.input_channels, 10, 10) if spec.input_mode == "logical10" else (1, spec.input_channels, 224, 224)
        with torch.inference_mode():
            output = model(torch.zeros(shape))
        for hook_handle in hooks:
            hook_handle.remove()
        for call_index, (name, layer_type, output_shape, parameters) in enumerate(calls, 1):
            rows.append((model_id, call_index, name, layer_type, "x".join(map(str, output_shape)), parameters))
        if tuple(output.shape) != (1, 2):
            raise RuntimeError(f"Unexpected final shape for {model_id}: {tuple(output.shape)}")
        del model
    with (REPORT_ROOT / "v2_layer_output_shapes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_id", "call_index", "module_path", "layer_type", "output_shape", "direct_parameters"))
        writer.writerows(rows)


def build_markdown(summary: dict, legacy: dict, overfit: dict) -> str:
    legacy_by_id = {item["model_id"]: item for item in legacy["results"]}
    lines = [
        "# Six-model V2 fair short-test report",
        "",
        "All V2 models use the same frozen 11,700/1,500/1,500 indices, the same training-only target scaler, standardized two-target MSE, effective batch 64, and three seeds. The legacy column is a single historical seed, so its apparent difference has no uncertainty estimate.",
        "",
        "## Architecture and result comparison",
        "",
        "| Model | Main V2 change | Parameters old -> V2 | Tavg MAE old -> V2 mean +/- std | DeltaT MAE old -> V2 mean +/- std |",
        "|---|---|---:|---:|---:|",
    ]
    for model_id, model in summary["models"].items():
        old = legacy_by_id[LEGACY_IDS[model_id]]
        tavg = model["overall"]["tavg"]["mae"]
        delta = model["overall"]["delta_t"]["mae"]
        lines.append(
            f"| {MODEL_LABELS[model_id]} | {CHANGE_SUMMARY[model_id]} | {old['parameter_count']:,} -> {model['parameter_count']:,} | "
            f"{old['test_metrics']['tavg']['mae']:.4f} -> {tavg['mean']:.4f} +/- {tavg['std']:.4f} | "
            f"{old['test_metrics']['delta_t']['mae']:.4f} -> {delta['mean']:.4f} +/- {delta['std']:.4f} |"
        )
    lines.extend([
        "",
        "## Validation and runtime",
        "",
        "| Model | Best epoch mean +/- std | Time/run (s) | Torch peak MiB | GPU utilization mean | GPU temperature max (C) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for model_id, model in summary["models"].items():
        run = model["runtime"]
        lines.append(
            f"| {MODEL_LABELS[model_id]} | {run['best_epoch']['mean']:.1f} +/- {run['best_epoch']['std']:.1f} | "
            f"{run['elapsed_seconds']['mean']:.1f} +/- {run['elapsed_seconds']['std']:.1f} | "
            f"{run['torch_peak_allocated_mb']['mean']:.0f} | {run['gpu_utilization_percent_mean']['mean']:.1f}% | "
            f"{run['temperature_celsius_max']['mean']:.1f} |"
        )
    lines.extend([
        "",
        "## Verification",
        "",
        f"- Unit/shape tests passed before training.",
        f"- Fixed 256-sample memorization diagnostic: status `{overfit['status']}`, all six passed = `{overfit['all_six_pass']}`.",
        "- Test evaluation occurred once per run, only after loading the best validation checkpoint.",
        "- Test predictions, full histories, per-run configs, checkpoints, band counts and band errors are retained.",
        "",
        "## Interpretation boundary",
        "",
        "A gain can come from several controlled V2 changes together: architecture, per-model learning rate, normalization, longer validation-controlled training, and equalized effective batch. This experiment does not identify a single causal change. Models whose best epoch is at the maximum may still be under-converged. Mean-regression is assessed using calibration slope, bias, residual plots and sparse-band errors rather than overall MAE alone.",
        "",
        "## Artifacts",
        "",
        "- `aggregate_metrics.json`: all mean/std metrics and band results.",
        "- `overall_metrics_mean_std.csv`: complete target metrics.",
        "- `band_metrics_mean_std.csv`: band count, MAE and bias.",
        "- `runtime_gpu_mean_std.csv`: time, memory, GPU utilization, power and temperature.",
        "- `v2_layer_output_shapes.csv`: actual leaf-layer call order, output size and direct parameter count.",
        "- `training_validation_curves_six_v2.png`: all seed histories.",
        "- `legacy_vs_v2_test_mae.png`: old single-seed versus V2 three-seed comparison.",
        "- `*_prediction_six_v2.png` and `*_residual_six_v2.png`: test behavior using the three-seed mean prediction.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    grouped = load_results()
    summary = aggregate(grouped)
    (REPORT_ROOT / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_metric_csvs(summary)
    plot_learning_curves(grouped)
    plot_predictions_and_residuals(grouped)
    legacy = json.loads(LEGACY_SUMMARY.read_text(encoding="utf-8"))
    overfit = json.loads(OVERFIT_SUMMARY.read_text(encoding="utf-8"))
    plot_legacy_comparison(summary, legacy)
    export_layer_shapes(summary)
    (REPORT_ROOT / "README.md").write_text(build_markdown(summary, legacy, overfit), encoding="utf-8")
    print(f"Report written to {REPORT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
