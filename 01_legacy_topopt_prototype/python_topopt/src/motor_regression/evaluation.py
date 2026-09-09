from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MotorTopologyDataset, SampleRecord, TargetScaler, group_value


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        residual = predicted[:, index] - actual[:, index]
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        denominator = float(np.sum((actual[:, index] - np.mean(actual[:, index])) ** 2))
        r2 = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else None
        result[name] = {"mae": mae, "rmse": rmse, "r2": r2}
    return result


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    scaler: TargetScaler,
) -> tuple[np.ndarray, np.ndarray]:
    actual_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device)).cpu().numpy()
            actual_parts.append(targets.numpy())
            predicted_parts.append(outputs)
    actual = scaler.inverse_transform(np.concatenate(actual_parts, axis=0))
    predicted = scaler.inverse_transform(np.concatenate(predicted_parts, axis=0))
    return actual, predicted


def save_predictions(
    path: Path,
    *,
    records: Sequence[SampleRecord],
    split_indices: dict[str, list[int]],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    target_names: Sequence[str],
    group_mode: str,
) -> None:
    fields = [
        "sample_key", "run_id", "sample_id", "split", "group", "generation", "fitness_band"
    ]
    for name in target_names:
        fields.extend([f"actual_{name}", f"predicted_{name}", f"residual_{name}"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            actual, predicted = predictions[split_name]
            for position, record_index in enumerate(split_indices[split_name]):
                record = records[record_index]
                row: dict[str, Any] = {
                    "sample_key": record.key,
                    "run_id": record.run_id,
                    "sample_id": record.sample_id,
                    "split": split_name,
                    "group": group_value(record, group_mode),
                    "generation": record.generation,
                    "fitness_band": record.fitness_band,
                }
                for target_index, name in enumerate(target_names):
                    row[f"actual_{name}"] = float(actual[position, target_index])
                    row[f"predicted_{name}"] = float(predicted[position, target_index])
                    row[f"residual_{name}"] = float(predicted[position, target_index] - actual[position, target_index])
                writer.writerow(row)


def save_plots(
    output_directory: Path,
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
    target_names: Sequence[str],
) -> None:
    for index, name in enumerate(target_names):
        lower = float(min(actual[:, index].min(), predicted[:, index].min()))
        upper = float(max(actual[:, index].max(), predicted[:, index].max()))
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(actual[:, index], predicted[:, index], s=22, alpha=0.75)
        axis.plot([lower, upper], [lower, upper], "r--", linewidth=1)
        axis.set_xlabel(f"Actual {name}")
        axis.set_ylabel(f"Predicted {name}")
        axis.set_title(f"Test set: actual vs predicted {name}")
        figure.tight_layout()
        figure.savefig(output_directory / f"actual_vs_predicted_{name}.png", dpi=180)
        plt.close(figure)

        residual = predicted[:, index] - actual[:, index]
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(predicted[:, index], residual, s=22, alpha=0.75)
        axis.axhline(0.0, color="red", linestyle="--", linewidth=1)
        axis.set_xlabel(f"Predicted {name}")
        axis.set_ylabel("Residual (predicted - actual)")
        axis.set_title(f"Test residuals: {name}")
        figure.tight_layout()
        figure.savefig(output_directory / f"residuals_{name}.png", dpi=180)
        plt.close(figure)


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
