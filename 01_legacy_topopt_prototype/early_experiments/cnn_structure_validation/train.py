from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from model import StructureCNN, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mse": mse, "rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(error))),
        "r2": 1 - float(np.sum(error ** 2)) / denominator if denominator else float("nan"),
    }


def train(dataset: Path, output: Path, config: dict[str, Any], mode: str) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    data = np.load(dataset, allow_pickle=False)
    images = data["images"].astype(np.float32)
    t_min = data["t_min"].astype(np.float32)
    targets = data["target_j"].astype(np.float32)
    sample_ids = data["sample_ids"].astype(str)
    splits = data["splits"].astype(str)
    topology_ids = data["topology_ids"].astype(str)
    train_indices = np.where(splits == "train")[0]
    validation_indices = np.where(splits == "validation")[0]
    if mode == "overfit":
        train_indices = train_indices[: int(config["training"]["overfit_samples"])]
        validation_indices = train_indices
        epochs = int(config["training"]["overfit_epochs"])
        patience = epochs + 1
    else:
        epochs = int(config["training"]["epochs"])
        patience = int(config["training"]["patience"])
    t_mean = float(t_min[train_indices].mean())
    t_std = float(t_min[train_indices].std()) or 1.0
    y_mean = float(targets[train_indices].mean())
    y_std = float(targets[train_indices].std()) or 1.0
    image_train = torch.from_numpy(images[train_indices])
    t_train = torch.from_numpy((t_min[train_indices] - t_mean) / t_std)
    y_train = torch.from_numpy((targets[train_indices] - y_mean) / y_std)
    image_val = torch.from_numpy(images[validation_indices])
    t_val = torch.from_numpy((t_min[validation_indices] - t_mean) / t_std)
    y_val = torch.from_numpy((targets[validation_indices] - y_mean) / y_std)
    model = StructureCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    criterion = torch.nn.MSELoss()
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(image_train, t_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(image_val, t_val), y_val)
        train_value, val_value = float(loss.item()), float(val_loss.item())
        history.append({"epoch": epoch, "train_mse_scaled": train_value, "validation_mse_scaled": val_value})
        if val_value < best_loss - float(config["training"]["min_delta"]):
            best_loss, best_epoch, stale = val_value, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint created")
    model.load_state_dict(best_state)
    model.eval()
    predictions: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, float]] = {}
    for split, indices in (("train", train_indices), ("validation", validation_indices)):
        image_tensor = torch.from_numpy(images[indices])
        t_tensor = torch.from_numpy((t_min[indices] - t_mean) / t_std)
        with torch.no_grad():
            scaled = model(image_tensor, t_tensor).numpy()
        predicted = scaled * y_std + y_mean
        actual = targets[indices]
        split_metrics[split] = regression_metrics(actual, predicted)
        for index, truth, estimate in zip(indices, actual.reshape(-1), predicted.reshape(-1)):
            predictions.append({
                "sample_id": sample_ids[index], "topology_id": topology_ids[index], "split": split,
                "actual_j": float(truth), "predicted_j": float(estimate),
                "absolute_error": float(abs(estimate - truth)),
            })
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": mode, "seed": config["seed"], "model_parameters": parameter_count(model),
        "train_samples": len(train_indices), "validation_samples": len(validation_indices),
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "input_image_shape": list(images.shape[1:]), "channel_order": ["Background", "Air", "Iron", "Copper", "Magnet"],
        "t_min_mean": t_mean, "t_min_std": t_std, "target_mean": y_mean, "target_std": y_std,
        "train_metrics": split_metrics["train"], "validation_metrics": split_metrics["validation"],
    }
    torch.save({
        "model_state_dict": model.state_dict(), "config": config,
        "t_min_mean": t_mean, "t_min_std": t_std, "target_mean": y_mean, "target_std": y_std,
        "input_image_shape": list(images.shape[1:]), "channel_order": ["Background", "Air", "Iron", "Copper", "Magnet"],
    }, output / "model.pt")
    with (output / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_history(history, output / "loss_curve.png", mode)
    if mode == "full":
        plot_validation(predictions, output / "validation_comparison.png", output / "validation_comparison.svg")
    return result


def plot_history(history: list[dict[str, Any]], path: Path, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in history], [r["train_mse_scaled"] for r in history], label="train")
    ax.plot([r["epoch"] for r in history], [r["validation_mse_scaled"] for r in history], label="validation")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Scaled MSE")
    ax.set_title(f"Structure CNN loss ({mode})"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_validation(rows: list[dict[str, Any]], png: Path, svg: Path) -> None:
    values = [r for r in rows if r["split"] == "validation"]
    x = np.arange(1, len(values) + 1)
    actual = np.asarray([r["actual_j"] for r in values])
    predicted = np.asarray([r["predicted_j"] for r in values])
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (truth, estimate) in enumerate(zip(actual, predicted), 1):
        ax.plot([i, i], [truth, estimate], color="#aab2bd", linewidth=0.9)
    ax.plot(x, predicted, color="#1565c0", marker="o", label="Structure CNN prediction")
    ax.scatter(x, actual, marker="D", s=48, facecolors="none", edgecolors="#e53935", linewidths=1.5, label="Actual J", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([r["sample_id"].replace("SAMPLE_", "") for r in values], rotation=45, ha="right")
    ax.set_xlabel("Independent validation topology"); ax.set_ylabel("Objective value J")
    ax.set_title("Independent validation: structure CNN vs. actual J")
    ax.grid(linestyle="--", alpha=0.28); ax.legend()
    ax.text(0.985, 0.03, f"MAE = {np.mean(abs(predicted-actual)):.5f}\nRMSE = {np.sqrt(np.mean((predicted-actual)**2)):.5f}",
            transform=ax.transAxes, ha="right", va="bottom", bbox={"boxstyle":"round,pad=0.35","facecolor":"white","alpha":0.85})
    fig.tight_layout(); fig.savefig(png, dpi=180); fig.savefig(svg); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conv2D on material topology images.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("overfit", "full"), default="full")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = train(args.dataset.resolve(), args.output.resolve(), config, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
