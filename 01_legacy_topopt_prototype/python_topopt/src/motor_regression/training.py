from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .audit import audit_records, write_audit_report
from .config import resolved_run_paths
from .data import (
    MotorTopologyDataset,
    SampleRecord,
    TargetScaler,
    assign_topology_families,
    group_value,
    grouped_split,
    load_records,
)
from .evaluation import predict, regression_metrics, save_metrics, save_plots, save_predictions
from .models import build_model


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_reused_split(
    records: Sequence[SampleRecord],
    split_path: Path,
    *,
    group_mode: str,
) -> dict[str, list[int]]:
    """Load an existing split by sample key and verify exact, leak-free coverage."""

    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    source = payload.get("samples")
    names = ("train", "val", "test")
    if not isinstance(source, dict) or set(source) != set(names):
        raise ValueError(f"reused split {split_path} must contain train/val/test sample lists")
    index_by_key = {record.key: index for index, record in enumerate(records)}
    if len(index_by_key) != len(records):
        raise ValueError("record keys must be unique before reusing a split")
    requested = [str(key) for name in names for key in source[name]]
    if len(requested) != len(set(requested)):
        raise ValueError(f"reused split {split_path} contains duplicate sample keys")
    missing = set(index_by_key) - set(requested)
    unknown = set(requested) - set(index_by_key)
    if missing or unknown:
        raise ValueError(
            f"reused split does not exactly cover current records: "
            f"missing={len(missing)}, unknown={len(unknown)}"
        )
    split = {name: [index_by_key[str(key)] for key in source[name]] for name in names}
    groups = {
        name: {group_value(records[index], group_mode) for index in indices}
        for name, indices in split.items()
    }
    if (
        not groups["train"].isdisjoint(groups["val"])
        or not groups["train"].isdisjoint(groups["test"])
        or not groups["val"].isdisjoint(groups["test"])
    ):
        raise ValueError(f"reused split {split_path} leaks {group_mode} groups across partitions")
    return split


def prepare_data(
    config: dict[str, Any], project_root: Path, output_directory: Path
) -> tuple[list[SampleRecord], dict[str, list[int]], TargetScaler, dict[str, Any]]:
    specs = config["data"]["targets"]
    names = [str(spec["name"]) for spec in specs]
    categories = [int(value) for value in config["data"]["material_categories"]]
    records = load_records(resolved_run_paths(config, project_root), target_specs=specs)
    records = assign_topology_families(
        records, int(config["split"].get("topology_family_hamming_threshold", 12))
    )
    audit = audit_records(records, target_names=names, expected_categories=categories)
    write_audit_report(audit, output_directory)
    if audit["missing_matrix_values"] or audit["unknown_material_cell_count"]:
        raise ValueError("data audit found invalid material matrix values")
    if any(values["missing_or_nonfinite"] for values in audit["targets"].values()):
        raise ValueError("data audit found missing or non-finite targets")
    reuse_from = config["split"].get("reuse_from")
    reused_split_path: Path | None = None
    if reuse_from:
        candidate = Path(str(reuse_from))
        reused_split_path = (
            candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
        )
        split = load_reused_split(
            records,
            reused_split_path,
            group_mode=str(config["split"]["group_by"]),
        )
    else:
        split = grouped_split(
            records,
            group_mode=str(config["split"]["group_by"]),
            fractions=(
                float(config["split"]["train_fraction"]),
                float(config["split"]["val_fraction"]),
                float(config["split"]["test_fraction"]),
            ),
            seed=int(config["split"]["random_seed"]),
            stratification_bins=int(config["split"].get("target_stratification_bins", 10)),
            band_balance_weight=float(config["split"].get("band_balance_weight", 4.0)),
            target_balance_weight=float(config["split"].get("target_balance_weight", 1.0)),
            search_restarts=int(config["split"].get("search_restarts", 128)),
        )
    train_targets = np.stack([records[index].targets for index in split["train"]])
    scaler = TargetScaler.fit(train_targets, names)
    split_payload = {
        "group_by": config["split"]["group_by"],
        "random_seed": int(config["split"]["random_seed"]),
        "reused_from": str(reused_split_path) if reused_split_path else None,
        "stratification": {
            "fitness_bands": True,
            "target_stratification_bins": int(config["split"].get("target_stratification_bins", 10)),
            "band_balance_weight": float(config["split"].get("band_balance_weight", 4.0)),
            "target_balance_weight": float(config["split"].get("target_balance_weight", 1.0)),
            "search_restarts": int(config["split"].get("search_restarts", 128)),
        },
        "counts": {name: len(indices) for name, indices in split.items()},
        "samples": {
            name: [records[index].key for index in indices] for name, indices in split.items()
        },
        "groups": {
            name: sorted({group_value(records[index], config["split"]["group_by"]) for index in indices})
            for name, indices in split.items()
        },
        "fitness_band_counts": {
            name: {
                f"B{band}": sum(records[index].fitness_band == f"B{band}" for index in indices)
                for band in range(1, 8)
            }
            for name, indices in split.items()
        },
        "target_summary": {
            name: {
                target_name: {
                    "mean": float(np.mean([records[index].targets[target_index] for index in indices])),
                    "std": float(np.std([records[index].targets[target_index] for index in indices])),
                }
                for target_index, target_name in enumerate(names)
            }
            for name, indices in split.items()
        },
    }
    (output_directory / "data_split.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_directory / "target_scaler.json").write_text(
        json.dumps(scaler.as_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return records, split, scaler, audit


def _loader(
    records: Sequence[SampleRecord],
    indices: Sequence[int],
    *,
    categories: Sequence[int],
    scaler: TargetScaler,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = MotorTopologyDataset(records, indices, categories=categories, scaler=scaler)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=False,
        drop_last=False,
    )


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * inputs.shape[0]
            count += inputs.shape[0]
    return total / max(1, count)


def _write_history(history: list[dict[str, float]], output_directory: Path) -> None:
    with (output_directory / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(history)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Standardized MSE")
    axis.set_yscale("log")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "training_curve.png", dpi=180)
    plt.close(figure)


def train_one_model(
    model_name: str,
    *,
    config: dict[str, Any],
    records: Sequence[SampleRecord],
    split: dict[str, list[int]],
    scaler: TargetScaler,
    output_directory: Path,
) -> dict[str, Any]:
    model_output = output_directory / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    seed = int(config["training"]["random_seed"])
    set_reproducible_seed(seed)
    device = resolve_device(str(config["training"].get("device", "auto")))
    categories = [int(value) for value in config["data"]["material_categories"]]
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"].get("data_loader_workers", 0))
    loaders = {
        name: _loader(
            records,
            indices,
            categories=categories,
            scaler=scaler,
            batch_size=batch_size,
            shuffle=name == "train",
            seed=seed,
            num_workers=workers,
        )
        for name, indices in split.items()
    }
    model = build_model(
        model_name,
        input_channels=len(categories),
        output_dim=len(config["data"]["targets"]),
        circular_width=bool(config["model"].get("circular_width_padding", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"]["scheduler"]["factor"]),
        patience=int(config["training"]["scheduler"]["patience"]),
        min_lr=float(config["training"]["scheduler"]["min_lr"]),
    )
    criterion = nn.MSELoss()
    best_loss = math.inf
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint = model_output / "best_checkpoint.pt"
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    for epoch in range(1, max_epochs + 1):
        train_loss = _epoch(model, loaders["train"], criterion, device=device, optimizer=optimizer)
        val_loss = _epoch(model, loaders["val"], criterion, device=device, optimizer=None)
        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": learning_rate}
        )
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            stale = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "input_channels": len(categories),
                    "output_dim": len(config["data"]["targets"]),
                    "material_categories": categories,
                    "matrix_shape": [18, 10],
                    "target_scaler": scaler.as_dict(),
                    "target_specs": config["data"]["targets"],
                    "circular_width_padding": bool(config["model"].get("circular_width_padding", False)),
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= patience:
            break
    _write_history(history, model_output)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    prediction_sets = {
        name: predict(model, loader, device=device, scaler=scaler) for name, loader in loaders.items()
    }
    target_names = [str(spec["name"]) for spec in config["data"]["targets"]]
    metrics = {
        "model": model_name,
        "device": str(device),
        "best_epoch": int(saved["epoch"]),
        "epochs_completed": len(history),
        "best_standardized_validation_mse": float(saved["validation_loss"]),
        "splits": {
            name: regression_metrics(actual, predicted, target_names)
            for name, (actual, predicted) in prediction_sets.items()
        },
        "splits_by_fitness_band": {},
    }
    for split_name, (actual, predicted) in prediction_sets.items():
        bands = np.asarray([records[index].fitness_band for index in split[split_name]])
        metrics["splits_by_fitness_band"][split_name] = {}
        for band in (f"B{index}" for index in range(1, 8)):
            mask = bands == band
            if np.any(mask):
                metrics["splits_by_fitness_band"][split_name][band] = {
                    "sample_count": int(np.count_nonzero(mask)),
                    "targets": regression_metrics(actual[mask], predicted[mask], target_names),
                }
    save_metrics(model_output / "metrics.json", metrics)
    save_predictions(
        model_output / "predictions.csv",
        records=records,
        split_indices=split,
        predictions=prediction_sets,
        target_names=target_names,
        group_mode=str(config["split"]["group_by"]),
    )
    save_plots(
        model_output,
        actual=prediction_sets["test"][0],
        predicted=prediction_sets["test"][1],
        target_names=target_names,
    )
    return metrics


def run_training(
    config: dict[str, Any],
    *,
    project_root: Path,
    output_directory: Path,
    model_names: Sequence[str],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "config.yaml").write_text(
        yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    records, split, scaler, audit = prepare_data(config, project_root, output_directory)
    metrics = {
        name: train_one_model(
            name,
            config=config,
            records=records,
            split=split,
            scaler=scaler,
            output_directory=output_directory,
        )
        for name in model_names
    }
    result = {"audit": audit, "models": metrics}
    (output_directory / "model_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
