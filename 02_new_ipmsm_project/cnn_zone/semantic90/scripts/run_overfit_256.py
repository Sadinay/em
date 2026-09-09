"""Overfit the same 256 samples with logical-10x10 and FEM-semantic models."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset


PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.models import (  # noqa: E402
    Logical10MiniInception,
    Semantic90MiniInception,
    parameter_count,
)
from cnn_zone.semantic90.src.topology_renderer import FEMSemanticRenderer  # noqa: E402


DEFAULT_DATASET = PROJECT / "data_zone" / "processed" / "ipmsm_topology_dataset" / "training_corrected_physical_three_state"
DEFAULT_LOOKUP = ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
DEFAULT_SPLIT = ROOT / "outputs" / "splits" / "scheme_a_performance_80_10_10.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "overfit_256"
SEED = 20260822


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_diverse_indices(targets: np.ndarray, train_indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministically sample across a 16x16 rank grid in the two targets."""
    candidate_targets = targets[train_indices]
    rank_t = np.argsort(np.argsort(candidate_targets[:, 0], kind="stable"), kind="stable")
    rank_d = np.argsort(np.argsort(candidate_targets[:, 1], kind="stable"), kind="stable")
    bin_t = np.minimum(15, 16 * rank_t // len(train_indices))
    bin_d = np.minimum(15, 16 * rank_d // len(train_indices))
    cells = 16 * bin_t + bin_d
    rng = np.random.default_rng(seed)
    chosen = []
    for cell in rng.permutation(256):
        members = np.flatnonzero(cells == cell)
        if len(members):
            chosen.append(int(train_indices[rng.choice(members)]))
    remaining = count - len(chosen)
    if remaining > 0:
        pool = np.setdiff1d(train_indices, np.asarray(chosen, dtype=np.int64), assume_unique=False)
        chosen.extend(rng.choice(pool, size=remaining, replace=False).tolist())
    return np.asarray(chosen[:count], dtype=np.int64)


def logical_images(codes: torch.Tensor) -> torch.Tensor:
    # Gene vectors are angular-major. CNN grid axes are [radius, angle].
    grids = codes.reshape(-1, 10, 10).transpose(1, 2)
    return functional.one_hot(grids.to(torch.long), num_classes=3).permute(0, 3, 1, 2).to(torch.float32)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    denominator = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(error**2, axis=0) / denominator
    return {
        "tavg": {"mae": float(mae[0]), "rmse": float(rmse[0]), "r2": float(r2[0])},
        "delta_t": {"mae": float(mae[1]), "rmse": float(rmse[1]), "r2": float(r2[1])},
    }


def save_preflight_audit_images(
    output_dir: Path,
    selected: np.ndarray,
    base_dataset: GeneCodeDataset,
    targets: np.ndarray,
    lookup: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_targets = targets[selected]
    order = np.argsort(selected_targets[:, 0])
    positions = np.unique(np.linspace(0, len(order) - 1, 6, dtype=int))
    audit_indices = selected[order[positions]]
    codes = torch.stack([base_dataset[int(index)][0] for index in audit_indices])
    renderer = FEMSemanticRenderer(lookup)
    with torch.inference_mode():
        images = renderer(codes).numpy()
    semantic_colors = ListedColormap(
        ["#ffffff", "#9dd5ed", "#e52b2b", "#c51b8a", "#56636f", "#e3f3ff", "#8b8b8b", "#df9130", "#f5f5f5"]
    )
    logical_colors = ListedColormap(["#9dd5ed", "#e52b2b", "#56636f"])
    for number, (sample_index, code, rendered) in enumerate(zip(audit_indices, codes.numpy(), images), start=1):
        class_image = np.zeros(rendered.shape[1:], dtype=np.uint8)
        for channel in range(7, -1, -1):
            class_image[rendered[channel] > 0.5] = channel + 1
        logical_grid = code.reshape(10, 10).T
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
        axes[0].imshow(logical_grid, origin="lower", cmap=logical_colors, vmin=-0.5, vmax=2.5, interpolation="nearest")
        axes[0].set_title("10×10 physical topology")
        axes[0].set_xlabel("Angular index"); axes[0].set_ylabel("Radial index")
        axes[0].set_xticks(range(10)); axes[0].set_yticks(range(10))
        axes[1].imshow(class_image, origin="lower", cmap=semantic_colors, vmin=-0.5, vmax=8.5, interpolation="nearest")
        axes[1].set_title("224×224 real-FEM semantic topology")
        axes[1].set_xlabel("Physical x pixel"); axes[1].set_ylabel("Physical y pixel")
        fig.suptitle(
            f"Training audit sample={sample_index}, Tavg={targets[sample_index,0]:.5f}, "
            f"DeltaT={targets[sample_index,1]:.5f}"
        )
        fig.savefig(output_dir / f"training_topology_{number:02d}.png", dpi=180)
        plt.close(fig)


def train_one(
    mode: str,
    subset: Subset,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    lookup: Path,
    output: Path,
    device: torch.device,
    epochs: int,
) -> dict:
    if mode == "logical10":
        model = Logical10MiniInception().to(device)
        batch_size = 32
        renderer = None
    elif mode == "semantic224":
        model = Semantic90MiniInception().to(device)
        batch_size = 8
        renderer = FEMSemanticRenderer(lookup).to(device)
    else:
        raise ValueError(mode)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    evaluation_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = torch.nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mean_device, std_device = target_mean.to(device), target_std.to(device)
    history = []
    best_loss = float("inf")
    best_eval_objective = float("inf")
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for codes, target in loader:
            codes, target = codes.to(device, non_blocking=True), target.to(device, non_blocking=True)
            standardized = (target - mean_device) / std_device
            inputs = logical_images(codes) if renderer is None else renderer(codes)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                prediction = model(inputs)
                loss = criterion(prediction[:, 0], standardized[:, 0]) + criterion(prediction[:, 1], standardized[:, 1])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(codes)
        scheduler.step()
        epoch_loss = total_loss / len(subset)
        best_loss = min(best_loss, epoch_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            model.eval()
            actual_batches, prediction_batches = [], []
            with torch.inference_mode():
                for codes, target in evaluation_loader:
                    codes = codes.to(device)
                    inputs = logical_images(codes) if renderer is None else renderer(codes)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                        standardized_prediction = model(inputs)
                    physical_prediction = standardized_prediction.float() * std_device + mean_device
                    actual_batches.append(target.numpy())
                    prediction_batches.append(physical_prediction.cpu().numpy())
            score = metrics(np.vstack(actual_batches), np.vstack(prediction_batches))
            eval_objective = (
                score["tavg"]["rmse"] / float(target_std[0])
                + score["delta_t"]["rmse"] / float(target_std[1])
            )
            if eval_objective < best_eval_objective:
                best_eval_objective = eval_objective
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(
                {"epoch": epoch, "loss": epoch_loss, "eval_objective": eval_objective,
                 "learning_rate": optimizer.param_groups[0]["lr"], **score}
            )
            print(
                f"{mode} epoch={epoch:03d} loss={epoch_loss:.6f} "
                f"Tavg_MAE={score['tavg']['mae']:.5f} Delta_MAE={score['delta_t']['mae']:.5f} "
                f"R2=({score['tavg']['r2']:.4f},{score['delta_t']['r2']:.4f})",
                flush=True,
            )
            if score["tavg"]["r2"] > 0.995 and score["delta_t"]["r2"] > 0.995:
                break
    elapsed = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    actual_batches, prediction_batches = [], []
    with torch.inference_mode():
        for codes, target in evaluation_loader:
            codes = codes.to(device)
            inputs = logical_images(codes) if renderer is None else renderer(codes)
            prediction = model(inputs).float() * std_device + mean_device
            actual_batches.append(target.numpy()); prediction_batches.append(prediction.cpu().numpy())
    actual, predicted = np.vstack(actual_batches), np.vstack(prediction_batches)
    final_metrics = metrics(actual, predicted)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(), "mode": mode, "target_mean": target_mean,
            "target_std": target_std, "seed": SEED, "parameter_count": parameter_count(model),
        },
        output / "checkpoint.pt",
    )
    np.savez_compressed(output / "predictions.npz", actual=actual, predicted=predicted)
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for index, (name, unit) in enumerate((("Tavg", "N·m"), ("DeltaT", "N·m"))):
        axes[index].scatter(actual[:, index], predicted[:, index], s=18, alpha=0.75)
        low = min(actual[:, index].min(), predicted[:, index].min())
        high = max(actual[:, index].max(), predicted[:, index].max())
        axes[index].plot([low, high], [low, high], "r--", linewidth=1.2)
        axes[index].set_xlabel(f"Actual {name} ({unit})"); axes[index].set_ylabel(f"Predicted {name} ({unit})")
        axes[index].set_title(f"{mode}: 256-sample memorization")
    fig.savefig(output / "prediction_comparison.png", dpi=180)
    plt.close(fig)
    return {
        "mode": mode,
        "samples": len(subset),
        "epochs_completed": int(history[-1]["epoch"]),
        "elapsed_seconds": elapsed,
        "parameter_count": parameter_count(model),
        "best_training_loss": best_loss,
        "best_evaluation_objective": best_eval_objective,
        "metrics_on_same_256_training_samples": final_metrics,
        "purpose": "pipeline/memorization check only; not a generalization result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--modes", nargs="+", choices=["logical10", "semantic224"], default=["logical10", "semantic224"])
    args = parser.parse_args()
    configure_reproducibility(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dataset = GeneCodeDataset(args.dataset, "all")
    targets = np.asarray(base_dataset.targets)
    with np.load(args.split) as split:
        train_indices = np.asarray(split["train"], dtype=np.int64)
    selected = choose_diverse_indices(targets, train_indices, 256, SEED)
    subset = Subset(base_dataset, selected.tolist())
    selected_targets = torch.from_numpy(np.asarray(targets[selected], dtype=np.float32))
    target_mean = selected_targets.mean(dim=0)
    target_std = selected_targets.std(dim=0, unbiased=False).clamp_min(1e-8)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "selected_sample_indices.npy", selected)
    (args.output / "target_scaler.json").write_text(
        json.dumps({"mean": target_mean.tolist(), "std": target_std.tolist(), "source": "only the selected 256 training samples"}, indent=2),
        encoding="utf-8",
    )
    save_preflight_audit_images(args.output / "topology_audit_224", selected, base_dataset, targets, args.lookup)
    results = []
    print(f"device={device}, samples=256, target_mean={target_mean.tolist()}, target_std={target_std.tolist()}")
    for mode in args.modes:
        configure_reproducibility(SEED)
        result = train_one(mode, subset, target_mean, target_std, args.lookup, args.output / mode, device, args.epochs)
        results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "status": "complete",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "same_sample_indices_for_all_modes": True,
        "target_scaler_uses_only_selected_training_samples": True,
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
