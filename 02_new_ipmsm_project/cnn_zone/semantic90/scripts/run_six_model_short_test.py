"""Train the six audited CNNs on one fixed 11,700-sample Scheme-A subset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.models import (  # noqa: E402
    Logical10MiniInception,
    Logical10ResNet20,
    Logical10SmallCNN,
    Semantic90MiniInception,
    Semantic90ResNet18,
    Semantic90VGG16,
    parameter_count,
)
from cnn_zone.semantic90.src.topology_renderer import FEMSemanticRenderer  # noqa: E402


DEFAULT_DATASET = (
    PROJECT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)
DEFAULT_LOOKUP = ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
DEFAULT_SPLIT = ROOT / "outputs" / "splits" / "scheme_a_performance_80_10_10.npz"
DEFAULT_MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "test_11700"
DEFAULT_REPORT_ROOT = PROJECT / "reports" / "test"
SEED = 20260822
TARGET_NAMES = ("tavg", "delta_t")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    input_mode: str
    epochs: int
    batch_size: int


MODEL_SPECS = (
    ModelSpec("mini_inception", "logical10", "logical10", 10, 64),
    ModelSpec("resnet20", "logical10", "logical10", 10, 64),
    ModelSpec("small_cnn", "logical10", "logical10", 10, 64),
    ModelSpec("mini_inception", "semantic224", "semantic224", 2, 8),
    ModelSpec("resnet18", "semantic224", "semantic224", 2, 8),
    ModelSpec("vgg16", "semantic224", "semantic224", 2, 8),
)


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def logical_images(codes: torch.Tensor) -> torch.Tensor:
    """Convert angular-major material vectors to `[B,3,radius,angle]`."""
    grids = codes.reshape(-1, 10, 10).transpose(1, 2)
    return functional.one_hot(grids.long(), num_classes=3).permute(0, 3, 1, 2).float()


def stratified_subsample(
    targets: np.ndarray, candidate_indices: np.ndarray, count: int, seed: int, bins: int = 10
) -> np.ndarray:
    """Preserve the two-target empirical distribution using rank-based 2-D strata."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if count > len(candidate_indices):
        raise ValueError(f"Requested {count} from only {len(candidate_indices)} candidates")
    values = np.asarray(targets[candidate_indices], dtype=np.float64)
    rank_a = np.argsort(np.argsort(values[:, 0], kind="stable"), kind="stable")
    rank_b = np.argsort(np.argsort(values[:, 1], kind="stable"), kind="stable")
    strata = bins * np.minimum(bins - 1, bins * rank_a // len(values))
    strata += np.minimum(bins - 1, bins * rank_b // len(values))
    sizes = np.bincount(strata, minlength=bins * bins)
    exact = count * sizes / len(values)
    quotas = np.floor(exact).astype(np.int64)
    remaining = count - int(quotas.sum())
    priority = np.argsort(-(exact - quotas), kind="stable")
    for cell in priority[:remaining]:
        if quotas[cell] < sizes[cell]:
            quotas[cell] += 1
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cell, quota in enumerate(quotas):
        members = candidate_indices[strata == cell]
        if quota:
            chosen.extend(rng.choice(members, size=int(quota), replace=False).tolist())
    chosen_array = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen_array)
    if len(chosen_array) != count:
        raise RuntimeError(f"Stratified sampling produced {len(chosen_array)} instead of {count}")
    return chosen_array


def build_model(spec: ModelSpec) -> nn.Module:
    key = (spec.family, spec.name)
    factories = {
        ("logical10", "mini_inception"): Logical10MiniInception,
        ("logical10", "resnet20"): Logical10ResNet20,
        ("logical10", "small_cnn"): Logical10SmallCNN,
        ("semantic224", "mini_inception"): Semantic90MiniInception,
        ("semantic224", "resnet18"): Semantic90ResNet18,
        ("semantic224", "vgg16"): Semantic90VGG16,
    }
    return factories[key]()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    error = predicted - actual
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    denominator = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(error**2, axis=0) / np.maximum(denominator, 1e-12)
    return {
        name: {"mae": float(mae[i]), "rmse": float(rmse[i]), "r2": float(r2[i])}
        for i, name in enumerate(TARGET_NAMES)
    }


def make_inputs(
    codes: torch.Tensor, input_mode: str, renderer: FEMSemanticRenderer | None
) -> torch.Tensor:
    if input_mode == "logical10":
        return logical_images(codes)
    if renderer is None:
        raise RuntimeError("The semantic renderer was not initialized")
    return renderer(codes)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    input_mode: str,
    renderer: FEMSemanticRenderer | None,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    actual_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []
    index_batches: list[np.ndarray] = []
    for dataset_indices, codes, targets in loader:
        codes = codes.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        inputs = make_inputs(codes, input_mode, renderer)
        if input_mode == "semantic224":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            standardized_prediction = model(inputs)
            standardized_actual = (targets_device - target_mean) / target_std
            loss = functional.mse_loss(standardized_prediction, standardized_actual)
        prediction = standardized_prediction.float() * target_std + target_mean
        total_loss += float(loss) * len(codes)
        actual_batches.append(targets.numpy())
        predicted_batches.append(prediction.cpu().numpy())
        index_batches.append(dataset_indices.numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(index_batches),
        np.vstack(actual_batches),
        np.vstack(predicted_batches),
    )


class IndexedSubset(Subset):
    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        codes, targets = self.dataset[global_index]
        return torch.tensor(global_index, dtype=torch.int64), codes, targets

    def __getitems__(self, items: list[int]):
        return [self[item] for item in items]


def save_predictions(path: Path, indices: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "sample_index",
                "actual_tavg_nm",
                "predicted_tavg_nm",
                "error_tavg_nm",
                "actual_delta_t_nm",
                "predicted_delta_t_nm",
                "error_delta_t_nm",
            )
        )
        for sample_index, truth, prediction in zip(indices, actual, predicted):
            writer.writerow(
                (
                    int(sample_index),
                    float(truth[0]),
                    float(prediction[0]),
                    float(prediction[0] - truth[0]),
                    float(truth[1]),
                    float(prediction[1]),
                    float(prediction[1] - truth[1]),
                )
            )


def save_prediction_figure(
    report_path: Path, label: str, actual: np.ndarray, predicted: np.ndarray, metrics: dict
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for index, (title, unit) in enumerate((("Mean torque", "N m"), ("Torque ripple DeltaT", "N m"))):
        axes[index].hexbin(actual[:, index], predicted[:, index], gridsize=38, mincnt=1, cmap="viridis")
        low = float(min(actual[:, index].min(), predicted[:, index].min()))
        high = float(max(actual[:, index].max(), predicted[:, index].max()))
        axes[index].plot((low, high), (low, high), "r--", linewidth=1.2)
        key = TARGET_NAMES[index]
        axes[index].set_title(
            f"{title}\nMAE={metrics[key]['mae']:.4f}, R2={metrics[key]['r2']:.4f}"
        )
        axes[index].set_xlabel(f"Actual ({unit})")
        axes[index].set_ylabel(f"Predicted ({unit})")
        axes[index].grid(alpha=0.18)
    fig.suptitle(f"Independent test subset: {label}")
    fig.savefig(report_path, dpi=170)
    plt.close(fig)


def train_model(
    spec: ModelSpec,
    train_set: IndexedSubset,
    validation_set: IndexedSubset,
    test_set: IndexedSubset,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    lookup: Path,
    model_dir: Path,
    report_root: Path,
    device: torch.device,
    seed: int,
) -> dict:
    configure_runtime(seed)
    model = build_model(spec).to(device)
    if spec.input_mode == "semantic224":
        model = model.to(memory_format=torch.channels_last)
        renderer: FEMSemanticRenderer | None = FEMSemanticRenderer(lookup).to(device)
    else:
        renderer = None
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    generator = torch.Generator().manual_seed(seed)
    common_loader = {
        "batch_size": spec.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common_loader)
    validation_loader = DataLoader(validation_set, shuffle=False, **common_loader)
    test_loader = DataLoader(test_set, shuffle=False, **common_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **asdict(spec),
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss": "mean MSE of two standardized targets",
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)",
        "seed": seed,
        "parameter_count": parameter_count(model),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "test_samples": len(test_set),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_validation_loss = float("inf")
    history: list[dict] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"START {spec.family}/{spec.name}: epochs={spec.epochs}, batch={spec.batch_size}, "
        f"parameters={parameter_count(model):,}",
        flush=True,
    )
    for epoch in range(1, spec.epochs + 1):
        model.train()
        running_loss = 0.0
        for _, codes, targets in train_loader:
            codes = codes.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            standardized = (targets - target_mean_device) / target_std_device
            inputs = make_inputs(codes, spec.input_mode, renderer)
            if spec.input_mode == "semantic224":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                prediction = model(inputs)
                loss = functional.mse_loss(prediction, standardized)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            running_loss += float(loss.detach()) * len(codes)
        train_loss = running_loss / len(train_set)
        validation_loss, _, validation_actual, validation_predicted = evaluate(
            model,
            validation_loader,
            spec.input_mode,
            renderer,
            target_mean_device,
            target_std_device,
            device,
        )
        scheduler.step(validation_loss)
        validation_metrics = regression_metrics(validation_actual, validation_predicted)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_metrics": validation_metrics,
        }
        history.append(row)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_family": spec.family,
                    "model_name": spec.name,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "target_mean": target_mean.cpu(),
                    "target_std": target_std.cpu(),
                    "config": config,
                },
                model_dir / "best_checkpoint.pt",
            )
        print(
            f"  epoch {epoch:02d}/{spec.epochs}: train={train_loss:.5f}, val={validation_loss:.5f}, "
            f"Tavg_MAE={validation_metrics['tavg']['mae']:.4f}, "
            f"DeltaT_MAE={validation_metrics['delta_t']['mae']:.4f}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    checkpoint = torch.load(model_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_indices, test_actual, test_predicted = evaluate(
        model,
        test_loader,
        spec.input_mode,
        renderer,
        target_mean_device,
        target_std_device,
        device,
    )
    test_metrics = regression_metrics(test_actual, test_predicted)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
    )
    result = {
        "model_id": f"{spec.family}/{spec.name}",
        "epochs": spec.epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": best_validation_loss,
        "test_standardized_mse": test_loss,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": peak_memory_mb,
        "parameter_count": parameter_count(model),
        "checkpoint": str((model_dir / "best_checkpoint.pt").resolve()),
    }
    (model_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (model_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_predictions(model_dir / "test_predictions.csv", test_indices, test_actual, test_predicted)
    figure_name = f"{spec.family}_{spec.name}_prediction.png"
    save_prediction_figure(
        report_root / figure_name,
        f"{spec.family}/{spec.name}",
        test_actual,
        test_predicted,
        test_metrics,
    )
    print(
        f"DONE {spec.family}/{spec.name}: {elapsed:.1f}s, test MAE="
        f"({test_metrics['tavg']['mae']:.4f}, {test_metrics['delta_t']['mae']:.4f})",
        flush=True,
    )
    del model, optimizer, amp_scaler, renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_comparison_figure(results: list[dict], report_root: Path) -> None:
    labels = [result["model_id"].replace("logical10/", "10/").replace("semantic224/", "224/") for result in results]
    x = np.arange(len(labels))
    width = 0.36
    tavg = [result["test_metrics"]["tavg"]["mae"] for result in results]
    delta = [result["test_metrics"]["delta_t"]["mae"] for result in results]
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, tavg, width, label="Mean torque MAE")
    ax.bar(x + width / 2, delta, width, label="Torque ripple DeltaT MAE")
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("MAE (N m)")
    ax.set_title("Six-model short-test comparison (independent Scheme-A test subset)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(report_root / "six_model_mae_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--train-count", type=int, default=11_700)
    parser.add_argument("--validation-count", type=int, default=1_500)
    parser.add_argument("--test-count", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional IDs such as logical10/resnet20 or semantic224/vgg16",
    )
    args = parser.parse_args()
    configure_runtime(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this six-model training run")
    device = torch.device("cuda:0")
    args.model_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    base_dataset = GeneCodeDataset(args.dataset, "all")
    targets = np.asarray(base_dataset.targets)
    with np.load(args.split) as split:
        train_pool = np.asarray(split["train"], dtype=np.int64)
        validation_pool = np.asarray(split["validation"], dtype=np.int64)
        test_pool = np.asarray(split["test"], dtype=np.int64)
    train_indices = stratified_subsample(targets, train_pool, args.train_count, args.seed)
    validation_indices = stratified_subsample(targets, validation_pool, args.validation_count, args.seed + 1)
    test_indices = stratified_subsample(targets, test_pool, args.test_count, args.seed + 2)
    if any(
        len(np.intersect1d(first, second))
        for first, second in (
            (train_indices, validation_indices),
            (train_indices, test_indices),
            (validation_indices, test_indices),
        )
    ):
        raise RuntimeError("Subsample leakage detected")
    split_output = args.report_root / "fixed_scheme_a_subsample_11700_1500_1500.npz"
    np.savez_compressed(
        split_output,
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
        source_split=str(args.split.resolve()),
        seed=args.seed,
    )
    train_targets = torch.from_numpy(np.asarray(targets[train_indices], dtype=np.float32))
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0, unbiased=False).clamp_min(1e-8)
    scaler_record = {
        "target_order": ["Tavg_Nm", "DeltaT_Nm"],
        "mean": target_mean.tolist(),
        "std": target_std.tolist(),
        "statistics_source": "only the fixed 11,700-sample training subset",
    }
    (args.report_root / "target_scaler.json").write_text(
        json.dumps(scaler_record, indent=2), encoding="utf-8"
    )
    train_set = IndexedSubset(base_dataset, train_indices.tolist())
    validation_set = IndexedSubset(base_dataset, validation_indices.tolist())
    test_set = IndexedSubset(base_dataset, test_indices.tolist())
    selected_ids = set(args.models or ())
    specs = [spec for spec in MODEL_SPECS if not selected_ids or f"{spec.family}/{spec.name}" in selected_ids]
    unknown = selected_ids - {f"{spec.family}/{spec.name}" for spec in MODEL_SPECS}
    if unknown:
        raise ValueError(f"Unknown model IDs: {sorted(unknown)}")
    run_manifest = {
        "status": "running",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "source_split": str(args.split.resolve()),
        "subsample_split": str(split_output.resolve()),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "test_samples": len(test_set),
        "same_indices_for_all_models": True,
        "target_scaler_training_only": True,
        "results": [],
    }
    summary_path = args.report_root / "run_summary.json"
    summary_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(
        f"CUDA confirmed: {torch.cuda.get_device_name(0)}; samples="
        f"{len(train_set)}/{len(validation_set)}/{len(test_set)}; models={len(specs)}",
        flush=True,
    )
    for model_number, spec in enumerate(specs):
        model_dir = args.model_root / spec.family / spec.name
        result = train_model(
            spec,
            train_set,
            validation_set,
            test_set,
            target_mean,
            target_std,
            args.lookup,
            model_dir,
            args.report_root,
            device,
            args.seed + model_number,
        )
        run_manifest["results"].append(result)
        summary_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    save_comparison_figure(run_manifest["results"], args.report_root)
    run_manifest["status"] = "complete"
    summary_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
