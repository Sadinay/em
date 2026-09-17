"""Reproducible two-target training for the SPMSM V3 experiment matrix."""

from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Sampler

from .dataset import IndexedSubset, SPMSMGeneDataset
from .models_v2 import (
    Logical6x20MiniInceptionV2,
    Logical6x20ResNet20V2,
    Logical6x20SmallCNNV2,
    Semantic224MiniInceptionV2,
    Semantic224ResNet18V2,
    Semantic224VGG16V2,
    parameter_count,
)
from .training_inputs import TorchGap90Renderer, TorchSemanticRenderer, enable_angular_circular_padding, logical6x20


TARGET_NAMES = ("tavg", "delta_t")


@dataclass(frozen=True)
class TrainingSpec:
    experiment_id: str
    input_mode: str
    architecture: str
    lookup: str | None
    learning_rate: float
    weight_decay: float
    max_epochs: int
    minimum_epochs: int
    early_stopping_patience: int
    scheduler_patience: int
    physical_batch_size: int
    effective_batch_size: int = 64
    circular_angular_padding: bool = False
    gap_mapping: str | None = None


class EffectiveBatchSampler(Sampler[list[int]]):
    def __init__(self, sample_count: int, physical: int, effective: int, seed: int, epoch: int) -> None:
        self.sample_count = sample_count
        self.physical = physical
        self.effective = effective
        self.seed = seed
        self.epoch = epoch

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.sample_count)
        for group_start in range(0, self.sample_count, self.effective):
            group = order[group_start : group_start + self.effective]
            for start in range(0, len(group), self.physical):
                yield group[start : start + self.physical].tolist()

    def __len__(self) -> int:
        full, remainder = divmod(self.sample_count, self.effective)
        return full * math.ceil(self.effective / self.physical) + (math.ceil(remainder / self.physical) if remainder else 0)

    @property
    def optimizer_steps(self) -> int:
        return math.ceil(self.sample_count / self.effective)


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_model(spec: TrainingSpec) -> nn.Module:
    if spec.architecture == "small_cnn_v2":
        model = Logical6x20SmallCNNV2()
    elif spec.architecture == "resnet20_v2" and spec.input_mode in ("logical6x20", "gap90_6x97"):
        model = Logical6x20ResNet20V2()
    elif spec.architecture == "mini_inception_v2" and spec.input_mode == "logical6x20":
        model = Logical6x20MiniInceptionV2()
    elif spec.architecture == "mini_inception_v2":
        model = Semantic224MiniInceptionV2()
    elif spec.architecture == "vgg16_v2":
        model = Semantic224VGG16V2()
    elif spec.architecture == "resnet18_v2":
        model = Semantic224ResNet18V2()
    else:
        raise KeyError(f"Unsupported architecture/input combination: {spec}")
    if spec.circular_angular_padding:
        enable_angular_circular_padding(model)
    return model


def build_renderer(spec: TrainingSpec, root: Path, device: torch.device):
    if spec.input_mode == "logical6x20":
        return None
    if spec.input_mode == "gap90_6x97":
        if spec.gap_mapping is None:
            raise ValueError(f"Missing gap mapping for {spec.experiment_id}")
        return TorchGap90Renderer(root / spec.gap_mapping).to(device)
    if spec.lookup is None:
        raise ValueError(f"Missing lookup for {spec.experiment_id}")
    return TorchSemanticRenderer(root / spec.lookup).to(device)


def make_inputs(bits: torch.Tensor, spec: TrainingSpec, renderer) -> torch.Tensor:
    if spec.input_mode == "logical6x20":
        return logical6x20(bits)
    if spec.input_mode == "gap90_6x97":
        return renderer(bits)
    return renderer(bits).contiguous(memory_format=torch.channels_last)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    output = {}
    for column, name in enumerate(TARGET_NAMES):
        truth = actual[:, column]
        estimate = predicted[:, column]
        error = estimate - truth
        denominator = max(float(np.sum((truth - truth.mean()) ** 2)), 1e-12)
        output[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "r2": float(1.0 - np.sum(error**2) / denominator),
            "bias": float(np.mean(error)),
            "absolute_error_p95": float(np.percentile(np.abs(error), 95)),
        }
    return output


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset: SPMSMGeneDataset,
    indices: np.ndarray,
    spec: TrainingSpec,
    renderer,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(
        IndexedSubset(dataset, indices),
        batch_size=max(spec.physical_batch_size, 64 if spec.input_mode in ("logical6x20", "gap90_6x97") else spec.physical_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    total_loss = 0.0
    count = 0
    all_indices, all_actual, all_predicted = [], [], []
    mean_device = target_mean.to(device)
    std_device = target_std.to(device)
    for sample_indices, bits, targets in loader:
        bits = bits.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        inputs = make_inputs(bits, spec, renderer)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            standardized = model(inputs)
            standardized_target = (targets_device - mean_device) / std_device
            total_loss += float(functional.mse_loss(standardized, standardized_target, reduction="sum"))
        predicted = standardized.float() * std_device + mean_device
        count += len(bits)
        all_indices.append(sample_indices.numpy())
        all_actual.append(targets.numpy())
        all_predicted.append(predicted.cpu().numpy())
    actual = np.vstack(all_actual)
    predicted = np.vstack(all_predicted)
    return total_loss / (count * 2), regression_metrics(actual, predicted), np.concatenate(all_indices), actual, predicted


def save_predictions(path: Path, indices: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> None:
    values = np.column_stack((indices, actual[:, 0], predicted[:, 0], predicted[:, 0] - actual[:, 0], actual[:, 1], predicted[:, 1], predicted[:, 1] - actual[:, 1]))
    np.savetxt(
        path,
        values,
        delimiter=",",
        header="sample_index,actual_tavg,predicted_tavg,error_tavg,actual_delta_t,predicted_delta_t,error_delta_t",
        comments="",
    )


def probe_spec(spec: TrainingSpec, root: Path, device: torch.device) -> dict:
    configure_reproducibility(10101)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(spec).to(device)
    renderer = build_renderer(spec, root, device)
    if spec.input_mode not in ("logical6x20", "gap90_6x97"):
        model = model.to(memory_format=torch.channels_last)
    bits = torch.randint(0, 2, (spec.physical_batch_size, 120), dtype=torch.uint8, device=device)
    targets = torch.zeros((spec.physical_batch_size, 2), device=device)
    model.train()
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        prediction = model(make_inputs(bits, spec, renderer))
        loss = functional.mse_loss(prediction, targets)
    loss.backward()
    if prediction.shape != targets.shape or not torch.isfinite(loss):
        raise RuntimeError("Probe produced invalid output")
    torch.cuda.synchronize()
    result = {
        "experiment_id": spec.experiment_id,
        "physical_batch_size": spec.physical_batch_size,
        "parameter_count": parameter_count(model),
        "output_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "status": "passed",
    }
    del model, renderer, bits, targets, prediction, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_one(
    spec: TrainingSpec,
    seed: int,
    dataset: SPMSMGeneDataset,
    split: dict[str, np.ndarray],
    root: Path,
    output_dir: Path,
    device: torch.device,
    resume: bool,
    deadline: float | None,
) -> dict:
    configure_reproducibility(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(dataset.targets[split["train"]], dtype=np.float32)
    target_mean = torch.from_numpy(targets.mean(axis=0))
    target_std = torch.from_numpy(targets.std(axis=0).clip(min=1e-8))
    model = build_model(spec).to(device)
    renderer = build_renderer(spec, root, device)
    if spec.input_mode not in ("logical6x20", "gap90_6x97"):
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=spec.scheduler_patience, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=128.0)
    mean_device = target_mean.to(device)
    std_device = target_std.to(device)
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    starting_epoch = 1
    elapsed_before = 0.0
    last_path = output_dir / "last_checkpoint.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["experiment_id"] != spec.experiment_id or int(state["seed"]) != seed:
            raise RuntimeError("Resume checkpoint identity mismatch")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        no_improvement = int(state["no_improvement"])
        starting_epoch = int(state["epoch"]) + 1
        elapsed_before = float(state["elapsed_seconds"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
        print(f"RESUME {spec.experiment_id} seed={seed} from epoch {starting_epoch - 1}", flush=True)
    elif last_path.exists() and not resume:
        raise RuntimeError(f"Existing checkpoint requires --resume: {last_path}")

    config = {
        **asdict(spec),
        "seed": seed,
        "train_samples": int(len(split["train"])),
        "validation_samples": int(len(split["validation"])),
        "test_samples": int(len(split["test"])),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "target_scaler_source": "training set only",
        "parameter_count": parameter_count(model),
        "optimizer": "AdamW",
        "loss": "mean MSE over independently standardized Tavg and DeltaT",
        "best_checkpoint_selection": "validation standardized MSE only",
    }
    atomic_json(output_dir / "config.json", config)
    started = perf_counter()
    print(f"START {spec.experiment_id} seed={seed} params={parameter_count(model):,} physical={spec.physical_batch_size} effective={spec.effective_batch_size}", flush=True)
    for epoch in range(starting_epoch, spec.max_epochs + 1):
        if deadline is not None and history:
            estimated_epoch_seconds = float(
                history[-1].get("epoch_seconds", elapsed_before / max(len(history), 1))
            )
            remaining_seconds = deadline - perf_counter()
            if remaining_seconds < estimated_epoch_seconds * 1.05:
                del model, renderer, optimizer, scheduler, scaler
                torch.cuda.empty_cache()
                return {
                    "status": "paused_time_limit_before_next_epoch",
                    "epoch": int(history[-1]["epoch"]),
                    "best_epoch": best_epoch,
                    "remaining_seconds": max(0.0, remaining_seconds),
                    "estimated_next_epoch_seconds": estimated_epoch_seconds,
                    "output_dir": str(output_dir),
                }
        epoch_started = perf_counter()
        sampler = EffectiveBatchSampler(len(split["train"]), spec.physical_batch_size, spec.effective_batch_size, seed, epoch)
        loader = DataLoader(IndexedSubset(dataset, split["train"]), batch_sampler=sampler, num_workers=0, pin_memory=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        processed = 0
        group_processed = 0
        group_size = min(spec.effective_batch_size, len(split["train"]))
        epoch_loss_sum = 0.0
        steps = 0
        for _, bits, batch_targets in loader:
            bits = bits.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            standardized_target = (batch_targets - mean_device) / std_device
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                prediction = model(make_inputs(bits, spec, renderer))
                loss_sum = functional.mse_loss(prediction, standardized_target, reduction="sum")
                backward_loss = loss_sum / (group_size * 2)
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(backward_loss).backward()
            batch_count = len(bits)
            processed += batch_count
            group_processed += batch_count
            epoch_loss_sum += float(loss_sum.detach())
            if group_processed == group_size:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                group_processed = 0
                remaining = len(split["train"]) - processed
                group_size = min(spec.effective_batch_size, remaining) if remaining else 0
        if processed != len(split["train"]) or group_processed:
            raise RuntimeError("Incomplete accumulated training batch")
        if steps != sampler.optimizer_steps:
            raise RuntimeError(f"Optimizer step mismatch: {steps} != {sampler.optimizer_steps}")
        train_loss = epoch_loss_sum / (processed * 2)
        validation_loss, validation_metrics, _, _, _ = evaluate(model, dataset, split["validation"], spec, renderer, target_mean, target_std, device)
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1e-12
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            no_improvement = 0
            atomic_torch_save(
                output_dir / "best_checkpoint.pt",
                {"experiment_id": spec.experiment_id, "seed": seed, "epoch": epoch, "model_state": model.state_dict(), "target_mean": target_mean, "target_std": target_std, "config": config},
            )
        else:
            no_improvement += 1
        history.append({
            "epoch": epoch,
            "epoch_seconds": perf_counter() - epoch_started,
            "train_standardized_mse": train_loss,
            "validation_standardized_mse": validation_loss,
            "validation_metrics": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "optimizer_steps": steps,
        })
        elapsed = elapsed_before + perf_counter() - started
        state = {
            "experiment_id": spec.experiment_id,
            "seed": seed,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "no_improvement": no_improvement,
            "elapsed_seconds": elapsed,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
        }
        atomic_torch_save(last_path, state)
        atomic_json(output_dir / "history.json", history)
        metrics = validation_metrics
        print(
            f"  epoch={epoch:03d}/{spec.max_epochs} train={train_loss:.5f} val={validation_loss:.5f} "
            f"MAE=({metrics['tavg']['mae']:.4f},{metrics['delta_t']['mae']:.4f}) "
            f"R2=({metrics['tavg']['r2']:.4f},{metrics['delta_t']['r2']:.4f}) lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if deadline is not None and perf_counter() >= deadline:
            del model, renderer, optimizer, scheduler, scaler
            torch.cuda.empty_cache()
            return {"status": "paused_time_limit", "epoch": epoch, "best_epoch": best_epoch, "output_dir": str(output_dir)}
        if epoch >= spec.minimum_epochs and no_improvement >= spec.early_stopping_patience:
            print(f"EARLY STOP {spec.experiment_id} after {no_improvement} non-improving epochs", flush=True)
            break

    best = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_loss, test_metrics, sample_indices, actual, predicted = evaluate(model, dataset, split["test"], spec, renderer, target_mean, target_std, device)
    save_predictions(output_dir / "test_predictions.csv", sample_indices, actual, predicted)
    result = {
        "status": "complete",
        "experiment_id": spec.experiment_id,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_standardized_mse": best_loss,
        "test_standardized_mse": test_loss,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed_before + perf_counter() - started,
        "checkpoint": str((output_dir / "best_checkpoint.pt").resolve()),
    }
    atomic_json(output_dir / "result.json", result)
    print(f"DONE {spec.experiment_id} seed={seed} best={best_epoch} test_MAE=({test_metrics['tavg']['mae']:.4f},{test_metrics['delta_t']['mae']:.4f})", flush=True)
    del model, renderer, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result
