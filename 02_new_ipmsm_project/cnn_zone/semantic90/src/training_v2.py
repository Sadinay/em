"""Fair, reproducible training infrastructure for the six v2 CNN regressors."""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import subprocess
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .models_v2 import V2_MODEL_CLASSES, parameter_count
from .topology_renderer import FEMSemanticRenderer


TARGET_NAMES = ("tavg", "delta_t")
TARGET_DIMENSION = 2
EFFECTIVE_BATCH_SIZE = 64
MAX_CONSECUTIVE_AMP_OVERFLOWS = 8


@dataclass(frozen=True)
class V2TrainingSpec:
    model_id: str
    input_mode: str
    learning_rate: float
    weight_decay: float
    max_epochs: int
    scheduler_patience: int
    early_stopping_patience: int
    physical_batch_candidates: tuple[int, ...]
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    input_channels: int = 3


V2_TRAINING_SPECS = {
    "logical10/mini_inception_v2": V2TrainingSpec(
        "logical10/mini_inception_v2", "logical10", 1e-3, 1e-4, 30, 3, 6, (64,), input_channels=3
    ),
    "logical10/resnet20_v2": V2TrainingSpec(
        "logical10/resnet20_v2", "logical10", 3e-4, 1e-4, 30, 3, 6, (64,), input_channels=3
    ),
    "logical10/small_cnn_v2": V2TrainingSpec(
        "logical10/small_cnn_v2", "logical10", 1e-3, 1e-4, 25, 3, 6, (64,), input_channels=3
    ),
    "semantic224/mini_inception_v2": V2TrainingSpec(
        "semantic224/mini_inception_v2", "semantic224", 3e-4, 1e-4, 30, 3, 6, (8, 16, 24, 32), input_channels=8
    ),
    "semantic224/resnet18_v2": V2TrainingSpec(
        "semantic224/resnet18_v2", "semantic224", 3e-4, 1e-4, 30, 3, 6, (8, 16, 24), input_channels=8
    ),
    "semantic224/vgg16_v2": V2TrainingSpec(
        "semantic224/vgg16_v2", "semantic224", 1e-4, 1e-4, 30, 3, 5, (8, 16), input_channels=8
    ),
}


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def logical_images(codes: torch.Tensor) -> torch.Tensor:
    grids = codes.reshape(-1, 10, 10).transpose(1, 2)
    return functional.one_hot(grids.long(), num_classes=3).permute(0, 3, 1, 2).float()


class IndexedDataset(Dataset):
    """Wrap a global gene dataset while preserving each global sample index."""

    def __init__(self, dataset: Dataset, indices: np.ndarray | list[int]) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        global_index = int(self.indices[item])
        codes, target = self.dataset[global_index]
        return torch.tensor(global_index, dtype=torch.int64), codes, target


class EffectiveBatchSampler(Sampler[list[int]]):
    """Shuffle effective batches, then split each into bounded physical microbatches."""

    def __init__(
        self,
        sample_count: int,
        physical_batch_size: int,
        effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
        seed: int = 0,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if physical_batch_size <= 0 or effective_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if physical_batch_size > effective_batch_size:
            raise ValueError("physical_batch_size cannot exceed effective_batch_size")
        self.sample_count = sample_count
        self.physical_batch_size = physical_batch_size
        self.effective_batch_size = effective_batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(self.sample_count)
        for group_start in range(0, self.sample_count, self.effective_batch_size):
            group = order[group_start : group_start + self.effective_batch_size]
            for start in range(0, len(group), self.physical_batch_size):
                yield group[start : start + self.physical_batch_size].tolist()

    def __len__(self) -> int:
        full_groups, remainder = divmod(self.sample_count, self.effective_batch_size)
        microbatches_per_full_group = math.ceil(self.effective_batch_size / self.physical_batch_size)
        return full_groups * microbatches_per_full_group + (
            math.ceil(remainder / self.physical_batch_size) if remainder else 0
        )

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(self.sample_count / self.effective_batch_size)


def build_v2_model(spec: V2TrainingSpec) -> nn.Module:
    return V2_MODEL_CLASSES[spec.model_id](input_channels=spec.input_channels)


def make_inputs(
    codes: torch.Tensor,
    input_mode: str,
    renderer: FEMSemanticRenderer | None,
) -> torch.Tensor:
    if input_mode == "logical10":
        return logical_images(codes)
    if renderer is None:
        raise RuntimeError("Semantic renderer is required")
    return renderer(codes).contiguous(memory_format=torch.channels_last)


def target_scaler(targets: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    return tensor.mean(dim=0), tensor.std(dim=0, unbiased=False).clamp_min(1e-8)


def basic_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    error = predicted - actual
    absolute_error = np.abs(error)
    output: dict[str, dict[str, float]] = {}
    for target_index, name in enumerate(TARGET_NAMES):
        truth = actual[:, target_index]
        estimate = predicted[:, target_index]
        target_error = error[:, target_index]
        target_absolute = absolute_error[:, target_index]
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        target_range = max(float(truth.max() - truth.min()), 1e-12)
        if float(np.var(truth)) > 1e-15:
            calibration_slope, calibration_intercept = np.polyfit(truth, estimate, 1)
        else:
            calibration_slope, calibration_intercept = float("nan"), float("nan")
        output[name] = {
            "mae": float(target_absolute.mean()),
            "rmse": float(np.sqrt(np.mean(target_error**2))),
            "r2": float(1.0 - np.sum(target_error**2) / max(denominator, 1e-12)),
            "normalized_mae_by_actual_range": float(target_absolute.mean() / target_range),
            "bias": float(target_error.mean()),
            "calibration_slope_predicted_vs_actual": float(calibration_slope),
            "calibration_intercept_nm": float(calibration_intercept),
            "absolute_error_p50": float(np.percentile(target_absolute, 50)),
            "absolute_error_p90": float(np.percentile(target_absolute, 90)),
            "absolute_error_p95": float(np.percentile(target_absolute, 95)),
            "fraction_absolute_error_le_0_05_nm": float(np.mean(target_absolute <= 0.05)),
            "fraction_absolute_error_le_0_10_nm": float(np.mean(target_absolute <= 0.10)),
        }
    return output


TAVG_BANDS = (
    ("<0.5", -np.inf, 0.5),
    ("0.5-1.0", 0.5, 1.0),
    ("1.0-1.5", 1.0, 1.5),
    ("1.5-2.0", 1.5, 2.0),
    ("2.0-2.5", 2.0, 2.5),
    ("2.5-3.0", 2.5, 3.0),
    (">=3.0", 3.0, np.inf),
)
DELTA_T_BANDS = (
    ("<0.5", -np.inf, 0.5),
    ("0.5-0.75", 0.5, 0.75),
    ("0.75-1.0", 0.75, 1.0),
    ("1.0-1.5", 1.0, 1.5),
    ("1.5-2.0", 1.5, 2.0),
    (">=2.0", 2.0, np.inf),
)


def band_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for target_index, (target_name, bands) in enumerate(
        (("tavg", TAVG_BANDS), ("delta_t", DELTA_T_BANDS))
    ):
        target_result = []
        truth = actual[:, target_index]
        error = predicted[:, target_index] - truth
        for label, low, high in bands:
            mask = (truth >= low) & (truth < high)
            count = int(mask.sum())
            target_result.append(
                {
                    "band": label,
                    "count": count,
                    "mae": float(np.mean(np.abs(error[mask]))) if count else None,
                    "bias": float(np.mean(error[mask])) if count else None,
                }
            )
        result[target_name] = target_result
    return result


def complete_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {"overall": basic_metrics(actual, predicted), "bands": band_metrics(actual, predicted)}


class NvidiaSmiMonitor:
    """Low-frequency driver-level utilization monitor for one local GPU."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    creationflags=flags,
                    check=False,
                )
                values = [float(part.strip()) for part in completed.stdout.splitlines()[0].split(",")]
                self.samples.append((values[0], values[1], values[2], values[3]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)
        if not self.samples:
            return {"sample_count": 0}
        values = np.asarray(self.samples, dtype=np.float64)
        return {
            "sample_count": len(values),
            "gpu_utilization_percent_mean": float(values[:, 0].mean()),
            "gpu_utilization_percent_p50": float(np.percentile(values[:, 0], 50)),
            "gpu_utilization_percent_p90": float(np.percentile(values[:, 0], 90)),
            "gpu_utilization_percent_max": float(values[:, 0].max()),
            "driver_memory_used_mb_peak": float(values[:, 1].max()),
            "power_watts_mean": float(values[:, 2].mean()),
            "temperature_celsius_max": float(values[:, 3].max()),
        }


def probe_physical_batches(
    spec: V2TrainingSpec,
    lookup: Path,
    sample_codes: torch.Tensor,
    device: torch.device,
) -> dict:
    """Test requested physical batches and return the largest stable candidate."""
    results = []
    for candidate in spec.physical_batch_candidates:
        configure_reproducibility(10101 + candidate)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        status = "passed"
        error = None
        model = renderer = codes = inputs = targets = prediction = loss = None
        try:
            model = build_v2_model(spec).to(device)
            renderer = None
            if spec.input_mode == "semantic224":
                model = model.to(memory_format=torch.channels_last)
                renderer = FEMSemanticRenderer(lookup).to(device)
            codes = sample_codes[:candidate].to(device)
            inputs = make_inputs(codes, spec.input_mode, renderer)
            targets = torch.zeros((candidate, 2), device=device)
            model.train()
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                prediction = model(inputs)
                loss = functional.mse_loss(prediction, targets)
            loss.backward()
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError("non-finite forward or loss")
            torch.cuda.synchronize()
            peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
        except (torch.cuda.OutOfMemoryError, RuntimeError, FloatingPointError) as exc:
            status = "failed"
            error = str(exc)
            peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
        finally:
            del model, renderer, codes, inputs, targets, prediction, loss
            gc.collect()
            torch.cuda.empty_cache()
        results.append(
            {
                "physical_batch_size": candidate,
                "status": status,
                "torch_peak_allocated_mb": peak_mb,
                "error": error,
            }
        )
    stable = [item["physical_batch_size"] for item in results if item["status"] == "passed"]
    if not stable:
        raise RuntimeError(f"No stable physical batch for {spec.model_id}")
    return {"model_id": spec.model_id, "candidates": results, "selected": max(stable)}


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    spec: V2TrainingSpec,
    renderer: FEMSemanticRenderer | None,
    mean_device: torch.Tensor,
    std_device: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    squared_error_sum = 0.0
    sample_count = 0
    indices_list: list[np.ndarray] = []
    actual_list: list[np.ndarray] = []
    predicted_list: list[np.ndarray] = []
    for indices, codes, targets in loader:
        codes = codes.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        inputs = make_inputs(codes, spec.input_mode, renderer)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            standardized_prediction = model(inputs)
            standardized_target = (targets_device - mean_device) / std_device
            squared_error_sum += float(
                functional.mse_loss(standardized_prediction, standardized_target, reduction="sum")
            )
        physical_prediction = standardized_prediction.float() * std_device + mean_device
        sample_count += len(codes)
        indices_list.append(indices.numpy())
        actual_list.append(targets.numpy())
        predicted_list.append(physical_prediction.cpu().numpy())
    return (
        squared_error_sum / (sample_count * TARGET_DIMENSION),
        np.concatenate(indices_list),
        np.vstack(actual_list),
        np.vstack(predicted_list),
    )


def save_predictions_csv(
    path: Path, indices: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> None:
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
        for index, truth, estimate in zip(indices, actual, predicted):
            writer.writerow(
                (
                    int(index),
                    float(truth[0]),
                    float(estimate[0]),
                    float(estimate[0] - truth[0]),
                    float(truth[1]),
                    float(estimate[1]),
                    float(estimate[1] - truth[1]),
                )
            )


def evaluate_v2_checkpoint_sets(
    spec: V2TrainingSpec,
    base_dataset: Dataset,
    evaluation_sets: dict[str, np.ndarray],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    physical_batch_size: int,
    lookup: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, dict]:
    """Evaluate one best checkpoint on named held-out sets and save predictions/metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_id") != spec.model_id:
        raise RuntimeError("Checkpoint model ID does not match the requested evaluation model")
    checkpoint_mean = torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32).cpu()
    checkpoint_std = torch.as_tensor(checkpoint["target_std"], dtype=torch.float32).cpu()
    if not torch.allclose(checkpoint_mean, target_mean.cpu(), rtol=0.0, atol=1e-7):
        raise RuntimeError("Checkpoint target mean does not match the V3 training scaler")
    if not torch.allclose(checkpoint_std, target_std.cpu(), rtol=0.0, atol=1e-7):
        raise RuntimeError("Checkpoint target std does not match the V3 training scaler")

    model = build_v2_model(spec).to(device)
    renderer: FEMSemanticRenderer | None = None
    if spec.input_mode == "semantic224":
        model = model.to(memory_format=torch.channels_last)
        renderer = FEMSemanticRenderer(lookup).to(device)
    model.load_state_dict(checkpoint["model_state"])
    evaluation_batch = max(
        physical_batch_size,
        64 if spec.input_mode == "logical10" else physical_batch_size,
    )
    mean_device = target_mean.to(device)
    std_device = target_std.to(device)
    results: dict[str, dict] = {}
    for name, indices in evaluation_sets.items():
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
            raise ValueError(f"Unsafe evaluation-set name: {name!r}")
        indexed = IndexedDataset(base_dataset, indices)
        loader = DataLoader(
            indexed,
            batch_size=evaluation_batch,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        loss, sample_indices, actual, predicted = evaluate_model(
            model,
            loader,
            spec,
            renderer,
            mean_device,
            std_device,
            device,
        )
        result = {
            "status": "complete",
            "evaluation_set": name,
            "sample_count": int(len(indices)),
            "standardized_mse": float(loss),
            "metrics": complete_metrics(actual, predicted),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
        }
        save_predictions_csv(
            output_dir / f"{name}_predictions.csv",
            sample_indices,
            actual,
            predicted,
        )
        temporary = (output_dir / f"{name}_result.json").with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(output_dir / f"{name}_result.json")
        results[name] = result
        print(
            f"EVAL {spec.model_id} {name} n={len(indices):,} MAE="
            f"({result['metrics']['overall']['tavg']['mae']:.4f},"
            f"{result['metrics']['overall']['delta_t']['mae']:.4f})",
            flush=True,
        )
    del model, renderer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def train_v2_model(
    spec: V2TrainingSpec,
    base_dataset: Dataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    physical_batch_size: int,
    lookup: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
    stop_r2: float | None = None,
    disable_dropout_during_training: bool = False,
    resume_training: bool = False,
) -> dict:
    configure_reproducibility(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_v2_model(spec).to(device)
    renderer: FEMSemanticRenderer | None = None
    if spec.input_mode == "semantic224":
        model = model.to(memory_format=torch.channels_last)
        renderer = FEMSemanticRenderer(lookup).to(device)
    train_set = IndexedDataset(base_dataset, train_indices)
    validation_set = IndexedDataset(base_dataset, validation_indices)
    test_set = IndexedDataset(base_dataset, test_indices)
    sampler = EffectiveBatchSampler(
        len(train_set), physical_batch_size, spec.effective_batch_size, seed
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    evaluation_batch = max(physical_batch_size, 64 if spec.input_mode == "logical10" else physical_batch_size)
    validation_loader = DataLoader(
        validation_set,
        batch_size=evaluation_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=evaluation_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=spec.scheduler_patience,
        min_lr=1e-6,
    )
    amp_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=128.0,
        growth_interval=2000,
    )
    mean_device, std_device = target_mean.to(device), target_std.to(device)
    config_record = {
        **asdict(spec),
        "physical_batch_size": physical_batch_size,
        "optimizer": "AdamW",
        "loss": "mean MSE over two independently standardized targets",
        "scheduler": "ReduceLROnPlateau(factor=0.5,min_lr=1e-6)",
        "amp": True,
        "amp_grad_scaler_init_scale": 128.0,
        "amp_overflow_policy": (
            "skip the affected optimizer update, reduce GradScaler scale, and fail only "
            f"after more than {MAX_CONSECUTIVE_AMP_OVERFLOWS} consecutive overflows"
        ),
        "channels_last": spec.input_mode == "semantic224",
        "seed": seed,
        "dropout_disabled_for_memorization_diagnostic": disable_dropout_during_training,
        "epoch_checkpoint_resume_enabled": resume_training,
        "parameter_count": parameter_count(model),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "test_samples": len(test_set),
        "expected_optimizer_steps_per_epoch": sampler.optimizer_steps_per_epoch,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )
    history: list[dict] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    total_optimizer_steps = 0
    amp_overflow_steps_total = 0
    consecutive_amp_overflows = 0
    maximum_consecutive_amp_overflows = 0
    maximum_gradient_norm = 0.0
    starting_epoch = 1
    elapsed_before_resume = 0.0
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    if resume_training and last_checkpoint_path.exists():
        resume_state = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        if resume_state["model_id"] != spec.model_id or int(resume_state["seed"]) != seed:
            raise RuntimeError("Last checkpoint model ID or seed does not match this run")
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        amp_scaler.load_state_dict(resume_state["amp_scaler_state"])
        history = resume_state["history"]
        best_validation_loss = float(resume_state["best_validation_loss"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        total_optimizer_steps = int(resume_state["total_optimizer_steps"])
        amp_overflow_steps_total = int(resume_state.get("amp_overflow_steps_total", 0))
        consecutive_amp_overflows = int(resume_state.get("consecutive_amp_overflows", 0))
        maximum_consecutive_amp_overflows = int(
            resume_state.get("maximum_consecutive_amp_overflows", 0)
        )
        maximum_gradient_norm = float(resume_state["maximum_gradient_norm"])
        elapsed_before_resume = float(resume_state["elapsed_seconds"])
        starting_epoch = int(resume_state["epoch"]) + 1
        random.setstate(resume_state["python_random_state"])
        np.random.set_state(resume_state["numpy_random_state"])
        torch.set_rng_state(resume_state["torch_cpu_random_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_state["torch_cuda_random_states"]]
            )
        print(
            f"RESUME {spec.model_id} seed={seed} from completed epoch {starting_epoch - 1}",
            flush=True,
        )
    monitor = NvidiaSmiMonitor()
    monitor.start()
    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(
        f"START {spec.model_id} seed={seed} physical={physical_batch_size} "
        f"effective={spec.effective_batch_size} params={parameter_count(model):,}",
        flush=True,
    )
    try:
        for epoch in range(starting_epoch, spec.max_epochs + 1):
            sampler.set_epoch(epoch)
            model.train()
            if disable_dropout_during_training:
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.eval()
            optimizer.zero_grad(set_to_none=True)
            standardized_squared_error_sum = 0.0
            processed_samples = 0
            group_processed_samples = 0
            group_size = min(spec.effective_batch_size, len(train_set))
            epoch_optimizer_steps = 0
            epoch_amp_overflow_steps = 0
            for _, codes, targets in train_loader:
                codes = codes.to(device, non_blocking=True)
                targets_device = targets.to(device, non_blocking=True)
                inputs = make_inputs(codes, spec.input_mode, renderer)
                standardized_target = (targets_device - mean_device) / std_device
                with torch.amp.autocast(
                    "cuda", enabled=device.type == "cuda", dtype=torch.float16
                ):
                    prediction = model(inputs)
                    loss_sum = functional.mse_loss(
                        prediction, standardized_target, reduction="sum"
                    )
                    backward_loss = loss_sum / (group_size * TARGET_DIMENSION)
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("Non-finite training loss")
                amp_scaler.scale(backward_loss).backward()
                standardized_squared_error_sum += float(loss_sum.detach())
                batch_samples = len(codes)
                processed_samples += batch_samples
                group_processed_samples += batch_samples
                if group_processed_samples == group_size:
                    amp_scaler.unscale_(optimizer)
                    gradient_norm = float(
                        torch.nn.utils.get_total_norm(
                            [
                                parameter.grad
                                for parameter in model.parameters()
                                if parameter.grad is not None
                            ]
                        )
                    )
                    gradient_is_finite = math.isfinite(gradient_norm)
                    scale_before_step = float(amp_scaler.get_scale())
                    if not gradient_is_finite and not amp_scaler.is_enabled():
                        raise FloatingPointError("Non-finite gradient norm without AMP")
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    scale_after_step = float(amp_scaler.get_scale())
                    optimizer.zero_grad(set_to_none=True)
                    epoch_optimizer_steps += 1
                    total_optimizer_steps += 1
                    if gradient_is_finite:
                        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                        consecutive_amp_overflows = 0
                    else:
                        epoch_amp_overflow_steps += 1
                        amp_overflow_steps_total += 1
                        consecutive_amp_overflows += 1
                        maximum_consecutive_amp_overflows = max(
                            maximum_consecutive_amp_overflows,
                            consecutive_amp_overflows,
                        )
                        print(
                            f"  AMP overflow: skipped optimizer update at epoch={epoch} "
                            f"step={epoch_optimizer_steps}; scale "
                            f"{scale_before_step:.0f}->{scale_after_step:.0f}",
                            flush=True,
                        )
                        if scale_after_step >= scale_before_step:
                            raise FloatingPointError(
                                "Non-finite AMP gradient did not reduce the loss scale"
                            )
                        if consecutive_amp_overflows > MAX_CONSECUTIVE_AMP_OVERFLOWS:
                            raise FloatingPointError(
                                "Too many consecutive AMP gradient overflows"
                            )
                    group_processed_samples = 0
                    remaining = len(train_set) - processed_samples
                    group_size = min(spec.effective_batch_size, remaining) if remaining else 0
            if processed_samples != len(train_set) or group_processed_samples != 0:
                raise RuntimeError("Incomplete final accumulated batch")
            if epoch_optimizer_steps != sampler.optimizer_steps_per_epoch:
                raise RuntimeError(
                    f"Optimizer steps {epoch_optimizer_steps} != expected {sampler.optimizer_steps_per_epoch}"
                )
            training_loss = standardized_squared_error_sum / (
                len(train_set) * TARGET_DIMENSION
            )
            validation_loss, _, validation_actual, validation_predicted = evaluate_model(
                model,
                validation_loader,
                spec,
                renderer,
                mean_device,
                std_device,
                device,
            )
            validation_metrics = basic_metrics(validation_actual, validation_predicted)
            scheduler.step(validation_loss)
            improved = validation_loss < best_validation_loss - 1e-12
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_id": spec.model_id,
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        "target_mean": target_mean.cpu(),
                        "target_std": target_std.cpu(),
                        "config": config_record,
                    },
                    output_dir / "best_checkpoint.pt",
                )
            else:
                epochs_without_improvement += 1
            history.append(
                {
                    "epoch": epoch,
                    "training_standardized_mse": training_loss,
                    "validation_standardized_mse": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "optimizer_steps": epoch_optimizer_steps,
                    "successful_optimizer_steps": (
                        epoch_optimizer_steps - epoch_amp_overflow_steps
                    ),
                    "amp_overflow_steps": epoch_amp_overflow_steps,
                    "validation_metrics": validation_metrics,
                }
            )
            elapsed_at_checkpoint = elapsed_before_resume + perf_counter() - started
            last_state = {
                "model_id": spec.model_id,
                "seed": seed,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "amp_scaler_state": amp_scaler.state_dict(),
                "history": history,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "total_optimizer_steps": total_optimizer_steps,
                "amp_overflow_steps_total": amp_overflow_steps_total,
                "consecutive_amp_overflows": consecutive_amp_overflows,
                "maximum_consecutive_amp_overflows": maximum_consecutive_amp_overflows,
                "maximum_gradient_norm": maximum_gradient_norm,
                "elapsed_seconds": elapsed_at_checkpoint,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_cpu_random_state": torch.get_rng_state(),
                "torch_cuda_random_states": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
            }
            temporary_checkpoint = last_checkpoint_path.with_suffix(".tmp")
            torch.save(last_state, temporary_checkpoint)
            temporary_checkpoint.replace(last_checkpoint_path)
            temporary_history = (output_dir / "history.json").with_suffix(".tmp")
            temporary_history.write_text(json.dumps(history, indent=2), encoding="utf-8")
            temporary_history.replace(output_dir / "history.json")
            print(
                f"  epoch={epoch:02d}/{spec.max_epochs} train={training_loss:.5f} "
                f"val={validation_loss:.5f} MAE=({validation_metrics['tavg']['mae']:.4f},"
                f"{validation_metrics['delta_t']['mae']:.4f}) "
                f"R2=({validation_metrics['tavg']['r2']:.4f},"
                f"{validation_metrics['delta_t']['r2']:.4f}) lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            if stop_r2 is not None and all(
                validation_metrics[name]["r2"] >= stop_r2 for name in TARGET_NAMES
            ):
                print(f"  overfit threshold R2>={stop_r2:.3f} reached", flush=True)
                break
            if epochs_without_improvement >= spec.early_stopping_patience:
                print(
                    f"  early stopping after {epochs_without_improvement} non-improving epochs",
                    flush=True,
                )
                break
    finally:
        elapsed_seconds = elapsed_before_resume + perf_counter() - started
        gpu_monitor = monitor.stop()
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(
        output_dir / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_loss, indices, actual, predicted = evaluate_model(
        model,
        test_loader,
        spec,
        renderer,
        mean_device,
        std_device,
        device,
    )
    metrics = complete_metrics(actual, predicted)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / 1024**2)
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "status": "complete",
        "model_id": spec.model_id,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_standardized_mse": best_validation_loss,
        "test_standardized_mse": test_loss,
        "metrics": metrics,
        "elapsed_seconds": elapsed_seconds,
        "torch_peak_allocated_mb": peak_memory_mb,
        "gpu_monitor": gpu_monitor,
        "maximum_gradient_norm": maximum_gradient_norm,
        "amp_overflow_steps_total": amp_overflow_steps_total,
        "maximum_consecutive_amp_overflows": maximum_consecutive_amp_overflows,
        "successful_optimizer_steps_total": (
            total_optimizer_steps - amp_overflow_steps_total
        ),
        "optimizer_steps_total": total_optimizer_steps,
        "optimizer_steps_per_completed_epoch": sampler.optimizer_steps_per_epoch,
        "checkpoint": str((output_dir / "best_checkpoint.pt").resolve()),
    }
    save_predictions_csv(output_dir / "test_predictions.csv", indices, actual, predicted)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"DONE {spec.model_id} seed={seed} best={best_epoch} test_MAE="
        f"({metrics['overall']['tavg']['mae']:.4f},{metrics['overall']['delta_t']['mae']:.4f}) "
        f"elapsed={elapsed_seconds:.1f}s",
        flush=True,
    )
    del model, renderer, optimizer, amp_scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def overfit_spec(spec: V2TrainingSpec, max_epochs: int = 300) -> V2TrainingSpec:
    """Disable regularization while retaining each architecture's stable LR."""
    return replace(
        spec,
        weight_decay=0.0,
        effective_batch_size=32,
        max_epochs=max_epochs,
        scheduler_patience=10,
        early_stopping_patience=30,
    )
