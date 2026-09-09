"""Run the SPMSM V3 matrix and supplementary logical ResNet20 with resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CNN_ROOT = ROOT / "cnn_zone"
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))
from src.dataset import SPMSMGeneDataset  # noqa: E402
from src.training import TrainingSpec, atomic_json, probe_spec, train_one  # noqa: E402


DATASET_DIR = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
SPLIT_PATH = CNN_ROOT / "outputs" / "splits" / "scheme_a_tavg_bands_train40000_val6483_test6483.npz"
MODEL_ROOT = CNN_ROOT / "models" / "v3_40000_6runs"
REPORT_ROOT = ROOT / "reports" / "V3" / "03_完整审核资料" / "训练过程记录"
DEFAULT_SEEDS = (20260903,)


def semantic_spec(experiment_id: str, input_mode: str, architecture: str, lookup: str, circular: bool = False) -> TrainingSpec:
    is_vgg = architecture == "vgg16_v2"
    return TrainingSpec(
        experiment_id=experiment_id,
        input_mode=input_mode,
        architecture=architecture,
        lookup=lookup,
        learning_rate=1e-4 if is_vgg else 3e-4,
        weight_decay=1e-4,
        max_epochs=40,
        minimum_epochs=30,
        early_stopping_patience=8,
        scheduler_patience=3,
        physical_batch_size=8 if is_vgg else 16,
        effective_batch_size=64,
        circular_angular_padding=circular,
    )


SPECS = {
    "logical6x20/small_cnn_v2": TrainingSpec(
        "logical6x20/small_cnn_v2", "logical6x20", "small_cnn_v2", None,
        1e-3, 1e-4, 100, 20, 12, 3, 64, 64, False,
    ),
    "logical6x20/mini_inception_v2": TrainingSpec(
        "logical6x20/mini_inception_v2", "logical6x20", "mini_inception_v2", None,
        1e-3, 1e-4, 100, 20, 12, 3, 64, 64, False,
    ),
    "logical6x20/resnet20_v2": TrainingSpec(
        "logical6x20/resnet20_v2", "logical6x20", "resnet20_v2", None,
        3e-4, 1e-4, 100, 20, 12, 3, 64, 64, False,
    ),
}

for field, lookup, circular in (
    ("xy90_224", "cnn_zone/outputs/lookups/spmsm_xy224_full_motor.npz", False),
    ("polar90_224", "cnn_zone/outputs/lookups/spmsm_polar90_224_full_motor.npz", False),
    ("xy360_224", "cnn_zone/outputs/lookups/spmsm_xy360_224_full_motor_ss4.npz", False),
    ("polar360_224", "cnn_zone/outputs/lookups/spmsm_polar360_224_full_motor_ss4.npz", True),
):
    for architecture in ("vgg16_v2",):
        experiment_id = f"{field}/{architecture}"
        SPECS[experiment_id] = semantic_spec(experiment_id, field, architecture, lookup, circular)


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def load_split(dataset_size: int) -> dict[str, np.ndarray]:
    with np.load(SPLIT_PATH) as source:
        split = {name: np.asarray(source[name], dtype=np.int64) for name in ("train", "validation", "test")}
    expected = {"train": 40_000, "validation": 6_483, "test": 6_483}
    if {name: len(values) for name, values in split.items()} != expected:
        raise RuntimeError("Unexpected split sizes")
    for name, values in split.items():
        if len(np.unique(values)) != len(values) or values.min() < 0 or values.max() >= dataset_size:
            raise RuntimeError(f"Invalid {name} indices")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if len(np.intersect1d(split[left], split[right])):
            raise RuntimeError(f"Split leakage between {left} and {right}")
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume checkpoints and skip completed runs")
    parser.add_argument("--probe-only", action="store_true", help="Run one real forward/backward batch for every selected configuration")
    parser.add_argument("--dry-run", action="store_true", help="Validate the plan without allocating models")
    parser.add_argument("--time-limit-minutes", type=float, default=None, help="Pause safely after the current epoch once this wall-time budget is reached")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    args = parser.parse_args()
    if args.time_limit_minutes is not None and args.time_limit_minutes <= 0:
        raise ValueError("--time-limit-minutes must be positive")

    dataset = SPMSMGeneDataset(DATASET_DIR)
    split = load_split(len(dataset))
    missing_lookups = [spec.lookup for spec in (SPECS[name] for name in args.models) if spec.lookup and not (ROOT / spec.lookup).exists()]
    if missing_lookups:
        raise FileNotFoundError(f"Missing lookup files: {missing_lookups}")
    plan = {
        "status": "validated_no_training_started",
        "experiment": "SPMSM V3 40,000-sample matrix plus supplementary logical ResNet20",
        "models": args.models,
        "seeds": args.seeds,
        "expected_runs": len(args.models) * len(args.seeds),
        "split_sizes": {name: int(len(values)) for name, values in split.items()},
        "split_hashes_sha256": {name: index_hash(values) for name, values in split.items()},
        "specifications": {name: SPECS[name].__dict__ for name in args.models},
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "selection": {
            "logical": ["small_cnn_v2", "mini_inception_v2", "resnet20_v2"],
            "semantic224": ["vgg16_v2"],
            "evidence": "VGG16 retained for all 224 views by the simplified training decision",
        },
        "test_policy": "test split is evaluated only after a run finishes; validation chooses the checkpoint",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT_ROOT / "training_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("DRY RUN COMPLETE: no training started", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")

    if args.probe_only:
        results = []
        for name in args.models:
            print(f"PROBE {name}", flush=True)
            results.append(probe_spec(SPECS[name], ROOT, device))
            print(f"  PASS peak={results[-1]['peak_allocated_mb']:.1f} MB", flush=True)
        report = {"status": "all_passed", "gpu": torch.cuda.get_device_name(0), "results": results}
        atomic_json(REPORT_ROOT / "batch_probe.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    deadline = perf_counter() + args.time_limit_minutes * 60.0 if args.time_limit_minutes else None
    summary_path = REPORT_ROOT / "run_summary.json"
    summary = {**plan, "status": "running", "resume_requested": bool(args.resume), "completed_or_paused": []}
    atomic_json(summary_path, summary)
    try:
        for name in args.models:
            spec = SPECS[name]
            for seed in args.seeds:
                output = MODEL_ROOT / name / f"seed_{seed}"
                result_path = output / "result.json"
                if args.resume and result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
                    print(f"SKIP complete {name} seed={seed}", flush=True)
                    continue
                if deadline is not None and perf_counter() >= deadline:
                    summary["status"] = "paused_time_limit"
                    atomic_json(summary_path, summary)
                    print("PAUSED before next run: time limit reached", flush=True)
                    return
                result = train_one(spec, seed, dataset, split, ROOT, output, device, args.resume, deadline)
                summary["completed_or_paused"].append(result)
                atomic_json(summary_path, summary)
                if result["status"].startswith("paused_time_limit"):
                    summary["status"] = "paused_time_limit"
                    atomic_json(summary_path, summary)
                    print("PAUSED safely at an epoch checkpoint; use --resume", flush=True)
                    return
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        atomic_json(summary_path, summary)
        print("INTERRUPTED: the last completed epoch checkpoint remains resumable", flush=True)
        raise
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        atomic_json(summary_path, summary)
        raise
    summary["status"] = "complete"
    atomic_json(summary_path, summary)
    print(f"COMPLETE: {plan['expected_runs']} runs", flush=True)


if __name__ == "__main__":
    main()
