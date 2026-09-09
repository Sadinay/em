"""Train or resume the selected V3 regressors on the frozen 30k split.

The 5,000-sample validation subset controls scheduling and early stopping.  The
original 1,500-sample V2 test subset remains the core comparison set.  After
each run, the best checkpoint is also evaluated once on the complete Scheme-A
validation and test pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.training_v2 import (  # noqa: E402
    V2_TRAINING_SPECS,
    evaluate_v2_checkpoint_sets,
    target_scaler,
    train_v2_model,
)


DATASET = (
    PROJECT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)
LOOKUP = ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
BATCH_PROBE = ROOT / "outputs" / "v2_validation" / "batch_probe.json"
SELECTION_ROOT = PROJECT / "reports" / "v3_30000_selection"
FROZEN_SPLIT = SELECTION_ROOT / "fixed_v3_split_30000_5000_1500_full14655.npz"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_v3_training_selection.py"
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v3_30000"
REPORT_ROOT = PROJECT / "reports" / "v3_30000"

DEFAULT_SEEDS = (20260823, 20260824)
SELECTED_MODEL_IDS = (
    "logical10/mini_inception_v2",
    "logical10/resnet20_v2",
    "logical10/small_cnn_v2",
    "semantic224/vgg16_v2",
)
V3_TRAINING_SPECS = {
    "logical10/mini_inception_v2": replace(
        V2_TRAINING_SPECS["logical10/mini_inception_v2"],
        max_epochs=100,
    ),
    "logical10/resnet20_v2": replace(
        V2_TRAINING_SPECS["logical10/resnet20_v2"],
        max_epochs=100,
    ),
    "logical10/small_cnn_v2": replace(
        V2_TRAINING_SPECS["logical10/small_cnn_v2"],
        max_epochs=100,
    ),
    "semantic224/vgg16_v2": replace(
        V2_TRAINING_SPECS["semantic224/vgg16_v2"],
        max_epochs=35,
    ),
}


def write_json_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def indices_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def ensure_split(rebuild: bool) -> None:
    if rebuild or not FROZEN_SPLIT.exists():
        subprocess.run([sys.executable, str(PREPARE_SCRIPT)], cwd=PROJECT, check=True)
    if not FROZEN_SPLIT.exists():
        raise FileNotFoundError(f"V3 split was not created: {FROZEN_SPLIT}")


def load_and_validate_split(dataset_size: int) -> dict[str, np.ndarray]:
    with np.load(FROZEN_SPLIT) as split:
        arrays = {
            name: np.asarray(split[name], dtype=np.int64)
            for name in ("train", "validation", "core_test", "full_validation", "full_test")
        }
    expected = {
        "train": 30_000,
        "validation": 5_000,
        "core_test": 1_500,
        "full_validation": 14_656,
        "full_test": 14_655,
    }
    actual = {name: len(indices) for name, indices in arrays.items()}
    if actual != expected:
        raise RuntimeError(f"V3 split sizes {actual} do not equal {expected}")
    for name, indices in arrays.items():
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{name} contains duplicate indices")
        if indices.min() < 0 or indices.max() >= dataset_size:
            raise RuntimeError(f"{name} contains an out-of-range sample index")

    train = arrays["train"]
    validation = arrays["validation"]
    core_test = arrays["core_test"]
    full_validation = arrays["full_validation"]
    full_test = arrays["full_test"]
    if not np.all(np.isin(validation, full_validation)):
        raise RuntimeError("The 5,000-sample validation set is not inside full validation")
    if not np.all(np.isin(core_test, full_test)):
        raise RuntimeError("The 1,500-sample core test is not inside full Scheme-A test")
    for left_name, left, right_name, right in (
        ("train", train, "full_validation", full_validation),
        ("train", train, "full_test", full_test),
        ("full_validation", full_validation, "full_test", full_test),
    ):
        overlap = len(np.intersect1d(left, right))
        if overlap:
            raise RuntimeError(f"{left_name}/{right_name} overlap by {overlap} samples")
    return arrays


def load_batches() -> dict[str, int]:
    probe = json.loads(BATCH_PROBE.read_text(encoding="utf-8"))
    batches = {item["model_id"]: int(item["selected"]) for item in probe["models"]}
    missing = set(SELECTED_MODEL_IDS) - set(batches)
    if missing:
        raise RuntimeError(f"Batch probe is missing selected V3 models: {sorted(missing)}")
    return batches


def completed_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record if record.get("status") == "complete" else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted epoch checkpoint and skip completed runs/evaluations",
    )
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Deterministically regenerate and revalidate the frozen V3 split before training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, split, models and GPU visibility without starting training",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds; defaults to the two reviewed V3 seeds",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SELECTED_MODEL_IDS,
        default=list(SELECTED_MODEL_IDS),
        help="Optional selected-model subset",
    )
    args = parser.parse_args()

    ensure_split(args.rebuild_split)
    dataset = GeneCodeDataset(DATASET, "all")
    split = load_and_validate_split(len(dataset))
    batches = load_batches()
    targets = np.asarray(dataset.targets, dtype=np.float32)
    mean, std = target_scaler(targets[split["train"]])

    plan = {
        "experiment": "V3 targeted 30,000-sample four-model regression",
        "dataset": str(DATASET.resolve()),
        "split": str(FROZEN_SPLIT.resolve()),
        "split_sizes": {name: int(len(indices)) for name, indices in split.items()},
        "split_hashes_sha256": {
            name: indices_sha256(indices) for name, indices in split.items()
        },
        "target_order": ["Tavg_Nm", "DeltaT_Nm"],
        "target_scaler_source": "V3 training set only",
        "target_mean": mean.tolist(),
        "target_std": std.tolist(),
        "models": args.models,
        "seeds": args.seeds,
        "max_epochs": {
            model_id: V3_TRAINING_SPECS[model_id].max_epochs for model_id in args.models
        },
        "physical_batches": {model_id: batches[model_id] for model_id in args.models},
        "effective_batch_size": 64,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "evaluation": {
            "early_stopping": "validation (5,000)",
            "v2_comparison": "core_test (1,500)",
            "post_training_audit": ["full_validation (14,656)", "full_test (14,655)"],
        },
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("DRY RUN COMPLETE: no training was started", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V3 training; use --dry-run for CPU-only validation")
    device = torch.device("cuda:0")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FROZEN_SPLIT, REPORT_ROOT / FROZEN_SPLIT.name)
    for artifact_name in (
        "selection_summary.json",
        "evaluation_split_distribution.csv",
        "tavg_band_distribution.csv",
        "delta_t_band_distribution.csv",
        "joint_tavg_delta_distribution.csv",
    ):
        source = SELECTION_ROOT / artifact_name
        if source.exists():
            shutil.copy2(source, REPORT_ROOT / artifact_name)
    write_json_atomic(
        REPORT_ROOT / "target_scaler.json",
        {
            "target_order": ["Tavg_Nm", "DeltaT_Nm"],
            "mean": mean.tolist(),
            "std": std.tolist(),
            "statistics_source": "only the frozen 30,000-sample V3 training set",
        },
    )

    summary_path = REPORT_ROOT / "run_summary.json"
    summary = {
        **plan,
        "status": "running",
        "resume_requested": bool(args.resume),
        "results": [],
    }
    write_json_atomic(summary_path, summary)
    try:
        for model_id in args.models:
            spec = V3_TRAINING_SPECS[model_id]
            family, name = model_id.split("/")
            for seed in args.seeds:
                output = MODEL_ROOT / family / name / f"seed_{seed}"
                result_path = output / "result.json"
                result = completed_json(result_path) if args.resume else None
                if result is not None:
                    print(f"SKIP complete training {model_id} seed={seed}", flush=True)
                else:
                    if not args.resume and (output / "last_checkpoint.pt").exists():
                        raise RuntimeError(
                            f"Existing checkpoint found at {output}; rerun with --resume"
                        )
                    result = train_v2_model(
                        spec,
                        dataset,
                        split["train"],
                        split["validation"],
                        split["core_test"],
                        mean,
                        std,
                        batches[model_id],
                        LOOKUP,
                        output,
                        device,
                        seed,
                        resume_training=args.resume,
                    )

                evaluation_names = {
                    "full_validation": split["full_validation"],
                    "full_test": split["full_test"],
                }
                if args.resume:
                    evaluation_names = {
                        evaluation_name: indices
                        for evaluation_name, indices in evaluation_names.items()
                        if completed_json(output / f"{evaluation_name}_result.json") is None
                    }
                if evaluation_names:
                    evaluate_v2_checkpoint_sets(
                        spec,
                        dataset,
                        evaluation_names,
                        mean,
                        std,
                        batches[model_id],
                        LOOKUP,
                        Path(result["checkpoint"]),
                        output,
                        device,
                    )
                else:
                    print(f"SKIP complete full evaluation {model_id} seed={seed}", flush=True)

                supplemental = {
                    evaluation_name: json.loads(
                        (output / f"{evaluation_name}_result.json").read_text(encoding="utf-8")
                    )
                    for evaluation_name in ("full_validation", "full_test")
                }
                run_record = {
                    "model_id": model_id,
                    "seed": seed,
                    "training_and_core_test": result,
                    "supplemental_evaluations": supplemental,
                }
                summary["results"] = [
                    old
                    for old in summary["results"]
                    if not (old["model_id"] == model_id and old["seed"] == seed)
                ]
                summary["results"].append(run_record)
                write_json_atomic(summary_path, summary)
    except KeyboardInterrupt:
        summary["status"] = "paused"
        write_json_atomic(summary_path, summary)
        print("PAUSED: rerun the same command with --resume", flush=True)
        raise
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        write_json_atomic(summary_path, summary)
        raise

    summary["status"] = "complete"
    summary["completed_results"] = len(summary["results"])
    summary["expected_results"] = len(args.models) * len(args.seeds)
    write_json_atomic(summary_path, summary)
    print(f"COMPLETE: {summary['expected_results']} V3 model/seed runs", flush=True)


if __name__ == "__main__":
    main()
