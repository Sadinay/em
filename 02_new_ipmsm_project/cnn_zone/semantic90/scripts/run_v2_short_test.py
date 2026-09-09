"""Run the six v2 regressors on the frozen 11,700/1,500/1,500 split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.training_v2 import (  # noqa: E402
    V2_TRAINING_SPECS,
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
FROZEN_SPLIT = PROJECT / "reports" / "test" / "fixed_scheme_a_subsample_11700_1500_1500.npz"
LEGACY_SCALER = PROJECT / "reports" / "test" / "target_scaler.json"
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v2_test_11700"
REPORT_ROOT = PROJECT / "reports" / "v2_test_11700"
DEFAULT_SEEDS = (20260822, 20260823, 20260824)


def load_and_validate_split(dataset_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = np.load(FROZEN_SPLIT)
    train = np.asarray(split["train"], dtype=np.int64)
    validation = np.asarray(split["validation"], dtype=np.int64)
    test = np.asarray(split["test"], dtype=np.int64)
    expected = (11700, 1500, 1500)
    if tuple(map(len, (train, validation, test))) != expected:
        raise RuntimeError(f"Frozen split sizes do not equal {expected}")
    joined = np.concatenate((train, validation, test))
    if len(np.unique(joined)) != len(joined):
        raise RuntimeError("Frozen train/validation/test indices overlap or contain duplicates")
    if joined.min() < 0 or joined.max() >= dataset_size:
        raise RuntimeError("Frozen split contains an out-of-range sample index")
    return train, validation, test


def write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Skip complete model/seed runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(V2_TRAINING_SPECS),
        default=list(V2_TRAINING_SPECS),
        help="Optional subset; defaults to all six v2 model IDs",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v2 fair short test")
    device = torch.device("cuda:0")
    dataset = GeneCodeDataset(DATASET, "all")
    train, validation, test = load_and_validate_split(len(dataset))
    targets = np.asarray(dataset.targets, dtype=np.float32)
    mean, std = target_scaler(targets[train])
    legacy_scaler = json.loads(LEGACY_SCALER.read_text(encoding="utf-8"))
    if not np.allclose(mean.numpy(), legacy_scaler["mean"], rtol=0.0, atol=1e-7):
        raise RuntimeError("New training-only target mean differs from the frozen legacy scaler")
    if not np.allclose(std.numpy(), legacy_scaler["std"], rtol=0.0, atol=1e-7):
        raise RuntimeError("New training-only target std differs from the frozen legacy scaler")
    probe = json.loads(BATCH_PROBE.read_text(encoding="utf-8"))
    batches = {item["model_id"]: int(item["selected"]) for item in probe["models"]}

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FROZEN_SPLIT, REPORT_ROOT / FROZEN_SPLIT.name)
    scaler_record = {
        "target_order": ["Tavg_Nm", "DeltaT_Nm"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "statistics_source": "only the same frozen 11,700-sample training subset",
        "verified_equal_to_legacy_short_test_scaler": True,
    }
    (REPORT_ROOT / "target_scaler.json").write_text(
        json.dumps(scaler_record, indent=2), encoding="utf-8"
    )
    summary_path = REPORT_ROOT / "run_summary.json"
    summary = {
        "status": "running",
        "experiment": "six v2 models, three-seed fair short test",
        "dataset": str(DATASET.resolve()),
        "split": str(FROZEN_SPLIT.resolve()),
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "indices_disjoint": True,
        "effective_batch_size": 64,
        "seeds": args.seeds,
        "models": args.models,
        "results": [],
    }
    write_summary(summary_path, summary)
    try:
        for model_id in args.models:
            spec = V2_TRAINING_SPECS[model_id]
            family, name = model_id.split("/")
            for seed in args.seeds:
                output = MODEL_ROOT / family / name / f"seed_{seed}"
                result_path = output / "result.json"
                if args.resume and result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result.get("status") != "complete":
                        raise RuntimeError(f"Incomplete result cannot be skipped: {result_path}")
                    print(f"SKIP complete {model_id} seed={seed}", flush=True)
                else:
                    result = train_v2_model(
                        spec,
                        dataset,
                        train,
                        validation,
                        test,
                        mean,
                        std,
                        batches[model_id],
                        LOOKUP,
                        output,
                        device,
                        seed,
                        resume_training=args.resume,
                    )
                summary["results"] = [
                    old
                    for old in summary["results"]
                    if not (old["model_id"] == model_id and old["seed"] == seed)
                ]
                summary["results"].append(result)
                write_summary(summary_path, summary)
    except KeyboardInterrupt:
        summary["status"] = "paused"
        write_summary(summary_path, summary)
        raise
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        write_summary(summary_path, summary)
        raise
    expected_results = len(args.models) * len(args.seeds)
    summary["status"] = "complete"
    summary["completed_results"] = len(summary["results"])
    summary["expected_results"] = expected_results
    write_summary(summary_path, summary)
    print(f"COMPLETE: {expected_results} model/seed runs", flush=True)


if __name__ == "__main__":
    main()
