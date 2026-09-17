"""Guarded entry point for the eight frozen v3 CNN comparisons.

Running this file without ``--train`` performs only identity and forward/backward
checks.  Formal training is always sequential and requires the explicit flag.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CNN_ROOT = PROJECT / "cnn_zone"
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))

from src.dataset import IndexedSubset  # noqa: E402
from src.models_v2 import parameter_count  # noqa: E402
from src.training import (  # noqa: E402
    TrainingSpec,
    atomic_json,
    build_model,
    build_renderer,
    configure_reproducibility,
    make_inputs,
    train_one,
)


POOL = HERE / "data/candidate_pool"
SPLIT_AUDIT = HERE / "data/split_audit.json"
SPLIT_MEMBERSHIP = HERE / "data/split_membership.csv"
OUTPUT = HERE / "models/training_runs"
DEFAULT_SEED = 20260917
GAP_MAPPING = "experiments/cnn_comprehensive_v3/data/gap90/geometry.json"


def spec(name: str, input_mode: str, architecture: str, lookup: str | None,
         learning_rate: float, maximum_epochs: int, minimum_epochs: int,
         early_stopping_patience: int, physical_batch_size: int,
         circular_padding: bool = False) -> TrainingSpec:
    return TrainingSpec(
        experiment_id=name,
        input_mode=input_mode,
        architecture=architecture,
        lookup=lookup,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        max_epochs=maximum_epochs,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
        scheduler_patience=3,
        physical_batch_size=physical_batch_size,
        effective_batch_size=64,
        circular_angular_padding=circular_padding,
        gap_mapping=GAP_MAPPING if input_mode == "gap90_6x97" else None,
    )


SPECS = {
    "gene_6x20_smallcnn": spec("gene_6x20_smallcnn", "logical6x20", "small_cnn_v2", None, 1e-3, 100, 20, 12, 64),
    "gene_6x20_resnet20": spec("gene_6x20_resnet20", "logical6x20", "resnet20_v2", None, 3e-4, 100, 20, 12, 64),
    "gene_gap90_6x97_smallcnn": spec("gene_gap90_6x97_smallcnn", "gap90_6x97", "small_cnn_v2", None, 1e-3, 100, 20, 12, 64),
    "gene_gap90_6x97_resnet20": spec("gene_gap90_6x97_resnet20", "gap90_6x97", "resnet20_v2", None, 3e-4, 100, 20, 12, 64),
    "polar90_vgg16": spec("polar90_vgg16", "polar90_224", "vgg16_v2", "cnn_zone/outputs/lookups/spmsm_polar90_224_full_motor.npz", 1e-4, 40, 30, 8, 8),
    "polar90_resnet18": spec("polar90_resnet18", "polar90_224", "resnet18_v2", "cnn_zone/outputs/lookups/spmsm_polar90_224_full_motor.npz", 1e-4, 40, 30, 8, 16),
    "polar360_vgg16": spec("polar360_vgg16", "polar360_224", "vgg16_v2", "cnn_zone/outputs/lookups/spmsm_polar360_224_full_motor_ss4.npz", 1e-4, 40, 30, 8, 8, True),
    "polar360_resnet18": spec("polar360_resnet18", "polar360_224", "resnet18_v2", "cnn_zone/outputs/lookups/spmsm_polar360_224_full_motor_ss4.npz", 1e-4, 40, 30, 8, 16, True),
}


class CandidateDataset(Dataset):
    def __init__(self) -> None:
        bits_path = POOL / "topology_bits_120.npy"
        targets_path = POOL / "targets_tavg_delta_nm.npy"
        if not bits_path.exists() or not targets_path.exists():
            with (POOL / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 33299:
                raise ValueError(f"Expected 33299 manifest rows, got {len(rows)}")
            bits = np.fromiter((int(bit) for row in rows for bit in row["bits"]),
                               dtype=np.uint8, count=len(rows) * 120).reshape(len(rows), 120)
            targets = np.asarray([[float(row["tavg_nm"]), float(row["delta_t_nm"])]
                                  for row in rows], dtype=np.float32)
            np.save(bits_path, bits)
            np.save(targets_path, targets)
        self.bits = np.load(POOL / "topology_bits_120.npy", mmap_mode="r")
        self.targets = np.load(POOL / "targets_tavg_delta_nm.npy", mmap_mode="r")
        if self.bits.shape != (33299, 120) or self.targets.shape != (33299, 2):
            raise ValueError(f"Unexpected candidate arrays: {self.bits.shape}, {self.targets.shape}")

    def __len__(self) -> int:
        return len(self.bits)

    def __getitem__(self, index: int):
        bits = torch.from_numpy(np.array(self.bits[index], dtype=np.uint8, copy=True))
        target = torch.from_numpy(np.array(self.targets[index], dtype=np.float32, copy=True))
        return torch.tensor(index, dtype=torch.int64), bits, target


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_split() -> tuple[dict[str, np.ndarray], dict]:
    audit = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "frozen" or file_sha256(SPLIT_MEMBERSHIP) != audit["membership_sha256"]:
        raise RuntimeError("Frozen split identity check failed")
    paths = {name: HERE / f"data/{name}_indices.npy" for name in ("train", "validation", "test")}
    if any(not path.exists() for path in paths.values()):
        with SPLIT_MEMBERSHIP.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for name, path in paths.items():
            values = np.asarray([int(row["candidate_index"]) for row in rows if row["split"] == name],
                                dtype=np.int64)
            np.save(path, values)
    split = {name: np.load(path) for name, path in paths.items()}
    expected = {"train": 23309, "validation": 4995, "test": 4995}
    if {name: len(values) for name, values in split.items()} != expected:
        raise RuntimeError("Frozen split count check failed")
    joined = np.concatenate(tuple(split.values()))
    if len(np.unique(joined)) != 33299 or not np.array_equal(np.sort(joined), np.arange(33299)):
        raise RuntimeError("Frozen split overlaps or misses candidates")
    return split, audit


def preflight(names: list[str], dataset: CandidateDataset, split: dict[str, np.ndarray],
              device: torch.device, seed: int) -> dict:
    rows = []
    sample_indices = split["train"][:2]
    bits = torch.stack([dataset[int(index)][1] for index in sample_indices]).to(device)
    for name in names:
        item = SPECS[name]
        configure_reproducibility(seed)
        model = build_model(item).to(device)
        renderer = build_renderer(item, PROJECT, device)
        if item.input_mode not in ("logical6x20", "gap90_6x97"):
            model = model.to(memory_format=torch.channels_last)
        model.train()
        inputs = make_inputs(bits, item, renderer)
        prediction = model(inputs)
        loss = functional.mse_loss(prediction, torch.zeros_like(prediction))
        loss.backward()
        if prediction.shape != (2, 2) or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid preflight for {name}")
        rows.append({
            "configuration": name,
            "input_shape": list(inputs.shape),
            "output_shape": list(prediction.shape),
            "parameter_count": parameter_count(model),
            "status": "forward_backward_passed",
        })
        del model, renderer, inputs, prediction, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {"status": "passed_no_training", "device": str(device), "checks": rows}
    atomic_json(HERE / "reports/training_preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Start formal sequential training")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete runs and skip completed runs")
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS) + ("all",), default=["all"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    names = list(SPECS) if "all" in args.models else args.models
    split, split_audit = load_frozen_split()
    dataset = CandidateDataset()
    if args.train and not torch.cuda.is_available():
        raise RuntimeError("Formal training requires a CUDA GPU")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checked = preflight(names, dataset, split, device, args.seed)
    plan = {
        "status": "ready" if not args.train else "training",
        "configurations": names,
        "seed": args.seed,
        "split_counts": split_audit["total_counts"],
        "checkpoint_selection": "validation standardized MSE only",
        "test_access": "once after each completed run; never used for checkpoint selection or normalization",
        "normalization": "target mean/std fitted on the frozen training split only",
        "execution": "sequential on one GPU",
        "preflight": checked["status"],
    }
    atomic_json(HERE / "reports/training_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.train:
        print("PRECHECK COMPLETE: no training started")
        return
    for name in names:
        output_dir = OUTPUT / name / f"seed_{args.seed}"
        result_path = output_dir / "result.json"
        if result_path.exists():
            if args.resume:
                print(f"SKIP completed run: {name}", flush=True)
                continue
            raise RuntimeError(f"Completed output already exists for {name}; use --resume to skip it")
        train_one(SPECS[name], args.seed, dataset, split, PROJECT, output_dir,
                  torch.device(args.device), args.resume, None)


if __name__ == "__main__":
    main()
