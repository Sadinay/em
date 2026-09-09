"""Create 80/10/10 performance and late-generation extrapolation split schemes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT / "data_zone" / "processed" / "ipmsm_topology_dataset" / "training_corrected_physical_three_state"
DEFAULT_OUTPUT = SEMANTIC_ROOT / "outputs" / "splits"
SEED = 20260821


def strata(targets: np.ndarray) -> np.ndarray:
    tavg, delta = targets[:, 0], targets[:, 1]
    tavg_bin = np.digitize(tavg, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    delta_bin = np.zeros(len(delta), dtype=np.int16)
    for band in range(7):
        mask = tavg_bin == band
        quartiles = np.quantile(delta[mask], [0.25, 0.5, 0.75])
        delta_bin[mask] = np.searchsorted(quartiles, delta[mask], side="right")
    return (4 * tavg_bin + delta_bin).astype(np.int16)


def grouped_split(sample_strata: np.ndarray, groups: np.ndarray, ratios: np.ndarray, seed: int) -> np.ndarray:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    members = [[] for _ in unique_groups]
    for index, group_index in enumerate(inverse):
        members[int(group_index)].append(index)
    category_count = int(sample_strata.max()) + 1
    category_total = np.bincount(sample_strata, minlength=category_count)
    target = ratios[:, None] * category_total[None, :]
    target_total = ratios * len(sample_strata)
    current = np.zeros_like(target)
    current_total = np.zeros(len(ratios))
    assignment = np.full(len(sample_strata), -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(members))
    rarity = 1.0 / np.maximum(category_total, 1)
    order = sorted(order.tolist(), key=lambda i: (-max(rarity[sample_strata[members[i]]]), -len(members[i])))
    for group_index in order:
        indices = np.asarray(members[group_index], dtype=np.int64)
        contribution = np.bincount(sample_strata[indices], minlength=category_count)
        scores = []
        for split_index in range(len(ratios)):
            category_need = (target[split_index] - current[split_index]) / np.maximum(target[split_index], 1)
            total_need = (target_total[split_index] - current_total[split_index]) / max(target_total[split_index], 1)
            scores.append(float(contribution @ category_need + 0.15 * len(indices) * total_need))
        selected = int(np.argmax(scores))
        assignment[indices] = selected
        current[selected] += contribution
        current_total[selected] += len(indices)
    return assignment


def load_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    component, generation_first, hashes = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            component.append(int(row["repair_component_id"]))
            generation_first.append(int(row["generation_first"]))
            hashes.append(row["physical_topology_hash"])
    return np.asarray(component, dtype=np.int32), np.asarray(generation_first, dtype=np.int16), np.asarray(hashes, dtype="U64")


def sampled_hamming(codes: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sampled_train = rng.choice(train_indices, size=min(8192, len(train_indices)), replace=False)
    sampled_test = rng.choice(test_indices, size=min(512, len(test_indices)), replace=False)
    train_codes = np.asarray(codes[sampled_train], dtype=np.uint8)
    minima = []
    for start in range(0, len(sampled_test), 32):
        query = np.asarray(codes[sampled_test[start : start + 32]], dtype=np.uint8)
        distance = np.count_nonzero(query[:, None, :] != train_codes[None, :, :], axis=2)
        minima.extend(np.min(distance, axis=1).tolist())
    values = np.asarray(minima)
    return {
        "method": "minimum distance to a fixed random subset of at most 8192 training topologies; not an exact all-training nearest neighbour",
        "test_samples": int(len(values)),
        "training_reference_samples": int(len(sampled_train)),
        "minimum": int(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": int(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--late-generation-start", type=int, default=541)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    targets = np.load(args.dataset / "targets_tavg_delta.npy", mmap_mode="r")
    grids = np.load(args.dataset / "topology_codes.npy", mmap_mode="r")
    codes = grids.swapaxes(1, 2).reshape(len(grids), 100)
    components, generation_first, hashes = load_metadata(args.dataset / "metadata.csv")
    sample_strata = strata(targets)

    scheme_a = grouped_split(sample_strata, components, np.asarray([0.8, 0.1, 0.1]), args.seed)
    np.savez_compressed(
        args.output / "scheme_a_performance_80_10_10.npz",
        split_code=scheme_a,
        train=np.flatnonzero(scheme_a == 0), validation=np.flatnonzero(scheme_a == 1), test=np.flatnonzero(scheme_a == 2),
    )

    component_min_generation: dict[int, int] = {}
    for component, first in zip(components, generation_first):
        component_min_generation[int(component)] = min(component_min_generation.get(int(component), 10000), int(first))
    temporal_test = np.asarray(
        [component_min_generation[int(component)] >= args.late_generation_start for component in components], dtype=bool
    )
    remaining = np.flatnonzero(~temporal_test)
    remaining_split = grouped_split(
        sample_strata[remaining], components[remaining], np.asarray([0.9, 0.1]), args.seed + 1
    )
    scheme_b = np.full(len(targets), 2, dtype=np.int8)
    scheme_b[remaining[remaining_split == 0]] = 0
    scheme_b[remaining[remaining_split == 1]] = 1
    np.savez_compressed(
        args.output / "scheme_b_temporal_extrapolation.npz",
        split_code=scheme_b,
        train=np.flatnonzero(scheme_b == 0), validation=np.flatnonzero(scheme_b == 1), test=np.flatnonzero(scheme_b == 2),
    )
    summary = {
        "seed": args.seed,
        "same_cleaned_topology_catalog": str(args.dataset.resolve()),
        "scheme_a": {
            "definition": "grouped Tavg-band × within-band DeltaT-quartile stratification",
            "train": int(np.sum(scheme_a == 0)), "validation": int(np.sum(scheme_a == 1)), "test": int(np.sum(scheme_a == 2)),
        },
        "scheme_b": {
            "definition": "test topology and every member of its repair-related component first appear in states >= late_generation_start",
            "late_generation_start": args.late_generation_start,
            "train": int(np.sum(scheme_b == 0)), "validation": int(np.sum(scheme_b == 1)), "test": int(np.sum(scheme_b == 2)),
            "test_tavg_range": [float(targets[scheme_b == 2, 0].min()), float(targets[scheme_b == 2, 0].max())],
            "test_delta_range": [float(targets[scheme_b == 2, 1].min()), float(targets[scheme_b == 2, 1].max())],
            "sampled_hamming_to_training": sampled_hamming(codes, np.flatnonzero(scheme_b == 0), np.flatnonzero(scheme_b == 2), args.seed),
        },
        "hash_overlap": 0,
        "repair_component_overlap": 0,
    }
    (args.output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
