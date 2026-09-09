"""Create the deterministic 40,000-sample SPMSM training subset.

Validation and test indices are copied unchanged from the audited 80/10/10 split.
The two rare edge Tavg bands are retained in full; the remaining quota is
distributed proportionally over the five central bands.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
SOURCE_SPLIT = ROOT / "cnn_zone" / "outputs" / "splits" / "scheme_a_tavg_bands_80_10_10.npz"
OUTPUT_SPLIT = ROOT / "cnn_zone" / "outputs" / "splits" / "scheme_a_tavg_bands_train40000_val6483_test6483.npz"
REPORT_DIR = ROOT / "reports" / "spmsm_topology_dataset"
TARGET_TRAIN = 40_000
SEED = 20260903


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def largest_remainder(capacities: np.ndarray, total: int) -> np.ndarray:
    raw = capacities / capacities.sum() * total
    result = np.floor(raw).astype(int)
    remainder = total - int(result.sum())
    order = np.argsort(-(raw - result), kind="stable")
    result[order[:remainder]] += 1
    return result


def main() -> None:
    source = np.load(SOURCE_SPLIT)
    targets = np.load(DATASET / "targets_tavg_delta.npy")
    train = np.asarray(source["train"], dtype=np.int64)
    validation = np.asarray(source["validation"], dtype=np.int64)
    test = np.asarray(source["test"], dtype=np.int64)
    edges = np.asarray(source["tavg_edges"], dtype=np.float64)
    labels = np.asarray(source["tavg_labels"]).astype(str)
    band = np.digitize(targets[:, 0], edges[1:-1], right=False)

    candidates = [train[band[train] == index] for index in range(len(labels))]
    capacities = np.asarray([len(values) for values in candidates], dtype=int)
    quotas = np.zeros_like(capacities)
    quotas[0] = capacities[0]
    quotas[-1] = capacities[-1]
    quotas[1:-1] = largest_remainder(capacities[1:-1], TARGET_TRAIN - quotas[0] - quotas[-1])
    if np.any(quotas > capacities) or int(quotas.sum()) != TARGET_TRAIN:
        raise AssertionError("Invalid per-band quotas")

    rng = np.random.default_rng(SEED)
    selected_parts: list[np.ndarray] = []
    unused_parts: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for index, values in enumerate(candidates):
        permuted = rng.permutation(values)
        chosen = np.sort(permuted[: quotas[index]])
        unused = np.sort(permuted[quotas[index] :])
        selected_parts.append(chosen)
        unused_parts.append(unused)
        rows.append(
            {
                "tavg_band": labels[index],
                "source_train": int(capacities[index]),
                "selected_train": int(quotas[index]),
                "unused_train_pool": int(capacities[index] - quotas[index]),
                "retention_percent": round(100.0 * quotas[index] / capacities[index], 3),
                "selection_policy": "retain_all_rare_edge_band" if index in (0, len(labels) - 1) else "proportional_seeded_sample",
            }
        )

    selected = np.sort(np.concatenate(selected_parts))
    unused = np.sort(np.concatenate(unused_parts))
    if len(selected) != TARGET_TRAIN or len(np.intersect1d(selected, unused)):
        raise AssertionError("Training subset partition failed")
    if len(np.intersect1d(selected, validation)) or len(np.intersect1d(selected, test)):
        raise AssertionError("Training subset leaks into validation or test")
    if not np.array_equal(np.sort(np.concatenate((selected, unused))), np.sort(train)):
        raise AssertionError("Selected and unused indices do not reconstruct the source training split")

    split_code = np.full(len(targets), -1, dtype=np.int8)
    split_code[unused] = 3
    split_code[selected] = 0
    split_code[validation] = 1
    split_code[test] = 2
    OUTPUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_SPLIT,
        split_code=split_code,
        train=selected,
        validation=validation,
        test=test,
        unused_train_pool=unused,
        tavg_edges=edges,
        tavg_labels=labels,
        seed=np.int64(SEED),
        source_split=str(SOURCE_SPLIT.resolve()),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "train40000_tavg_band_allocation.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "ready_no_training_started",
        "policy": "keep validation/test unchanged; retain both rare edge Tavg bands; proportionally sample central bands",
        "seed": SEED,
        "source_train": int(len(train)),
        "selected_train": int(len(selected)),
        "unused_train_pool": int(len(unused)),
        "validation": int(len(validation)),
        "test": int(len(test)),
        "index_overlap": {
            "train_validation": int(len(np.intersect1d(selected, validation))),
            "train_test": int(len(np.intersect1d(selected, test))),
            "validation_test": int(len(np.intersect1d(validation, test))),
        },
        "validation_unchanged": bool(np.array_equal(validation, source["validation"])),
        "test_unchanged": bool(np.array_equal(test, source["test"])),
        "index_hashes_sha256": {
            "train": digest(selected),
            "validation": digest(validation),
            "test": digest(test),
            "unused_train_pool": digest(unused),
        },
        "bands": rows,
        "artifacts": {"split_npz": str(OUTPUT_SPLIT.resolve()), "allocation_csv": str(csv_path.resolve())},
    }
    (REPORT_DIR / "train40000_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
