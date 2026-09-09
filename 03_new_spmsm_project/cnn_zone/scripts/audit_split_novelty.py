"""Audit train/test gene novelty and a nearest-neighbour FEMM-label baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
SPLIT = PROJECT / "cnn_zone" / "outputs" / "splits" / "scheme_a_tavg_bands_train40000_val6483_test6483.npz"
OUTPUT = PROJECT / "reports" / "V3" / "03_完整审核资料" / "原始审核包" / "summary_reports" / "split_novelty_audit.json"


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    result = {}
    for column, name in enumerate(("tavg", "delta_t")):
        error = predicted[:, column] - actual[:, column]
        denominator = float(np.sum((actual[:, column] - actual[:, column].mean()) ** 2))
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "r2": float(1.0 - np.sum(error**2) / denominator),
        }
    return result


def main() -> None:
    grids = np.load(DATA / "topology_bits.npy", mmap_mode="r")
    genes = np.asarray(grids.swapaxes(1, 2).reshape(len(grids), 120), dtype=np.uint8)
    targets = np.asarray(np.load(DATA / "targets_tavg_delta.npy", mmap_mode="r"), dtype=np.float32)
    with np.load(SPLIT) as source:
        train_indices = np.asarray(source["train"], dtype=np.int64)
        test_indices = np.asarray(source["test"], dtype=np.int64)
    train = torch.from_numpy(genes[train_indices].astype(np.float16)).cuda()
    train_ones = train.sum(dim=1)
    test = genes[test_indices]
    nearest_distances = np.empty(len(test), dtype=np.int16)
    nearest_positions = np.empty(len(test), dtype=np.int64)
    batch_size = 256
    with torch.inference_mode():
        for start in range(0, len(test), batch_size):
            block = torch.from_numpy(test[start : start + batch_size].astype(np.float16)).cuda()
            distances = block.sum(dim=1, keepdim=True) + train_ones.unsqueeze(0) - 2 * (block @ train.T)
            values, positions = distances.min(dim=1)
            end = start + len(block)
            nearest_distances[start:end] = values.cpu().numpy().astype(np.int16)
            nearest_positions[start:end] = positions.cpu().numpy()
    nearest_targets = targets[train_indices[nearest_positions]]
    actual = targets[test_indices]
    mean_prediction = np.broadcast_to(targets[train_indices].mean(axis=0), actual.shape)
    best_prediction_file = (
        PROJECT / "cnn_zone" / "models" / "v3_40000_6runs" / "polar90_224" /
        "vgg16_v2" / "seed_20260903" / "test_predictions.csv"
    )
    best_data = np.genfromtxt(best_prediction_file, delimiter=",", names=True)
    if not np.array_equal(np.asarray(best_data["sample_index"], dtype=np.int64), test_indices):
        raise RuntimeError("Best-model prediction rows do not match the frozen test split")
    best_prediction = np.column_stack((best_data["predicted_tavg"], best_data["predicted_delta_t"]))
    distance_bands = {}
    for label, lower, upper in (("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20), (">20", 21, 120)):
        mask = (nearest_distances >= lower) & (nearest_distances <= upper)
        distance_bands[label] = {
            "count": int(mask.sum()),
            "polar90_vgg16": metrics(actual[mask], best_prediction[mask]),
            "one_nearest_gene": metrics(actual[mask], nearest_targets[mask]),
        }
    unique_all = np.unique(np.packbits(genes, axis=1), axis=0).shape[0]
    distribution = {
        str(distance): int(np.sum(nearest_distances == distance))
        for distance in np.unique(nearest_distances)
    }
    thresholds = {
        f"le_{distance}": {
            "count": int(np.sum(nearest_distances <= distance)),
            "percent": float(100 * np.mean(nearest_distances <= distance)),
        }
        for distance in (0, 1, 2, 3, 4, 5, 10)
    }
    result = {
        "status": "complete",
        "gene_bits": 120,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "exact_unique_gene_count_in_clean_dataset": int(unique_all),
        "exact_duplicate_train_test_count": int(np.sum(nearest_distances == 0)),
        "nearest_train_hamming_distance": {
            "minimum": int(nearest_distances.min()),
            "median": float(np.median(nearest_distances)),
            "p90": float(np.percentile(nearest_distances, 90)),
            "p95": float(np.percentile(nearest_distances, 95)),
            "maximum": int(nearest_distances.max()),
            "distribution": distribution,
            "thresholds": thresholds,
        },
        "training_mean_baseline": metrics(actual, mean_prediction),
        "one_nearest_gene_femm_label_baseline": metrics(actual, nearest_targets),
        "best_model_by_nearest_train_hamming_band": distance_bands,
        "correlation_hamming_distance_with_best_model_absolute_error": {
            "tavg_pearson": float(np.corrcoef(nearest_distances, np.abs(best_prediction[:, 0] - actual[:, 0]))[0, 1]),
            "delta_t_pearson": float(np.corrcoef(nearest_distances, np.abs(best_prediction[:, 1] - actual[:, 1]))[0, 1]),
        },
        "interpretation": "No distance-zero row means no exact chromosome leakage. Small nonzero distances indicate interpolation among closely related GA chromosomes.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
