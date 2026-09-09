"""Build auditable data for the V3 four-model disagreement diagnostic.

The four architecture predictions are each the mean of their two completed V3
seeds.  The script evaluates the complete frozen Scheme-A test population,
selects the 500 largest consensus-vs-FEMM standardized joint errors, and draws
500 non-overlapping random controls from the remainder.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v3_30000"
REPORT_ROOT = PROJECT / "reports" / "v3_30000"
DATA_ROOT = (
    PROJECT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)

MODELS = {
    "mini_inception": "logical10/mini_inception_v2",
    "resnet20": "logical10/resnet20_v2",
    "small_cnn": "logical10/small_cnn_v2",
    "vgg16_224": "semantic224/vgg16_v2",
}
SEEDS = (20260823, 20260824)
RANDOM_SEED = 20260827
HIGH_ERROR_COUNT = 500
RANDOM_COUNT = 500


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    if rank:
        left, right = rankdata(left), rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / denominator),
    }


def load_predictions() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    reference_indices: np.ndarray | None = None
    reference_actual: np.ndarray | None = None
    predictions: dict[str, np.ndarray] = {}
    for short_name, model_id in MODELS.items():
        family, model_name = model_id.split("/")
        seed_predictions = []
        for seed in SEEDS:
            path = (
                MODEL_ROOT
                / family
                / model_name
                / f"seed_{seed}"
                / "full_test_predictions.csv"
            )
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
            actual = np.asarray(
                [
                    [float(row["actual_tavg_nm"]), float(row["actual_delta_t_nm"])]
                    for row in rows
                ],
                dtype=np.float64,
            )
            predicted = np.asarray(
                [
                    [
                        float(row["predicted_tavg_nm"]),
                        float(row["predicted_delta_t_nm"]),
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )
            if reference_indices is None:
                reference_indices, reference_actual = indices, actual
            elif not np.array_equal(reference_indices, indices) or not np.allclose(
                reference_actual, actual, rtol=0.0, atol=1e-7
            ):
                raise RuntimeError(f"Prediction rows differ for {model_id} seed {seed}")
            seed_predictions.append(predicted)
        predictions[short_name] = np.mean(np.stack(seed_predictions), axis=0)
    assert reference_indices is not None and reference_actual is not None
    return reference_indices, reference_actual, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indices, actual, by_model = load_predictions()
    scaler = json.loads((REPORT_ROOT / "target_scaler.json").read_text(encoding="utf-8"))
    target_std = np.asarray(scaler["std"], dtype=np.float64)
    stacked = np.stack([by_model[name] for name in MODELS], axis=1)
    consensus = stacked.mean(axis=1)
    disagreement = stacked.std(axis=1, ddof=1)
    spread = stacked.max(axis=1) - stacked.min(axis=1)
    absolute_error = np.abs(consensus - actual)
    joint_error = np.sqrt(np.mean(((consensus - actual) / target_std) ** 2, axis=1))
    joint_disagreement = np.sqrt(np.mean((disagreement / target_std) ** 2, axis=1))

    order = np.argsort(-joint_error, kind="stable")
    high_positions = order[:HIGH_ERROR_COUNT]
    remaining = order[HIGH_ERROR_COUNT:]
    random_positions = np.random.default_rng(RANDOM_SEED).choice(
        remaining, size=RANDOM_COUNT, replace=False
    )
    selected_positions = np.concatenate((high_positions, random_positions))
    selected_groups = np.array(
        ["high_error"] * HIGH_ERROR_COUNT + ["random_control"] * RANDOM_COUNT
    )
    selection_ranks = np.concatenate(
        (np.arange(1, HIGH_ERROR_COUNT + 1), np.arange(1, RANDOM_COUNT + 1))
    )

    topology = np.load(DATA_ROOT / "topology_codes.npy", mmap_mode="r")
    samples = []
    for group, group_rank, position in zip(
        selected_groups, selection_ranks, selected_positions
    ):
        grid = np.asarray(topology[indices[position]], dtype=np.int64)
        flat = grid.reshape(-1)
        model_errors = {}
        model_deviations = {}
        for name in MODELS:
            error = by_model[name][position] - actual[position]
            deviation = by_model[name][position] - consensus[position]
            model_errors[name] = {
                "abs_error_tavg_nm": float(abs(error[0])),
                "abs_error_delta_t_nm": float(abs(error[1])),
                "joint_standardized_error": float(
                    np.sqrt(np.mean((error / target_std) ** 2))
                ),
            }
            model_deviations[name] = float(
                np.sqrt(np.mean((deviation / target_std) ** 2))
            )
        record = {
            "selection_group": str(group),
            "selection_rank": int(group_rank),
            "sample_index": int(indices[position]),
            "gene_code_flat_100": "".join(map(str, flat.tolist())),
            "gene_rows": ["".join(map(str, row.tolist())) for row in grid],
            "actual_tavg_nm": float(actual[position, 0]),
            "actual_delta_t_nm": float(actual[position, 1]),
            "predictions": {
                name: {
                    "tavg_nm": float(by_model[name][position, 0]),
                    "delta_t_nm": float(by_model[name][position, 1]),
                }
                for name in MODELS
            },
            "per_model_errors_vs_femm": model_errors,
            "per_model_joint_deviation_from_consensus": model_deviations,
            "check": {
                "mean_tavg_nm": float(consensus[position, 0]),
                "sigma_tavg_nm": float(disagreement[position, 0]),
                "spread_tavg_nm": float(spread[position, 0]),
                "abs_error_tavg_nm": float(absolute_error[position, 0]),
                "mean_delta_t_nm": float(consensus[position, 1]),
                "sigma_delta_t_nm": float(disagreement[position, 1]),
                "spread_delta_t_nm": float(spread[position, 1]),
                "abs_error_delta_t_nm": float(absolute_error[position, 1]),
                "joint_standardized_error": float(joint_error[position]),
                "joint_standardized_disagreement": float(joint_disagreement[position]),
                "highest_tavg_prediction_model": max(
                    MODELS, key=lambda name: by_model[name][position, 0]
                ),
                "lowest_tavg_prediction_model": min(
                    MODELS, key=lambda name: by_model[name][position, 0]
                ),
                "highest_delta_t_prediction_model": max(
                    MODELS, key=lambda name: by_model[name][position, 1]
                ),
                "lowest_delta_t_prediction_model": min(
                    MODELS, key=lambda name: by_model[name][position, 1]
                ),
            },
        }
        samples.append(record)

    high_error_threshold_10 = float(np.quantile(joint_error, 0.90))
    population_summary = {
        "sample_count": int(len(indices)),
        "target_std": {
            "tavg_nm": float(target_std[0]),
            "delta_t_nm": float(target_std[1]),
        },
        "correlations": {
            "pearson_joint_disagreement_vs_joint_error": correlation(
                joint_disagreement, joint_error
            ),
            "spearman_joint_disagreement_vs_joint_error": correlation(
                joint_disagreement, joint_error, rank=True
            ),
            "pearson_sigma_tavg_vs_abs_error_tavg": correlation(
                disagreement[:, 0], absolute_error[:, 0]
            ),
            "spearman_sigma_tavg_vs_abs_error_tavg": correlation(
                disagreement[:, 0], absolute_error[:, 0], rank=True
            ),
            "pearson_sigma_delta_t_vs_abs_error_delta_t": correlation(
                disagreement[:, 1], absolute_error[:, 1]
            ),
            "spearman_sigma_delta_t_vs_abs_error_delta_t": correlation(
                disagreement[:, 1], absolute_error[:, 1], rank=True
            ),
        },
        "overall": {
            "joint_error_mean": float(joint_error.mean()),
            "joint_error_median": float(np.median(joint_error)),
            "joint_disagreement_mean": float(joint_disagreement.mean()),
            "joint_disagreement_median": float(np.median(joint_disagreement)),
            "top_10_percent_error_threshold": high_error_threshold_10,
        },
        "model_metrics": {},
        "pairwise": [],
        "disagreement_enrichment": [],
        "disagreement_deciles": [],
    }

    for name, predicted in by_model.items():
        population_summary["model_metrics"][name] = {
            "tavg": regression_metrics(actual[:, 0], predicted[:, 0]),
            "delta_t": regression_metrics(actual[:, 1], predicted[:, 1]),
        }

    for left, right in combinations(MODELS, 2):
        population_summary["pairwise"].append(
            {
                "model_a": left,
                "model_b": right,
                "mean_abs_difference_tavg_nm": float(
                    np.mean(np.abs(by_model[left][:, 0] - by_model[right][:, 0]))
                ),
                "mean_abs_difference_delta_t_nm": float(
                    np.mean(np.abs(by_model[left][:, 1] - by_model[right][:, 1]))
                ),
                "correlation_tavg": correlation(
                    by_model[left][:, 0], by_model[right][:, 0]
                ),
                "correlation_delta_t": correlation(
                    by_model[left][:, 1], by_model[right][:, 1]
                ),
            }
        )

    high_error_flag = joint_error >= high_error_threshold_10
    population_high_rate = float(high_error_flag.mean())
    for percentage in (1, 5, 10, 20):
        count = max(1, int(np.ceil(len(indices) * percentage / 100)))
        positions = np.argsort(-joint_disagreement, kind="stable")[:count]
        high_rate = float(high_error_flag[positions].mean())
        population_summary["disagreement_enrichment"].append(
            {
                "top_disagreement_percent": percentage,
                "sample_count": count,
                "mean_joint_error": float(joint_error[positions].mean()),
                "median_joint_error": float(np.median(joint_error[positions])),
                "high_error_top10_rate": high_rate,
                "high_error_lift_vs_population": float(high_rate / population_high_rate),
            }
        )

    sorted_positions = np.argsort(joint_disagreement, kind="stable")
    for decile_index, positions in enumerate(np.array_split(sorted_positions, 10), start=1):
        population_summary["disagreement_deciles"].append(
            {
                "disagreement_decile": decile_index,
                "label": f"D{decile_index} ({'lowest' if decile_index == 1 else 'highest' if decile_index == 10 else ''})".strip(),
                "sample_count": int(len(positions)),
                "mean_joint_disagreement": float(joint_disagreement[positions].mean()),
                "mean_joint_error": float(joint_error[positions].mean()),
                "median_joint_error": float(np.median(joint_error[positions])),
                "p90_joint_error": float(np.quantile(joint_error[positions], 0.90)),
                "mean_abs_error_tavg_nm": float(absolute_error[positions, 0].mean()),
                "mean_abs_error_delta_t_nm": float(absolute_error[positions, 1].mean()),
                "high_error_top10_rate": float(high_error_flag[positions].mean()),
            }
        )

    output = {
        "method": {
            "model_count": len(MODELS),
            "models": MODELS,
            "architecture_prediction": "mean of V3 seeds 20260823 and 20260824",
            "model_mean": "arithmetic mean across the four architecture predictions",
            "model_sigma": "sample standard deviation across four architectures (ddof=1)",
            "joint_error": "RMS of consensus-minus-FEMM errors standardized by V3 training target std",
            "joint_disagreement": "RMS of architecture sample sigmas standardized by V3 training target std",
            "high_error_selection": f"largest {HIGH_ERROR_COUNT} joint standardized consensus errors on full_test",
            "random_selection": f"uniform without replacement from remaining full_test samples; seed={RANDOM_SEED}",
        },
        "population_summary": population_summary,
        "selected_sample_summary": {
            group: {
                "sample_count": int(np.sum(selected_groups == group)),
                "mean_joint_standardized_error": float(
                    joint_error[selected_positions[selected_groups == group]].mean()
                ),
                "median_joint_standardized_error": float(
                    np.median(joint_error[selected_positions[selected_groups == group]])
                ),
                "mean_joint_standardized_disagreement": float(
                    joint_disagreement[selected_positions[selected_groups == group]].mean()
                ),
                "median_joint_standardized_disagreement": float(
                    np.median(joint_disagreement[selected_positions[selected_groups == group]])
                ),
            }
            for group in ("high_error", "random_control")
        },
        "integrity": {
            "selected_sample_count": int(len(selected_positions)),
            "selected_unique_sample_count": int(len(np.unique(selected_positions))),
            "high_error_count": int(np.sum(selected_groups == "high_error")),
            "random_control_count": int(np.sum(selected_groups == "random_control")),
            "groups_overlap": bool(
                len(np.intersect1d(high_positions, random_positions)) > 0
            ),
            "all_prediction_rows_aligned": True,
        },
        "interpretation_notes": [
            "The 500 high-error rows are deliberately outcome-selected using FEMM truth; do not use the combined 1,000 rows to estimate population prevalence or an unbiased correlation.",
            "Use population_summary for unbiased association/enrichment evidence across all 14,655 frozen test genes.",
            "A useful FEMM acquisition channel should show increasing true error as disagreement rises and should enrich high-error cases in top-disagreement subsets.",
            "Architecture predictions are two-seed means; sigma measures disagreement among four architecture ensembles, not seed uncertainty within one architecture.",
        ],
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} selected samples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
