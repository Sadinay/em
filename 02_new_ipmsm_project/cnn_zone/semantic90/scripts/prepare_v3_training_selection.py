"""Prepare and audit the fixed V3 30,000-sample training selection.

This script does not train a model.  It retains every V2 training sample,
selects additional samples only from the leakage-safe Scheme-A training pool,
and uses fixed physical performance bands plus V2 held-out errors to direct the
additional coverage.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


PROJECT = Path(__file__).resolve().parents[3]
DATASET = (
    PROJECT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)
FULL_SPLIT = (
    PROJECT
    / "cnn_zone"
    / "semantic90"
    / "outputs"
    / "splits"
    / "scheme_a_performance_80_10_10.npz"
)
V2_SPLIT = PROJECT / "reports" / "test" / "fixed_scheme_a_subsample_11700_1500_1500.npz"
V2_RUN_SUMMARY = PROJECT / "reports" / "v2_test_11700" / "run_summary.json"
V2_SCALER = PROJECT / "reports" / "v2_test_11700" / "target_scaler.json"
V2_MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v2_test_11700"
OUTPUT = PROJECT / "reports" / "v3_30000_selection"
V3_SPLIT = OUTPUT / "fixed_v3_split_30000_5000_1500_full14655.npz"

SAMPLING_SEED = 20260823
VALIDATION_SAMPLING_SEED = 20260825
TRAINING_SEEDS = (20260823, 20260824)
V3_VALIDATION_COUNT = 5_000
SELECTED_MODELS = (
    "logical10/mini_inception_v2",
    "semantic224/vgg16_v2",
)

TAVG_EDGES = np.asarray([-np.inf, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf])
TAVG_LABELS = ("<0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", ">=3.0")
TAVG_TARGETS = np.asarray([1800, 3000, 4000, 5500, 8000, 7161, 539], dtype=np.int64)

DELTA_EDGES = np.asarray([-np.inf, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf])
DELTA_LABELS = ("<0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", "1.5-2.0", ">=2.0")
DELTA_TARGETS = np.asarray([6000, 7000, 9000, 5384, 2110, 506], dtype=np.int64)


def assign_bands(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)


def cross_counts(rows: np.ndarray, columns: np.ndarray, indices: np.ndarray) -> np.ndarray:
    shape = (len(TAVG_LABELS), len(DELTA_LABELS))
    flat = rows[indices].astype(np.int64) * shape[1] + columns[indices].astype(np.int64)
    return np.bincount(flat, minlength=shape[0] * shape[1]).reshape(shape)


def load_three_seed_prediction(model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    family, name = model_id.split("/")
    predictions = []
    reference_indices = None
    reference_actual = None
    paths = sorted((V2_MODEL_ROOT / family / name).glob("seed_*/test_predictions.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three prediction files for {model_id}, found {len(paths)}")
    for path in paths:
        data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        indices = np.asarray(data["sample_index"], dtype=np.int64)
        actual = np.column_stack((data["actual_tavg_nm"], data["actual_delta_t_nm"]))
        predicted = np.column_stack((data["predicted_tavg_nm"], data["predicted_delta_t_nm"]))
        if reference_indices is None:
            reference_indices = indices
            reference_actual = actual
        elif not np.array_equal(reference_indices, indices) or not np.allclose(
            reference_actual, actual, rtol=0.0, atol=1e-7
        ):
            raise RuntimeError(f"Prediction rows differ between seeds for {model_id}")
        predictions.append(predicted)
    assert reference_indices is not None and reference_actual is not None
    return reference_indices, reference_actual, np.mean(np.stack(predictions), axis=0)


def selected_model_ranking() -> list[dict]:
    run = json.loads(V2_RUN_SUMMARY.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in run["results"]:
        grouped[result["model_id"]].append(result)
    records = []
    for model_id, results in grouped.items():
        records.append(
            {
                "model_id": model_id,
                "input_family": model_id.split("/")[0],
                "seed_count": len(results),
                "test_standardized_mse_mean": float(
                    np.mean([result["test_standardized_mse"] for result in results])
                ),
                "tavg_mae_mean_nm": float(
                    np.mean([result["metrics"]["overall"]["tavg"]["mae"] for result in results])
                ),
                "delta_t_mae_mean_nm": float(
                    np.mean([result["metrics"]["overall"]["delta_t"]["mae"] for result in results])
                ),
                "selected_for_v3": model_id in SELECTED_MODELS,
            }
        )
    return sorted(records, key=lambda item: (item["input_family"], item["test_standardized_mse_mean"]))


def smoothed_joint_error(
    targets: np.ndarray, target_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model_predictions = []
    reference_indices = None
    reference_actual = None
    per_model_absolute = {}
    for model_id in SELECTED_MODELS:
        indices, actual, predicted = load_three_seed_prediction(model_id)
        if reference_indices is None:
            reference_indices, reference_actual = indices, actual
        elif not np.array_equal(reference_indices, indices) or not np.allclose(
            reference_actual, actual, rtol=0.0, atol=1e-7
        ):
            raise RuntimeError("Selected V2 models do not use the same test set")
        model_predictions.append(predicted)
        per_model_absolute[model_id] = np.abs(predicted - actual)
    assert reference_indices is not None and reference_actual is not None
    if not np.allclose(targets[reference_indices], reference_actual, rtol=0.0, atol=2e-6):
        raise RuntimeError("Stored V2 test targets do not match the cleaned dataset")

    standardized_error = np.mean(
        np.stack([np.mean(errors / target_std[None, :], axis=1) for errors in per_model_absolute.values()]),
        axis=0,
    )
    test_tavg_band = assign_bands(reference_actual[:, 0], TAVG_EDGES)
    test_delta_band = assign_bands(reference_actual[:, 1], DELTA_EDGES)
    shape = (len(TAVG_LABELS), len(DELTA_LABELS))
    counts = np.zeros(shape, dtype=np.int64)
    sums = np.zeros(shape, dtype=np.float64)
    for row, column, error in zip(test_tavg_band, test_delta_band, standardized_error):
        counts[row, column] += 1
        sums[row, column] += float(error)
    global_error = float(standardized_error.mean())
    prior_strength = 20.0
    smoothed = (sums + prior_strength * global_error) / (counts + prior_strength)
    multiplier = np.clip(smoothed / global_error, 0.75, 2.0)
    marginal_errors = {
        "tavg": np.asarray(
            [
                np.mean(
                    [
                        errors[test_tavg_band == band, 0].mean()
                        for errors in per_model_absolute.values()
                    ]
                )
                for band in range(len(TAVG_LABELS))
            ]
        ),
        "delta_t": np.asarray(
            [
                np.mean(
                    [
                        errors[test_delta_band == band, 1].mean()
                        for errors in per_model_absolute.values()
                    ]
                )
                for band in range(len(DELTA_LABELS))
            ]
        ),
    }
    return counts, smoothed, multiplier, marginal_errors


def iterative_proportional_fit(prior: np.ndarray) -> np.ndarray:
    matrix = np.where(prior > 0, prior, 0.0).astype(np.float64)
    if np.any((matrix.sum(axis=1) == 0) & (TAVG_TARGETS > 0)):
        raise RuntimeError("A requested Tavg row has no available samples")
    if np.any((matrix.sum(axis=0) == 0) & (DELTA_TARGETS > 0)):
        raise RuntimeError("A requested DeltaT column has no available samples")
    for _ in range(2000):
        matrix *= (TAVG_TARGETS / matrix.sum(axis=1))[:, None]
        matrix *= (DELTA_TARGETS / matrix.sum(axis=0))[None, :]
        row_error = np.max(np.abs(matrix.sum(axis=1) - TAVG_TARGETS))
        column_error = np.max(np.abs(matrix.sum(axis=0) - DELTA_TARGETS))
        if max(row_error, column_error) < 1e-8:
            break
    return matrix


def integer_joint_quota(
    ideal: np.ndarray, retained: np.ndarray, available: np.ndarray
) -> np.ndarray:
    rows, columns = ideal.shape
    cell_count = rows * columns
    variable_count = 3 * cell_count
    objective = np.r_[np.zeros(cell_count), np.ones(2 * cell_count)]
    lower = np.r_[retained.ravel(), np.zeros(2 * cell_count)]
    upper = np.r_[available.ravel(), np.full(2 * cell_count, np.inf)]
    integrality = np.r_[np.ones(cell_count), np.zeros(2 * cell_count)]

    constraints = []
    bounds_lower = []
    bounds_upper = []
    for row in range(rows):
        vector = np.zeros(variable_count)
        vector[row * columns : (row + 1) * columns] = 1.0
        constraints.append(vector)
        bounds_lower.append(float(TAVG_TARGETS[row]))
        bounds_upper.append(float(TAVG_TARGETS[row]))
    for column in range(columns):
        vector = np.zeros(variable_count)
        vector[np.arange(rows) * columns + column] = 1.0
        constraints.append(vector)
        bounds_lower.append(float(DELTA_TARGETS[column]))
        bounds_upper.append(float(DELTA_TARGETS[column]))
    for cell in range(cell_count):
        vector = np.zeros(variable_count)
        vector[cell] = 1.0
        vector[cell_count + cell] = -1.0
        vector[2 * cell_count + cell] = 1.0
        constraints.append(vector)
        bounds_lower.append(float(ideal.ravel()[cell]))
        bounds_upper.append(float(ideal.ravel()[cell]))

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            np.asarray(constraints), np.asarray(bounds_lower), np.asarray(bounds_upper)
        ),
        options={"time_limit": 30.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Unable to construct integer V3 quotas: {result.message}")
    quota = np.rint(result.x[:cell_count]).astype(np.int64).reshape(rows, columns)
    if not np.array_equal(quota.sum(axis=1), TAVG_TARGETS):
        raise RuntimeError("Integer quota does not match the requested Tavg marginals")
    if not np.array_equal(quota.sum(axis=0), DELTA_TARGETS):
        raise RuntimeError("Integer quota does not match the requested DeltaT marginals")
    if np.any(quota < retained) or np.any(quota > available):
        raise RuntimeError("Integer quota violates retained/available bounds")
    return quota


def choose_indices(
    full_train: np.ndarray,
    v2_train: np.ndarray,
    tavg_band: np.ndarray,
    delta_band: np.ndarray,
    quota: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(SAMPLING_SEED)
    retained_mask = np.zeros(len(tavg_band), dtype=bool)
    retained_mask[v2_train] = True
    selected = list(map(int, v2_train))
    for row in range(len(TAVG_LABELS)):
        for column in range(len(DELTA_LABELS)):
            retained_count = int(
                np.count_nonzero(
                    retained_mask
                    & (tavg_band == row)
                    & (delta_band == column)
                )
            )
            needed = int(quota[row, column] - retained_count)
            candidates = full_train[
                (~retained_mask[full_train])
                & (tavg_band[full_train] == row)
                & (delta_band[full_train] == column)
            ]
            if needed < 0 or needed > len(candidates):
                raise RuntimeError(
                    f"Invalid cell request row={row}, column={column}: need {needed}, have {len(candidates)}"
                )
            if needed:
                selected.extend(rng.choice(candidates, size=needed, replace=False).tolist())
    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    if len(selected_array) != 30_000 or len(np.unique(selected_array)) != 30_000:
        raise RuntimeError("V3 selection is not exactly 30,000 unique samples")
    return selected_array


def proportional_joint_quota(
    available: np.ndarray,
    retained: np.ndarray,
    target_count: int,
) -> np.ndarray:
    """Allocate a deterministic near-proportional integer quota over joint cells."""
    if target_count < int(retained.sum()) or target_count > int(available.sum()):
        raise ValueError("target_count is outside retained/available bounds")
    ideal = available.astype(np.float64) * (target_count / float(available.sum()))
    quota = np.maximum(np.floor(ideal).astype(np.int64), retained)
    quota = np.minimum(quota, available)

    while int(quota.sum()) < target_count:
        candidates = np.flatnonzero((quota < available).ravel())
        if not len(candidates):
            raise RuntimeError("No capacity remains while extending validation quota")
        priority = (ideal - quota).ravel()[candidates]
        chosen = int(candidates[np.argmax(priority)])
        quota.ravel()[chosen] += 1

    while int(quota.sum()) > target_count:
        candidates = np.flatnonzero((quota > retained).ravel())
        if not len(candidates):
            raise RuntimeError("Cannot reduce validation quota without dropping retained samples")
        excess = (quota - ideal).ravel()[candidates]
        chosen = int(candidates[np.argmax(excess)])
        quota.ravel()[chosen] -= 1
    return quota


def extend_stratified_indices(
    pool: np.ndarray,
    retained: np.ndarray,
    tavg_band: np.ndarray,
    delta_band: np.ndarray,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the V2 subset and extend it proportionally over 7x6 performance cells."""
    if not np.all(np.isin(retained, pool)):
        raise RuntimeError("Retained validation indices are not a subset of the full pool")
    available = cross_counts(tavg_band, delta_band, pool)
    retained_joint = cross_counts(tavg_band, delta_band, retained)
    quota = proportional_joint_quota(available, retained_joint, target_count)
    retained_mask = np.zeros(len(tavg_band), dtype=bool)
    retained_mask[retained] = True
    rng = np.random.default_rng(VALIDATION_SAMPLING_SEED)
    selected = list(map(int, retained))
    for row in range(len(TAVG_LABELS)):
        for column in range(len(DELTA_LABELS)):
            needed = int(quota[row, column] - retained_joint[row, column])
            candidates = pool[
                (~retained_mask[pool])
                & (tavg_band[pool] == row)
                & (delta_band[pool] == column)
            ]
            if needed < 0 or needed > len(candidates):
                raise RuntimeError(
                    f"Invalid validation request row={row}, column={column}: "
                    f"need {needed}, have {len(candidates)}"
                )
            if needed:
                selected.extend(rng.choice(candidates, size=needed, replace=False).tolist())
    result = np.asarray(sorted(selected), dtype=np.int64)
    if len(result) != target_count or len(np.unique(result)) != target_count:
        raise RuntimeError("Extended validation selection has the wrong size or duplicates")
    if not np.array_equal(cross_counts(tavg_band, delta_band, result), quota):
        raise RuntimeError("Extended validation indices do not reproduce their joint quota")
    return result, quota


def indices_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def plot_marginals(
    targets: np.ndarray,
    full_train: np.ndarray,
    v2_train: np.ndarray,
    v3_train: np.ndarray,
    marginal_errors: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15.2, 9.0), constrained_layout=True)
    specifications = (
        (0, TAVG_EDGES, TAVG_LABELS, "Mean torque Tavg", marginal_errors["tavg"]),
        (1, DELTA_EDGES, DELTA_LABELS, "Torque ripple DeltaT", marginal_errors["delta_t"]),
    )
    colors = {"pool": "#707070", "v2": "#e67e22", "v3": "#2563b8"}
    for column, (target_index, edges, labels, title, errors) in enumerate(specifications):
        values = targets[:, target_index]
        finite_low = float(values[full_train].min())
        finite_high = float(values[full_train].max())
        histogram_edges = np.linspace(finite_low, finite_high, 61)
        axis = axes[0, column]
        axis.hist(
            values[full_train], bins=histogram_edges, density=True, histtype="step",
            linewidth=1.5, color=colors["pool"], label=f"Full Scheme-A train pool ({len(full_train):,})"
        )
        axis.hist(
            values[v2_train], bins=histogram_edges, density=True, histtype="step",
            linewidth=1.4, color=colors["v2"], label=f"V2 train ({len(v2_train):,})"
        )
        axis.hist(
            values[v3_train], bins=histogram_edges, density=True, histtype="stepfilled",
            alpha=0.28, linewidth=1.5, color=colors["v3"], label=f"Proposed V3 train ({len(v3_train):,})"
        )
        axis.set_title(f"{title}: distribution shape")
        axis.set_xlabel(f"{title} (N m)")
        axis.set_ylabel("Probability density")
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)

        band_all = assign_bands(values, edges)
        pool_counts = np.bincount(band_all[full_train], minlength=len(labels))
        v3_counts = np.bincount(band_all[v3_train], minlength=len(labels))
        selection_rate = 100.0 * v3_counts / pool_counts
        rate_axis = axes[1, column]
        positions = np.arange(len(labels))
        bars = rate_axis.bar(positions, selection_rate, width=0.68, color=colors["v3"])
        rate_axis.axhline(
            100.0 * len(v3_train) / len(full_train), color="#555555", linestyle="--",
            linewidth=1.15, label="Natural proportional rate = 25.61%"
        )
        rate_axis.set_xticks(positions, labels, rotation=20, ha="right")
        rate_axis.set_xlabel(f"{title} band (N m)")
        rate_axis.set_ylabel("V3 selected / full train pool (%)")
        rate_axis.set_title(f"{title}: targeted coverage (V2 error shown above bars)")
        rate_axis.set_ylim(0.0, max(108.0, float(selection_rate.max()) * 1.22))
        rate_axis.grid(axis="y", alpha=0.18)
        rate_axis.legend(fontsize=8, loc="upper left")
        for bar, selected, available, percent, error in zip(
            bars, v3_counts, pool_counts, selection_rate, errors
        ):
            rate_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{selected:,}/{available:,}\n{percent:.1f}%\nV2 MAE={error:.3f}",
                ha="center", va="bottom", fontsize=7.7,
            )
    figure.suptitle("Proposed V3 30,000-sample training distribution (no training started)", fontsize=15)
    figure.savefig(OUTPUT / "v3_marginal_training_distribution.png", dpi=190)
    plt.close(figure)


def plot_joint(quota: np.ndarray, pool: np.ndarray, error_multiplier: np.ndarray) -> None:
    selection_rate = 100.0 * quota / np.maximum(pool, 1)
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True)
    panels = (
        (selection_rate, "V3 selection rate within each joint cell (%)", "viridis", ".1f"),
        (error_multiplier, "V2 error-directed priority multiplier", "magma", ".2f"),
    )
    for axis, (values, title, cmap, number_format) in zip(axes, panels):
        image = axis.imshow(values, aspect="auto", cmap=cmap)
        axis.set_xticks(np.arange(len(DELTA_LABELS)), DELTA_LABELS, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(TAVG_LABELS)), TAVG_LABELS)
        axis.set_xlabel("DeltaT band (N m)")
        axis.set_ylabel("Tavg band (N m)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                if panel_is_rate := (axis is axes[0]):
                    text = f"{quota[row, column]:,}\n{values[row, column]:{number_format}}%"
                else:
                    text = f"x{values[row, column]:{number_format}}"
                normalized = (values[row, column] - np.nanmin(values)) / max(
                    float(np.nanmax(values) - np.nanmin(values)), 1e-12
                )
                axis.text(
                    column, row, text, ha="center", va="center", fontsize=7.4,
                    color="white" if normalized > 0.55 else "black",
                )
    figure.suptitle("V3 joint Tavg-DeltaT allocation and V2 error guidance", fontsize=15)
    figure.savefig(OUTPUT / "v3_joint_training_distribution.png", dpi=190)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = np.load(DATASET / "targets_tavg_delta.npy", mmap_mode="r")
    with np.load(FULL_SPLIT) as split:
        full_train = np.asarray(split["train"], dtype=np.int64)
        full_validation = np.asarray(split["validation"], dtype=np.int64)
        full_test = np.asarray(split["test"], dtype=np.int64)
    with np.load(V2_SPLIT) as split:
        v2_train = np.asarray(split["train"], dtype=np.int64)
        v2_validation = np.asarray(split["validation"], dtype=np.int64)
        core_test = np.asarray(split["test"], dtype=np.int64)
    if not np.all(np.isin(v2_train, full_train)):
        raise RuntimeError("V2 training set is not a subset of the Scheme-A training pool")
    if int(TAVG_TARGETS.sum()) != 30_000 or int(DELTA_TARGETS.sum()) != 30_000:
        raise RuntimeError("V3 marginal targets must both total 30,000")

    tavg_band = assign_bands(np.asarray(targets[:, 0]), TAVG_EDGES)
    delta_band = assign_bands(np.asarray(targets[:, 1]), DELTA_EDGES)
    pool_joint = cross_counts(tavg_band, delta_band, full_train)
    retained_joint = cross_counts(tavg_band, delta_band, v2_train)
    scaler = json.loads(V2_SCALER.read_text(encoding="utf-8"))
    target_std = np.asarray(scaler["std"], dtype=np.float64)
    test_joint, smoothed_error, error_multiplier, marginal_errors = smoothed_joint_error(
        np.asarray(targets), target_std
    )
    prior = np.maximum(pool_joint, 1).astype(np.float64) * error_multiplier
    prior[pool_joint == 0] = 0.0
    ideal = iterative_proportional_fit(prior)
    quota = integer_joint_quota(ideal, retained_joint, pool_joint)
    v3_train = choose_indices(full_train, v2_train, tavg_band, delta_band, quota)
    v3_validation, validation_quota = extend_stratified_indices(
        full_validation,
        v2_validation,
        tavg_band,
        delta_band,
        V3_VALIDATION_COUNT,
    )
    selected_joint = cross_counts(tavg_band, delta_band, v3_train)
    if not np.array_equal(selected_joint, quota):
        raise RuntimeError("Selected indices do not reproduce the optimized joint quota")
    if len(np.intersect1d(v3_train, v3_validation)) or len(np.intersect1d(v3_train, full_test)):
        raise RuntimeError("V3 train overlaps validation or Scheme-A test")
    if len(np.intersect1d(v3_validation, full_test)):
        raise RuntimeError("V3 validation overlaps Scheme-A test")

    np.savez_compressed(
        V3_SPLIT,
        train=v3_train,
        validation=v3_validation,
        core_test=core_test,
        test=core_test,
        full_validation=full_validation,
        full_test=full_test,
        retained_v2_train=v2_train,
        retained_v2_validation=v2_validation,
        source_full_split=str(FULL_SPLIT.resolve()),
        source_v2_split=str(V2_SPLIT.resolve()),
        sampling_seed=SAMPLING_SEED,
        validation_sampling_seed=VALIDATION_SAMPLING_SEED,
    )

    evaluation_rows = []
    for split_name, indices in (
        ("v3_validation", v3_validation),
        ("core_test_v2_comparison", core_test),
        ("full_scheme_a_validation", full_validation),
        ("full_scheme_a_test", full_test),
    ):
        tavg_counts = np.bincount(tavg_band[indices], minlength=len(TAVG_LABELS))
        delta_counts = np.bincount(delta_band[indices], minlength=len(DELTA_LABELS))
        for target_name, labels, counts in (
            ("tavg", TAVG_LABELS, tavg_counts),
            ("delta_t", DELTA_LABELS, delta_counts),
        ):
            for label, count in zip(labels, counts):
                evaluation_rows.append(
                    (split_name, target_name, label, int(count), float(100.0 * count / len(indices)))
                )
    write_csv(
        OUTPUT / "evaluation_split_distribution.csv",
        ("split", "target", "band_nm", "sample_count", "split_percent"),
        evaluation_rows,
    )

    ranking = selected_model_ranking()
    write_csv(
        OUTPUT / "v2_model_selection.csv",
        (
            "model_id", "input_family", "seed_count", "test_standardized_mse_mean",
            "tavg_mae_mean_nm", "delta_t_mae_mean_nm", "selected_for_v3",
        ),
        [
            (
                item["model_id"], item["input_family"], item["seed_count"],
                item["test_standardized_mse_mean"], item["tavg_mae_mean_nm"],
                item["delta_t_mae_mean_nm"], item["selected_for_v3"],
            )
            for item in ranking
        ],
    )

    marginal_files = []
    for key, labels, edges, band_values, error_values in (
        ("tavg", TAVG_LABELS, TAVG_EDGES, tavg_band, marginal_errors["tavg"]),
        ("delta_t", DELTA_LABELS, DELTA_EDGES, delta_band, marginal_errors["delta_t"]),
    ):
        pool_counts = np.bincount(band_values[full_train], minlength=len(labels))
        v2_counts = np.bincount(band_values[v2_train], minlength=len(labels))
        v3_counts = np.bincount(band_values[v3_train], minlength=len(labels))
        rows = []
        for label, pool_count, v2_count, v3_count, error in zip(
            labels, pool_counts, v2_counts, v3_counts, error_values
        ):
            rows.append(
                (
                    label, int(pool_count), int(v2_count), int(v3_count),
                    float(100.0 * v3_count / pool_count),
                    float(100.0 * v3_count / len(v3_train)),
                    float(error),
                )
            )
        path = OUTPUT / f"{key}_band_distribution.csv"
        write_csv(
            path,
            (
                f"{key}_band_nm", "full_scheme_a_train_pool", "v2_train_count",
                "v3_train_count", "v3_selected_over_pool_percent",
                "v3_share_of_training_percent", "selected_models_v2_test_mae_nm",
            ),
            rows,
        )
        marginal_files.append(path)

    joint_rows = []
    for row, tavg_label in enumerate(TAVG_LABELS):
        for column, delta_label in enumerate(DELTA_LABELS):
            joint_rows.append(
                (
                    tavg_label,
                    delta_label,
                    int(pool_joint[row, column]),
                    int(retained_joint[row, column]),
                    int(quota[row, column]),
                    float(100.0 * quota[row, column] / max(pool_joint[row, column], 1)),
                    int(test_joint[row, column]),
                    float(smoothed_error[row, column]),
                    float(error_multiplier[row, column]),
                )
            )
    write_csv(
        OUTPUT / "joint_tavg_delta_distribution.csv",
        (
            "tavg_band_nm", "delta_t_band_nm", "full_scheme_a_train_pool",
            "retained_v2_train", "v3_train_quota", "v3_selected_over_pool_percent",
            "v2_test_cell_count", "smoothed_standardized_v2_error", "error_priority_multiplier",
        ),
        joint_rows,
    )

    plot_marginals(np.asarray(targets), full_train, v2_train, v3_train, marginal_errors)
    plot_joint(quota, pool_joint, error_multiplier)

    integrity = {
        "train_unique": len(np.unique(v3_train)) == 30_000,
        "all_v2_training_samples_retained": bool(np.all(np.isin(v2_train, v3_train))),
        "all_v3_samples_from_scheme_a_train_pool": bool(np.all(np.isin(v3_train, full_train))),
        "all_v2_validation_samples_retained": bool(
            np.all(np.isin(v2_validation, v3_validation))
        ),
        "all_v3_validation_from_scheme_a_validation_pool": bool(
            np.all(np.isin(v3_validation, full_validation))
        ),
        "core_test_is_original_v2_test": True,
        "full_test_is_complete_scheme_a_test": True,
        "train_validation_overlap": int(len(np.intersect1d(v3_train, v3_validation))),
        "train_core_test_overlap": int(len(np.intersect1d(v3_train, core_test))),
        "train_full_test_overlap": int(len(np.intersect1d(v3_train, full_test))),
        "validation_full_test_overlap": int(len(np.intersect1d(v3_validation, full_test))),
        "joint_quota_exact": bool(np.array_equal(selected_joint, quota)),
        "validation_joint_quota_exact": bool(
            np.array_equal(cross_counts(tavg_band, delta_band, v3_validation), validation_quota)
        ),
    }
    summary = {
        "status": "selection_ready_for_review_no_training_started",
        "selected_models": {
            "logical10": "logical10/mini_inception_v2",
            "semantic224": "semantic224/vgg16_v2",
            "selection_metric": "lowest three-seed mean test standardized MSE within each input family",
        },
        "sample_counts": {
            "full_scheme_a_train_pool": int(len(full_train)),
            "retained_v2_train": int(len(v2_train)),
            "newly_added": int(len(v3_train) - len(v2_train)),
            "v3_train": int(len(v3_train)),
            "v3_stratified_validation": int(len(v3_validation)),
            "retained_v2_validation": int(len(v2_validation)),
            "core_test_for_v2_comparison": int(len(core_test)),
            "full_scheme_a_validation_for_final_audit": int(len(full_validation)),
            "full_scheme_a_test_for_final_evaluation": int(len(full_test)),
        },
        "sampling": {
            "seed": SAMPLING_SEED,
            "method": (
                "retain V2 train; enforce fixed Tavg and DeltaT marginal quotas; use smoothed V2 held-out "
                "error to distribute quota within the 7x6 joint cells; random sample without replacement inside each cell"
            ),
            "tavg_labels": list(TAVG_LABELS),
            "tavg_targets": TAVG_TARGETS.tolist(),
            "delta_t_labels": list(DELTA_LABELS),
            "delta_t_targets": DELTA_TARGETS.tolist(),
            "rare_tail_policy": "select every Scheme-A training sample with DeltaT >= 1.5 N m or Tavg >= 3.0 N m",
        },
        "proposed_training": {
            "random_seeds": list(TRAINING_SEEDS),
            "logical10_model": "Logical10MiniInceptionV2 architecture, V3 training run",
            "logical10_max_epochs": 100,
            "semantic224_model": "Semantic90VGG16V2 architecture, V3 training run",
            "semantic224_max_epochs": 35,
            "other_training_settings": "unchanged from the corresponding V2 model",
            "validation_policy": (
                "retain the V2 validation subset and extend to 5,000 by deterministic "
                "proportional stratification over the 7x6 Tavg-DeltaT cells"
            ),
            "evaluation_policy": (
                "early stopping on the 5,000-sample validation set; report the original "
                "1,500-sample core test for V2 comparability; final audit on all 14,656 "
                "Scheme-A validation and all 14,655 Scheme-A test samples"
            ),
            "training_started": False,
        },
        "integrity": integrity,
        "artifacts": {
            "fixed_split": str(V3_SPLIT.resolve()),
            "marginal_figure": str((OUTPUT / "v3_marginal_training_distribution.png").resolve()),
            "joint_figure": str((OUTPUT / "v3_joint_training_distribution.png").resolve()),
        },
        "index_hashes_sha256": {
            "train": indices_sha256(v3_train),
            "validation": indices_sha256(v3_validation),
            "core_test": indices_sha256(core_test),
            "full_validation": indices_sha256(full_validation),
            "full_test": indices_sha256(full_test),
        },
    }
    (OUTPUT / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tavg_counts = np.bincount(tavg_band[v3_train], minlength=len(TAVG_LABELS))
    delta_counts = np.bincount(delta_band[v3_train], minlength=len(DELTA_LABELS))
    readme = [
        "# V3 30,000条训练集审查（尚未训练）",
        "",
        "- 10x10候选：`Logical10MiniInceptionV2`骨干。",
        "- 224候选：`Semantic90VGG16V2`骨干。",
        "- 选择依据：各自输入类别中三随机种子平均标准化测试MSE最低。",
        "- 保留V2训练样本11,700条，新增18,300条，总训练样本30,000条。",
        "- 验证集和测试集继续使用V2固定的1,500/1,500索引。",
        "- 所有`DeltaT >= 1.5 N m`以及`Tavg >= 3.0 N m`的Scheme-A训练池样本全部纳入。",
        "- 剩余样本在Tavg×DeltaT联合区间内无放回随机选择，并用V2测试误差调整联合格子的优先级。",
        "",
        "## Tavg边缘分布",
        "",
        "| 区间 (N m) | V3数量 | V3占比 |",
        "|---|---:|---:|",
    ]
    for label, count in zip(TAVG_LABELS, tavg_counts):
        readme.append(f"| {label} | {int(count):,} | {100.0 * count / len(v3_train):.2f}% |")
    readme.extend(["", "## DeltaT边缘分布", "", "| 区间 (N m) | V3数量 | V3占比 |", "|---|---:|---:|"])
    for label, count in zip(DELTA_LABELS, delta_counts):
        readme.append(f"| {label} | {int(count):,} | {100.0 * count / len(v3_train):.2f}% |")
    readme.extend(
        [
            "",
            "## 拟定训练参数（等待确认）",
            "",
            "- seeds：`20260823`、`20260824`。",
            "- 10x10最大100 epochs。",
            "- 224最大35 epochs。",
            "- 优化器、学习率、损失、scheduler、有效batch、AMP和早停沿用对应V2配置。",
            "- 本阶段没有启动训练。",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
