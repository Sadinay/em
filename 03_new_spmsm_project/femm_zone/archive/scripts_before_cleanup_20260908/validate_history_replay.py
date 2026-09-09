"""Replay three representative SPMSM history genes with six FEMM angles."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from spmsm_mapping import DEFAULT_MAT, DEFAULT_TEMPLATE, build_topology, history_gene


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "femm_zone" / "results" / "history_replay_validation"
ANGLES = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
POLE_PAIRS = 4
CURRENT_AMPLITUDE_A = 3.5


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def candidate_selection(mat_path: Path) -> list[dict]:
    data = loadmat(
        mat_path,
        variable_names=["population_all", "population_noChange_all", "Tavg_all", "DeltaT_all", "Fitvalue_all"],
        squeeze_me=True,
    )
    corrected = np.asarray(data["population_all"], dtype=np.uint8)
    before = np.asarray(data["population_noChange_all"], dtype=np.uint8)
    tavg = np.asarray(data["Tavg_all"], dtype=np.float64)
    delta = np.asarray(data["DeltaT_all"], dtype=np.float64)
    fitness = np.asarray(data["Fitvalue_all"], dtype=np.float64)
    unchanged = np.all(corrected == before, axis=1)
    values = tavg[unchanged]
    targets = (
        ("low_tavg", float(np.min(values))),
        ("median_tavg", float(np.median(values))),
        ("high_tavg", float(np.max(values))),
    )
    selected: list[dict] = []
    used_genes: set[bytes] = set()
    for candidate_id, target in targets:
        for flat_index in np.argsort(np.abs(tavg - target), axis=None):
            row, state = np.unravel_index(int(flat_index), tavg.shape)
            if not unchanged[row, state]:
                continue
            gene = corrected[row, :, state]
            key = np.packbits(gene).tobytes()
            if key in used_genes:
                continue
            used_genes.add(key)
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "state_index_0based": int(state),
                    "population_row_0based": int(row),
                    "historical_tavg_nm": float(tavg[row, state]),
                    "historical_delta_t_nm": float(delta[row, state]),
                    "historical_fitness": float(fitness[row, state]),
                    "pm_cells": int(gene.sum()),
                    "before_after_gene_identical": True,
                }
            )
            break
    final_state = corrected.shape[2] - 1
    final_rows = np.flatnonzero(unchanged[:, final_state])
    final_values = tavg[final_rows, final_state]
    final_targets = (
        ("final_low_tavg", float(np.min(final_values))),
        ("final_median_tavg", float(np.median(final_values))),
        ("final_high_tavg", float(np.max(final_values))),
    )
    for candidate_id, target in final_targets:
        for row in np.argsort(np.abs(tavg[:, final_state] - target)):
            if not unchanged[row, final_state]:
                continue
            gene = corrected[row, :, final_state]
            key = np.packbits(gene).tobytes()
            if key in used_genes:
                continue
            used_genes.add(key)
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "state_index_0based": int(final_state),
                    "population_row_0based": int(row),
                    "historical_tavg_nm": float(tavg[row, final_state]),
                    "historical_delta_t_nm": float(delta[row, final_state]),
                    "historical_fitness": float(fitness[row, final_state]),
                    "pm_cells": int(gene.sum()),
                    "before_after_gene_identical": True,
                }
            )
            break
    return selected


def angle_key(angle: float) -> str:
    return f"angle_{angle:g}".replace(".", "p")


def run_one(base_fem: Path, output_dir: Path, angle: float, timeout_seconds: float, maximum_attempts: int) -> dict:
    result_path = output_dir / "result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("status") == "complete":
            return cached
    script = Path(__file__).with_name("solve_one_angle.py")
    command = [
        sys.executable,
        str(script),
        "--base-fem",
        str(base_fem),
        "--output-dir",
        str(output_dir),
        "--angle-deg",
        str(angle),
        "--pole-pairs",
        str(POLE_PAIRS),
        "--amplitude-a",
        str(CURRENT_AMPLITUDE_A),
    ]
    last_error = ""
    for attempt in range(1, maximum_attempts + 1):
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            (output_dir / f"attempt_{attempt}_stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (output_dir / f"attempt_{attempt}_stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode == 0 and result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if payload.get("status") == "complete":
                    return payload
            last_error = f"returncode={completed.returncode}; stderr={completed.stderr[-1000:]}"
        except subprocess.TimeoutExpired as exc:
            last_error = f"timeout after {timeout_seconds}s: {exc}"
        time.sleep(1)
    raise RuntimeError(f"FEMM solve failed after {maximum_attempts} attempts: {last_error}")


def metric_values(torques: np.ndarray, angles: np.ndarray, multiplier: int, method: str) -> tuple[float, float]:
    values = multiplier * np.asarray(torques, dtype=np.float64)
    if method == "trapezoidal":
        tavg = float(np.trapezoid(values, angles) / (angles[-1] - angles[0]))
    elif method == "mean6":
        tavg = float(np.mean(values))
    elif method == "mean_first5":
        tavg = float(np.mean(values[:-1]))
    else:
        raise ValueError(method)
    return tavg, float(np.ptp(values))


def analyze(selected: list[dict], results_by_candidate: dict[str, list[dict]]) -> dict:
    angles = np.asarray(ANGLES, dtype=np.float64)
    configurations = []
    for multiplier in (1, -1, 2, -2, 4, -4):
        for method in ("trapezoidal", "mean6", "mean_first5"):
            rows = []
            for candidate in selected:
                result_rows = results_by_candidate[candidate["candidate_id"]]
                raw = np.asarray([float(item["torque_nm"]) for item in result_rows])
                tavg, delta = metric_values(raw, angles, multiplier, method)
                historical_tavg = float(candidate["historical_tavg_nm"])
                historical_delta = float(candidate["historical_delta_t_nm"])
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "historical_tavg_nm": historical_tavg,
                        "recomputed_tavg_nm": tavg,
                        "absolute_error_tavg_nm": abs(tavg - historical_tavg),
                        "relative_error_tavg": abs(tavg - historical_tavg) / abs(historical_tavg),
                        "historical_delta_t_nm": historical_delta,
                        "recomputed_delta_t_nm": delta,
                        "absolute_error_delta_t_nm": abs(delta - historical_delta),
                        "relative_error_delta_t": abs(delta - historical_delta) / abs(historical_delta),
                        "raw_femm_torques_nm": raw.tolist(),
                    }
                )
            configurations.append(
                {
                    "torque_multiplier": multiplier,
                    "tavg_method": method,
                    "mean_relative_error_tavg": float(np.mean([row["relative_error_tavg"] for row in rows])),
                    "mean_relative_error_delta_t": float(np.mean([row["relative_error_delta_t"] for row in rows])),
                    "mean_combined_relative_error": float(
                        np.mean([row["relative_error_tavg"] + row["relative_error_delta_t"] for row in rows])
                    ),
                    "candidates": rows,
                }
            )
    configurations.sort(key=lambda item: item["mean_combined_relative_error"])
    expected = next(
        item
        for item in configurations
        if item["torque_multiplier"] == -2 and item["tavg_method"] == "trapezoidal"
    )
    return {
        "status": "complete",
        "angles_mechanical_deg": list(ANGLES),
        "pole_pairs": POLE_PAIRS,
        "current_amplitude_a": CURRENT_AMPLITUDE_A,
        "material_mapping": {"0": "Air", "1": "N38 permanent magnet"},
        "candidate_selection": selected,
        "best_enumerated_configuration": configurations[0],
        "expected_configuration_from_saved_workspace_parameters_and_02_protocol": expected,
        "all_configurations": configurations,
    }


def save_csv(path: Path, summary: dict) -> None:
    expected = summary["expected_configuration_from_saved_workspace_parameters_and_02_protocol"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "candidate_id",
            "state_index_0based",
            "population_row_0based",
            "pm_cells",
            "historical_tavg_nm",
            "recomputed_tavg_nm",
            "relative_error_tavg_percent",
            "historical_delta_t_nm",
            "recomputed_delta_t_nm",
            "relative_error_delta_t_percent",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        selected = {row["candidate_id"]: row for row in summary["candidate_selection"]}
        for row in expected["candidates"]:
            source = selected[row["candidate_id"]]
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "state_index_0based": source["state_index_0based"],
                    "population_row_0based": source["population_row_0based"],
                    "pm_cells": source["pm_cells"],
                    "historical_tavg_nm": row["historical_tavg_nm"],
                    "recomputed_tavg_nm": row["recomputed_tavg_nm"],
                    "relative_error_tavg_percent": 100 * row["relative_error_tavg"],
                    "historical_delta_t_nm": row["historical_delta_t_nm"],
                    "recomputed_delta_t_nm": row["recomputed_delta_t_nm"],
                    "relative_error_delta_t_percent": 100 * row["relative_error_delta_t"],
                }
            )


def run(output_root: Path, workers: int, timeout_seconds: float, maximum_attempts: int) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    selected = candidate_selection(DEFAULT_MAT)
    atomic_json(output_root / "selected_candidates.json", {"candidates": selected})
    base_models: dict[str, Path] = {}
    for candidate in selected:
        bits, provenance = history_gene(
            DEFAULT_MAT,
            candidate["state_index_0based"],
            candidate["population_row_0based"],
            "population_all",
        )
        base_models[candidate["candidate_id"]] = build_topology(
            bits,
            DEFAULT_TEMPLATE,
            DEFAULT_MAT,
            output_root / candidate["candidate_id"] / "base",
            provenance,
        )

    futures = {}
    results_by_candidate: dict[str, list[dict]] = {item["candidate_id"]: [] for item in selected}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for candidate in selected:
            candidate_id = candidate["candidate_id"]
            for angle in ANGLES:
                future = executor.submit(
                    run_one,
                    base_models[candidate_id],
                    output_root / candidate_id / angle_key(angle),
                    angle,
                    timeout_seconds,
                    maximum_attempts,
                )
                futures[future] = (candidate_id, angle)
        completed_rows: dict[tuple[str, float], dict] = {}
        for future in as_completed(futures):
            candidate_id, angle = futures[future]
            completed_rows[(candidate_id, angle)] = future.result()
            print(f"complete: {candidate_id}, angle={angle:g} deg", flush=True)
    for candidate in selected:
        candidate_id = candidate["candidate_id"]
        results_by_candidate[candidate_id] = [completed_rows[(candidate_id, angle)] for angle in ANGLES]

    summary = analyze(selected, results_by_candidate)
    atomic_json(output_root / "analysis_summary.json", summary)
    save_csv(output_root / "expected_configuration_comparison.csv", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use 1..3 isolated FEMM workers")
    summary = run(args.output_root, args.workers, args.timeout_seconds, args.maximum_attempts)
    print(json.dumps(summary["expected_configuration_from_saved_workspace_parameters_and_02_protocol"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
