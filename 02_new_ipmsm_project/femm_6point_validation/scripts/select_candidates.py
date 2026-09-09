"""Deterministically select eight historical candidates for six-angle validation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat


VALIDATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VALIDATION_ROOT.parent
MAT_PATH = PROJECT_ROOT / "data_zone" / "raw" / "workspace_600.mat"
OUTPUT_PATH = VALIDATION_ROOT / "selected_candidates.csv"


def bits_text(bits: np.ndarray) -> str:
    return "".join(str(int(value)) for value in np.asarray(bits).reshape(-1))


def select() -> list[dict[str, object]]:
    data = loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
    before = np.asarray(data["population_noChange_all"], dtype=np.uint8).transpose(2, 0, 1)
    after = np.asarray(data["population_all"], dtype=np.uint8).transpose(2, 0, 1)
    tavg = np.asarray(data["Tavg_all"], dtype=np.float64).T
    delta_t = np.asarray(data["DeltaT_all"], dtype=np.float64).T
    fitness = np.asarray(data["Fitvalue_all"], dtype=np.float64).T

    flat = before.reshape(-1, 200)
    packed = np.packbits(flat, axis=1)
    unique, inverse, counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
    inverse_grid = inverse.reshape(before.shape[:2])
    order = np.argsort(inverse, kind="stable")
    sorted_group = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]
    ends = np.r_[starts[1:], len(order)]
    sorted_tavg = tavg.reshape(-1)[order]
    sorted_delta = delta_t.reshape(-1)[order]
    tavg_range = np.fromiter(
        (sorted_tavg[start:end].max() - sorted_tavg[start:end].min() for start, end in zip(starts, ends)),
        dtype=np.float64,
        count=len(unique),
    )
    delta_range = np.fromiter(
        (sorted_delta[start:end].max() - sorted_delta[start:end].min() for start, end in zip(starts, ends)),
        dtype=np.float64,
        count=len(unique),
    )
    consistent = (tavg_range <= 1e-10) & (delta_range <= 1e-10)

    # Pick near fixed within-state quantiles, then order two widely separated
    # samples first for the smoke test.
    specifications = [
        ("low_tavg", 0, "tavg", 0.10),
        ("low_tavg", 100, "tavg", 0.10),
        ("medium_tavg", 300, "tavg", 0.50),
        ("medium_tavg", 600, "tavg", 0.50),
        ("high_tavg", 100, "tavg", 0.90),
        ("high_tavg", 300, "tavg", 0.90),
        ("high_delta_t", 0, "delta", 0.90),
        ("high_delta_t", 600, "delta", 0.90),
    ]
    chosen: list[dict[str, object]] = []
    used_groups: set[int] = set()
    for category, state, metric, quantile in specifications:
        unchanged = np.all(before[state] == after[state], axis=1)
        eligible = np.flatnonzero(unchanged & consistent[inverse_grid[state]])
        values = (tavg if metric == "tavg" else delta_t)[state, eligible]
        target = float(np.quantile(values, quantile))
        for row in eligible[np.argsort(np.abs(values - target), kind="stable")]:
            group = int(inverse_grid[state, row])
            if group in used_groups:
                continue
            used_groups.add(group)
            gene_before = before[state, row]
            gene_after = after[state, row]
            chosen.append(
                {
                    "category": category,
                    "state_index_0based": state,
                    "generation_label": "initial" if state == 0 else str(state),
                    "individual_0based": int(row),
                    "individual_1based": int(row) + 1,
                    "historical_tavg": float(tavg[state, row]),
                    "historical_delta_t": float(delta_t[state, row]),
                    "historical_fitness": float(fitness[state, row]),
                    "before_after_equal": bool(np.array_equal(gene_before, gene_after)),
                    "repeat_count": int(counts[group]),
                    "repeat_tavg_range": float(tavg_range[group]),
                    "repeat_delta_t_range": float(delta_range[group]),
                    "topology_key_hex": unique[group].tobytes().hex(),
                    "gene_before_correction": bits_text(gene_before),
                    "gene_after_correction": bits_text(gene_after),
                    "gene_sha256": hashlib.sha256(np.packbits(gene_before).tobytes()).hexdigest(),
                }
            )
            break

    # Smoke test: one early low-torque and one later high-torque topology.
    output_order = [0, 5, 1, 2, 3, 4, 6, 7]
    ordered = [chosen[index] for index in output_order]
    for index, row in enumerate(ordered, start=1):
        row["candidate_id"] = f"candidate_{index:03d}"
        row["run_stage"] = "smoke" if index <= 2 else "pending_after_smoke"
    return ordered


def main() -> None:
    rows = select()
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {len(rows)} candidates: {OUTPUT_PATH}")
    for row in rows:
        print(
            row["candidate_id"], row["run_stage"], row["category"],
            f"state={row['state_index_0based']}", f"row={row['individual_0based']}",
            f"Tavg={row['historical_tavg']:.6g}", f"DeltaT={row['historical_delta_t']:.6g}",
        )


if __name__ == "__main__":
    main()
