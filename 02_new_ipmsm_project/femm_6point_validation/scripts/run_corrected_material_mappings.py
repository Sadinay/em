"""Fair material-mapping comparison after fixing every PM direction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_corrected_3deg_angles as experiment
from build_femm_topology import DEFAULT_MAT, DEFAULT_TEMPLATE, MAPPING_CANDIDATES, build_topology
from run_validation import validation_lock


ROOT = experiment.VALIDATION_ROOT / "corrected_pm_direction" / "mapping_comparison"


def selected() -> dict[str, str]:
    with (experiment.VALIDATION_ROOT / "selected_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        return next(row for row in csv.DictReader(stream) if row["candidate_id"] == "candidate_002")


def run_mapping(mapping: str, workers: int, timeout_seconds: float, maximum_attempts: int) -> None:
    row = selected()
    bits = np.fromiter((int(char) for char in row["gene_after_correction"]), dtype=np.uint8, count=200)
    mapping_root = ROOT / mapping
    base_dir = mapping_root / "base"
    build_topology(
        bits,
        DEFAULT_TEMPLATE,
        DEFAULT_MAT,
        base_dir,
        provenance={
            "kind": "corrected_pm_direction_material_mapping_comparison",
            "candidate_id": "candidate_002",
            "state_index_0based": int(row["state_index_0based"]),
            "individual_0based": int(row["individual_0based"]),
        },
        code_to_class=MAPPING_CANDIDATES[mapping],
        mapping_version=f"candidate_{mapping}_pm_direction_fixed_v2",
    )
    experiment.BASE = base_dir / "model.fem"
    experiment.ROOT = mapping_root / "angles_3deg"
    experiment.CHECKPOINT = experiment.ROOT / "checkpoint.json"
    experiment.SUMMARY = experiment.ROOT / "analysis_summary.json"
    experiment.CURVE = experiment.ROOT / f"candidate_002_{mapping}_angles_0_to_15.csv"
    experiment.REUSE_15_PATH = None
    experiment.VARIANT_DESCRIPTION = "reference nonlinear Pure Iron with 21-point BH curve"
    experiment.MAPPING_NAME = mapping
    experiment.run(workers, timeout_seconds, maximum_attempts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--maximum-attempts", type=int, default=2)
    parser.add_argument("--mappings", nargs="+", choices=("air2", "air3"), default=("air2", "air3"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("Use 1..3 workers")
    with validation_lock():
        for mapping in args.mappings:
            run_mapping(mapping, args.workers, args.timeout_seconds, args.maximum_attempts)


if __name__ == "__main__":
    main()
