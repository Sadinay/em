"""Collect audited phase scans and rank common phase candidates without scale fitting."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from compare_current_modes_5genes import DEFAULT_OUTPUT as BASELINE
from replay_fixed_current_5genes import save, digest

OUTPUT = BASELINE.parent / "phase360_search_20260908"
EARLIER = BASELINE.parent / "initial_current_phase_20260908"


def norm_phase(value: float) -> float:
    return round(float(value) % 360, 6)


def gather(output: Path) -> dict:
    old = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    candidates = old["spec"]["candidates"]
    index = {}
    sources = []
    source_paths = [BASELINE / "summary.json", EARLIER / "summary.json", *sorted(output.glob("*/summary.json"))]
    for path in source_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["status"] != "complete":
            continue
        phase_by_id = {s["id"]: s.get("phase_offset_electrical_deg", 0) for s in data["spec"]["scenarios"]
                       if s["source"] == "Is_amp" and s["angle_factor"] == 1}
        taken = 0
        for r in data["results"]:
            if r["scenario_id"] not in phase_by_id:
                continue
            phase = norm_phase(phase_by_id[r["scenario_id"]])
            row = {**copy.deepcopy(r), "initial_phase_electrical_deg": phase,
                   "source_summary_path": str(path), "source_scenario_id": r["scenario_id"]}
            key = (phase, row["candidate_id"])
            if key in index:
                assert np.max(np.abs(np.asarray(index[key]["comparison_torques_nm"]) - row["comparison_torques_nm"])) < 1e-8
            index[key] = row
            taken += 6
        sources.append({"summary_path": str(path), "sha256": digest(path), "included_samples": taken,
                        "new_in_this_360_search": output in path.parents})
    phase_rows = []
    for phase in sorted({p for p, cid in index}):
        if not all((phase, c["candidate_id"]) in index for c in candidates):
            continue
        rows = [index[phase, c["candidate_id"]] for c in candidates]
        t_err = np.array([r["tavg_error_percent"] for r in rows])
        d_err = np.array([r["ripple_error_percent_if_relative"] for r in rows])
        both = np.concatenate([t_err, d_err])
        absolute_d_err = np.array([r["peak_to_peak_error_percent_if_absolute"] for r in rows])
        phase_rows.append({"phase_deg": phase, "tavg_mape_percent": float(t_err.mean()),
                           "ripple_mape_percent": float(d_err.mean()),
                           "joint_mape_percent": float(both.mean()),
                           "worst_relative_error_percent": float(both.max()),
                           "tavg_max_error_percent": float(t_err.max()),
                           "ripple_max_error_percent": float(d_err.max()),
                           "joint_rmse_percent": float(np.sqrt(np.mean(both**2))),
                           "absolute_delta_joint_mape_percent": float(np.mean(np.concatenate([t_err, absolute_d_err]))),
                           "all_ten_within_5_percent": bool(np.all(both <= 5)),
                           "all_ten_within_10_percent": bool(np.all(both <= 10)),
                           "rows": rows})
    coarse_phases = list(range(0, 360, 30))
    available = {r["phase_deg"] for r in phase_rows}
    ranked = {key: sorted([{k: v for k, v in r.items() if k != "rows"} for r in phase_rows], key=lambda r: r[key])[:10]
              for key in ["worst_relative_error_percent", "joint_mape_percent", "tavg_mape_percent", "ripple_mape_percent"]}
    result = {"status": "complete_for_available_batches", "coarse_phases_degrees": coarse_phases,
              "coarse_complete": all(p in available for p in coarse_phases),
              "all_available_phases_degrees": sorted(available), "phase_count": len(phase_rows),
              "new_solves": sum(s["included_samples"] for s in sources if s["new_in_this_360_search"]),
              "reused_solves": sum(s["included_samples"] for s in sources if not s["new_in_this_360_search"]),
              "score_note": "A phase is evaluated across all five genes. Joint MAPE equally averages ten absolute relative errors (five mean torque, five ripple). Worst error is the maximum of those ten; this prevents a good metric concealing a bad one. No fitted torque scaling or separate phase per gene.",
              "comparison_convention": "Retain -2*FEMM raw gap torque; six-point arithmetic mean; MAT DeltaT tentatively interpreted as relative peak-to-peak. These historical conventions remain unconfirmed.",
              "candidates": candidates, "phase_results": phase_rows, "rankings": ranked,
              "matches_all_ten_within_5_percent": [r["phase_deg"] for r in phase_rows if r["all_ten_within_5_percent"]],
              "matches_all_ten_within_10_percent": [r["phase_deg"] for r in phase_rows if r["all_ten_within_10_percent"]],
              "sources": sources}
    save(output / "search_summary.json", result)
    print(json.dumps({"coarse_complete": result["coarse_complete"], "phase_count": len(phase_rows),
                      "new_solves": result["new_solves"], "reused_solves": result["reused_solves"],
                      "best_worst_error": ranked["worst_relative_error_percent"][:3],
                      "best_joint_mape": ranked["joint_mape_percent"][:3],
                      "best_tavg_mape": ranked["tavg_mape_percent"][:3]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    gather(args.output.resolve())
