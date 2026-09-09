"""Six documented current interpretations, five MAT genes, inner angle 0:3:15."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from replay_fixed_current_5genes import ANGLES, SELECTION, digest, save, verify_input
from spmsm_mapping import DEFAULT_MAT, DEFAULT_TEMPLATE, build_topology, history_gene

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "femm_zone/results/current_modes_5genes_20260908"
MULTIPLIER = -2.0  # Existing comparison convention, NOT a newly fitted physical scale.


def current_at(spec: dict, scenario: dict, angle: float) -> tuple[dict, float]:
    theta = (scenario["angle_factor"] * spec["mat_parameters"]["P"] * angle
             + scenario.get("phase_offset_electrical_deg", 0.0))
    offsets = {"A": 0, "B": -120, "C": 120}
    if scenario["source"] == "Is_amp":
        currents = {phase: spec["mat_parameters"]["Is_amp"] * math.cos(math.radians(theta + shift))
                    for phase, shift in offsets.items()}
    else:
        d, q = spec["mat_parameters"]["Id"], spec["mat_parameters"]["Iq"]
        currents = {phase: d * math.cos(math.radians(theta + shift)) - q * math.sin(math.radians(theta + shift))
                    for phase, shift in offsets.items()}
    assert abs(sum(currents.values())) < 1e-12
    return currents, theta


def worker(spec_path: Path, sid: str, cid: str) -> None:
    import femm
    import win32com.client

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    scenario = next(s for s in spec["scenarios"] if s["id"] == sid)
    out = spec_path.parent / sid / cid
    base = spec_path.parent / "bases" / cid / "model.fem"
    connected = False
    started = time.perf_counter()
    try:
        femm.HandleToFEMM = win32com.client.DispatchEx("femm.ActiveFEMM")
        connected = True
        femm.main_minimize()
        for angle in ANGLES:
            target = out / f"angle_{angle:03d}"
            target.mkdir(parents=True, exist_ok=True)
            model = target / "model.fem"
            if (target / "result.json").exists() or model.with_suffix(".ans").exists():
                raise FileExistsError(f"Fresh solve refuses existing result: {target}")
            shutil.copy2(base, model)
            currents, phase = current_at(spec, scenario, angle)
            femm.opendocument(str(model.resolve()))
            femm.mi_modifyboundprop("sliding_airgap", 10, angle)
            # The template has Outer Angle=0; leave it untouched in every case.
            for name, value in currents.items():
                femm.mi_modifycircprop(name, 1, value)
            femm.mi_saveas(str(model.resolve()))
            verified = verify_input(model, angle, currents)
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            raw = femm.mo_gapintegral("sliding_airgap", 0)
            if isinstance(raw, complex):
                assert abs(raw.imag) < 1e-10
                raw = raw.real
            raw = float(raw)
            assert math.isfinite(raw) and model.with_suffix(".ans").exists()
            save(target / "result.json", {
                "status": "complete", "fresh_solve": True, "scenario_id": sid,
                "candidate_id": cid, "inner_angle_deg": angle, "outer_angle_deg": 0,
                "current_angle_deg": phase, "currents_a": currents,
                "raw_gap_torque_nm": raw, "comparison_torque_nm": MULTIPLIER * raw,
                "verified_input": verified, "ans_sha256": digest(model.with_suffix(".ans")),
            })
            print(f"{sid}/{cid} inner={angle} raw={raw:.9f}", flush=True)
            femm.mo_close()
            femm.mi_close()
        save(out / "worker_status.json", {"status": "complete", "solves": len(ANGLES), "seconds": time.perf_counter() - started})
    except Exception:
        save(out / "worker_status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        if connected:
            try:
                femm.closefemm()
            except Exception:
                pass


def analyze(output: Path) -> dict:
    spec = json.loads((output / "run_spec.json").read_text(encoding="utf-8"))
    results, samples, aggregates = [], [], []
    for scenario in spec["scenarios"]:
        group = []
        for candidate in spec["candidates"]:
            sid, cid = scenario["id"], candidate["candidate_id"]
            raw = []
            for angle in ANGLES:
                target = output / sid / cid / f"angle_{angle:03d}"
                record = json.loads((target / "result.json").read_text(encoding="utf-8"))
                assert record["status"] == "complete" and record["fresh_solve"]
                currents, theta = current_at(spec, scenario, angle)
                verify_input(target / "model.fem", angle, currents)
                assert digest(target / "model.ans") == record["ans_sha256"]
                raw.append(record["raw_gap_torque_nm"])
                samples.append({"scenario_id": sid, "candidate_id": cid, "inner_angle_deg": angle,
                                "outer_angle_deg": 0, "current_angle_deg": theta,
                                **{f"i{k.lower()}_a": v for k, v in currents.items()},
                                "raw_gap_torque_nm": raw[-1], "comparison_torque_nm": MULTIPLIER * raw[-1]})
            torque = MULTIPLIER * np.asarray(raw)
            avg, pp = float(torque.mean()), float(np.ptp(torque))
            ripple = pp / abs(avg) if avg != 0 else None
            reference_avg = candidate["historical_tavg_nm"]
            reference_delta = candidate["historical_delta_t"]
            row = {"scenario_id": sid, **candidate, "raw_gap_torques_nm": raw,
                   "comparison_torques_nm": torque.tolist(), "raw_tavg_nm": float(np.mean(raw)),
                   "tavg_nm": avg, "peak_to_peak_nm": pp, "ripple_relative": ripple,
                   "ripple_percent": 100 * ripple if ripple is not None else None,
                   "near_zero_average_display_flag": abs(avg) < 0.01,
                   "trapezoidal_tavg_nm": float(np.trapezoid(torque, ANGLES) / 15),
                   "tavg_error_percent": 100 * abs(avg - reference_avg) / abs(reference_avg),
                   "ripple_error_percent_if_relative": 100 * abs(ripple - reference_delta) / abs(reference_delta) if ripple is not None else None,
                   "peak_to_peak_error_percent_if_absolute": 100 * abs(pp - reference_delta) / abs(reference_delta)}
            results.append(row)
            group.append(row)
        aggregate = {"scenario_id": scenario["id"], "label": scenario["label"],
                     **{name: float(np.mean([r[name] for r in group])) for name in
                        ["tavg_error_percent", "ripple_error_percent_if_relative", "peak_to_peak_error_percent_if_absolute"]}}
        aggregates.append(aggregate)
    before = spec["input_sha256"]
    assert before == {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    save(output / "input_integrity.json", {"unchanged": True, "sha256": before})
    summary = {"status": "complete", "fresh_solves": len(samples), "spec": spec,
               "aggregate": aggregates, "results": results}
    save(output / "summary.json", summary)
    # Native JSON is the archival numerical source; CSV is a convenience export.
    for name, rows in [("torque_samples.csv", samples), ("metrics.csv", [{k: v for k, v in r.items() if not isinstance(v, list)} for r in results])]:
        with (output / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(aggregates, ensure_ascii=False, indent=2), flush=True)
    return summary


def run(output: Path, workers: int) -> None:
    if output.exists():
        raise FileExistsError(f"A fresh output directory is required: {output}")
    output.mkdir(parents=True)
    inp = loadmat(DEFAULT_MAT, variable_names=["inp"], squeeze_me=True, struct_as_record=False)["inp"]
    mat_parameters = {key: float(getattr(inp, key)) for key in ["P", "Is_amp", "Id", "Iq", "Ia", "Ib", "Ic", "steps", "rotorschift", "PM2schiftangle"]}
    assert mat_parameters["P"] == 4 and mat_parameters["steps"] == len(ANGLES)
    before = {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    candidates = []
    for n, (cid, state, row) in enumerate(SELECTION, start=1):
        bits, provenance = history_gene(DEFAULT_MAT, state, row)
        original, _ = history_gene(DEFAULT_MAT, state, row, "population_noChange_all")
        assert np.array_equal(bits, original)
        build_topology(bits, DEFAULT_TEMPLATE, DEFAULT_MAT, output / "bases" / cid, provenance)
        stored = provenance["stored_results"]
        candidates.append({"gene_label": f"G{n}", "candidate_id": cid,
                           "state_index_0based": state, "population_row_0based": row,
                           "pm_cells": int(bits.sum()), "historical_tavg_nm": float(stored["tavg_nm"]),
                           "historical_delta_t": float(stored["delta_t_nm"])})
    scenarios = []
    for source, prefix, label in [("Is_amp", "is35", "Is_amp=3.5"), ("Id_Iq", "dq5", "Id=5 / Iq=0")]:
        for factor, suffix, mode in [(0, "fixed", "固定三相"), (1, "plus", "正向更新"), (-1, "minus", "反向更新")]:
            scenarios.append({"id": f"{prefix}_{suffix}", "source": source, "angle_factor": factor,
                              "label": f"{label} · {mode}", "phase_offset_electrical_deg": 0})
    spec = {"mat_parameters": mat_parameters, "angles_mechanical_deg": ANGLES,
            "outer_angle_deg": 0, "min_angle_deg": 15, "input_sha256": before,
            "torque_multiplier": MULTIPLIER,
            "torque_scale_note": "Retain prior empirical -2*gap torque for comparison; physical/history scale NOT established; no fitting in this run.",
            "ripple_definition": "(max(T)-min(T))/abs(mean(T)); six-point arithmetic mean; interpreting historical DeltaT this way remains a hypothesis.",
            "current_formula": "theta_e=factor*P*inner_angle; Is_amp source: I*cos(theta_e+[0,-120,120]); Id/Iq source: Id*cos(theta_e+offset)-Iq*sin(theta_e+offset)",
            "current_convention_note": "Amplitude-invariant inverse Park; zero electrical phase at Inner Angle=0 is a test convention, not established historical d-axis alignment. No RMS conversion, no fitted phase shift.",
            "material_note": "Original 03 FEM nonlinear Pure Iron/N38 and unchanged geometry/magnetization; only mapped genes, Inner Angle, and currents differ.",
            "sources": ["https://www.mathworks.com/help/mcb/ref/surfacemountpmsm.html", "https://www.femm.info/doku/doku.php?id=rotormotion"],
            "candidates": candidates, "scenarios": scenarios}
    save(output / "run_spec.json", spec)
    def launch(scenario: dict, candidate: dict) -> None:
        sid, cid = scenario["id"], candidate["candidate_id"]
        target = output / sid / cid
        target.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(output / "run_spec.json"), "--scenario", sid, "--candidate", cid]
        with (target / "stdout.txt").open("w", encoding="utf-8") as stdout, (target / "stderr.txt").open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            raise RuntimeError(f"Worker failed: {sid}/{cid}; see stderr.txt")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(launch, s, c): f"{s['id']}/{c['candidate_id']}" for s in scenarios for c in candidates}
        done = 0
        for future in as_completed(futures):
            future.result()
            done += len(ANGLES)
            print(f"Complete {done}/{len(futures)*len(ANGLES)}: {futures[future]}", flush=True)
    analyze(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--scenario")
    parser.add_argument("--candidate")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.scenario, args.candidate)
    elif args.analyze_only:
        analyze(args.output.resolve())
    else:
        run(args.output.resolve(), args.workers)
