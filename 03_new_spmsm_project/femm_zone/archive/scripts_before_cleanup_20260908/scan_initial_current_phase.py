"""Test a common current phase offset for the same five genes at 3.5 A."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from compare_current_modes_5genes import DEFAULT_OUTPUT as BASELINE, analyze, current_at
from replay_fixed_current_5genes import save, digest
from spmsm_mapping import DEFAULT_MAT, DEFAULT_TEMPLATE

OUTPUT = BASELINE.parent / "initial_current_phase_20260908"


def run(output: Path, phases: list[float]) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    spec = copy.deepcopy(json.loads((BASELINE / "run_spec.json").read_text(encoding="utf-8")))
    assert spec["input_sha256"] == {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    shutil.copytree(BASELINE / "bases", output / "bases")
    spec["scenarios"] = [{"id": "phase_" + ("p" if phase >= 0 else "m") + f"{abs(phase):g}".replace(".", "p"),
                           "source": "Is_amp", "angle_factor": 1, "phase_offset_electrical_deg": phase,
                           "label": f"Is_amp=3.5; initial electrical phase={phase:g} deg"} for phase in phases]
    spec["current_formula"] = "theta_e=P*inner_angle+phase_offset; ABC=Is_amp*cos(theta_e+[0,-120,120] deg)"
    spec["current_convention_note"] = "The requested initial electrical phases are listed explicitly in scenarios and applied in common to all five genes. The scan varies phase only; it does not alter amplitude, current direction, torque scale, or genes. No per-gene phase fitting."
    spec["requested_initial_phases_degrees"] = phases
    spec["baseline_path"] = str(BASELINE)
    spec["baseline_scenario"] = "is35_plus"
    spec["mat_angle_evidence_note"] = "MAT rotorschift=22.5 and PM2schiftangle=67.5 describe geometry-related saved fields; neither is demonstrated to be the current offset. No explicit saved initial current phase was identified."
    save(output / "run_spec.json", spec)
    # Direct numerical checks of the new offset operation before invoking FEMM.
    by_phase = {s["phase_offset_electrical_deg"]: s for s in spec["scenarios"]}
    if 120 in by_phase:
        currents, theta = current_at(spec, by_phase[120], 0)
        assert theta == 120 and abs(currents["A"] + 1.75) < 1e-12
        assert abs(currents["B"] - 3.5) < 1e-12 and abs(currents["C"] + 1.75) < 1e-12
    zero = {"source": "Is_amp", "angle_factor": 1, "phase_offset_electrical_deg": 0}
    for angle in spec["angles_mechanical_deg"]:
        current, theta = current_at(spec, zero, angle)
        assert theta == 4 * angle
        baseline = json.loads((BASELINE / "is35_plus" / spec["candidates"][0]["candidate_id"] / f"angle_{angle:03d}" / "result.json").read_text())
        assert all(abs(current[k] - baseline["currents_a"][k]) < 1e-12 for k in current)

    def launch(scenario: dict, candidate: dict) -> None:
        sid, cid = scenario["id"], candidate["candidate_id"]
        target = output / sid / cid
        target.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("compare_current_modes_5genes.py")),
               "--worker", str(output / "run_spec.json"), "--scenario", sid, "--candidate", cid]
        with (target / "stdout.txt").open("w") as stdout, (target / "stderr.txt").open("w") as stderr:
            process = subprocess.run(cmd, stdout=stdout, stderr=stderr, timeout=600,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            raise RuntimeError(f"FEMM worker failed: {sid}/{cid}")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(launch, s, c): f"{s['id']}/{c['candidate_id']}" for s in spec["scenarios"] for c in spec["candidates"]}
        for n, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(f"Complete {6*n}/{6*len(futures)}: {futures[future]}", flush=True)
    analyze(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--phases", nargs="+", type=float, default=[120, -3, -1, 1, 3])
    args = parser.parse_args()
    run(args.output.resolve(), args.phases)
