"""Check the 1.25 degree increment implied by saved T, wmech and stps; literal ABC."""
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from replay_fixed_current_5genes import ROOT, DEFAULT_MAT, DEFAULT_TEMPLATE, angle_directory, digest, save, verify_input

OUTPUT = ROOT / "femm_zone/results/mat_literal_current_probe_20260908/timing_grid"
SOURCE = OUTPUT.parent / "template_iron"


def analyze():
    spec = json.loads((OUTPUT / "run_spec.json").read_text(encoding="utf-8"))
    rows = []
    for c in spec["candidates"]:
        raw = []
        for a in spec["angles_mechanical_deg"]:
            folder = OUTPUT / c["candidate_id"] / angle_directory(a)
            rec = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            assert rec["status"] == "complete"
            verify_input(folder / "model.fem", a, spec["fixed_currents_a"])
            raw.append(rec["torque_nm_raw"])
        torque = -2 * np.array(raw)
        pp = float(np.ptp(torque))
        rows.append({**c, "raw_torques_nm": raw, "torques_nm_previous_scale": torque.tolist(),
                     "tavg_nm": float(np.trapezoid(torque, spec["angles_mechanical_deg"]) / 15),
                     "absolute_peak_to_peak_nm": pp,
                     "delta_error_if_absolute_percent": 100 * abs(pp-c["historical_delta_t"]) / c["historical_delta_t"]})
    result = {"status": "complete", "fresh_solves": 65, "spec": spec, "candidates": rows,
              "mean_delta_error_if_absolute_percent": float(np.mean([r["delta_error_if_absolute_percent"] for r in rows]))}
    save(OUTPUT / "summary.json", result)
    print(json.dumps({"mean_delta_error_if_absolute_percent": result["mean_delta_error_if_absolute_percent"], "values": [{"id":r["candidate_id"],"tavg":r["tavg_nm"],"pp":r["absolute_peak_to_peak_nm"]} for r in rows]}, indent=2))


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    inp = loadmat(DEFAULT_MAT, variable_names=["inp"], squeeze_me=True, struct_as_record=False)["inp"]
    step = float(np.rad2deg(inp.wmech * inp.T / inp.stps))
    angles = np.arange(0, 15 + step/2, step).tolist()
    assert np.isclose(step, 1.25) and len(angles) == 13
    spec = json.loads((SOURCE / "run_spec.json").read_text(encoding="utf-8"))
    spec.update({"angles_mechanical_deg": angles, "step_source": "degrees(inp.wmech * inp.T / inp.stps)",
                 "timing_values": {k: float(getattr(inp,k)) for k in ("wmech","T","stps","period","Nrpm","P")},
                 "scope": "First 15 mechanical degrees at the time-derived increment; not a claimed full 72-sample historical replay."})
    save(OUTPUT / "run_spec.json", spec)
    before = {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    for c in spec["candidates"]:
        dest = OUTPUT / c["candidate_id"] / "base/model.fem"
        dest.parent.mkdir(parents=True)
        shutil.copy2(SOURCE / c["candidate_id"] / "base/model.fem", dest)
    worker = Path(__file__).with_name("replay_fixed_current_5genes.py")
    def launch(c):
        folder = OUTPUT / c["candidate_id"]
        with (folder/"stdout.txt").open("w", encoding="utf-8") as so, (folder/"stderr.txt").open("w", encoding="utf-8") as se:
            res = subprocess.run([sys.executable,"-u",str(worker),"--worker",str(OUTPUT/"run_spec.json"),"--candidate",c["candidate_id"]],
                                 stdout=so,stderr=se,timeout=600,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if res.returncode:
            raise RuntimeError(folder)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(launch,c):c["candidate_id"] for c in spec["candidates"]}
        for f in as_completed(tasks):
            f.result()
            print("Completed 13 solves:",tasks[f],flush=True)
    assert before == {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    save(OUTPUT/"input_integrity.json",{"unchanged":True,"sha256":before})
    analyze()


if __name__ == "__main__":
    if "--analyze-only" in sys.argv:
        analyze()
    else:
        run()
