from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

from encoding.chromosome import Chromosome
from femm_runner.historical_inset_commands import apply_historical_inset
from femm_runner.merged_copper_v5_commands import apply_merged_copper_v5
from femm_runner.process import (
    OwnedFemmWindowHider,
    femm_process_snapshot,
    femm_startup_lock,
    hide_process_windows,
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _circuit_currents(job: dict, ia: float, ib: float, ic: float) -> tuple[tuple[str, float], ...]:
    mode = job.get("circuit_mode", "split_six_phase_signs")
    if mode == "three_phase_abc":
        return (("A", ia), ("B", ib), ("C", ic))
    if mode == "split_six_phase_signs":
        return (
            ("A+", ia),
            ("A-", -ia),
            ("B+", ib),
            ("B-", -ib),
            ("C+", ic),
            ("C-", -ic),
        )
    raise ValueError(f"unsupported circuit_mode {mode!r}")


def _scan_torque(api, job: dict) -> tuple[list[float], list[dict[str, float]]]:
    torques: list[float] = []
    timings: list[dict[str, float]] = []
    current = float(job["stator_current_a"])
    for angle_value in job["torque_angles_deg"]:
        angle = float(angle_value)
        angle_start = time.perf_counter()
        theta_e = math.radians(
            float(job.get("electrical_angle_multiplier", 4.0)) * angle
            + float(job.get("electrical_phase_offset_deg", 90.0))
        )
        ia = current * math.cos(theta_e)
        ib = current * math.cos(theta_e - 2.0 * math.pi / 3.0)
        ic = current * math.cos(theta_e - 4.0 * math.pi / 3.0)
        api.mi_modifyboundprop(job["air_gap_boundary_name"], 10, angle)
        for name, value in _circuit_currents(job, ia, ib, ic):
            # The historical MATLAB code uses property index 1 here.
            api.mi_modifycircprop(name, 1, value)
        mesh_seconds = None
        if job.get("explicit_mesh_before_solve", False):
            mesh_start = time.perf_counter()
            api.mi_createmesh()
            mesh_seconds = time.perf_counter() - mesh_start
        solve_start = time.perf_counter()
        api.mi_analyze(1)
        solve_seconds = time.perf_counter() - solve_start
        api.mi_loadsolution()
        mesh_nodes = int(api.mo_numnodes())
        mesh_elements = int(api.mo_numelements())
        post_start = time.perf_counter()
        api.mo_groupselectblock(int(job["rotor_group"]))
        torque = float(api.mo_blockintegral(22))
        api.mo_clearblock()
        api.mo_close()
        if not math.isfinite(torque):
            raise ValueError(f"non-finite torque at angle {angle}: {torque}")
        torques.append(torque)
        timings.append(
            {
                "angle_deg": angle,
                "mesh_seconds": mesh_seconds,
                "solve_seconds": solve_seconds,
                "postprocess_seconds": time.perf_counter() - post_start,
                "total_angle_seconds": time.perf_counter() - angle_start,
                "mesh_nodes": mesh_nodes,
                "mesh_elements": mesh_elements,
            }
        )
    return torques, timings


def execute(job_path: Path) -> dict:
    import femm

    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_path = Path(job["result_path"])
    opened = False
    window_hider: OwnedFemmWindowHider | None = None
    started = time.perf_counter()
    try:
        # COM startup is serialized only for the short open/snapshot section.
        # Solving remains fully parallel.  Recording only the newly-created
        # femm.exe prevents one worker from claiming another worker's fkn.exe.
        owned: list[int] = []
        with femm_startup_lock():
            before_processes = femm_process_snapshot()
            femm.openfemm(1)
            opened = True
            manifest_path_value = job.get("process_manifest_path")
            if manifest_path_value:
                after_open = femm_process_snapshot()
                owned = sorted(
                    after_open.get("femm.exe", set())
                    - before_processes.get("femm.exe", set())
                )
                _write_json_atomic(
                    Path(manifest_path_value),
                    {"worker_pid": os.getpid(), "femm_pids": owned},
                )
            if job.get("hide_femm_window", True):
                try:
                    femm.main_minimize()
                except Exception:
                    pass
                hide_process_windows(owned)
                window_hider = OwnedFemmWindowHider(owned)
                window_hider.start()
        femm.opendocument(str(Path(job["model_path"])))
        topology_seconds = 0.0
        geometry_mode = job.get("geometry_mode")
        if geometry_mode is None and job.get("apply_historical_inset", False):
            geometry_mode = "historical_inset"
        if geometry_mode:
            topology_start = time.perf_counter()
            chromosome = Chromosome.from_iterable(job["genes"])
            if geometry_mode == "historical_inset":
                apply_historical_inset(femm, chromosome)
            elif geometry_mode == "merged_copper_v5":
                apply_merged_copper_v5(femm, chromosome)
            else:
                raise ValueError(f"unsupported geometry_mode {geometry_mode!r}")
            topology_seconds = time.perf_counter() - topology_start
        femm.mi_saveas(str(Path(job["model_path"])))
        operation = job.get("operation", "solve_angles")
        if operation == "prepare_candidate":
            torques: list[float] = []
            angle_timings: list[dict[str, float]] = []
        elif operation in {"solve_angle", "solve_angles"}:
            torques, angle_timings = _scan_torque(femm, job)
        else:
            raise ValueError(f"unsupported FEMM worker operation {operation!r}")
        payload = {
            "status": "SUCCEEDED",
            "operation": operation,
            "geometry_builder_version": job.get("geometry_builder_version"),
            "torque_values_nm": torques,
            "timings": {
                "topology_seconds": topology_seconds,
                "angles": angle_timings,
                "total_seconds": time.perf_counter() - started,
            },
        }
        _write_json_atomic(result_path, payload)
        return payload
    except Exception as exc:
        payload = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "total_seconds": time.perf_counter() - started,
        }
        _write_json_atomic(result_path, payload)
        raise
    finally:
        if window_hider is not None:
            window_hider.close()
        if opened:
            try:
                femm.mo_close()
            except Exception:
                pass
            try:
                femm.mi_close()
            except Exception:
                pass
            try:
                femm.closefemm()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    execute(args.job)


if __name__ == "__main__":
    main()
