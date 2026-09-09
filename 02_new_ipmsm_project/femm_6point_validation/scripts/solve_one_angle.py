"""Isolated FEMM worker for one candidate and one mechanical angle."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import traceback
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def currents(angle_mechanical_deg: float, pole_pairs: int, amplitude_a: float) -> dict[str, float]:
    theta_e = math.radians(pole_pairs * angle_mechanical_deg)
    return {
        "A": amplitude_a * math.cos(theta_e),
        "B": amplitude_a * math.cos(theta_e - 2 * math.pi / 3),
        "C": amplitude_a * math.cos(theta_e + 2 * math.pi / 3),
    }


def solve(base_fem: Path, output_dir: Path, angle_deg: float, pole_pairs: int, amplitude_a: float) -> dict:
    import femm

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.fem"
    ans_path = model_path.with_suffix(".ans")
    if ans_path.exists():
        ans_path.unlink()
    result_path = output_dir / "result.json"
    shutil.copy2(base_fem, model_path)
    phase_currents = currents(angle_deg, pole_pairs, amplitude_a)
    started = time.perf_counter()
    opened = False
    payload = {
        "status": "running",
        "angle_mechanical_deg": angle_deg,
        "angle_electrical_deg": pole_pairs * angle_deg,
        "pole_pairs": pole_pairs,
        "current_amplitude_a": amplitude_a,
        "phase_currents_a": phase_currents,
        "rotor_angle_method": "mi_modifyboundprop('sliding_airgap', 10, angle_mechanical_deg)",
        "torque_method": "mo_gapintegral('sliding_airgap', 0)",
        "base_fem": str(base_fem.resolve()),
        "model_fem": str(model_path.resolve()),
    }
    atomic_json(result_path, payload)
    try:
        femm.openfemm(1)
        opened = True
        femm.opendocument(str(model_path.resolve()))
        femm.mi_modifyboundprop("sliding_airgap", 10, angle_deg)
        for circuit, value in phase_currents.items():
            femm.mi_modifycircprop(circuit, 1, value)
        femm.mi_saveas(str(model_path.resolve()))
        prepared_at = time.perf_counter()
        femm.mi_analyze(1)
        analyzed_at = time.perf_counter()
        femm.mi_loadsolution()
        torque = femm.mo_gapintegral("sliding_airgap", 0)
        finished_at = time.perf_counter()
        if isinstance(torque, complex):
            if abs(torque.imag) > 1e-10:
                raise ValueError(f"Unexpected complex DC torque: {torque!r}")
            torque = torque.real
        torque = float(torque)
        if not math.isfinite(torque):
            raise ValueError(f"Non-finite torque: {torque}")
        payload.update(
            {
                "status": "complete",
                "torque_nm": torque,
                "timing_seconds": {
                    "prepare": prepared_at - started,
                    "analyze": analyzed_at - prepared_at,
                    "postprocess": finished_at - analyzed_at,
                    "total": finished_at - started,
                },
                "ans_path": str(ans_path.resolve()),
                "ans_exists": ans_path.exists(),
                "ans_size_bytes": ans_path.stat().st_size if ans_path.exists() else None,
            }
        )
        atomic_json(result_path, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return payload
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(result_path, payload)
        raise
    finally:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angle-deg", type=float, required=True)
    parser.add_argument("--pole-pairs", type=int, default=4)
    parser.add_argument("--amplitude-a", type=float, default=3.5)
    args = parser.parse_args()
    solve(args.base_fem, args.output_dir, args.angle_deg, args.pole_pairs, args.amplitude_a)


if __name__ == "__main__":
    main()
