"""Solve one generated SPMSM topology at one mechanical angle."""

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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def phase_currents(
    angle_mechanical_deg: float,
    pole_pairs: int,
    amplitude_a: float,
    current_angle_factor: int = 1,
    current_phase_offset_deg: float = 0.0,
) -> dict[str, float]:
    theta_e = math.radians(current_angle_factor * pole_pairs * angle_mechanical_deg + current_phase_offset_deg)
    return {
        "A": amplitude_a * math.cos(theta_e),
        "B": amplitude_a * math.cos(theta_e - 2 * math.pi / 3),
        "C": amplitude_a * math.cos(theta_e + 2 * math.pi / 3),
    }


def solve(
    base_fem: Path,
    output_dir: Path,
    angle_deg: float,
    pole_pairs: int,
    amplitude_a: float,
    rotor_angle_factor: int = 1,
    current_angle_factor: int = 1,
    current_phase_offset_deg: float = 0.0,
    discard_ans: bool = False,
) -> dict:
    import femm

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.fem"
    result_path = output_dir / "result.json"
    shutil.copy2(base_fem, model_path)
    ans_path = model_path.with_suffix(".ans")
    if ans_path.exists():
        ans_path.unlink()
    currents = phase_currents(
        angle_deg,
        pole_pairs,
        amplitude_a,
        current_angle_factor=current_angle_factor,
        current_phase_offset_deg=current_phase_offset_deg,
    )
    applied_rotor_angle = rotor_angle_factor * angle_deg
    payload = {
        "status": "running",
        "angle_mechanical_deg": angle_deg,
        "angle_electrical_deg": pole_pairs * angle_deg,
        "pole_pairs": pole_pairs,
        "current_amplitude_a": amplitude_a,
        "rotor_angle_factor": rotor_angle_factor,
        "current_angle_factor": current_angle_factor,
        "current_phase_offset_deg": current_phase_offset_deg,
        "applied_rotor_angle_deg": applied_rotor_angle,
        "phase_currents_a": currents,
        "rotor_angle_method": "mi_modifyboundprop('sliding_airgap',10,angle_deg)",
        "torque_method": "mo_gapintegral('sliding_airgap',0)",
        "base_fem": str(base_fem.resolve()),
    }
    atomic_json(result_path, payload)
    started = time.perf_counter()
    opened = False
    try:
        femm.openfemm(1)
        opened = True
        femm.opendocument(str(model_path.resolve()))
        femm.mi_modifyboundprop("sliding_airgap", 10, applied_rotor_angle)
        for circuit, value in currents.items():
            femm.mi_modifycircprop(circuit, 1, value)
        femm.mi_saveas(str(model_path.resolve()))
        prepared = time.perf_counter()
        femm.mi_analyze(1)
        analyzed = time.perf_counter()
        femm.mi_loadsolution()
        torque = femm.mo_gapintegral("sliding_airgap", 0)
        finished = time.perf_counter()
        if isinstance(torque, complex):
            if abs(torque.imag) > 1e-10:
                raise ValueError(f"Unexpected complex torque: {torque!r}")
            torque = torque.real
        torque = float(torque)
        if not math.isfinite(torque):
            raise ValueError(f"Non-finite torque: {torque}")
        ans_size = ans_path.stat().st_size if ans_path.exists() else None
        payload.update(
            {
                "status": "complete",
                "torque_nm": torque,
                "ans_path": str(ans_path.resolve()),
                "ans_exists_before_optional_cleanup": ans_path.exists(),
                "ans_size_bytes_before_optional_cleanup": ans_size,
                "timing_seconds": {
                    "prepare": prepared - started,
                    "analyze": analyzed - prepared,
                    "postprocess": finished - analyzed,
                    "total": finished - started,
                },
            }
        )
        if discard_ans and ans_path.exists():
            ans_path.unlink()
            payload["ans_discarded_after_torque_extraction"] = True
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
    parser.add_argument("--rotor-angle-factor", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--current-angle-factor", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--current-phase-offset-deg", type=float, default=0.0)
    parser.add_argument("--discard-ans", action="store_true")
    args = parser.parse_args()
    solve(
        args.base_fem,
        args.output_dir,
        args.angle_deg,
        args.pole_pairs,
        args.amplitude_a,
        rotor_angle_factor=args.rotor_angle_factor,
        current_angle_factor=args.current_angle_factor,
        current_phase_offset_deg=args.current_phase_offset_deg,
        discard_ans=args.discard_ans,
    )


if __name__ == "__main__":
    main()
