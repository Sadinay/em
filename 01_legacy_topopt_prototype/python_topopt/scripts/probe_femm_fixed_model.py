from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from femm_runner.process import run_isolated_worker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a pre-built FEM model in an isolated FEMM subprocess."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--angles", type=float, nargs="+", default=[0.0])
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--current", type=float, default=3.5)
    parser.add_argument(
        "--work-root", type=Path, default=PROJECT_ROOT / "outputs" / "femm_probes"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.model.resolve()
    if not source.is_file():
        raise SystemExit(f"model not found: {source}")
    case_dir = args.work_root.resolve() / (
        f"probe_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
    )
    case_dir.mkdir(parents=True)
    model = case_dir / "model.fem"
    result = case_dir / "result.json"
    job_path = case_dir / "job.json"
    shutil.copy2(source, model)
    job = {
        "model_path": str(model),
        "result_path": str(result),
        "apply_historical_inset": False,
        "torque_angles_deg": args.angles,
        "stator_current_a": args.current,
        "rotor_group": 1,
        "air_gap_boundary_name": "Air gap",
    }
    job_path.write_text(
        json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outcome = run_isolated_worker(job_path, timeout_seconds=args.timeout)
    (case_dir / "worker.stdout.log").write_text(outcome.stdout, encoding="utf-8")
    (case_dir / "worker.stderr.log").write_text(outcome.stderr, encoding="utf-8")
    print(f"case_dir={case_dir}")
    print(
        f"timed_out={outcome.timed_out} return_code={outcome.return_code} "
        f"terminated_pids={outcome.terminated_pids}"
    )
    if result.is_file():
        print(result.read_text(encoding="utf-8"))
    if outcome.timed_out or outcome.return_code not in (0, None):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
