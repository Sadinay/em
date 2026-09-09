from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_io.matlab import load_best_result  # noqa: E402
from evaluator.femm import IsolatedFemmEvaluator  # noqa: E402
from femm_runner.config import FemmRunConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one historical chromosome through isolated FEMM."
    )
    parser.add_argument(
        "--best-result",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "FP"
            / "IA_test_Float"
            / "best_result_20260306_174158.mat"
        ),
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=WORKSPACE_ROOT / "FP" / "CLONALG" / "blank_18x10.FEM",
    )
    parser.add_argument("--angles", type=float, nargs="+", default=[0.0])
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--geometry-mode",
        choices=("historical_inset", "merged_copper_v5"),
        default="historical_inset",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "femm_cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical = load_best_result(args.best_result)
    evaluator = IsolatedFemmEvaluator(
        FemmRunConfig(
            base_model=args.base_model,
            work_root=args.work_root,
            torque_angles_deg=tuple(args.angles),
            timeout_seconds=args.timeout,
            maximum_retries=args.retries,
            geometry_mode=args.geometry_mode,
        )
    )
    result = evaluator.evaluate(historical.chromosome)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.status != "SUCCEEDED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
