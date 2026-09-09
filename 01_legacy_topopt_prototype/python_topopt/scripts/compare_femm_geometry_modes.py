from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_io.matlab import load_best_result  # noqa: E402
from evaluator.femm import IsolatedFemmEvaluator  # noqa: E402
from femm_runner.config import FemmRunConfig  # noqa: E402


COUNT_PATTERN = re.compile(
    r"^\[(NumPoints|NumSegments|NumArcSegments|NumBlockLabels)\]\s*=\s*(\d+)",
    re.MULTILINE,
)


def geometry_counts(model: Path) -> dict[str, int]:
    text = model.read_text(encoding="utf-8", errors="replace")
    return {name: int(value) for name, value in COUNT_PATTERN.findall(text)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B test historical_inset and merged_copper_v5 in FEMM."
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
    parser.add_argument(
        "--work-root", type=Path, default=PROJECT_ROOT / "outputs" / "femm_ab"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "femm_geometry_ab.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chromosome = load_best_result(args.best_result).chromosome
    records: dict[str, dict] = {}
    for mode in ("historical_inset", "merged_copper_v5"):
        evaluator = IsolatedFemmEvaluator(
            FemmRunConfig(
                base_model=args.base_model,
                work_root=args.work_root / mode,
                torque_angles_deg=tuple(args.angles),
                timeout_seconds=args.timeout,
                maximum_retries=0,
                geometry_mode=mode,
            )
        )
        result = evaluator.evaluate(chromosome)
        record = result.to_dict()
        case_dir = Path(result.metadata["case_dir"])
        record["geometry_counts"] = geometry_counts(case_dir / "model.fem")
        records[mode] = record
        if result.status != "SUCCEEDED":
            raise SystemExit(f"{mode} failed; see {case_dir}")

    old = records["historical_inset"]
    new = records["merged_copper_v5"]
    old_angle = old["metadata"]["timings"]["angles"][0]
    new_angle = new["metadata"]["timings"]["angles"][0]
    comparison = {
        "modes": records,
        "comparison": {
            "first_angle_torque_difference_nm": (
                new["torque_values_nm"][0] - old["torque_values_nm"][0]
            ),
            "first_angle_torque_relative_change": (
                new["torque_values_nm"][0] / old["torque_values_nm"][0] - 1.0
            ),
            "first_angle_solve_speedup": (
                old_angle["solve_seconds"] / new_angle["solve_seconds"]
            ),
            "first_angle_mesh_element_reduction": (
                1.0 - new_angle["mesh_elements"] / old_angle["mesh_elements"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
