"""Test whether the MAT mu_Eisen=4000 linear material explains DeltaT."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from diagnose_angle_current_directions import CANDIDATE_IDS, solve_task
from validate_history_replay import ANGLES, DEFAULT_OUTPUT, atomic_json


def linearize_pure_iron(text: str) -> str:
    pattern = re.compile(r"<BeginBlock>.*?<EndBlock>", flags=re.S)
    blocks = pattern.findall(text)
    target = next(block for block in blocks if re.search(r'<BlockName>\s*=\s*"Pure Iron"', block))
    replacement = re.sub(r"(<Mu_x>\s*=\s*)[^\r\n]+", r"\g<1>4000", target)
    replacement = re.sub(r"(<Mu_y>\s*=\s*)[^\r\n]+", r"\g<1>4000", replacement)
    replacement = re.sub(r"(<BHPoints>\s*=\s*)21\s*\r?\n(?:\s*[^\r\n]+\r?\n){21}", r"\g<1>0\n", replacement)
    if "<BHPoints> = 0" not in replacement:
        raise AssertionError("Failed to remove the Pure Iron BH curve")
    return text.replace(target, replacement)


def run(output_root: Path, workers: int, timeout: float) -> dict:
    selection = json.loads((output_root / "selected_candidates.json").read_text(encoding="utf-8"))["candidates"]
    by_id = {row["candidate_id"]: row for row in selection}
    diagnostic_root = output_root / "material_diagnostics" / "linear_iron_mu4000"
    base_models = {}
    for candidate_id in CANDIDATE_IDS:
        source = output_root / candidate_id / "base" / "model.fem"
        base_dir = diagnostic_root / candidate_id / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        destination = base_dir / "model.fem"
        destination.write_text(linearize_pure_iron(source.read_text(encoding="utf-8")), encoding="utf-8", newline="")
        shutil.copy2(output_root / candidate_id / "base" / "manifest.json", base_dir / "source_manifest.json")
        base_models[candidate_id] = destination

    futures = {}
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for candidate_id in CANDIDATE_IDS:
            for angle in ANGLES:
                future = executor.submit(
                    solve_task,
                    base_models[candidate_id],
                    diagnostic_root / candidate_id / f"angle_{angle:g}".replace(".", "p"),
                    angle,
                    1,
                    1,
                    timeout,
                )
                futures[future] = (candidate_id, angle)
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(f"complete: {key[0]}, angle={key[1]:g}", flush=True)

    comparisons = []
    for candidate_id in CANDIDATE_IDS:
        raw = np.asarray([results[(candidate_id, angle)]["torque_nm"] for angle in ANGLES])
        values = -2 * raw
        tavg = float(np.trapezoid(values, np.asarray(ANGLES)) / (ANGLES[-1] - ANGLES[0]))
        delta = float(np.ptp(values))
        historical_tavg = float(by_id[candidate_id]["historical_tavg_nm"])
        historical_delta = float(by_id[candidate_id]["historical_delta_t_nm"])
        comparisons.append(
            {
                "candidate_id": candidate_id,
                "historical_tavg_nm": historical_tavg,
                "recomputed_tavg_nm": tavg,
                "relative_error_tavg": abs(tavg - historical_tavg) / abs(historical_tavg),
                "historical_delta_t_nm": historical_delta,
                "recomputed_delta_t_nm": delta,
                "relative_error_delta_t": abs(delta - historical_delta) / abs(historical_delta),
                "sign_and_sector_scaled_torques_nm": values.tolist(),
            }
        )
    summary = {
        "status": "complete",
        "variant": "Pure Iron linear mu_x=mu_y=4000, BHPoints=0",
        "angles_mechanical_deg": list(ANGLES),
        "torque_multiplier": -2,
        "mean_relative_error_tavg": float(np.mean([row["relative_error_tavg"] for row in comparisons])),
        "mean_relative_error_delta_t": float(np.mean([row["relative_error_delta_t"] for row in comparisons])),
        "candidates": comparisons,
    }
    atomic_json(diagnostic_root / "analysis_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.workers, args.timeout_seconds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
