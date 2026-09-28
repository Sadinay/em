"""Run the guarded baseline-vs-merged six-angle FEMM comparison.

This is deliberately separate from the GA adapter.  The production path is
not changed unless the resulting report passes all numerical checks.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
from scipy.io import loadmat


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from femm_zone import femm_config as physical  # noqa: E402
from femm_zone.scripts import spmsm_mapping as mapping  # noqa: E402
from femm_zone.scripts.safe_boundary_merge import merge_equivalent_design_cells  # noqa: E402

OUT = PROJECT / "femm_zone" / "workspaces" / "safe_boundary_merge_validation_20260922"
# Acceptance remains deliberately strict.  A looser engineering tolerance may
# be reported separately, but must not be used to claim numerical equivalence.
ANGLE_ABS_TOL_NM = 5e-4
METRIC_ABS_TOL_NM = 5e-4
REL_TOL = 2e-4


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representative_genes() -> dict[str, np.ndarray]:
    low = np.zeros(120, dtype=np.uint8); low[:12] = 1
    medium = np.zeros(120, dtype=np.uint8); medium[:60] = 1
    high = np.ones(120, dtype=np.uint8); high[:12] = 0
    return {"low_pm12_connected": low, "medium_pm60_connected": medium,
            "high_pm108_connected": high}


def ans_nodes(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="strict")
    match = re.search(r"^\[Solution\]\s*\r?\n(\d+)", text, re.M)
    if not match:
        raise ValueError(f"Cannot read solution node count: {path}")
    return int(match.group(1))


def set_design_max_area(text: str, rows, max_area: float) -> str:
    match = re.search(r"\[NumBlockLabels\]\s*=\s*(\d+)\s*\r?\n", text)
    count = int(match.group(1)); lines = text[match.end():].splitlines(keepends=True)[:count]
    targets = {int(row["label_index_1based"]) - 1 for row in rows}
    for index in targets:
        fields = lines[index].split(); fields[3] = format(max_area, ".12g")
        lines[index] = "\t".join(fields) + "\n"
    end = match.end() + sum(len(line) for line in text[match.end():].splitlines(keepends=True)[:count])
    return text[:match.end()] + "".join(lines) + text[end:]


def run(output_dir: Path = OUT, min_angle: float | None = None,
        design_max_area: float | None = None) -> dict:
    import femm
    import win32com.client

    cfg = physical.load_config()
    if min_angle is not None:
        cfg["problem"]["MinAngle"] = float(min_angle)
    template = physical.TEMPLATE_FILE.read_text(encoding="utf-8")
    positions = loadmat(physical.MAT_FILE, variable_names=["MaterialPosition"],
                        squeeze_me=True)["MaterialPosition"]
    rows = mapping.match_material_positions(template, positions)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    femm.HandleToFEMM = win32com.client.DispatchEx("femm.ActiveFEMM")
    femm.main_minimize()
    try:
        for name, bits in representative_genes().items():
            plain = mapping.replace_cell_materials(template, bits)
            mapping.validate_generated_model(plain, rows, bits)
            if design_max_area is not None:
                plain = set_design_max_area(plain, rows, design_max_area)
            merged, audit = merge_equivalent_design_cells(plain, bits, rows)
            gene = mapping.genotype_sha256(bits)
            raw = {"baseline": [], "merged": []}
            elapsed = {"baseline": [], "merged": []}
            meshes = {"baseline": [], "merged": []}
            for variant, base in (("baseline", plain), ("merged", merged)):
                for angle in cfg["inner_angles_deg"]:
                    folder = output_dir / name / variant / f"angle_{angle:g}"
                    folder.mkdir(parents=True, exist_ok=True)
                    model = folder / "model.fem"
                    result_path = folder / "result.json"
                    prepared = physical.configure_fem(base, cfg, angle, 0)
                    prepared_sha = hashlib.sha256(prepared.encode("utf-8")).hexdigest()
                    if result_path.exists():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        if result.get("prepared_sha256") != prepared_sha:
                            raise ValueError(f"Changed prepared input in {folder}")
                    else:
                        model.write_text(prepared, encoding="utf-8", newline="")
                        started = time.perf_counter()
                        femm.opendocument(str(model))
                        femm.mi_analyze(1)
                        femm.mi_loadsolution()
                        torque = complex(femm.mo_gapintegral("sliding_airgap", 0))
                        if abs(torque.imag) > 1e-10 or not math.isfinite(torque.real):
                            raise ValueError(f"Invalid FEMM torque: {torque}")
                        femm.mo_close(); femm.mi_close()
                        ans = folder / "model.ans"
                        result = {"gene_id": gene, "variant": variant, "angle_deg": angle,
                                  "raw_torque_nm": float(torque.real),
                                  "elapsed_seconds": time.perf_counter() - started,
                                  "solution_nodes": ans_nodes(ans),
                                  "prepared_sha256": prepared_sha,
                                  "fem_sha256": file_sha(model), "ans_sha256": file_sha(ans)}
                        save(result_path, result)
                    raw[variant].append(float(result["raw_torque_nm"]))
                    elapsed[variant].append(float(result["elapsed_seconds"]))
                    meshes[variant].append(int(result["solution_nodes"]))
            base_metrics = physical.torque_metrics(raw["baseline"], cfg)
            merge_metrics = physical.torque_metrics(raw["merged"], cfg)
            diffs = [abs(a-b) for a, b in zip(raw["baseline"], raw["merged"])]
            angle_pass = all(math.isclose(a, b, abs_tol=ANGLE_ABS_TOL_NM, rel_tol=REL_TOL)
                             for a, b in zip(raw["baseline"], raw["merged"]))
            metric_pass = all(math.isclose(base_metrics[k], merge_metrics[k],
                                           abs_tol=METRIC_ABS_TOL_NM, rel_tol=REL_TOL)
                              for k in ("tavg_nm", "peak_to_peak_nm"))
            records.append({"name": name, "gene_id": gene, "magnet_cells": int(bits.sum()),
                            "bits": "".join(map(str, bits.tolist())), "merge_audit": audit,
                            "baseline": {"raw_torques_nm": raw["baseline"], **base_metrics,
                                         "elapsed_seconds": sum(elapsed["baseline"]),
                                         "mean_solution_nodes": float(np.mean(meshes["baseline"]))},
                            "merged": {"raw_torques_nm": raw["merged"], **merge_metrics,
                                       "elapsed_seconds": sum(elapsed["merged"]),
                                       "mean_solution_nodes": float(np.mean(meshes["merged"]))},
                            "comparison": {"max_angle_abs_error_nm": max(diffs),
                                           "tavg_abs_error_nm": abs(base_metrics["tavg_nm"]-merge_metrics["tavg_nm"]),
                                           "peak_to_peak_abs_error_nm": abs(base_metrics["peak_to_peak_nm"]-merge_metrics["peak_to_peak_nm"]),
                                           "angle_pass": angle_pass, "metric_pass": metric_pass,
                                           "passed": angle_pass and metric_pass}})
            save(output_dir / "partial_results.json", records)
    finally:
        try:
            femm.closefemm()
        except Exception:
            pass
    passed = all(item["comparison"]["passed"] for item in records)
    report = {"status": "passed" if passed else "failed", "production_default_enabled": False,
              "tolerances": {"angle_abs_nm": ANGLE_ABS_TOL_NM,
                             "metric_abs_nm": METRIC_ABS_TOL_NM, "relative": REL_TOL},
              "mesh_control": {"MinAngle": cfg["problem"]["MinAngle"],
                               "design_max_area_mm2": design_max_area}, "cases": records}
    save(output_dir / "validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-angle", type=float)
    parser.add_argument("--design-max-area", type=float)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.output, args.min_angle, args.design_max_area)
    print(json.dumps({"status": result["status"], "report": str(args.output / "validation.json")},
                     ensure_ascii=False))
