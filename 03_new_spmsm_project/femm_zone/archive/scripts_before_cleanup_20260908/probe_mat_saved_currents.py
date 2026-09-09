"""Probe the literal MAT ABC values, with no inferred current phase or dq transform."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from diagnose_linear_iron import linearize_pure_iron
from replay_fixed_current_5genes import DEFAULT_MAT, DEFAULT_TEMPLATE, ROOT, digest, save, verify_input

OUTPUT = ROOT / "femm_zone/results/mat_literal_current_probe_20260908"
SOURCE = ROOT / "femm_zone/results/fixed_current_5genes_20260908"


def analyze() -> None:
    summary = {"status": "complete", "fresh_solves": 60, "scenarios": {}}
    for mode in ("template_iron", "mat_linear_iron"):
        spec = json.loads((OUTPUT / mode / "run_spec.json").read_text(encoding="utf-8"))
        rows = []
        for c in spec["candidates"]:
            cid = c["candidate_id"]
            raw = []
            for angle in spec["angles_mechanical_deg"]:
                directory = OUTPUT / mode / cid / f"angle_{angle:03d}"
                d = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                assert d["status"] == "complete"
                verify_input(directory / "model.fem", angle, spec["fixed_currents_a"])
                raw.append(d["torque_nm_raw"])
            values = -2 * np.asarray(raw)
            mean = float(np.mean(values))
            pp = float(np.ptp(values))
            delta = c["historical_delta_t"]
            comparisons = {}
            for label, pred in {"absolute_peak_to_peak_previous_scale": pp,
                                "absolute_peak_to_peak_raw": pp / 2,
                                "absolute_peak_to_peak_sector4": pp * 2,
                                "half_peak_to_peak_previous_scale": pp / 2,
                                "relative_peak_to_peak": pp / max(abs(mean), 1e-12)}.items():
                comparisons[label] = {"value": pred, "error_percent": 100 * abs(pred - delta) / abs(delta)}
            rows.append({**c, "raw_torques_nm": raw, "historical_scale_torques_nm": values.tolist(),
                         "tavg_nm_previous_scale": mean, "peak_to_peak_nm_previous_scale": pp,
                         "tavg_error_percent": 100 * abs(mean - c["historical_tavg_nm"]) / abs(c["historical_tavg_nm"]),
                         "delta_definition_probes": comparisons})
        aggregate = {definition: float(np.mean([r["delta_definition_probes"][definition]["error_percent"] for r in rows]))
                     for definition in rows[0]["delta_definition_probes"]}
        summary["scenarios"][mode] = {"spec": spec, "candidates": rows, "mean_delta_error_percent": aggregate,
                                      "mean_tavg_error_percent": float(np.mean([r["tavg_error_percent"] for r in rows]))}
    save(OUTPUT / "summary.json", summary)
    report = ["# MAT 原样三相电流与材料参数验证", "", "日期：2026-09-08。", "",
              "本次直接 loadmat 读取 inp.Ia、inp.Ib、inp.Ic，三者均为 0 A。没有从 Is_amp 推导相电流，没有指定非零电流相位，也没有自行给 Id/Iq 添加坐标变换。", "",
              "设置：Inner Angle=0:3:15°，Outer Angle=0°，Min Angle=15°，Smart Mesh=On，Depth=36 mm，Precision=1e-8。5 个基因与前一轮相同。", "",
              "两组材料对照：第一组保留原 FEM 的非线性 Pure Iron；第二组使用 MAT 的 mu_Eisen=4000 作为线性铁心（移除 B–H 曲线）。后者是明确标注的参数解释试验，MAT 没有证明历史采用线性铁心。电流在两组中都直接使用 MAT 保存的三相数值。共 60 次新求解。", "",
              "原 MAT 中同时保存 Id=5、Iq=0、Is_amp=3.5；其与全零三相值的调用关系仍没有源码证据。原样三相测试只代表这个保存状态，不能自动当成生成 Tavg_all 的历史运行工况。", "",
              "由于零电流时可能仍有齿槽转矩，本次还检查其绝对峰峰值是否接近历史 DeltaT。下面明确区分绝对量和原始参考值，未把两者的定义认定为相同。转矩尺度沿用 −2×raw，并保留未缩放原值。", ""]
    for mode, section in summary["scenarios"].items():
        report += [f"## {mode}", "", "| 基因（代/行，0-based） | MAT Tavg | 本次Tavg N·m | MAT DeltaT | 本次绝对峰峰值 N·m | 若DeltaT是该峰峰值的误差 |", "|---|---:|---:|---:|---:|---:|"]
        for r in section["candidates"]:
            report.append(f"| {r['state_index_0based']}/{r['population_row_0based']} | {r['historical_tavg_nm']:.6f} | {r['tavg_nm_previous_scale']:.6f} | {r['historical_delta_t']:.6f} | {r['peak_to_peak_nm_previous_scale']:.6f} | {r['delta_definition_probes']['absolute_peak_to_peak_previous_scale']['error_percent']:.2f}% |")
        report += ["", "各候选波动定义的五样本平均相对误差：", "", "```json", json.dumps(section["mean_delta_error_percent"], indent=2), "```", ""]
    report += ["完整原始转矩、候选定义与误差见 [summary.json](summary.json)。每个角度保留 .fem、.ans 与 result.json。接近零平均值下的相对脉动率病态，不用于说明复现成功。"]
    (OUTPUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({k: {"mean_tavg_error_percent": v["mean_tavg_error_percent"], "mean_delta_error_percent": v["mean_delta_error_percent"]} for k,v in summary["scenarios"].items()}, indent=2), flush=True)


def run() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    before = {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    inp = loadmat(DEFAULT_MAT, variable_names=["inp"], squeeze_me=True, struct_as_record=False)["inp"]
    currents = {"A": float(inp.Ia), "B": float(inp.Ib), "C": float(inp.Ic)}
    old_spec = json.loads((SOURCE / "run_spec.json").read_text(encoding="utf-8"))
    assert inp.mu_Eisen == 4000
    tasks = []
    for mode in ("template_iron", "mat_linear_iron"):
        spec = {**old_spec, "status": "prepared", "fixed_currents_a": currents,
                "fixed_current_phase_electrical_deg": None, "current_source": "literal inp.Ia / inp.Ib / inp.Ic; no inferred phase",
                "material_scenario": mode}
        save(OUTPUT / mode / "run_spec.json", spec)
        for c in spec["candidates"]:
            cid = c["candidate_id"]
            base = OUTPUT / mode / cid / "base/model.fem"
            base.parent.mkdir(parents=True)
            text = (SOURCE / cid / "base/model.fem").read_text(encoding="utf-8")
            if mode == "mat_linear_iron":
                text = linearize_pure_iron(text)
            base.write_text(text, encoding="utf-8", newline="")
            tasks.append((mode, cid))
    worker = Path(__file__).with_name("replay_fixed_current_5genes.py")
    def launch(item):
        mode, cid = item
        d = OUTPUT / mode / cid
        with (d / "stdout.txt").open("w", encoding="utf-8") as so, (d / "stderr.txt").open("w", encoding="utf-8") as se:
            process = subprocess.run([sys.executable, "-u", str(worker), "--worker", str(OUTPUT / mode / "run_spec.json"), "--candidate", cid],
                                     stdout=so, stderr=se, timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            raise RuntimeError(f"Failed {mode}/{cid}; see stderr.txt")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(launch, task): task for task in tasks}
        for f in as_completed(futures):
            f.result()
            print("Completed six new solves:", futures[f], flush=True)
    assert before == {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    save(OUTPUT / "input_integrity.json", {"unchanged": True, "sha256": before})
    analyze()


if __name__ == "__main__":
    if "--analyze-only" in sys.argv:
        analyze()
    else:
        run()
