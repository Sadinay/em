"""Fresh five-gene FEMM replay: fixed ABC current, 0:3:15 mechanical degrees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from spmsm_mapping import DEFAULT_MAT, DEFAULT_TEMPLATE, build_topology, history_gene

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "femm_zone/results/fixed_current_5genes_20260908"
PREVIOUS = ROOT / "femm_zone/results/history_replay_validation"
SELECTION = [
    ("low_tavg", 0, 338),
    ("median_tavg", 31, 190),
    ("high_tavg", 193, 258),
    ("final_low_tavg", 200, 491),
    ("final_high_tavg", 200, 33),
]
ANGLES = [0, 3, 6, 9, 12, 15]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle_directory(angle: float) -> str:
    return f"angle_{int(angle):03d}" if float(angle).is_integer() else "angle_" + f"{angle:06.2f}".replace(".", "p")


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def field(text: str, name: str) -> float:
    return float(re.search(r"<" + re.escape(name) + r">\s*=\s*([^\r\n]+)", text)[1])


def verify_input(path: Path, angle: float, currents: dict) -> dict:
    text = path.read_text(encoding="utf-8")
    headers = {}
    for key in ("Frequency", "Precision", "MinAngle", "DoSmartMesh", "Depth"):
        headers[key] = float(re.search(r"\[" + key + r"\]\s*=\s*([^\r\n]+)", text)[1])
    assert headers == {"Frequency": 0, "Precision": 1e-8, "MinAngle": 15, "DoSmartMesh": 1, "Depth": 36}, headers
    gap = next(b for b in re.findall(r"<BeginBdry>.*?<EndBdry>", text, re.S) if '"sliding_airgap"' in b)
    assert field(gap, "BdryType") == 6
    assert abs(field(gap, "innerangle") - angle) < 1e-10
    assert field(gap, "outerangle") == 0
    saved_currents = {}
    for block in re.findall(r"<BeginCircuit>.*?<EndCircuit>", text, re.S):
        name = re.search(r'<CircuitName>\s*=\s*"([^"]+)"', block)[1]
        saved_currents[name] = field(block, "TotalAmps_re")
        assert field(block, "TotalAmps_im") == 0
    assert saved_currents.keys() == currents.keys()
    assert all(abs(saved_currents[k] - currents[k]) < 1e-12 for k in currents)
    return {"headers": headers, "inner_angle_deg": angle, "outer_angle_deg": 0,
            "fixed_currents_a": saved_currents, "fem_sha256": digest(path)}


def worker(spec_path: Path, candidate_id: str) -> None:
    import femm
    import win32com.client

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = spec_path.parent / candidate_id
    base = output / "base/model.fem"
    started = time.perf_counter()
    connected = False
    try:
        # DispatchEx requests a new COM server instead of attaching to a user's open FEMM.
        femm.HandleToFEMM = win32com.client.DispatchEx("femm.ActiveFEMM")
        connected = True
        femm.main_minimize()
        for angle in spec["angles_mechanical_deg"]:
            directory = output / angle_directory(angle)
            directory.mkdir(parents=True, exist_ok=True)
            model = directory / "model.fem"
            if (directory / "result.json").exists() or model.with_suffix(".ans").exists():
                raise FileExistsError(f"Fresh replay refuses cached results: {directory}")
            shutil.copy2(base, model)
            tick = time.perf_counter()
            femm.opendocument(str(model.resolve()))
            femm.mi_modifyboundprop("sliding_airgap", 10, angle)
            femm.mi_modifyboundprop("sliding_airgap", 11, 0)
            for circuit, current in spec["fixed_currents_a"].items():
                femm.mi_modifycircprop(circuit, 1, current)
            femm.mi_saveas(str(model.resolve()))
            verified = verify_input(model, angle, spec["fixed_currents_a"])
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            raw = femm.mo_gapintegral("sliding_airgap", 0)
            if isinstance(raw, complex):
                assert abs(raw.imag) < 1e-10
                raw = raw.real
            raw = float(raw)
            assert math.isfinite(raw)
            assert model.with_suffix(".ans").exists()
            save(directory / "result.json", {
                "status": "complete", "candidate_id": candidate_id,
                "angle_mechanical_deg": angle, "current_phase_electrical_deg": spec.get("fixed_current_phase_electrical_deg", 0),
                "torque_nm_raw": raw, "torque_nm_historical_scale": -2 * raw,
                "verified_input": verified, "elapsed_seconds": time.perf_counter() - tick,
                "ans_path": str(model.with_suffix(".ans")), "fresh_solve": True,
            })
            print(f"{candidate_id} angle={angle} raw={raw:.9f}", flush=True)
            femm.mo_close()
            femm.mi_close()
        save(output / "worker_status.json", {"status": "complete", "solves": len(spec["angles_mechanical_deg"]), "elapsed_seconds": time.perf_counter() - started})
    except Exception:
        save(output / "worker_status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        if connected:
            try:
                femm.closefemm()
            except Exception:
                pass


def metrics(raw: list[float], historical_tavg: float, historical_delta: float) -> dict:
    torque = -2 * np.asarray(raw)
    result = {"raw_femm_torques_nm": raw, "historical_scale_torques_nm": torque.tolist(),
              "peak_to_peak_nm": float(np.ptp(torque))}
    for method, avg in [("mean6", float(np.mean(torque))),
                        ("trapezoidal", float(np.trapezoid(torque, ANGLES) / 15))]:
        ripple = float(np.ptp(torque)) / max(abs(avg), 1e-12)
        result[method] = {
            "tavg_nm": avg, "ripple_relative": ripple,
            "tavg_relative_error_percent": 100 * abs(avg - historical_tavg) / abs(historical_tavg),
            "delta_relative_error_percent": 100 * abs(ripple - historical_delta) / abs(historical_delta),
        }
    return result


def analyze(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = json.loads((output / "run_spec.json").read_text(encoding="utf-8"))
    rows = []
    long_rows = []
    for candidate in spec["candidates"]:
        cid = candidate["candidate_id"]
        raw = []
        for angle in ANGLES:
            p = output / cid / f"angle_{angle:03d}"
            record = json.loads((p / "result.json").read_text(encoding="utf-8"))
            assert record["status"] == "complete"
            verify_input(p / "model.fem", angle, spec["fixed_currents_a"])
            raw.append(record["torque_nm_raw"])
            long_rows.append({"candidate_id": cid, "angle_mechanical_deg": angle,
                              "ia_a": spec["fixed_currents_a"]["A"], "ib_a": spec["fixed_currents_a"]["B"],
                              "ic_a": spec["fixed_currents_a"]["C"], "raw_torque_nm": raw[-1], "historical_scale_torque_nm": -2 * raw[-1]})
        row = {**candidate, "fixed_current": metrics(raw, candidate["historical_tavg_nm"], candidate["historical_delta_t"])}
        previous_raw = []
        for angle in ANGLES:
            old = json.loads((PREVIOUS / cid / f"angle_{angle}/result.json").read_text(encoding="utf-8"))
            old_model = PREVIOUS / cid / f"angle_{angle}/model.fem"
            expected = {
                "A": spec["amplitude_a"] * math.cos(math.radians(spec["pole_pairs"] * angle)),
                "B": spec["amplitude_a"] * math.cos(math.radians(spec["pole_pairs"] * angle - 120)),
                "C": spec["amplitude_a"] * math.cos(math.radians(spec["pole_pairs"] * angle + 120)),
            }
            verify_input(old_model, angle, expected)
            previous_raw.append(float(old["torque_nm"]))
        row["previous_synchronous_current"] = metrics(previous_raw, candidate["historical_tavg_nm"], candidate["historical_delta_t"])
        rows.append(row)
    aggregate = {}
    for condition in ("fixed_current", "previous_synchronous_current"):
        aggregate[condition] = {
            method: {key: float(np.mean([r[condition][method][key] for r in rows]))
                     for key in ("tavg_relative_error_percent", "delta_relative_error_percent")}
            for method in ("mean6", "trapezoidal")
        }
    save(output / "summary.json", {"status": "complete", "fresh_solves": 30, "spec": spec, "aggregate": aggregate, "candidates": rows})
    with (output / "torque_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, row in zip(axes.flat, rows):
        ax.plot(ANGLES, row["fixed_current"]["historical_scale_torques_nm"], "o-", label="Fixed ABC (new)")
        ax.plot(ANGLES, row["previous_synchronous_current"]["historical_scale_torques_nm"], "s--", label="Synchronous ABC (previous)")
        ax.axhline(row["historical_tavg_nm"], color="black", alpha=.6, linestyle=":", label="Historical MAT Tavg")
        ax.set(title=f"{row['candidate_id']} | g{row['state_index_0based']}, row{row['population_row_0based']}",
               xlabel="Inner Angle (mechanical deg)", ylabel="Torque (-2 x FEMM raw), N m", xticks=ANGLES)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02, .95, "Min Angle = 15 deg; Outer Angle = 0 deg\nFixed ABC = [3.5, -1.75, -1.75] A\n30 fresh FEMM solves, five unchanged history genes\nHistorical DeltaT compared as relative peak-to-peak\nNo historical angle-by-angle waveforms available\nTorque multiplier -2 retained; no fitting", va="top", fontsize=11)
    fig.savefig(output / "torque_comparison.png", dpi=180)
    plt.close(fig)
    near_zero = all(abs(r["fixed_current"]["mean6"]["tavg_nm"]) < .01 * abs(r["historical_tavg_nm"]) for r in rows)
    conclusion = ("本次固定电流方案未复现历史数据：5 个基因的平均转矩均接近零，且曲线在指定扫描区间内发生正负变化。不能据此认定只转气隙、固定电流就是历史 DeltaT 不一致的原因。结论限定于本次明确记录的固定电流值和相位。" if near_zero else "请根据下表同时检查平均转矩与脉动误差，不能只以一个指标判断历史复现成功。")
    report = ["# 03：固定电流、只转气隙的五基因复现验证", "", "日期：2026-09-08。", "",
              "完成 5 个历史基因 × 6 个机械角，共 30 次全新 FEMM 求解。", "", conclusion, "",
              "设置：Inner Angle=0°、3°、6°、9°、12°、15°；Outer Angle=0°；Min Angle=15°；Smart Mesh=On；Depth=36 mm；Precision=1e-8；Frequency=0。",
              "", "电流固定为 A/B/C=3.5、−1.75、−1.75 A。幅值取自 03 MAT 的 inp.Is_amp=3.5；固定相位选为当前相序下初始电角度 0°。这是本次明确采用的试验解释，MAT 本身没有证明历史程序采用这个固定相位。未使用 MAT 中另外保存的 Id=5/Iq=0 或全零 Ia/Ib/Ic。",
              "", "材料、几何、磁化方向保持 03 原始 FEM 模板及已验证基因映射；铁心仍使用模板的非线性 Pure Iron，不将 MAT 中 mu_Eisen=4000 当作新的线性材料设置。原始 MAT/FEM 未改动。",
              "", "所有求解前及汇总时均读取生成 FEM 文件检查：Min Angle、精度、深度、Smart Mesh、内外气隙角以及三相电流。每个角度的 .fem、.ans、result.json 均保留。",
              "", "## 与历史 MAT 对比", "", "主表采用六点算术平均。相对脉动=(max(T)-min(T))/abs(mean(T))。历史 DeltaT 按相对脉动解释，但原历史后处理代码仍缺失。转矩统一保留既有经验尺度 −2×FEMM raw，未重新拟合比例或方向。", "",
              "本次均值接近零时，相对脉动的分母也接近零，导致比值极大且对数值扰动敏感；下表保留计算值供审计，不应将其作为常规带载脉动率解释。判断复现失败应首先看平均转矩与波形；绝对峰峰值见 summary.json。", "",
              "| 基因（代/行，0-based） | 历史Tavg | 固定电流Tavg | Tavg误差 | 历史DeltaT | 固定电流相对脉动 | 脉动误差 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        m = r["fixed_current"]["mean6"]
        report.append(f"| {r['state_index_0based']}/{r['population_row_0based']} | {r['historical_tavg_nm']:.6f} | {m['tavg_nm']:.6f} | {m['tavg_relative_error_percent']:.2f}% | {r['historical_delta_t']:.6f} | {m['ripple_relative']:.6f} | {m['delta_relative_error_percent']:.2f}% |")
    report += ["", "## 五个相同基因的两种电流方式", "", "同步电流曲线来自之前已完成的求解，并核对其保存模型为相同 Min Angle=15°与同一角度下同步电流。它们不计入本次 30 次新求解。", "", "| 电流方式 | 平均方法 | Tavg平均相对误差 | DeltaT平均相对误差 |", "|---|---|---:|---:|"]
    for condition, names in [("fixed_current", "本次固定电流"), ("previous_synchronous_current", "之前同步电流")]:
        for method, label in [("mean6", "六点算术平均"), ("trapezoidal", "梯形角度平均")]:
            a = aggregate[condition][method]
            report.append(f"| {names} | {label} | {a['tavg_relative_error_percent']:.3f}% | {a['delta_relative_error_percent']:.3f}% |")
    report += ["", "[五基因转矩曲线](torque_comparison.png) · [完整数值](summary.json) · [30个角度原始数据](torque_samples.csv)", "", "本次只有六个指定角度的样本，峰峰值针对这六点计算，不能保证捕获连续波形的所有极值。两端点是否属于历史实际采样集合、历史电流相位及转矩后处理，仍需要原 fitness_GA5_SPMSM.m 确认。"]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


def run(output: Path, workers: int) -> None:
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True)
    inp = loadmat(DEFAULT_MAT, variable_names=["inp"], squeeze_me=True, struct_as_record=False)["inp"]
    before = {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    assert inp.P == 4 and inp.Is_amp == 3.5 and inp.steps == 6
    candidates = []
    for cid, state, row in SELECTION:
        bits, provenance = history_gene(DEFAULT_MAT, state, row)
        original, _ = history_gene(DEFAULT_MAT, state, row, "population_noChange_all")
        assert np.array_equal(bits, original)
        build_topology(bits, DEFAULT_TEMPLATE, DEFAULT_MAT, output / cid / "base", provenance)
        stored = provenance["stored_results"]
        candidates.append({"candidate_id": cid, "state_index_0based": state, "population_row_0based": row,
                           "pm_cells": int(bits.sum()), "historical_tavg_nm": stored["tavg_nm"],
                           "historical_delta_t": stored["delta_t_nm"], "before_after_identical": True})
    spec = {"status": "prepared", "angles_mechanical_deg": ANGLES, "min_angle_deg": 15,
            "pole_pairs": int(inp.P), "amplitude_a": float(inp.Is_amp),
            "fixed_currents_a": {"A": float(inp.Is_amp), "B": -float(inp.Is_amp)/2, "C": -float(inp.Is_amp)/2},
            "fixed_current_phase_electrical_deg": 0, "outer_angle_deg": 0,
            "torque_multiplier": -2, "multiplier_source": "previous mean-torque validation; not refitted",
            "mat_unused_current_parameters": {"Id": float(inp.Id), "Iq": float(inp.Iq), "Ia": float(inp.Ia), "Ib": float(inp.Ib), "Ic": float(inp.Ic)},
            "input_sha256": before, "candidates": candidates}
    save(output / "run_spec.json", spec)

    def launch(candidate: dict) -> None:
        cid = candidate["candidate_id"]
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(output / "run_spec.json"), "--candidate", cid]
        with (output / cid / "worker_stdout.txt").open("w", encoding="utf-8") as stdout, (output / cid / "worker_stderr.txt").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if completed.returncode:
            raise RuntimeError(f"Worker {cid} failed; see worker_stderr.txt")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = {executor.submit(launch, c): c["candidate_id"] for c in candidates}
        for future in as_completed(tasks):
            future.result()
            print(f"Completed 6 new solves: {tasks[future]}", flush=True)
    assert before == {str(p): digest(p) for p in (DEFAULT_MAT, DEFAULT_TEMPLATE)}
    save(output / "input_integrity.json", {"unchanged": True, "sha256": before})
    analyze(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--candidate")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.candidate)
    elif args.analyze_only:
        analyze(args.output.resolve())
    else:
        run(args.output.resolve(), args.workers)
