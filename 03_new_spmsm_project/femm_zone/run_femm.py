"""03 唯一运行入口：show / prepare / solve / report。默认只显示配置。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import femm_config as config
from scripts.spmsm_mapping import build_topology, history_gene


def save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("运行名称只使用字母、数字、下划线、点和短横线，并以字母或数字开头")
    root = config.RUN_ROOT.resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise ValueError("运行目录必须位于配置中的 RUN_ROOT 下")
    return target


def tag(angle: float) -> str:
    return f"{angle:.9g}".replace("-", "m").replace(".", "p")


def check_inputs(cfg: dict) -> None:
    for path, expected in cfg["input_sha256"].items():
        if config.sha256(Path(path)) != expected:
            raise ValueError(f"原始输入已变化：{path}；请用新目录重新prepare")


def prepare(name: str) -> dict:
    cfg, output = config.load_config(), run_path(name)
    output.mkdir(parents=True, exist_ok=False)  # 不覆盖已有运行
    manifest = {"status": "preparing", "config": cfg, "genes": [], "cases": []}
    save(output / "run.json", manifest)
    try:
        for gene in cfg["genes"]:
            bits, provenance = history_gene(Path(cfg["mat_file"]), gene["state"], gene["row"], cfg["gene_source"])
            base = build_topology(bits, Path(cfg["template_file"]), Path(cfg["mat_file"]), output / gene["id"] / "base", provenance)
            stored = provenance["stored_results"]
            manifest["genes"].append({**gene, "pm_cells": int(bits.sum()),
                                      "reference_pm_cells": stored["volume_pm_cells"],
                                      "reference_tavg_nm": stored["tavg_nm"], "reference_delta_t": stored["delta_t"]})
            text = base.read_text(encoding="utf-8")
            for phase in cfg["initial_phases_deg"]:
                for inner in cfg["inner_angles_deg"]:
                    model = output / gene["id"] / f"phase_{tag(phase)}" / f"angle_{tag(inner)}" / "model.fem"
                    model.parent.mkdir(parents=True)
                    model.write_text(config.configure_fem(text, cfg, inner, phase), encoding="utf-8", newline="")
                    manifest["cases"].append({"gene_id": gene["id"], "initial_phase_deg": phase,
                                              "inner_angle_deg": inner, "model": str(model.relative_to(output)),
                                              "currents_a": config.phase_currents(cfg, inner, phase),
                                              "prepared_sha256": config.sha256(model)})
        check_inputs(cfg)
        manifest["status"] = "prepared"
    except Exception as exc:
        manifest.update(status="prepare_failed", error=str(exc))
        raise
    finally:
        save(output / "run.json", manifest)
    return manifest


def solve(name: str) -> None:
    output = run_path(name)
    manifest = read(output / "run.json")
    if manifest["status"] != "prepared":
        raise ValueError("solve只接受prepared目录；已有/中断的运行请保留并用新名称prepare")
    cfg = manifest["config"]
    check_inputs(cfg)
    for case in manifest["cases"]:
        model = output / case["model"]
        if config.sha256(model) != case["prepared_sha256"] or (model.parent / "result.json").exists():
            raise ValueError(f"准备好的模型已变动或已有结果：{model}；请重新prepare")
    connected = False
    try:
        import femm
        import win32com.client

        femm.HandleToFEMM = win32com.client.DispatchEx("femm.ActiveFEMM")
        connected = True
        femm.main_minimize()
        manifest["status"] = "running"
        save(output / "run.json", manifest)
        # 顺序求解，避免后台队列在停止后继续启动新任务。
        for index, case in enumerate(manifest["cases"], 1):
            model = output / case["model"]
            femm.opendocument(str(model))  # 所有参数已由femm_config写入此文件
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            raw = complex(femm.mo_gapintegral(cfg["airgap"]["name"], 0))
            if abs(raw.imag) > 1e-10 or not math.isfinite(raw.real):
                raise ValueError(f"异常转矩：{raw}")
            save(model.parent / "result.json", {"raw_torque_nm": raw.real,
                                                "fem_sha256": config.sha256(model),
                                                "ans_sha256": config.sha256(model.with_suffix(".ans"))})
            femm.mo_close()
            femm.mi_close()
            print(f"{index}/{len(manifest['cases'])}: {case['gene_id']} phase={case['initial_phase_deg']} inner={case['inner_angle_deg']}", flush=True)
        check_inputs(cfg)
        manifest["status"] = "solved"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "solve_failed", error=repr(exc))
        raise
    finally:
        save(output / "run.json", manifest)
        if connected:
            femm.closefemm()  # 只关闭本次DispatchEx创建的实例


def report(name: str) -> dict:
    output = run_path(name)
    manifest = read(output / "run.json")
    if manifest["status"] != "solved":
        raise ValueError("只有完整求解的运行才能生成对照报告")
    cfg, rows = manifest["config"], []
    for phase in cfg["initial_phases_deg"]:
        for gene in manifest["genes"]:
            cases = [c for c in manifest["cases"] if c["gene_id"] == gene["id"] and c["initial_phase_deg"] == phase]
            raw = []
            for case in cases:
                model = output / case["model"]
                record = read(model.parent / "result.json")
                if config.sha256(model) != record["fem_sha256"] or config.sha256(model.with_suffix(".ans")) != record["ans_sha256"]:
                    raise ValueError(f"结果文件已变动：{model.parent}")
                raw.append(record["raw_torque_nm"])
            metrics = config.torque_metrics(raw, cfg)
            row = {**gene, "initial_phase_deg": phase, **metrics,
                   "reference_ripple_relative": gene["reference_delta_t"] / abs(gene["reference_tavg_nm"])}
            for metric, reference, key in [(metrics["tavg_nm"], gene["reference_tavg_nm"], "tavg_error_percent"),
                                            (metrics[config.REFERENCE_DELTA_T_METRIC], gene["reference_delta_t"], "delta_t_error_percent")]:
                row[key] = 100 * abs(metric / reference - 1) if metric is not None and reference else None
            rows.append(row)
    summary = {"status": "complete", "config": cfg, "rows": rows,
               "report_definition": {"reference_delta_t_metric": config.REFERENCE_DELTA_T_METRIC,
                                     "reference_delta_t_unit": "N*m",
                                     "note": "29°重算的四个基因证实DeltaT为峰峰转矩差；取代求解前冻结配置中的暂定相对波动解释。"}}
    summary["legacy_minus2_comparison"] = [
        {"gene_id": row["id"], "initial_phase_deg": row["initial_phase_deg"],
         "tavg_nm": -2 * math.fsum(row["raw_torques_nm"]) / len(row["raw_torques_nm"]),
         "note": "旧经验倍率对照，未用作本次计算值"} for row in rows]
    save(output / "summary.json", summary)
    flat = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot(rows, cfg, output)
    return summary


def plot(rows: list[dict], cfg: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    for phase in cfg["initial_phases_deg"]:
        selected = [r for r in rows if r["initial_phase_deg"] == phase]
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        x = np.arange(len(selected))
        for ax, key, ref, factor, title in [(axes[0], "tavg_nm", "reference_tavg_nm", 1, "平均转矩（N·m）"),
                                            (axes[1], config.REFERENCE_DELTA_T_METRIC, "reference_delta_t", 1, "转矩波动 ΔT = Tmax − Tmin（N·m）")]:
            actual = [r[key] * factor if r[key] is not None else np.nan for r in selected]
            a = ax.bar(x-.18, [r[ref] * factor for r in selected], .34, label="MAT参考", color="#26364c")
            b = ax.bar(x+.18, actual, .34, label="计算值", color="#1674d1")
            ax.bar_label(a, fmt="%.3f", fontsize=9)
            ax.bar_label(b, fmt="%.3f", fontsize=9)
            ax.set_xticks(x, [r["id"] for r in selected])
            ax.set_title(title)
            ax.margins(y=.2)
            ax.legend()
        offset = cfg["airgap"].get("initial_inner_angle_deg", 0)
        fig.suptitle(f"03 参考与计算对照｜初始内角 {offset:g}°｜初始电流相位 {phase:g}°")
        fig.tight_layout(rect=(0,.07,1,.95))
        fig.text(.05,.025,f"T={cfg['torque_multiplier']:g}×FEMM气隙积分；六点算术平均；MAT DeltaT为峰峰转矩差（N·m）。",fontsize=9)
        fig.savefig(output / f"comparison_phase_{tag(phase)}.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for ax, row in zip(axes.flat, selected):
            ax.plot(cfg["inner_angles_deg"], row["comparison_torques_nm"], "o-", color="#1674d1", label="计算转矩")
            ax.axhline(row["reference_tavg_nm"], color="#26364c", linestyle="--", label="参考平均转矩")
            ax.set_title(f"{row['id']}｜平均：计算 {row['tavg_nm']:.4f} / 参考 {row['reference_tavg_nm']:.4f} N·m\n"
                         f"波动：计算 {row['peak_to_peak_nm']:.4f} / 参考 {row['reference_delta_t']:.4f} N·m", fontsize=10)
            ax.set_xlabel("滑动气隙内角（°）")
            ax.set_ylabel("转矩（N·m）")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        fig.suptitle(f"03 转矩曲线｜初始内角 {offset:g}°｜{cfg['current']['amplitude_a']:g} A 正向同步电流")
        fig.tight_layout(rect=(0,.04,1,.95))
        fig.text(.06,.02,"虚线只表示MAT参考平均值；MAT未提供逐角度参考曲线。转矩波动为六点最大值与最小值之差。",fontsize=9)
        fig.savefig(output / f"torque_curves_phase_{tag(phase)}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("show", "prepare", "solve", "report"), nargs="?", default="show")
    parser.add_argument("--name", help="运行目录名称，位于femm_config.RUN_ROOT中")
    args = parser.parse_args()
    if args.command == "show":
        print(json.dumps(config.load_config(), ensure_ascii=False, indent=2))
        return
    if not args.name:
        parser.error("prepare / solve / report需要 --name")
    if args.command == "prepare":
        result = prepare(args.name)
        print(f"Prepared {len(result['cases'])} models without FEMM: {run_path(args.name)}")
    elif args.command == "solve":
        solve(args.name)
        report(args.name)
    else:
        report(args.name)


if __name__ == "__main__":
    main()
