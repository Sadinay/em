"""Compare one common initial phase across five historical genes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scan_initial_current_phase import OUTPUT, BASELINE
from replay_fixed_current_5genes import save, digest


def render(output: Path) -> None:
    fresh = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    baseline = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    assert fresh["status"] == baseline["status"] == "complete"
    candidates = fresh["spec"]["candidates"]
    phase_map = {s["id"]: s["phase_offset_electrical_deg"] for s in fresh["spec"]["scenarios"]}
    rows = [{**copy.deepcopy(r), "initial_phase_electrical_deg": phase_map[r["scenario_id"]], "provenance": "fresh FEMM solve"} for r in fresh["results"]]
    rows += [{**copy.deepcopy(r), "initial_phase_electrical_deg": 0.0, "provenance": "previous validated baseline"}
             for r in baseline["results"] if r["scenario_id"] == "is35_plus"]
    phases = sorted({r["initial_phase_electrical_deg"] for r in rows})
    by = {(r["initial_phase_electrical_deg"], r["candidate_id"]): r for r in rows}
    aggregate = []
    for phase in phases:
        selected = [by[phase, c["candidate_id"]] for c in candidates]
        aggregate.append({"initial_phase_electrical_deg": phase,
                          **{key: float(np.mean([r[key] for r in selected])) for key in
                             ["tavg_error_percent", "ripple_error_percent_if_relative", "peak_to_peak_error_percent_if_absolute"]},
                          "maximum_tavg_error_percent": max(r["tavg_error_percent"] for r in selected),
                          "maximum_ripple_error_percent": max(r["ripple_error_percent_if_relative"] for r in selected)})
    save(output / "phase_comparison.json", {"fresh_solves": fresh["fresh_solves"], "reused_baseline_solves": 30,
                                           "baseline_summary_sha256": digest(BASELINE / "summary.json"),
                                           "aggregate": aggregate, "results": rows})

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f6f8fc", "savefig.facecolor": "#f6f8fc"})
    colors = ["#1674d1", "#d9682f", "#169986", "#905ba5", "#b89b2e"]
    fig = plt.figure(figsize=(17, 12.5))
    fig.text(.05, .96, "03 初始电流相位验证｜3.5 A 正向同步更新", fontsize=23, weight="bold", color="#23334d")
    fig.text(.05, .925, "电流角 θe = 4 × 气隙内角 + φ0；内角 0°～15°、步长 3°；外角 0°；Min Angle=15°", fontsize=12, color="#5b6980")
    near = [p for p in phases if abs(p) <= 3]
    axes = [fig.add_axes([.085, .59, .38, .255]), fig.add_axes([.565, .59, .38, .255])]
    for c, color in zip(candidates, colors):
        cid = c["candidate_id"]
        axes[0].plot(near, [100*(by[p,cid]["tavg_nm"]/c["historical_tavg_nm"]-1) for p in near], "o-", color=color, label=c["gene_label"], lw=1.8)
        axes[1].plot(near, [100*(by[p,cid]["ripple_relative"]/c["historical_delta_t"]-1) for p in near], "o-", color=color, label=c["gene_label"], lw=1.8)
    for ax, title in zip(axes, ["0°附近：平均转矩相对参考的偏差", "0°附近：转矩波动相对参考的偏差"]):
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
        ax.axhline(0, color="#25334d", ls=":", lw=1.5)
        ax.set(xticks=near, xlabel="初始电流相位 φ0（电角度 °）", ylabel="(计算值 / 参考值 − 1) × 100%")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, ncol=5, fontsize=10, loc="lower left", bbox_to_anchor=(-.01, 1.15))
    tax = fig.add_axes([.05, .115, .90, .38])
    tax.axis("off")
    tax.set_title("完整对照｜每格上行：平均转矩 N·m；下行：相对转矩波动率", loc="left", fontsize=14, weight="bold", pad=12)
    table_rows = [["MAT 参考", *[f"{c['historical_tavg_nm']:.4f}\n{100*c['historical_delta_t']:.2f}%" for c in candidates]]]
    for phase in phases:
        table_rows.append([f"φ0 = {phase:+g}°" if phase else "φ0 = 0°（原方案）",
                           *[f"{by[phase,c['candidate_id']]['tavg_nm']:.4f}\n{by[phase,c['candidate_id']]['ripple_percent']:.2f}%" for c in candidates]])
    table = tax.table(cellText=table_rows, colLabels=["初始电角度", *[f"{c['gene_label']}  代{c['state_index_0based']}/行{c['population_row_0based']}" for c in candidates]],
                      cellLoc="center", colWidths=[.23,*([.154]*5)], bbox=[0,0,1,1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor("#dfe6ef")
        if r==0:
            cell.set_facecolor("#e2e9f3"); cell.set_text_props(weight="bold", color="#25334d")
        elif r==1:
            cell.set_facecolor("#eef1f6"); cell.set_text_props(weight="bold")
        else:
            phase=phases[r-2]
            cell.set_facecolor("#e5f0fb" if phase==0 else "#f1eafa" if phase==120 else "white")
            if c==0:
                cell.set_text_props(weight="bold")
    fig.text(.05, .069, "φ0=120°检验三相电流循环对应关系；φ0=0°复用已核验的 30 次结果，其余共新增 150 次求解。所有基因使用相同 φ0。", fontsize=10, color="#55657d")
    fig.text(.05, .045, "T 沿用此前 −2 × FEMM 气隙转矩的经验换算；波动率=(最大值−最小值)/|均值|×100%。历史尺度及 DeltaT 公式尚未确认。", fontsize=10, color="#55657d")
    fig.savefig(output / "phase_comparison.png", dpi=180)
    plt.close(fig)

    report = ["# 03 初始电流相位验证", "", "## 已确认的原始复现方式", "",
              "最初的 03 复现已同时更新滑动气隙内角和 3.5 A 三相电流：内角 0:3:15°，电流相位 0:12:60°。最初就使用了滑动气隙旋转，不存在漏转转子的证据。上一轮新旧同步结果最大差异约 7.4e-13 N·m。", "",
              "## 本次方法", "",
              "固定 Is_amp=3.5 A、P=4；θe=4θinner+φ0；Ia=3.5cosθe，Ib=3.5cos(θe−120°)，Ic=3.5cos(θe+120°)。只有 φ0 在试验之间变化，同一个 φ0 同时用于全部五个基因。内角 0、3、6、9、12、15°，外角仍为 0°，Min Angle=15°。", "",
              "φ0=±1°、±3°测试初始角的小偏差，φ0=120°测试三相电流循环对应关系。120°起始 ABC 为 −1.75、3.5、−1.75 A。该偏移是诊断假设，未宣称来自 MAT 中明确的电流初始角字段。150 次新求解；0°基线复用上一轮30次已验证的结果。", "",
              "MAT 中 rotorschift=22.5、PM2schiftangle=67.5 等为几何相关字段；N极参考中心坐标角为67.5°。这些线索不能直接等同于电流初相位；没有发现明确标注为初始电流角的已保存参数。机械位置零点与电流/磁轴零点也不能自动等同。", "",
              "变换定义参考：[MathWorks dq 变换及轴约定](https://www.mathworks.com/help/mcb/ref/surfacemountpmsm.html)。资料支持角度关系与变换定义，不证明历史脚本如何设置 φ0。", "",
              "转矩沿用之前 −2×FEMM 气隙积分的经验比较口径，未重新拟合比例；历史 DeltaT 暂按 (max(T)−min(T))/abs(mean(T)) 解释。两项历史口径仍未确定。", "",
              "![初始相位对照](phase_comparison.png)", "", "## 五基因误差汇总", "",
              "| 初始电角度 | Tavg平均相对误差 | 波动平均相对误差 | 单个基因最大Tavg误差 | 单个基因最大波动误差 |", "|---:|---:|---:|---:|---:|"]
    for a in aggregate:
        report.append(f"| {a['initial_phase_electrical_deg']:+g}° | {a['tavg_error_percent']:.3f}% | {a['ripple_error_percent_if_relative']:.3f}% | {a['maximum_tavg_error_percent']:.3f}% | {a['maximum_ripple_error_percent']:.3f}% |")
    report += ["", "这组测试没有找到能够同时使五个基因平均转矩和波动率对应参考值的统一初始相位。不能用其中一个基因的一项指标接近来证明历史初相位已还原；也不能据有限离散相位排除所有可能初相位。", "",
               "[完整比较数据](phase_comparison.json) · [新求解汇总](summary.json) · [电流与逐角度转矩](torque_samples.csv) · [原始输入完整性](input_integrity.json)"]
    (output / "REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(aggregate,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    render(args.output.resolve())
