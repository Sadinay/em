"""Render reproducible Chinese comparison figures from the six-mode FEMM run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np

from compare_current_modes_5genes import DEFAULT_OUTPUT, ANGLES

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "font.size": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#f6f8fc",
                     "axes.facecolor": "white", "savefig.facecolor": "#f6f8fc"})

ORDER = ["is35_fixed", "is35_plus", "is35_minus", "dq5_fixed", "dq5_plus", "dq5_minus"]
LABELS = {"reference": "MAT 参考", "is35_fixed": "3.5 A · 三相固定", "is35_plus": "3.5 A · 正向更新",
          "is35_minus": "3.5 A · 反向更新", "dq5_fixed": "Id=5, Iq=0 · 三相固定",
          "dq5_plus": "Id=5, Iq=0 · 正向更新", "dq5_minus": "Id=5, Iq=0 · 反向更新"}
PALETTE = {"reference": "#25324b", "is35_fixed": "#97a8be", "is35_plus": "#1674d1",
           "is35_minus": "#4ec0bf", "dq5_fixed": "#c3a6c2", "dq5_plus": "#d9682f", "dq5_minus": "#905ba5"}


def shade(color: str, alpha: float = .12):
    return tuple(1 - alpha + alpha * v for v in colors.to_rgb(color))


def ripple_text(row: dict) -> str:
    value = row["ripple_percent"]
    if value is None:
        return "未定义"
    if row["near_zero_average_display_flag"]:
        return f"{value:,.0f}%*"
    return f"{value:.2f}%"


def footnotes(fig, y=.04):
    fig.text(.045, y, "比较口径：T = −2 × FEMM 气隙积分原值（沿用此前经验换算，历史物理尺度尚未确认；本次未拟合）。", fontsize=10, color="#4e5c72")
    fig.text(.045, y-.022, "波动率 = (最大转矩 − 最小转矩) / |六点平均转矩| × 100%；MAT DeltaT 暂按该口径对照。* 均值接近零，比值对数值扰动敏感。", fontsize=10, color="#4e5c72")


def render(output: Path) -> None:
    data = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert data["status"] == "complete" and data["fresh_solves"] == 180
    spec = data["spec"]
    candidates = spec["candidates"]
    rows = {(r["scenario_id"], r["candidate_id"]): r for r in data["results"]}
    labels = [f"{c['gene_label']}\n代 {c['state_index_0based']} / 行 {c['population_row_0based']}" for c in candidates]
    x = np.arange(len(candidates))
    fig = plt.figure(figsize=(17, 13))
    fig.text(.045, .964, "03 电机｜六种电流方式与 MAT 参考值", fontsize=24, weight="bold", color="#1e2c43")
    fig.text(.045, .932, "同一批 5 个基因 · 内角 0°、3°、6°、9°、12°、15° · 外角固定 0° · Min Angle = 15° · 180 次新求解", fontsize=12, color="#59677b")
    ax = fig.add_axes([.075, .53, .89, .33])
    width = .115
    series = ["reference", *ORDER]
    for j, sid in enumerate(series):
        values = [c["historical_tavg_nm"] if sid == "reference" else rows[sid, c["candidate_id"]]["tavg_nm"] for c in candidates]
        bars = ax.bar(x + (j - 3) * width, values, width*.9, color=PALETTE[sid], label=LABELS[sid], zorder=3)
        if sid in ("reference", "is35_plus", "dq5_plus"):
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value + .1, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=PALETTE[sid])
    ax.axhline(0, color="#8e99aa", linewidth=.8)
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylabel("六点平均转矩 Tavg（N·m，按上述比较换算）", labelpad=12)
    ax.grid(axis="y", alpha=.18, zorder=0)
    ax.set_title("平均转矩对比", loc="left", pad=12, fontsize=15)
    ax.margins(y=.15)
    ax.legend(loc="lower left", bbox_to_anchor=(-.012, 1.10), ncol=4, frameon=False, fontsize=10, columnspacing=1.6)

    table_ax = fig.add_axes([.045, .105, .92, .355])
    table_ax.axis("off")
    table_ax.set_title("完整数值｜每格上行：平均转矩 N·m；下行：相对转矩波动率", loc="left", fontsize=14, pad=12)
    table_rows = []
    for sid in series:
        values = []
        for c in candidates:
            if sid == "reference":
                values.append(f"{c['historical_tavg_nm']:.4f}\n{100*c['historical_delta_t']:.2f}%")
            else:
                r = rows[sid, c["candidate_id"]]
                values.append(f"{r['tavg_nm']:.4f}\n{ripple_text(r)}")
        table_rows.append([LABELS[sid], *values])
    table = table_ax.table(cellText=table_rows,
                          colLabels=["电流方式", *[c["gene_label"] for c in candidates]],
                          cellLoc="center", colWidths=[.26, *([.148]*5)], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e6ef")
        cell.set_linewidth(.7)
        if r == 0:
            cell.set_facecolor("#e5ebf4")
            cell.set_text_props(weight="bold", color="#25324b")
        else:
            sid = series[r-1]
            cell.set_facecolor(shade(PALETTE[sid], .11 if sid != "reference" else .07))
            if col == 0 or sid == "reference":
                cell.set_text_props(weight="bold", color=PALETTE[sid])
    footnotes(fig, .062)
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(17, 14))
    fig.subplots_adjust(left=.07, right=.965, top=.87, bottom=.11, hspace=.5, wspace=.23)
    fig.suptitle("转矩随滑动气隙内角的变化｜同一组 5 个基因", x=.045, y=.975, ha="left", fontsize=22, weight="bold")
    handles = [Line2D([0], [0], color=PALETTE[sid], lw=2, linestyle="--" if "fixed" in sid else "-", label=LABELS[sid]) for sid in ORDER]
    handles.append(Line2D([0], [0], color=PALETTE["reference"], lw=1.5, linestyle=":", label="MAT 参考平均转矩"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .942), ncol=4, frameon=False, fontsize=11)
    for ax, c in zip(axes.flat, candidates):
        for sid in ORDER:
            r = rows[sid, c["candidate_id"]]
            ax.plot(ANGLES, r["comparison_torques_nm"], color=PALETTE[sid], lw=1.8,
                    linestyle="--" if "fixed" in sid else "-", marker="o" if "is35" in sid else "s", markersize=4)
        ax.axhline(c["historical_tavg_nm"], color=PALETTE["reference"], ls=":", lw=1.8)
        ax.axhline(0, color="#b3bfce", lw=.7)
        ax.set_title(f"{c['gene_label']}  代 {c['state_index_0based']} / 行 {c['population_row_0based']}\n参考 Tavg={c['historical_tavg_nm']:.4f} N·m；参考 DeltaT={c['historical_delta_t']:.6f}", loc="left", fontsize=12)
        ax.set(xticks=ANGLES, xlabel="Inner Angle（机械角 °）", ylabel="比较转矩（N·m）")
        ax.grid(alpha=.2)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.set_title("各方式的 5 基因平均相对误差", loc="left", fontsize=12, pad=12)
    info = []
    for a in data["aggregate"]:
        sid = a["scenario_id"]
        if "fixed" in sid:
            ripple = "均值≈0"
        else:
            ripple = f"{a['ripple_error_percent_if_relative']:.2f}%"
        info.append([LABELS[sid], f"{a['tavg_error_percent']:.2f}%", ripple])
    error_table = ax.table(cellText=info, colLabels=["电流方式", "Tavg 误差", "波动误差"],
                           cellLoc="center", colWidths=[.57, .215, .215], bbox=[0, .03, 1, .94])
    error_table.auto_set_font_size(False)
    error_table.set_fontsize(10)
    for (r, col), cell in error_table.get_celld().items():
        cell.set_edgecolor("#e0e6ef")
        cell.set_facecolor("#e5ebf4" if r == 0 else shade(PALETTE[ORDER[r-1]], .10))
        if r == 0:
            cell.set_text_props(weight="bold")
    footnotes(fig, .062)
    fig.savefig(output / "torque_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.97, top=.85, bottom=.14, hspace=.43, wspace=.15)
    fig.suptitle("实际写入 FEMM 的三相电流", x=.045, y=.965, ha="left", fontsize=22, weight="bold")
    fig.text(.045, .913, "正向 θe=+4θm；反向 θe=−4θm；固定 θe=0。Id/Iq 使用幅值不变逆 Park 换算，初始相位 0°为本次测试约定。", fontsize=11, color="#59677b")
    from compare_current_modes_5genes import current_at
    for ax, sid in zip(axes.flat, ORDER):
        scenario = next(s for s in spec["scenarios"] if s["id"] == sid)
        current_rows = [current_at(spec, scenario, a)[0] for a in ANGLES]
        for phase, color in [("A", "#d45b42"), ("B", "#157f94"), ("C", "#8665b5")]:
            ax.plot(ANGLES, [r[phase] for r in current_rows], "o-", color=color, label=f"I{phase.lower()}", markersize=4)
        ax.set_title(LABELS[sid], fontsize=12)
        ax.set(xticks=ANGLES, xlabel="Inner Angle（°）", ylabel="相电流（A）")
        ax.axhline(0, color="#a8b3c2", lw=.6)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, ncol=3, fontsize=10, loc="lower left")
    fig.text(.045, .04, "数值来源：workspace_200.mat / inp.Is_amp=3.5、Id=5、Iq=0、P=4；这些试验不证明历史脚本采用了其中某一种换算。", fontsize=10, color="#59677b")
    fig.savefig(output / "phase_currents.png", dpi=180)
    plt.close(fig)

    report = ["# 03：六种电流方式、五个基因的 FEMM 对照", "",
              "已完成 6 种方式 × 5 个基因 × 6 个内角 = 180 次新求解；每次保存并核对 .fem、.ans 与三相电流，原始 MAT 和 FEM 的 SHA256 未变化。", "",
              "![平均转矩和波动对照](comparison.png)", "", "![各角度转矩](torque_curves.png)", "", "![实际三相电流](phase_currents.png)", "",
              "## 设置及解释边界", "",
              "仅通过 sliding_airgap 的 Inner Angle 改变转子相对位置：0°、3°、6°、9°、12°、15°。Outer Angle 保持模板的 0°，没有执行外角修改。Min Angle=15°、Precision=1e-8、Depth=36 mm、Smart Mesh=On、Frequency=0。", "",
              "材料沿用 03 FEM 模板的非线性 Pure Iron 和 N38；基因、几何及磁化方向在六种方式之间相同。", "",
              "电流数值从 MAT 读取。Is_amp 方案采用 I·cos(θe+[0,−120°,120°])；Id/Iq 方案采用幅值不变逆 Park：Id·cos(θe+相位)−Iq·sin(θe+相位)。固定/正向/反向分别取 θe=0、+P·θm、−P·θm，P=4。", "",
              "初始电流角 0°是本次对照约定；MAT 中 Id/Iq 的实际历史使用方式及电机磁轴零点尚未确定。因此此处的 Id=5 方案是明确的换算试验，不能称作已还原真实转子 d 轴电流。没有把 3.5 与 5 强行解释成峰值/有效值关系，也没有调整相位或电流去拟合参考值。", "",
              "公式与滑动边界依据：[MathWorks 幅值不变 dq 变换](https://www.mathworks.com/help/mcb/ref/surfacemountpmsm.html)、[FEMM 转子运动说明](https://www.femm.info/doku/doku.php?id=rotormotion)。这些资料说明本次测试方法，不证明历史程序调用。", "",
              "## 转矩和波动计算口径", "",
              "FEMM 原始转矩是 mo_gapintegral('sliding_airgap',0)。主图沿用此前经验比较尺度 T=−2×原始值，未在本次拟合；该比例的历史/物理依据尚未确定。原始转矩和换算转矩均保存在 torque_samples.csv、summary.json。必须在这一限制下解释平均转矩是否接近参考。", "",
              "平均转矩使用六点算术平均，同时另存梯形角度平均。相对波动=(max(T)−min(T))/abs(mean(T))，图上显示为百分比。历史 MAT 只给 DeltaT，没有公式；主图暂按相对波动解释，同时在 metrics.csv 保存绝对峰峰值及将 DeltaT 解释成绝对量时的误差。", "",
              "主图星号只标记 |平均转矩|<0.01 N·m 的显示情况，没有把真实计算结果截断或改成零。固定电流的相对波动分母接近零，数值敏感；不能作为稳定带载脉动率解释。", "",
              "## 五基因汇总", "",
              "| 电流方式 | 平均转矩平均相对误差 | 波动平均相对误差（DeltaT按相对量） | 峰峰值平均相对误差（DeltaT按绝对量） |", "|---|---:|---:|---:|"]
    for a in data["aggregate"]:
        report.append(f"| {LABELS[a['scenario_id']]} | {a['tavg_error_percent']:.3f}% | {a['ripple_error_percent_if_relative']:.3f}% | {a['peak_to_peak_error_percent_if_absolute']:.3f}% |")
    report += ["", "## 各基因完整数值", "", "代、行均为 0-based；MAT DeltaT 保留原值，不在该列附加单位。", "",
               "| 基因（代/行） | 电流方式 | 参考Tavg | 计算Tavg | 参考DeltaT | 计算相对波动 | 绝对峰峰值 N·m |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in candidates:
        for sid in ORDER:
            r = rows[sid, c["candidate_id"]]
            report.append(f"| {c['gene_label']}（{c['state_index_0based']}/{c['population_row_0based']}） | {LABELS[sid]} | {c['historical_tavg_nm']:.6f} | {r['tavg_nm']:.6f} | {c['historical_delta_t']:.6f} | {r['ripple_relative']:.6f} | {r['peak_to_peak_nm']:.6f} |")
    report += ["", "没有一种已测试方式同时复现五个基因的平均转矩和参考波动。该结论限于本次零初相位、电流换算、材料、六点采样以及明确披露的转矩与 DeltaT 比较口径。", "",
               "[运行设置](run_spec.json) · [完整 JSON](summary.json) · [180 个角度样本](torque_samples.csv) · [指标数值](metrics.csv) · [原始输入完整性](input_integrity.json)"]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(str(output / "comparison.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output.resolve())
