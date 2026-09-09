"""Publish coarse/fine phase search and measured best-candidate comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_phase360_search import OUTPUT, gather

COLORS = ["#1674d1", "#d9682f", "#159b85", "#905ba5", "#b29425"]


def circ_distance(phase: float, center: float) -> float:
    return (phase-center+180) % 360-180


def render(output: Path = OUTPUT) -> None:
    data = gather(output)
    assert data["coarse_complete"]
    plan = json.loads((output / "fine_plan.json").read_text())
    points = {r["phase_deg"]: r for r in data["phase_results"]}
    candidates = data["candidates"]
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f6f8fc", "savefig.facecolor": "#f6f8fc"})
    fig, axes = plt.subplots(2,2,figsize=(17,11))
    fig.subplots_adjust(left=.075,right=.965,top=.865,bottom=.13,wspace=.22,hspace=.5)
    fig.text(.045,.955,"03 初始电流相位｜360°粗筛与局部细筛",fontsize=23,weight="bold",color="#243650")
    fig.text(.045,.914,f"3.5 A 正向同步；气隙内角 0:3:15°；外角 0°；Min Angle=15°；共 {data['phase_count']} 个不同初相位、{data['new_solves']} 次新增求解",fontsize=11,color="#5a687c")
    phases=[*data["coarse_phases_degrees"],360]
    coarse_points=[points[p%360] for p in phases]
    ax=axes[0,0]
    for n,(c,color) in enumerate(zip(candidates,COLORS)):
        ax.plot(phases,[p["rows"][n]["tavg_nm"] for p in coarse_points],"o-",color=color,lw=1.7,markersize=4,label=c["gene_label"])
        ax.axhline(c["historical_tavg_nm"],color=color,lw=.8,ls=":",alpha=.6)
    ax.axhline(0,color="#8695a9",lw=.6)
    ax.set_title("每30°粗筛：平均转矩（同色虚线为各基因参考值）",loc="left",fontsize=12,weight="bold")
    ax.set(xlabel="初始电流相位（电角度 °）",ylabel="平均转矩（N·m，沿用比较换算）",xticks=range(0,361,60))
    ax.legend(ncol=5,frameon=False,fontsize=9,loc="lower left",bbox_to_anchor=(0,1.08))
    ax.grid(alpha=.2)
    series=[("tavg_mape_percent","平均转矩：5基因平均误差","#1674d1"),
            ("ripple_mape_percent","转矩波动：5基因平均误差","#d9682f"),
            ("worst_relative_error_percent","10项指标中的最大误差","#905ba5")]
    ax=axes[0,1]
    for key,label,color in series:
        ax.semilogy(phases,[max(p[key],1e-5) for p in coarse_points],"o-",color=color,lw=1.7,markersize=4,label=label)
    ax.set_title("全周粗筛：同时检查平均转矩和波动",loc="left",fontsize=12,weight="bold")
    ax.set(xlabel="初始电流相位（电角度 °）",ylabel="相对误差（%，对数坐标）",xticks=range(0,361,60))
    ax.legend(frameon=False,fontsize=9,loc="upper left")
    ax.grid(alpha=.2)
    centers=[]
    for center in plan["fine_centers_deg"]:
        if not any(abs(circ_distance(center,x)) < 8 for x in centers): centers.append(center)
    for ax,center in zip(axes[1],centers[:2]):
        local=sorted([(circ_distance(p,center)+center,row) for p,row in points.items() if abs(circ_distance(p,center)) <= 5],key=lambda x:x[0])
        for key,label,color in series:
            ax.plot([p for p,row in local],[row[key] for p,row in local],"o-",color=color,lw=1.7,markersize=4,label=label)
        ax.set_title(f"{center:g}°附近：真实 FEMM 细筛结果",loc="left",fontsize=12,weight="bold")
        ax.set(xlabel="初始电流相位（电角度 °；负角与360°邻域等价）",ylabel="相对误差（%）")
        ax.legend(frameon=False,fontsize=9)
        ax.grid(alpha=.2)
    if len(centers)<2: axes[1,1].axis("off")
    fig.text(.045,.072,"粗筛覆盖0°～330°、每30°；图上360°复用0°。周期插值仅用于选择细筛位置，以上曲线上的试验点均来自真实FEMM结果。",fontsize=10,color="#57677f")
    fig.text(.045,.045,"比较口径：T=−2×FEMM气隙积分原值；波动=(最大T−最小T)/|均值|。历史尺度及MAT DeltaT定义尚未确认，本次没有调整。",fontsize=10,color="#57677f")
    fig.savefig(output/"phase360_overview.png",dpi=180)
    plt.close(fig)

    selected=[(0.0,"原方案")]
    for criterion,label in [("worst_relative_error_percent","最大单项误差最小"),("joint_mape_percent","两指标平均误差最小"),("tavg_mape_percent","平均转矩误差最小")]:
        phase=data["rankings"][criterion][0]["phase_deg"]
        if phase not in [p for p,l in selected]: selected.append((phase,label))
    fig=plt.figure(figsize=(17,12.5))
    fig.text(.045,.957,"最接近的已测初始电流相位｜逐基因对照",fontsize=23,weight="bold",color="#243650")
    fig.text(.045,.92,"每个初相位都同时用于全部5个基因；选择规则和未消除的误差一并保留，避免把单项接近当作复现成功。",fontsize=11,color="#5a687c")
    ax=fig.add_axes([.075,.53,.89,.31])
    x=np.arange(5)
    width=.75/(len(selected)+1)
    vals=[c["historical_tavg_nm"] for c in candidates]
    ax.bar(x-(len(selected)/2)*width,vals,width*.88,color="#26354c",label="MAT参考",zorder=3)
    for j,(phase,label) in enumerate(selected,start=1):
        vals=[r["tavg_nm"] for r in points[phase]["rows"]]
        ax.bar(x+(j-len(selected)/2)*width,vals,width*.88,color=COLORS[j-1],label=f"{phase:g}° · {label}",zorder=3)
    ax.set_xticks(x,[f"{c['gene_label']}\n代{c['state_index_0based']}/行{c['population_row_0based']}" for c in candidates])
    ax.set_ylabel("平均转矩（N·m，沿用比较换算）")
    ax.grid(axis="y",alpha=.2,zorder=0)
    ax.set_title("参考平均转矩与已测候选",loc="left",fontsize=14,weight="bold",pad=12)
    ax.legend(frameon=False,fontsize=10,ncol=2,loc="lower left",bbox_to_anchor=(0,1.12))
    tax=fig.add_axes([.045,.18,.92,.265]);tax.axis("off")
    tax.set_title("每格上行：平均转矩 N·m；下行：相对转矩波动率",loc="left",fontsize=14,weight="bold",pad=12)
    table_rows=[["MAT参考",*[f"{c['historical_tavg_nm']:.4f}\n{100*c['historical_delta_t']:.2f}%" for c in candidates]]]
    table_rows += [[f"{phase:g}°\n{label}",*[f"{r['tavg_nm']:.4f}\n{r['ripple_percent']:.2f}%" for r in points[phase]["rows"]]] for phase,label in selected]
    table=tax.table(cellText=table_rows,colLabels=["初始电角度",*[c["gene_label"] for c in candidates]],cellLoc="center",colWidths=[.25,*([.15]*5)],bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#dde5ef")
        cell.set_facecolor("#e2e9f3" if r==0 else "#eef2f7" if r==1 else "#e9f1fa" if r%2==0 else "white")
        if r<=1 or c==0: cell.set_text_props(weight="bold")
    best=data["rankings"]["worst_relative_error_percent"][0]
    fig.text(.045,.123,f"按最大单项误差选择：{best['phase_deg']:g}°；平均转矩平均误差 {best['tavg_mape_percent']:.2f}%；波动平均误差 {best['ripple_mape_percent']:.2f}%；最大单项误差 {best['worst_relative_error_percent']:.2f}%。",fontsize=11,color="#25354d")
    fig.text(.045,.083,"T=−2×FEMM气隙积分原值（沿用此前经验换算，未重新拟合）；图中MAT DeltaT暂按相对峰峰值显示为百分比。",fontsize=10,color="#57677f")
    fig.text(.045,.057,"初始相位搜索不能确认上述历史换算口径；有限离散相位的最小误差，也不等同于已证明连续全局最优。",fontsize=10,color="#57677f")
    fig.savefig(output/"phase360_best_comparison.png",dpi=180)
    plt.close(fig)

    report=["# 03：全360°初始电流相位粗筛与细筛", "",
            f"共归集 {data['phase_count']} 个不同初始电角度、每角度5基因×6个机械位置；本轮新增 {data['new_solves']} 次FEMM求解，另外复用 {data['reused_solves']} 次此前验证结果。", "",
            "粗筛为0°、30°、60°、90°、120°、150°、180°、210°、240°、270°、300°、330°。360°与0°等价，不重复求解。", "",
            "电流固定3.5 A幅值，θe=4θinner+φ0，Ia=3.5cosθe，Ib=3.5cos(θe−120°)，Ic=3.5cos(θe+120°)。内角0:3:15°、外角0°、Min Angle=15°。所有5个基因在每组试验使用同一φ0。几何、材料、磁化方向、转矩换算均保持原方案。", "",
            "## 筛选方法", "",
            "平均转矩和波动各有5个相对误差，共10项。联合平均误差为10项绝对相对误差的等权平均；最大单项误差为10项中的最大值。分别报告两种最优候选及平均转矩最优候选，不因某个指标接近而掩盖另一个指标不一致。", "",
            "30°粗筛结束后，对逐角度转矩作周期三次插值，仅用于选择细筛区间；既有±1°、±3°结果用于检查插值误差。候选位置随后均用真实FEMM求解。预测值没有作为计算结果写入最终对照表。具体方案与插值验证见 fine_plan.json。", "",
            f"初始1°细筛计划：{plan['fine_phases_including_existing']}。实际全部已测相位：{data['all_available_phases_degrees']}。", "",
            "## 比较口径与边界", "",
            "FEMM转矩为mo_gapintegral('sliding_airgap',0)。比较T沿用先前−2×原始值的经验换算，没有在本次拟合；六点算术平均，波动=(max(T)−min(T))/abs(mean(T))。MAT DeltaT暂按相对波动解释；历史转矩尺度和DeltaT公式均尚未确认。", "",
            "每个新增求解均保存.fem/.ans/result.json，并从FEM核对内外角、三相电流、Min Angle、深度、精度、Smart Mesh和频率。汇总核对解文件SHA256及原始MAT/FEM未被改动。", "",
            "![粗筛与细筛](phase360_overview.png)", "", "![候选对照](phase360_best_comparison.png)", "",
            "## 最接近的已测候选", "", "| 选择方式 | 初始电角度 | Tavg平均误差 | 波动平均误差 | 最大单项误差 |", "|---|---:|---:|---:|---:|"]
    for key,label in [("tavg_mape_percent","平均转矩误差最小"),("joint_mape_percent","联合平均误差最小"),("worst_relative_error_percent","最大单项误差最小")]:
        r=data["rankings"][key][0]
        report.append(f"| {label} | {r['phase_deg']:g}° | {r['tavg_mape_percent']:.4f}% | {r['ripple_mape_percent']:.4f}% | {r['worst_relative_error_percent']:.4f}% |")
    report += ["",f"10项误差均≤5%的已测相位：{data['matches_all_ten_within_5_percent']}；均≤10%的已测相位：{data['matches_all_ten_within_10_percent']}。这两个阈值仅用于展示是否同时接近，不是用户预先规定的验收标准。", "",
               "## 全部相位数值", "", "| 初始电角度 | Tavg平均相对误差 | 波动平均相对误差 | 联合平均误差 | 最大单项误差 |", "|---:|---:|---:|---:|---:|"]
    for r in data["phase_results"]:
        report.append(f"| {r['phase_deg']:g}° | {r['tavg_mape_percent']:.4f}% | {r['ripple_mape_percent']:.4f}% | {r['joint_mape_percent']:.4f}% | {r['worst_relative_error_percent']:.4f}% |")
    report += ["", "结论限于已完成的全周粗筛和局部细筛、上述固定设置及比较口径。有限离散相位的最优结果不能证明连续全局最优，也不单独证明历史代码采用了该角度。", "",
               "[归集数据及所有源文件哈希](search_summary.json) · [细筛计划](fine_plan.json)"]
    (output/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(str(output/"phase360_best_comparison.png"))


if __name__=="__main__":
    render()
