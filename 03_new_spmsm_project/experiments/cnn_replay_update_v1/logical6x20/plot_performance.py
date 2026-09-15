"""Render existing validation predictions only; no model loading or inference.

Training/selection remain frozen in ../update.py. This small report supplement
shows the unconstrained update candidates when the accepted choice falls back to f0.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT = Path(__file__).resolve().parent
OUT = ROOT = None
TITLE = STATUS = ""
TARGETS = ("tavg", "delta_t")
ROLES = ("old_validation", "dev_common")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_table(path, values):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def overlay():
    """Two colors for old/new validation, one panel per trained candidate."""
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    destination = OUT / "新旧基因性能对比"
    destination.mkdir(exist_ok=True)
    colors = {"old_validation": "#2478b5", "dev_common": "#eb861b"}
    names = {"old_validation": "旧验证基因", "dev_common": "新验证基因"}
    baseline = {role: rows(ROOT / "baseline" / (role + "_predictions.csv")) for role in ROLES}
    inputs = {str((ROOT / "baseline" / (r + "_predictions.csv")).relative_to(ROOT)):
              hashlib.sha256((ROOT / "baseline" / (r + "_predictions.csv")).read_bytes()).hexdigest() for r in ROLES}
    candidates, summaries, records = {}, {}, []
    for group in ("G-S", "F-S"):
        result = read(ROOT / "runs" / group / "result.json")
        step = result["best"]["unconstrained_step"]
        folder = ROOT / "runs" / group / f"step{step:05d}"
        candidates[group] = {"step": step, "data": {}}
        saved = read(folder / "metrics.json")
        summaries[group] = {}
        for role, count in (("old_validation", 6483), ("dev_common", 200)):
            path = folder / (role + "_predictions.csv")
            values = rows(path)
            assert len(values) == len({v["gene_id"] for v in values}) == count
            assert [v["gene_id"] for v in values] == [v["gene_id"] for v in baseline[role]]
            candidates[group]["data"][role] = values
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            summaries[group][role] = {}
            for target in TARGETS:
                truth = np.array([float(v[target+"_true_nm"]) for v in values])
                prediction = np.array([float(v[target+"_pred_nm"]) for v in values])
                original_truth = np.array([float(v[target+"_true_nm"]) for v in baseline[role]])
                assert np.array_equal(truth, original_truth) and np.isfinite(prediction).all()
                errors = prediction-truth
                base_errors = np.array([float(v[target+"_pred_nm"])-float(v[target+"_true_nm"]) for v in baseline[role]])
                mae, original_mae = np.abs(errors).mean(), np.abs(base_errors).mean()
                metric = {"mae": float(mae), "rmse": float(np.sqrt(np.mean(errors**2))),
                          "absolute_error_p95": float(np.percentile(np.abs(errors), 95)),
                          "f0_mae": float(original_mae), "mae_change_percent": float(100*(mae/original_mae-1))}
                assert all(abs(metric[k]-saved[role]["metrics"][target][k]) < 1e-12
                           for k in ("mae", "rmse", "absolute_error_p95"))
                summaries[group][role][target] = metric
                records.append({"model": group, "checkpoint_step": step, "role": role, "n": count,
                                "target": target, "unit": "N*m", **metric})
    write_table(destination / "性能指标.csv", records)
    for index, (target, label) in enumerate((("tavg", "平均转矩 Tavg"), ("delta_t", "转矩波动 DeltaT")), 1):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
        fig.subplots_adjust(left=.065, right=.98, bottom=.28, top=.84, wspace=.17)
        all_numbers = [float(v[target+suffix]) for c in candidates.values() for values in c["data"].values()
                       for v in values for suffix in ("_true_nm", "_pred_nm")]
        low, high = min(all_numbers), max(all_numbers)
        pad = (high-low)*.045
        for ax, (group, candidate) in zip(axes, candidates.items()):
            for role in ROLES:
                values = candidate["data"][role]
                x = [float(v[target+"_true_nm"]) for v in values]
                y = [float(v[target+"_pred_nm"]) for v in values]
                old = role == "old_validation"
                ax.scatter(x, y, s=6 if old else 24, alpha=.25 if old else .9,
                           c=colors[role], edgecolors="none" if old else "white", linewidths=0 if old else .35,
                           zorder=2 if old else 3, label=f"{names[role]}（n={len(values):,}）", rasterized=True)
            ax.plot([low-pad, high+pad], [low-pad, high+pad], "--", color="#ce3f3b", lw=1.35, label="理想预测 y=x", zorder=1)
            ax.set(xlim=(low-pad,high+pad), ylim=(low-pad,high+pad),
                   xlabel=f"真实{label}（N·m）", ylabel=f"预测{label}（N·m）")
            ax.set_title(f"{group} 更新模型 · 新验证最优检查点（{candidate['step']}步）", fontsize=12, pad=11)
            ax.grid(alpha=.18)
            legend = ax.legend(loc="upper left", fontsize=9, framealpha=.95)
            for handle in legend.legend_handles:
                if hasattr(handle, "set_alpha"):
                    handle.set_alpha(1)
            table_values = []
            for role in ROLES:
                m = summaries[group][role][target]
                table_values.append([names[role], f"{m['mae']:.5f}", f"{m['f0_mae']:.5f}",
                                     f"{m['rmse']:.5f}", f"{m['absolute_error_p95']:.5f}", f"{m['mae_change_percent']:+.1f}%"])
            table = ax.table(cellText=table_values,
                             colLabels=["样本", "当前MAE", "f0 MAE", "RMSE", "误差P95", "MAE变化"],
                             colWidths=[.20,.16,.16,.16,.16,.16], cellLoc="center", bbox=[0,-.335,1,.215])
            table.auto_set_font_size(False); table.set_fontsize(9)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#dddddd"); cell.set_linewidth(.6)
                if row == 0:
                    cell.set_facecolor("#f0f2f4"); cell.set_text_props(weight="bold")
                else:
                    cell.set_facecolor("#edf5fb" if row == 1 else "#fff4e7")
                    cell.get_text().set_color(colors[ROLES[row-1]])
        fig.suptitle(f"{TITLE} · {label}：旧、新基因预测性能", fontsize=18, y=.97)
        fig.text(.5,.912,"蓝色：旧验证基因 6,483 个     橙色：共同新验证基因 200 个     红虚线：真实值 = 预测值",
                 ha="center", fontsize=11)
        fig.text(.5,.031,"MAE变化与同一验证集上的训练前 f0 比较：正值为误差增加，负值为误差减少。误差单位均为 N·m。\n"
                 "展示最佳无约束候选，不等同于接受模型；最终测试集未使用。",
                 ha="center", va="bottom", fontsize=10, color="#505050", linespacing=1.7)
        filename = f"{index:02d}_{'平均转矩' if target=='tavg' else '转矩波动'}_真实值与预测值"
        fig.savefig(destination / (filename+".png"), dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(top=.78, bottom=.20, wspace=.22)
    for ax, (target, label) in zip(axes, (("tavg", "平均转矩 Tavg"), ("delta_t", "转矩波动 DeltaT"))):
        for shift, role in zip((-.18,.18), ROLES):
            changes = [summaries[g][role][target]["mae_change_percent"] for g in candidates]
            bars = ax.bar(np.arange(2)+shift, changes, .33, label=names[role], color=colors[role])
            for bar, value in zip(bars, changes):
                ax.text(bar.get_x()+bar.get_width()/2, value+(3 if value >= 0 else -3), f"{value:+.1f}%",
                        ha="center", va="bottom" if value>=0 else "top", fontsize=11, color=colors[role])
        ax.axhline(0, color="#555555", lw=1)
        ax.axhline(5, color="#777777", lw=.9, ls=":")
        ax.set(xticks=np.arange(2),xticklabels=list(candidates),title=label,
               ylabel="MAE相对自身 f0 的变化（%）")
        all_changes = [summaries[g][r][target]["mae_change_percent"] for g in candidates for r in ROLES]
        ax.set_ylim(min(-10, min(all_changes)-18), max(15, max(all_changes)+18))
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.5,.14), ncol=2, frameon=False, fontsize=10)
    fig.suptitle(TITLE+" · 新基因改善与旧基因退化",fontsize=17,y=.965)
    fig.text(.5,.86,"上方正值：误差增加，性能下降    |    下方负值：误差减少，性能改善",ha="center",fontsize=11)
    fig.text(.5,.045,"最佳无约束候选；旧验证 n=6,483，新验证 n=200。灰点线为旧分布 +5% 容限；接受状态见说明。",
             ha="center",fontsize=10,color="#505050")
    fig.savefig(destination / "03_新旧基因_MAE变化.png", dpi=200)
    plt.close(fig)
    manifest = {"experiment_model":TITLE, "accepted_status":STATUS, "models": {g:{"checkpoint_step":c["step"], "checkpoint":"best_unconstrained.pt"} for g,c in candidates.items()},
                "roles": {"old_validation":6483,"dev_common":200},"input_prediction_sha256":inputs,
                "metrics_verified_against_saved_evaluation":True,"new_inference":False,"test_used":False,
                "plot_source":"../../../plot_performance.py", "plot_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.suffix in (".png",".csv")}}
    (destination/"图表核验.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    print("Old/new overlay plots written: " + str(destination))


def new_gene_overlay():
    """Compare frozen and updated models on the same 200 new validation genes."""
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    destination = OUT / "新旧基因性能对比"
    destination.mkdir(exist_ok=True)
    folders = {"f0": ROOT / "baseline"}
    steps = {}
    for group in ("G-S", "F-S"):
        result = read(ROOT / "runs" / group / "result.json")
        steps[group] = result["best"]["unconstrained_step"]
        folders[group] = ROOT / "runs" / group / f"step{steps[group]:05d}"
    data, metrics, inputs = {}, {}, {}
    for name, folder in folders.items():
        path = folder / "dev_common_predictions.csv"
        values = rows(path)
        assert len(values) == len({v["gene_id"] for v in values}) == 200
        data[name] = values
        assert [v["gene_id"] for v in values] == [v["gene_id"] for v in data["f0"]]
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = read(folder / "metrics.json")["dev_common"]["metrics"]
        metrics[name] = {}
        for target in TARGETS:
            truth = np.array([float(v[target+"_true_nm"]) for v in values])
            pred = np.array([float(v[target+"_pred_nm"]) for v in values])
            assert np.isfinite(truth).all() and np.isfinite(pred).all()
            assert np.array_equal(truth, [float(v[target+"_true_nm"]) for v in data["f0"]])
            error = pred-truth
            metrics[name][target] = {"mae": float(np.abs(error).mean()),
                                     "rmse": float(np.sqrt(np.mean(error**2)))}
            for key, value in metrics[name][target].items():
                assert abs(value-saved[target][key]) < 1e-12
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5))
    fig.subplots_adjust(left=.08, right=.98, bottom=.10, top=.86, hspace=.34, wspace=.22)
    colors = ("#2478b5", "#eb861b")
    for row, (target, title) in enumerate((("tavg", "平均转矩 Tavg"), ("delta_t", "转矩波动 DeltaT"))):
        numbers = [float(v[target+suffix]) for values in data.values() for v in values
                   for suffix in ("_true_nm", "_pred_nm")]
        low, high = min(numbers), max(numbers)
        pad = max((high-low)*.055, .001)
        for col, group in enumerate(("G-S", "F-S")):
            ax = axes[row, col]
            for name, color, label in (("f0", colors[0], "老模型 f0"), (group, colors[1], "更新模型")):
                values = data[name]
                ax.scatter([float(v[target+"_true_nm"]) for v in values],
                           [float(v[target+"_pred_nm"]) for v in values], s=26, alpha=.75,
                           c=color, edgecolors="white", linewidths=.3, label=label, rasterized=True)
            ax.plot([low-pad, high+pad], [low-pad, high+pad], "--", color="#c74440", lw=1.2, label="理想预测 y=x", zorder=1)
            ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad),
                   xlabel=f"FEMM 真实值（N·m）", ylabel="CNN 预测值（N·m）")
            ax.set_title(f"{group} · {title}", fontsize=13, pad=12)
            ax.grid(alpha=.18)
            for j, name in enumerate(("f0", group)):
                m = metrics[name][target]
                ax.text(.035, .95-j*.07, f"{'老模型 f0' if name=='f0' else '更新 '+group}：MAE {m['mae']:.5f}  |  RMSE {m['rmse']:.5f}",
                        transform=ax.transAxes, va="top", fontsize=9.5, color=colors[j],
                        bbox={"facecolor":"white", "alpha":.88, "edgecolor":"none", "pad":2})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.925), ncol=3, frameon=False, fontsize=12)
    fig.suptitle(TITLE+" · 仅新基因：老模型与更新模型", fontsize=18, y=.979)
    fig.text(.5, .942, "同一批共同新验证基因 n=200；每个基因分别由老模型与更新模型预测", ha="center", fontsize=11)
    fig.text(.5, .029, f"G-S：第 {steps['G-S']:,} 步；F-S：第 {steps['F-S']:,} 步。采用各组新验证最优的无约束检查点。\n"
             "同一目标统一坐标范围；点越接近红色虚线，预测越准确。展示无约束候选；最终测试集未使用。",
             ha="center", fontsize=10, color="#505050", linespacing=1.7)
    outputs = [destination / "04_仅新基因_老模型与更新模型对比.png"]
    fig.savefig(outputs[0], dpi=200)
    plt.close(fig)
    manifest = {"experiment_model":TITLE, "accepted_status":STATUS, "role":"dev_common", "n":200, "candidate_steps":steps, "metrics":metrics,
                "input_prediction_sha256":inputs, "metrics_verified_against_saved_evaluation":True,
                "new_inference":False, "test_used":False, "command":"plot_performance.py",
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    (destination / "04_图表核验.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print("New-gene model comparison written: " + str(outputs[0]))




ARCHS = {"small_cnn_v2":"SmallCNN V2", "mini_inception_v2":"Mini-Inception V2", "resnet20_v2":"ResNet20 V2"}

def main():
    global ROOT, OUT, TITLE, STATUS
    version=EXPERIMENT.parent.name.rsplit("_",1)[1]
    links=[]
    for arch,name in ARCHS.items():
        ROOT=EXPERIMENT/arch
        OUT=EXPERIMENT/"report"/arch
        OUT.mkdir(parents=True,exist_ok=True)
        TITLE=f"{version} · {name}"
        status=[]
        for group in ("G-S","F-S"):
            result=read(ROOT/"runs"/group/"result.json")
            assert result["status"]=="complete"
            accepted="回退自身f0" if result["fallback_to_f0"] else f"接受第 {result['selected_step']} 步（best_feasible.pt）"
            status.append(f"{group}：图中为最佳无约束候选第 {result['best']['unconstrained_step']} 步；实际{accepted}。")
        STATUS=" ".join(status)
        overlay()
        new_gene_overlay()
        dest=OUT/"新旧基因性能对比"
        lines=[f"# {TITLE} 性能图", "", "四张图沿用此前VGG16的出图形式，仅复用已保存验证预测，不重新训练或推理，不读取测试集。", "",
               "01、02、03：蓝色是旧验证基因（6483个），橙色是共同新验证基因（200个）。",
               "04：仅使用同一批200个新验证基因，蓝色是该网络自己的冻结老模型f0，橙色是对应G/F更新候选。", "",
               "全部MAE变化均在同一验证集上对比该网络自身f0；正值为误差增加，负值为改善。转矩波动为DeltaT=max(T)-min(T)，误差单位均为N·m。", "",
               "## 检查点身份", "", *status, "",
               "通过接受约束的检查点可能与图中最佳无约束候选不同；本图不把候选当作最终接受模型。", "",
               "## 图表", ""]
        for p in sorted(dest.glob("*.png")):
            lines.extend([f"### {p.stem}", "", f"![{p.stem}]({p.name})", ""])
        lines += ["", "数据核验：ID一一对应、真实值一致、MAE/RMSE/P95与保存的验证指标一致。最终测试集未使用。", "",
                  "复现命令（em仓库根目录）：", "", f"`python ./03_new_spmsm_project/experiments/cnn_replay_update_{version}/logical6x20/plot_performance.py`"]
        (dest/"README.md").write_text("\n".join(lines)+"\n",encoding="utf8")
        assert len(list(dest.glob("*.png")))==4 and not list(dest.glob("*.pdf"))
        links.extend([f"## {name}", "", STATUS, ""])
        for p in sorted(dest.glob("*.png")):
            links.extend([f"### {p.stem}", "", f"![{name} · {p.stem}]({arch}/新旧基因性能对比/{p.name})", ""])
    (EXPERIMENT/"report/性能图索引.md").write_text(f"# {version} 三个6×20模型性能图\n\n每个网络4张PNG；最佳无约束候选与最终接受模型已明确区分。\n\n"+"\n".join(links)+"\n",encoding="utf8")

if __name__ == "__main__":
    main()
