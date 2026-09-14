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

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PROJECT = ROOT.parents[1]
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


def run():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    cfg = read(ROOT / "config.json")
    runs = {group: read(ROOT / "runs" / group / "result.json") for group in cfg["groups"]}
    assert all(r["status"] == "complete" for r in runs.values())
    baseline = read(ROOT / "baseline/metrics.json")
    locations = {"f0": ROOT / "baseline"}
    registry = {"f0": {"checkpoint": cfg["initial_checkpoint"], "sha256": cfg["initial_checkpoint_sha256"], "path_base": "03_new_spmsm_project"}}
    labels = {"f0": "冻结 f0"}
    for group, result in runs.items():
        name = "f_G_candidate" if group == "G-S" else "f_F_candidate"
        step = result["best"]["unconstrained_step"]
        locations[name] = ROOT / "runs" / group / f"step{step:05d}"
        labels[name] = f"{group}无约束候选 · {step}步"
        registry[name] = {"group": group, "checkpoint": f"runs/{group}/best_unconstrained.pt", "path_base": "cnn_replay_update_v1",
                          "sha256": result["checkpoint_sha256"]["best_unconstrained.pt"], "step": step,
                          "accepted_by_prespecified_rule": result["success"] and result["selected_step"] == step,
                          "accepted_choice": result["selected_checkpoint"], "fallback_to_f0": result["fallback_to_f0"],
                          "accepted_choice_path_base": "03_new_spmsm_project" if result["fallback_to_f0"] else "cnn_replay_update_v1",
                          "last_checkpoint": f"runs/{group}/last_checkpoint.pt"}
    metrics = {name: read(path / "metrics.json") for name, path in locations.items()}
    records = []
    for name, value in metrics.items():
        for role in ROLES:
            sources = {"all": value[role], **value[role].get("sources", {})}
            for source, sub in sources.items():
                reference = baseline[role] if source == "all" else baseline[role]["sources"][source]
                for target in TARGETS:
                    metric = sub["metrics"][target]
                    records.append({"model": name, "role": role, "source": source, "n": sub["n"], "target": target,
                        "mae_nm": metric["mae"], "rmse_nm": metric["rmse"], "p95_nm": metric["absolute_error_p95"],
                        "mae_change_pct": 100*(metric["mae"]/reference["metrics"][target]["mae"]-1),
                        "mae_improvement_pct": 100*(1-metric["mae"]/reference["metrics"][target]["mae"])})
    write_table(OUT / "candidate_validation_metrics.csv", records)
    (OUT / "model_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf8")
    for role in ROLES:
        prediction = {name: rows(folder / (role+"_predictions.csv")) for name, folder in locations.items()}
        assert all([r['gene_id'] for r in value] == [r['gene_id'] for r in prediction['f0']] for value in prediction.values())
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for j, target in enumerate(TARGETS):
            all_values = [float(r[key]) for value in prediction.values() for r in value for key in (target+"_true_nm", target+"_pred_nm")]
            low, high = min(all_values), max(all_values)
            pad = (high-low)*.04
            for column, (name, values) in enumerate(prediction.items()):
                x = np.array([float(r[target+"_true_nm"]) for r in values])
                y = np.array([float(r[target+"_pred_nm"]) for r in values])
                ax = axes[j,column]
                if role == "dev_common":
                    for source in "ULBP":
                        mask = np.array([r["source"] == source for r in values])
                        ax.scatter(x[mask],y[mask],s=15,alpha=.7,label=source)
                    ax.legend(fontsize=7)
                else:
                    ax.scatter(x,y,s=3,alpha=.25)
                ax.plot([low-pad,high+pad],[low-pad,high+pad],"k--",lw=1)
                ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),xlabel="真实值 (N·m)",ylabel="预测值 (N·m)")
                m=metrics[name][role]["metrics"][target]
                ax.set_title(f"{labels[name]} · {target}\nMAE {m['mae']:.5f} / RMSE {m['rmse']:.5f}",fontsize=10)
                ax.grid(alpha=.2)
        fig.suptitle(("旧验证6483" if role=="old_validation" else "新验证200")+"：冻结基线与无约束更新候选；同目标统一坐标轴")
        for extension in ("png","pdf"):
            fig.savefig(OUT/f"candidate_{role}_truth_prediction.{extension}",dpi=160)
        plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    x=np.arange(4);width=.24
    for j,target in enumerate(TARGETS):
        for offset,(name,value) in enumerate(metrics.items()):
            axes[j].bar(x+(offset-1)*width,[value['dev_common']['sources'][s]['metrics'][target]['mae'] for s in 'ULBP'],width,label=labels[name])
        axes[j].set(xticks=x,xticklabels=list('ULBP'),ylabel='MAE (N·m)',title=target+'：各来源50个新验证样本')
        axes[j].legend(fontsize=8);axes[j].grid(axis='y',alpha=.2)
    for extension in ('png','pdf'):
        fig.savefig(OUT/f"candidate_source_mae.{extension}",dpi=160)
    plt.close(fig)
    # A dimensionless score curve supplements the physical MAE curves in the main report.
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for group in runs:
        history=read(ROOT/'runs'/group/'history.json')
        trace=rows(ROOT/'runs'/group/'training_trace.csv')
        axes[0].plot([int(trace[min(i+99,len(trace)-1)]['step']) for i in range(0,len(trace),100)],
                     [np.mean([float(r['mse']) for r in trace[i:i+100]]) for i in range(0,len(trace),100)],label=group)
        for i,role in enumerate(ROLES,1):
            axes[i].plot([h['step'] for h in history],[h['metrics'][role]['standardized_mse'] for h in history],'o-',ms=3,label=group)
    for ax,title in zip(axes,['训练总损失（100步均值，含Dropout）','旧验证双目标标准化MSE','新验证双目标标准化MSE']):
        ax.set(title=title,xlabel='优化器更新次数',ylabel='标准化MSE');ax.legend();ax.grid(alpha=.2)
    axes[2].set_yscale('log')
    for extension in ('png','pdf'):
        fig.savefig(OUT/f"standardized_loss_curves.{extension}",dpi=160)
    plt.close(fig)
    evidence={'test_inference':False,'input_roles':list(ROLES),'models':list(locations),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'diagnostics_manifest.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    print('Candidate diagnostics written; no model inference or test access')


if __name__ == '__main__':
    run()
