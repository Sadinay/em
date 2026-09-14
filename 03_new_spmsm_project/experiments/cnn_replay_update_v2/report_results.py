"""Compact v1/v2 validation comparison. Saved validation outputs only; PNG only."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "cnn_replay_update_v1"
OUT = ROOT / "report"
TARGETS = ("tavg", "delta_t")
ROLES = ("old_validation", "dev_common")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def table(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_outputs():
    """Check saved states and validation tables without touching sealed labels."""
    import torch
    cfg = read(ROOT / 'config.json')
    restoration = read(ROOT / 'audit/v1_report_restoration.json')
    for name, expected in read(ROOT / 'audit/v1_reference.json')['sha256'].items():
        if name == restoration['file']:
            assert expected == restoration['original_sha256']
            assert sha(ROOT/'audit/v1_interpretation_before_restore.md') == expected
            expected = restoration['restored_sha256']
        assert sha(V1 / name) == expected, f'Unexpected v1 change: {name}'
    assert sha(ROOT / 'update.py') == sha(ROOT / 'source_snapshot/update.py')
    assert sha(ROOT.parents[1] / cfg['initial_checkpoint']) == cfg['initial_checkpoint_sha256']
    base_rows = {role:csv_rows(ROOT/'baseline'/(role+'_predictions.csv')) for role in ROLES}
    counts, checkpoint_steps = {}, {}
    for group in ('G-S','F-S'):
        folder = ROOT/'runs'/group
        result = read(folder/'result.json')
        history = read(folder/'history.json')
        assert [h['step'] for h in history] == list(range(0,5001,500))
        initial = read(folder/'initialization.json')
        assert initial['weights_equal_f0'] and initial['new_optimizer_state_empty'] and initial['all_parameters_trainable']
        assert result['successful_update_old_draws']==280000 and result['successful_update_new_draws']==40000
        for version, root in (('v1',V1),('v2',ROOT)):
            r=read(root/'runs'/group/'result.json')
            for name, expected in r['checkpoint_sha256'].items():
                assert sha(root/'runs'/group/name)==expected
        last=torch.load(folder/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        assert last['step']==5000 and last['history']==history
        optimizer_steps={int(s['step'].item()) for s in last['optimizer_state']['state'].values()}
        assert optimizer_steps=={5000}
        for key in ('target_mean','target_std'):
            assert np.array_equal(last[key].numpy(),np.asarray(cfg[key],dtype=np.float32))
        checkpoint_steps[group]={'last':last['step'],'optimizer_state_tensors':len(last['optimizer_state']['state']),
                                 'optimizer_steps':sorted(optimizer_steps)}
        del last
        qualifying=[h for h in history[1:] if all(h['metrics']['old_validation']['metrics'][t]['mae']<=1.05*history[0]['metrics']['old_validation']['metrics'][t]['mae'] for t in TARGETS)
                    and h['metrics']['dev_common']['standardized_mse'] < history[0]['metrics']['dev_common']['standardized_mse']-cfg['improvement_epsilon']]
        chosen=min(qualifying,key=lambda h:h['metrics']['dev_common']['standardized_mse']) if qualifying else None
        assert result['success']==bool(qualifying)
        assert result['selected_step']==(chosen['step'] if chosen else 0)
        assert (folder/'best_feasible.pt').exists()==bool(qualifying)
        for h in history:
            for role in ROLES:
                values=csv_rows(folder/f"step{h['step']:05d}"/(role+'_predictions.csv'))
                assert len(values)==len(base_rows[role])
                assert [r['gene_id'] for r in values]==[r['gene_id'] for r in base_rows[role]]
                for target in TARGETS:
                    truth=np.array([float(r[target+'_true_nm']) for r in values])
                    pred=np.array([float(r[target+'_pred_nm']) for r in values])
                    assert np.array_equal(truth,[float(r[target+'_true_nm']) for r in base_rows[role]])
                    assert np.isfinite(pred).all()
                    err=pred-truth
                    actual={'mae':np.abs(err).mean(),'rmse':np.sqrt(np.mean(err**2)),
                            'absolute_error_p95':np.percentile(np.abs(err),95)}
                    for key,value in actual.items():
                        assert abs(value-h['metrics'][role]['metrics'][target][key])<1e-12
                counts[group+'_'+role]=counts.get(group+'_'+role,0)+len(values)
    save(ROOT/'audit/final_output_verification.json',{'passed':True,'v1_configs_results_checkpoints_unchanged':True,
         'v1_report_restoration_verified':True,
         'f0_unchanged':True,'training_source_matches_snapshot':True,'checkpoint_steps':checkpoint_steps,
         'validation_rows_checked':counts,'all_prediction_ids_and_truth_match_allowed_baseline':True,
         'all_mae_rmse_p95_recomputed':True,'acceptance_and_fallback_recomputed':True,'test_used':False})


def run():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    cfg = read(ROOT / "config.json")
    base = read(ROOT / "baseline/metrics.json")
    assert base == read(V1 / "baseline/metrics.json"), "v1/v2 f0 evaluation differs"
    results = {(v, g): read(root / "runs" / g / "result.json")
               for v, root in (("v1", V1), ("v2", ROOT)) for g in ("G-S", "F-S")}
    assert all(r["status"] == "complete" for r in results.values())
    assert all(results["v2", g]["updates"] == 5000 for g in ("G-S", "F-S"))
    verify_outputs()
    histories = {(v, g): read(root / "runs" / g / "history.json")
                 for v, root in (("v1", V1), ("v2", ROOT)) for g in ("G-S", "F-S")}
    entries = {"f0": {"version":"f0", "group":"both", "kind":"baseline", "step":0,
                      "folder":ROOT / "baseline", "metrics":base}}
    registry, budgets, curves = {}, [], []
    for (version, group), result in results.items():
        root = V1 if version == "v1" else ROOT
        choices = {"unconstrained":result["best"]["unconstrained_step"], "final":result["updates"]}
        if version == "v2":
            choices["step2500"] = 2500
        if result["best"]["feasible_step"] is not None:
            choices["feasible"] = result["best"]["feasible_step"]
        for kind, step in choices.items():
            folder = root / "runs" / group / f"step{step:05d}"
            entries[f"{version}_{group}_{kind}"] = {"version":version, "group":group, "kind":kind,
                                                    "step":step, "folder":folder, "metrics":read(folder / "metrics.json")}
        registry[f"{version}_{group}"] = {
            "path_base":str(root.relative_to(ROOT.parent)),
            "best_unconstrained":{"path":f"runs/{group}/best_unconstrained.pt", "step":choices["unconstrained"],
                                  "sha256":result["checkpoint_sha256"]["best_unconstrained.pt"]},
            "best_feasible":None if "feasible" not in choices else {"path":f"runs/{group}/best_feasible.pt", "step":choices["feasible"],
                            "sha256":result["checkpoint_sha256"]["best_feasible.pt"]},
            "final":{"path":f"runs/{group}/last_checkpoint.pt", "step":result["updates"],
                     "sha256":result["checkpoint_sha256"]["last_checkpoint.pt"], "contains_resume_state":True},
            "accepted":{"path":result["selected_checkpoint"], "step":result["selected_step"],
                        "path_base":"03_new_spmsm_project" if result["fallback_to_f0"] else str(root.relative_to(ROOT.parent)),
                        "fallback_to_f0":result["fallback_to_f0"]}}
        budgets.append({"version":version, "group":group, "successful_updates":result["updates"],
                        "old_draws":result["sampled"]["old_draws"], "new_draws":result["sampled"]["new_draws"],
                        "old_unique":result["old_unique_covered"], "new_unique":result["new_unique_covered"],
                        "old_coverage_percent":100*result["old_unique_covered"]/40000,
                        "new_coverage_percent":100*result["new_unique_covered"]/1400,
                        "amp_skipped_attempts":result["sampled"]["amp_skipped_attempts"],
                        "elapsed_seconds":result["elapsed_seconds"], "final_lr":result["final_learning_rate"],
                        "stop_reason":result["stop_reason"]})
        for item in histories[version, group]:
            m = item["metrics"]
            curves.append({"version":version, "group":group, "step":item["step"],
                           "lr_after_validation":item["learning_rate"], "old_constraints_pass":item["feasible"],
                           "new_standardized_mse":m["dev_common"]["standardized_mse"],
                           **{t+"_old_mae_ratio":m["old_validation"]["metrics"][t]["mae"]/base["old_validation"]["metrics"][t]["mae"] for t in TARGETS}})
    records = []
    for name, entry in entries.items():
        for role in ROLES:
            m = entry["metrics"][role]
            for source, block in {"all":m, **m.get("sources", {})}.items():
                reference = base[role] if source == "all" else base[role]["sources"][source]
                for target in TARGETS:
                    metric = block["metrics"][target]
                    records.append({"model":name, "version":entry["version"], "group":entry["group"],
                        "kind":entry["kind"], "step":entry["step"], "role":role, "source":source, "n":block["n"],
                        "target":target, "mae_nm":metric["mae"], "rmse_nm":metric["rmse"],
                        "p95_nm":metric["absolute_error_p95"],
                        "mae_change_pct_vs_f0":100*(metric["mae"]/reference["metrics"][target]["mae"]-1)})
    table(OUT / "validation_metrics.csv", records)
    table(OUT / "training_budgets.csv", budgets)
    table(OUT / "validation_curves.csv", curves)
    save(OUT / "model_registry.json", registry)

    # Four distinct diagnostics, one compact figure; no redundant accepted-f0 scatter.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"G-S":"#2478b5", "F-S":"#e58124"}
    for (version, group), history in histories.items():
        x = [h["step"] for h in history]
        style = "--" if version == "v1" else "-"
        for j, target in enumerate(TARGETS):
            y = [100*(h["metrics"]["old_validation"]["metrics"][target]["mae"]/base["old_validation"]["metrics"][target]["mae"]-1) for h in history]
            axes[0,j].plot(x,y,style,marker="o",ms=3,color=colors[group],label=f"{version} {group}")
        axes[1,0].plot(x,[h["metrics"]["dev_common"]["standardized_mse"] for h in history],style,marker="o",ms=3,color=colors[group],label=f"{version} {group}")
        axes[1,1].step(x,[h["learning_rate"] for h in history],where="post",linestyle=style,color=colors[group],label=f"{version} {group}")
    for ax, label in zip(axes[0], ("平均转矩 Tavg", "转矩波动 DeltaT")):
        ax.axhline(5,color="#c74440",ls=":",label="旧误差 +5% 上限")
        ax.axhline(0,color="gray",lw=.7)
        ax.set(title=f"旧验证：{label}",ylabel="MAE 相对 f0 变化（%）")
    axes[1,0].set(title="共同新验证：双目标标准化 MSE",ylabel="标准化 MSE",yscale="log")
    axes[1,1].set(title="学习率（验证调度后，后续更新使用）",ylabel="学习率")
    for ax in axes.flat:
        ax.set_xlabel("成功优化器更新次数"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("v1 / v2 开发验证对比：虚线 v1（48旧+16新），实线 v2（56旧+8新）",fontsize=14)
    fig.savefig(OUT / "01_验证误差与学习率.png",dpi=180); plt.close(fig)

    prediction_inputs = {}
    def prediction(entry):
        path = entry["folder"] / "dev_common_predictions.csv"
        prediction_inputs[str(path.relative_to(ROOT.parent))] = sha(path)
        return csv_rows(path)
    base_rows = prediction(entries["f0"])
    fig, axes = plt.subplots(2,2,figsize=(12.5,10),constrained_layout=True)
    scatter_data = {}
    for group in ("G-S","F-S"):
        for version in ("v1","v2"):
            values = prediction(entries[f"{version}_{group}_unconstrained"])
            assert len(values)==200 and [r['gene_id'] for r in values]==[r['gene_id'] for r in base_rows]
            scatter_data[version,group]=values
    for row,target in enumerate(TARGETS):
        all_rows=base_rows+[r for values in scatter_data.values() for r in values]
        numbers=[float(r[target+s]) for r in all_rows for s in ("_true_nm","_pred_nm")]
        low,high=min(numbers),max(numbers);pad=(high-low)*.04
        for col,group in enumerate(("G-S","F-S")):
            ax=axes[row,col]
            for label,values,color in (("f0",base_rows,"#9aa1a8"),("v1",scatter_data["v1",group],"#2478b5"),("v2",scatter_data["v2",group],"#e58124")):
                x=np.array([float(r[target+"_true_nm"]) for r in values]); y=np.array([float(r[target+"_pred_nm"]) for r in values])
                assert np.array_equal(x,[float(r[target+"_true_nm"]) for r in base_rows])
                ax.scatter(x,y,s=15,alpha=.65,c=color,label=f"{label} MAE={np.abs(y-x).mean():.5f}")
            ax.plot([low-pad,high+pad],[low-pad,high+pad],"r--",lw=1)
            ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),xlabel="FEMM 真实值 (N·m)",ylabel="预测值 (N·m)",
                   title=group+" · "+("平均转矩" if target=="tavg" else "转矩波动"))
            ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle("仅共同新验证 200 个基因：f0 与各轮最佳无约束候选（不代表通过旧约束）",fontsize=13)
    fig.savefig(OUT / "02_新基因预测对比.png",dpi=180);plt.close(fig)

    def metric_line(name):
        e=entries[name];m=e['metrics']; old=m['old_validation']['metrics']; new=m['dev_common']['metrics']
        ratios=[100*(old[t]['mae']/base['old_validation']['metrics'][t]['mae']-1) for t in TARGETS]
        passed=all(x<=5+1e-10 for x in ratios)
        return f"| {name} | {e['step']} | {old['tavg']['mae']:.5f} ({ratios[0]:+.1f}%) | {old['delta_t']['mae']:.5f} ({ratios[1]:+.1f}%) | {new['tavg']['mae']:.5f} | {new['delta_t']['mae']:.5f} | {m['dev_common']['standardized_mse']:.5f} | {'通过' if passed else '未通过'} |\n"
    success=[g for g in ('G-S','F-S') if results['v2',g]['success']]
    text="# CNN 回放更新 v2：开发验证结果\n\n"
    text+=(f"**找到满足条件的更新模型：{'、'.join(success)}。**\n\n" if success else "**本轮未找到满足条件的改善模型，两组实际接受模型均回退 f0。**\n\n")
    text+="两组独立从同一原始 f0 初始化，seed=20260914；每个物理批次7旧+1新，累积8批，固定5000次成功更新。保留v1架构、编码、目标尺度、AdamW、初始学习率1e−5、权重衰减1e−4、梯度裁剪100及AMP；损失仍为sum-MSE/(64×2)。\n\n"
    text+="## 全体验证表现\n\nMAE单位N·m；括号为旧MAE相对f0变化。完整MAE、RMSE、P95及来源指标在 [validation_metrics.csv](validation_metrics.csv)。\n\n"
    text+="| 模型/用途 | 步数 | 旧Tavg MAE | 旧DeltaT MAE | 新Tavg MAE | 新DeltaT MAE | 新标准化MSE | 旧双约束 |\n|---|---:|---:|---:|---:|---:|---:|---|\n"
    for name in entries:text+=metric_line(name)
    text+="\n`unconstrained`为新验证最优候选，`feasible`须同时满足旧两项5%容限与新MSE优于f0，`final`为最终模型；`step2500`是固定预算中间节点。接受模型可能回退f0，不代表更新候选没有改善。\n\n"
    text+="## 分来源：重点检查 P\n\n每个来源50个新验证样本；下表为各轮最佳无约束候选，相对f0的MAE变化（负数改善）。\n\n| 来源 | 组 | v1 Tavg / DeltaT | v2 Tavg / DeltaT |\n|---|---|---:|---:|\n"
    for source in 'ULBP':
        for group in ('G-S','F-S'):
            values=[]
            for version in ('v1','v2'):
                m=entries[f'{version}_{group}_unconstrained']['metrics']['dev_common']['sources'][source]['metrics']
                values.append(' / '.join(f"{100*(m[t]['mae']/base['dev_common']['sources'][source]['metrics'][t]['mae']-1):+.1f}%" for t in TARGETS))
            text+=f"| {source} | {group} | {values[0]} | {values[1]} |\n"
    text+="\nP来源的变化并不一致："
    for group in ('G-S','F-S'):
        pm=entries[f'v2_{group}_unconstrained']['metrics']['dev_common']['sources']['P']['metrics']
        pr=base['dev_common']['sources']['P']['metrics']
        changes={t:100*(pm[t]['mae']/pr[t]['mae']-1) for t in TARGETS}
        text+=f"{group} 的Tavg/DeltaT MAE相对f0为 {changes['tavg']:+.1f}%/{changes['delta_t']:+.1f}%；"
    text+="因此不能把总体新预测改善表述为P来源全面改善。\n"
    text+="\n## 实际预算与学习率\n\n| 轮次/组 | 更新 | 旧/新抽取 | 旧/新唯一覆盖 | 分钟 | 结束学习率 | 原因 |\n|---|---:|---:|---:|---:|---:|---|\n"
    for r in budgets:
        text+=f"| {r['version']} {r['group']} | {r['successful_updates']} | {r['old_draws']}/{r['new_draws']} | {r['old_unique']}/40000；{r['new_unique']}/1400 | {r['elapsed_seconds']/60:.2f} | {r['final_lr']:.2g} | {r['stop_reason']} |\n"
    for group in ('G-S','F-S'):
        h=histories['v2',group]
        changes=[f"{b['step']}步后→{b['learning_rate']:.2g}" for a,b in zip(h,h[1:]) if a['learning_rate']!=b['learning_rate']]
        text+=f"\n{group} 学习率变化：{'; '.join(changes) if changes else '整个训练期间保持1e−5'}；AMP跳过尝试 {results['v2',group]['sampled']['amp_skipped_attempts']} 次。\n"
    text+="\n![验证曲线](01_验证误差与学习率.png)\n\n![新基因预测](02_新基因预测对比.png)\n\n"
    text+="## 两种预算比较\n\n| 组 | v1 2500步 新MSE | v2 2500步 新MSE | v2 5000步 新MSE |\n|---|---:|---:|---:|\n"
    for g in ('G-S','F-S'):
        scores=[entries[n]['metrics']['dev_common']['standardized_mse'] for n in (f'v1_{g}_final',f'v2_{g}_step2500',f'v2_{g}_final')]
        text+=f"| {g} | "+' | '.join(f'{x:.5f}' for x in scores)+' |\n'
    text+="\nv2第2500步与v1第2500步更新次数相同，但新样本曝光为20000对40000；v2第5000步与v1第2500步新样本曝光同为40000，但旧样本为280000对120000且计算预算不同。因此本轮同时改变配比与时长，不能将改善全部归因于配比。\n\n"
    text+="## 是否更接近新旧要求，瓶颈是什么？\n\n"
    balance={}
    for g in ('G-S','F-S'):
        balance[g]={}
        for version in ('v1','v2'):
            h=histories[version,g]
            eligible=[p for p in h if p['step']>0 and p['metrics']['dev_common']['standardized_mse']<base['dev_common']['standardized_mse']]
            best=min(eligible,key=lambda p:max(p['metrics']['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae'] for t in TARGETS))
            ratios={t:100*(best['metrics']['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae']-1) for t in TARGETS}
            balance[g][version]={'step':best['step'],'old_mae_changes_pct':ratios,'new_mse':best['metrics']['dev_common']['standardized_mse']}
        a,b=balance[g]['v1'],balance[g]['v2']
        closer=max(b['old_mae_changes_pct'].values())<max(a['old_mae_changes_pct'].values())
        text+=f"- {g}：以‘新MSE改善时，旧两目标中较大的MAE增幅’衡量，v2{'更接近' if closer else '没有更接近'}约束。v1最接近节点为{a['step']}步（最大增幅{max(a['old_mae_changes_pct'].values()):.1f}%），v2为{b['step']}步（Tavg {b['old_mae_changes_pct']['tavg']:+.1f}%，DeltaT {b['old_mae_changes_pct']['delta_t']:+.1f}%，新MSE {b['new_mse']:.5f}）。这仅是事后诊断，未改变模型选择规则。\n"
    if not success:
        text+="\n主要瓶颈是更新后旧分布的误差保持，而非未学会新结构；所有验证节点都未同时达到预设条件。提高旧回放比例并延长训练仍不足以消除新旧精度取舍，具体限制目标见上述两项旧MAE及曲线。当前证据不能区分参数遗忘、优化波动或新旧任务梯度冲突的各自贡献。\n"
    text+="\n最佳新验证候选与最终模型的差别："
    for group in ('G-S','F-S'):
        r=results['v2',group]
        final=entries[f'v2_{group}_final']['metrics']['dev_common']['standardized_mse']
        text+=f"{group} 最终新MSE相对最佳候选变化 {100*(final/r['best']['unconstrained_score']-1):+.1f}%；"
    text+="后续更新可能引起验证退化，不能默认最终权重最好；单次轨迹也不足以将波动唯一归因于过拟合。\n"
    text+="\nP来源变化应单独于总体改善解释；平均新误差下降不保证每个来源或每个基因改善。以上结果只反映一个随机种子，不能证明F必然优于G，也不能证明优于随机补样。新验证集已经参与选择与调参，本轮是开发实验；新旧最终测试均未用于训练、调参或预测。\n\n"
    text+="v1中文README、主报告正常；解释报告 `report/interpretation.md` 存在2135个字面问号，属于实际内容损坏。已保留原文副本，并依据保存的验证指标、结果和历史恢复可读报告，审计见 `../audit/v1_report_restoration.json`。UTF-8可解码本身不足以证明内容完整。所有v1配置、结果、检查点保留。本轮到此结束，不自动追加实验。\n\n"
    text+="下一步是否扩展多种子应取决于是否已有满足约束的设置；若仍无合格模型，宜先讨论一个明确的保旧对照（例如另行授权的分流或其他保持旧预测的方法），而非将本轮未合格方案直接认定为稳定改进。\n\n"
    text+="产物入口：[模型路径与用途](model_registry.json)、[预算](training_budgets.csv)、[验证曲线数值](validation_curves.csv)。训练源码在 `../update.py`，报告源码在 `../report_results.py`。\n"
    (OUT/'REPORT.md').write_text(text,encoding='utf8')
    save(OUT/'comparison_summary.json',{'success_groups':success,'closest_old_constraint_nodes':balance,'test_used':False})
    save(OUT/'report_manifest.json',{'source_sha256':sha(__file__),'input_prediction_sha256':prediction_inputs,
         'test_used':False,'new_inference':False,'outputs_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.png','.csv','.md')}})
    (ROOT/'STATUS.md').write_text('# v2 状态\n\n已完成两组各5000次成功更新及报告。\n\n接受结果：'+('、'.join(success) if success else '两组均回退 f0')+'。\n\n报告：`report/REPORT.md`；模型位置：`report/model_registry.json`；源码：`update.py`、`report_results.py`。最终测试集继续封存。\n',encoding='utf8')
    print('REPORT written: '+str(OUT/'REPORT.md'),flush=True)


if __name__=='__main__':
    run()
