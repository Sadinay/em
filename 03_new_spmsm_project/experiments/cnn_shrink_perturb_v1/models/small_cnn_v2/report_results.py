"""SP first-screen report. Saved allowed validation only; optional GPU checkpoint reload verification."""
from pathlib import Path
import sys as _layout_sys
_LAYOUT_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'cnn_zone').is_dir())
_layout_sys.path.insert(0, str(_LAYOUT_PROJECT.parent / 'maintenance'))
from experiment_paths import audit_dir, report_dir, artifact_sha, relocated

import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PROJECT = _LAYOUT_PROJECT
CONTROL=_LAYOUT_PROJECT/'experiments/cnn_replay_update_v2/models/small_cnn_v2'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

helpers=module('logical_report_helpers',CONTROL.parent/'report_results.py')
read,save,sha,rows,table=helpers.read,helpers.save,helpers.sha,helpers.rows,helpers.table
TARGETS=('tavg','delta_t')
ROLES=('old_validation','dev_common')
VARIANTS={'no_sp':CONTROL,'alpha_095':ROOT/'alpha_095','alpha_080':ROOT/'alpha_080','alpha_050':ROOT/'alpha_050'}
NAMES={'no_sp':'无SP','alpha_095':'α=0.95','alpha_080':'α=0.80','alpha_050':'α=0.50'}
COLORS={'no_sp':'#2478b5','alpha_095':'#ed8a25','alpha_080':'#30956b','alpha_050':'#a055b0'}


def metric_records(variant,group,kind,step,metrics,baseline):
    out=[]
    for role in ROLES:
        for source,block in {'all':metrics[role],**metrics[role].get('sources',{})}.items():
            bm=baseline[role] if source=='all' else baseline[role]['sources'][source]
            for target in TARGETS:
                m=block['metrics'][target]
                out.append({'variant':variant,'group':group,'kind':kind,'step':step,'role':role,'source':source,'n':block['n'],'target':target,
                    'mae_nm':m['mae'],'rmse_nm':m['rmse'],'p95_nm':m['absolute_error_p95'],
                    'mae_change_percent_vs_original_f0':100*(m['mae']/bm['metrics'][target]['mae']-1),
                    'standardized_mse':metrics[role]['standardized_mse'] if source=='all' else ''})
    return out


def check_predictions(folder,baseline_folder,metrics):
    for role in ROLES:
        values=rows(folder/(role+'_predictions.csv'));base=rows(baseline_folder/(role+'_predictions.csv'))
        assert len(values)==(6483 if role=='old_validation' else 200)
        assert len({r['gene_id'] for r in values})==len(values)
        assert [r['gene_id'] for r in values]==[r['gene_id'] for r in base]
        assert [r['source'] for r in values]==[r['source'] for r in base]
        for source,block in {'all':metrics[role],**metrics[role].get('sources',{})}.items():
            mask=np.ones(len(values),dtype=bool) if source=='all' else np.array([r['source']==source for r in values])
            assert int(mask.sum())==block['n']
            for t in TARGETS:
                y=np.array([float(r[t+'_true_nm']) for r in values]);p=np.array([float(r[t+'_pred_nm']) for r in values])
                assert np.isfinite(y).all() and np.isfinite(p).all() and np.array_equal(y,[float(r[t+'_true_nm']) for r in base])
                err=(p-y)[mask];m=block['metrics'][t]
                assert abs(np.abs(err).mean()-m['mae'])<1e-12
                assert abs(np.sqrt(np.mean(err**2))-m['rmse'])<1e-12
                assert abs(np.percentile(np.abs(err),95)-m['absolute_error_p95'])<1e-12


def verify_models():
    import torch
    runner=module('sp_runner_verify',ROOT/'update.py')
    data,meta,_,_=runner.load_data()
    f0=torch.load(runner.MODEL/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    checks=[]
    for variant,folder in VARIANTS.items():
        if variant=='no_sp':continue # Already reloaded and checked in v2; hash ledger checked below.
        runner.HERE=folder;runner.ALPHA=read(folder/'config.json')['sp']['alpha']
        cfg=read(folder/'config.json')
        model,renderer,spec,mean,std,device,opt,scheduler,scaler=runner.initialize(cfg)
        for group in ('G-S','F-S'):
            run=folder/'runs'/group
            result=read(run/'result.json')
            for name in result['checkpoint_sha256']:
                state=torch.load(run/name,map_location='cpu',weights_only=False)
                assert sha(run/name)==result['checkpoint_sha256'][name]
                step=state['step'];model.load_state_dict(state['model_state'],strict=True)
                if name=='last_checkpoint.pt':
                    assert step==20000 and all(int(s['step'])==step for s in state['optimizer_state']['state'].values())
                assert all(int(v)==int(f0['model_state'][k])+step for k,v in state['model_state'].items() if k.endswith('num_batches_tracked'))
                measured=runner.evaluation(model,renderer,spec,mean,std,device,data,meta)
                assert measured==read(run/f'step{step:05d}/metrics.json')
                checks.append({'variant':variant,'group':group,'checkpoint':name,'step':step,'full_validation_exact':True,'bn_successful_update_count_exact':True})
        del model,opt,scheduler,scaler
        torch.cuda.empty_cache()
    save(ROOT/'checkpoint_verification.json',{'passed':True,'test_used':False,'checks':checks})
    print('GPU full checkpoint reload verification passed',flush=True)


def run():
    out=report_dir(ROOT);out.mkdir(exist_ok=True)
    base=read(CONTROL/'baseline/metrics.json')
    frozen=read(ROOT/'source_reuse_audit.json')
    assert all(sha(PROJECT/p)==h for p,h in frozen['control_frozen_files'].items())
    assert sha(ROOT/'update.py')==frozen['new_source_sha256']
    metrics_rows=metric_records('f0','both','f0',0,base,base)
    entries={};results={};histories={};budget=[];registry={};initialization=[];directions=set();starts={}
    for variant,folder in VARIANTS.items():
        assert read(folder/'baseline/metrics.json')==base
        for group in ('G-S','F-S'):
            runfolder=folder/'runs'/group
            result=read(runfolder/'result.json');history=read(runfolder/'history.json')
            helpers.verify(runfolder,history,result,base)
            results[variant,group]=result;histories[variant,group]=history
            kinds={'initial':0,'unconstrained':result['best']['unconstrained_step'],'final':20000}
            if result['success']:kinds['feasible']=result['selected_step']
            for kind,step in kinds.items():
                path=runfolder/f'step{step:05d}'
                m=read(path/'metrics.json');check_predictions(path,CONTROL/'baseline',m)
                entries[variant,group,kind]={'step':step,'metrics':m,'folder':path}
                metrics_rows+=metric_records(variant,group,kind,step,m,base)
            accepted=entries[variant,group,'feasible'] if result['success'] else {'step':0,'metrics':base,'folder':CONTROL/'baseline'}
            entries[variant,group,'accepted']=accepted
            metrics_rows+=metric_records(variant,group,'accepted',accepted['step'],accepted['metrics'],base)
            cfg=read(folder/'config.json')
            budget.append({'variant':variant,'group':group,'successful_updates':result['updates'],'minutes':result['elapsed_seconds']/60,
                'initial_lr':cfg['initial_learning_rate'],'final_lr':result['final_learning_rate'],'physical_batch':64,'old_per_batch':56,'new_per_batch':8,
                **result['sampled'],'successful_old_draws':result['successful_update_old_draws'],'successful_new_draws':result['successful_update_new_draws'],
                'old_unique':result['old_unique_covered'],'new_unique':result['new_unique_covered'],'old_coverage_percent':100*result['old_unique_covered']/40000,
                'new_coverage_percent':100*result['new_unique_covered']/1400,'stop_reason':result['stop_reason'],'new_run':variant!='no_sp'})
            registry[variant+'/'+group]={'folder':str(runfolder.relative_to(PROJECT)),'unconstrained_step':kinds['unconstrained'],
                'feasible_step':result['selected_step'] if result['success'] else None,'final_step':20000,'accepted_step':accepted['step'],
                'fallback_to_f0':result['fallback_to_f0'],'accepted_checkpoint':str((folder/result['selected_checkpoint']).relative_to(PROJECT)) if result['success'] else cfg['initial_checkpoint'],
                'checkpoint_hashes':result['checkpoint_sha256']}
            if variant!='no_sp':
                info=read(runfolder/'initialization.json')['sp']
                assert info['applications']==1 and info['buffers_exactly_f0'] and info['training_rng_unchanged']
                directions.add(info['random_parameter_sha256']);starts[variant,group]=info['start_parameter_sha256']
                initialization.append({'variant':variant,'group':group,**{k:info[k] for k in ('alpha','beta','random_seed','random_parameter_sha256','start_parameter_sha256','delta_l2','relative_delta_l2','parameter_count')}})
    assert len(directions)==1 and all(starts[v,'G-S']==starts[v,'F-S'] for v in VARIANTS if v!='no_sp')
    table(out/'validation_metrics.csv',metrics_rows);table(out/'training_budgets.csv',budget);table(out/'initialization.csv',initialization)
    save(out/'model_registry.json',registry)
    # Record every 500-step evaluation, including SP zero separately from reference f0.
    trajectory=[]
    for (variant,group),history in histories.items():
        for h in history:
            trajectory.append({'variant':variant,'group':group,'step':h['step'],'lr':h['learning_rate'],
                'new_standardized_mse':h['metrics']['dev_common']['standardized_mse'],
                **{t+'_old_mae_change_percent':100*(h['metrics']['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae']-1) for t in TARGETS},
                'old_constraints_pass':h['feasible']})
    table(out/'validation_curves.csv',trajectory)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
    for row,group in enumerate(('G-S','F-S')):
        for variant in VARIANTS:
            h=histories[variant,group];x=[v['step'] for v in h]
            axes[row,0].plot(x,[v['metrics']['dev_common']['standardized_mse'] for v in h],color=COLORS[variant],label=NAMES[variant])
            for col,t in enumerate(TARGETS,1):
                axes[row,col].plot(x,[v['metrics']['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae'] for v in h],color=COLORS[variant],label=NAMES[variant])
            axes[row,3].step(x,[v['learning_rate'] for v in h],where='post',color=COLORS[variant],label=NAMES[variant])
        for col,title in enumerate(('新验证标准化MSE','旧Tavg MAE / 原f0','旧DeltaT MAE / 原f0','学习率')):
            ax=axes[row,col];ax.set(title=group+' · '+title,xlabel='成功更新次数');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
            if col in (1,2):ax.axhline(1.05,color='#c74440',ls='--',lw=1)
    fig.suptitle('SmallCNN · SP系数初筛：β=0.01；第0步已施加SP，门槛始终参照原f0',fontsize=16)
    fig.savefig(out/'01_验证与学习率曲线.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    for ax,group in zip(axes,('G-S','F-S')):
        for variant in VARIANTS:
            h=histories[variant,group][1:]
            xs=[max(v['metrics']['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae'] for t in TARGETS) for v in h]
            ys=[v['metrics']['dev_common']['standardized_mse'] for v in h]
            ax.scatter(xs,ys,s=18,alpha=.55,color=COLORS[variant],label=NAMES[variant])
            e=entries[variant,group,'accepted']
            if e['step']:
                x=max(e['metrics']['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae'] for t in TARGETS)
                ax.scatter([x],[e['metrics']['dev_common']['standardized_mse']],marker='*',s=180,color=COLORS[variant],edgecolor='black',linewidth=.5)
        ax.axvline(1.05,color='#c74440',ls='--');ax.axhline(base['dev_common']['standardized_mse'],color='#777777',ls=':')
        ax.set(title=group,xlabel='旧两目标 MAE 比率的较大值（相对原f0）',ylabel='新验证标准化MSE',xscale='log',yscale='log')
        ax.legend();ax.grid(alpha=.15)
    fig.suptitle('新旧精度取舍：红线左侧满足旧约束；灰线下方改善新预测；星号为实际接受更新模型',fontsize=13)
    fig.savefig(out/'02_新旧精度取舍.png',dpi=180);plt.close(fig)
    # Source changes of unconstrained candidates; positive is degradation. P is not hidden by aggregate improvements.
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row,group in enumerate(('G-S','F-S')):
        for col,t in enumerate(TARGETS):
            ax=axes[row,col]
            for i,variant in enumerate(VARIANTS):
                m=entries[variant,group,'unconstrained']['metrics']['dev_common']['sources']
                ys=[100*(m[s]['metrics'][t]['mae']/base['dev_common']['sources'][s]['metrics'][t]['mae']-1) for s in 'ULBP']
                ax.bar(np.arange(4)+(i-1.5)*.19,ys,.18,label=NAMES[variant],color=COLORS[variant])
            ax.axhline(0,color='#555555',lw=1);ax.set(xticks=np.arange(4),xticklabels=[f"{s} (n={base['dev_common']['sources'][s]['n']})" for s in 'ULBP'],title=group+' · '+t,ylabel='MAE相对原f0变化（%）');ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('最佳无约束候选的分来源表现：正值为退化，负值为改善',fontsize=16)
    fig.savefig(out/'03_分来源误差变化.png',dpi=180);plt.close(fig)
    write_report(out,base,results,entries,budget)
    save(out/'verification.json',{'passed':True,'original_control_files_unchanged':True,'source_unchanged':True,'all_selected_validation_predictions_metrics_checked':True,
        'acceptance_recomputed_against_original_f0':True,'shared_random_direction':True,'gf_initial_weights_identical':True,'test_used':False,
        'source_sha256':sha(__file__),'outputs_sha256':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.png','.csv','.md')}})
    print('SP report generated from saved validation only',flush=True)


def write_report(out,base,results,entries,budget):
    lines=['# SmallCNN：Shrink-and-Perturb＋经验回放首轮筛选','',
        '范围：只使用 Logical6x20SmallCNNV2（2×6×20空气/永磁体one-hot）；β固定0.01，α为0.95/0.80/0.50，各G/F两组。每组20,000次成功更新、物理batch64=56旧+8新，无微批次。原始SmallCNN f0（汇总时间线记为v0，原保存版本V3）始终作为选择门槛；新旧最终测试集未读取性能标签或推理。','',
        '无SP复用v2已有结果。完整配置（除实验标识和SP字段）、采样计划哈希、运行设备/PyTorch/CUDA、原f0验证完全一致；α=1且β=0的全状态及下一步训练与原路径精确一致。没有增加对照训练。','',
        '一次性SP覆盖所有可训练参数（含偏置及BN仿射）；原生随机模型在隔离CPU RNG上下文中生成，6组共享方向。所有BN缓冲状态原样继承f0，恢复训练时不重施SP。新优化器/调度器/AMP；损失仍为128个标准化目标误差的均值。该BN处理为本项目实现约定。','',
        '## 原始基线与SP后第0步','',
        '| 设置 | 旧Tavg MAE | 旧DeltaT MAE | 新Tavg MAE | 新DeltaT MAE | 新标准化MSE |','|---|---:|---:|---:|---:|---:|']
    def vals(m):return [m['old_validation']['metrics'][t]['mae'] for t in TARGETS]+[m['dev_common']['metrics'][t]['mae'] for t in TARGETS]+[m['dev_common']['standardized_mse']]
    lines.append('| 原始f0 | '+' | '.join(f'{v:.6f}' for v in vals(base))+' |')
    for v in VARIANTS:
        if v!='no_sp':lines.append(f'| {NAMES[v]}，β=.01 | '+' | '.join(f'{x:.6f}' for x in vals(entries[v,'G-S','initial']['metrics']))+' |')
    lines+=['','## 实际接受结果','',
        '| 设置 | 组 | 接受步数 | 新标准化MSE | 新Tavg MAE | 新DeltaT MAE | 旧Tavg变化 | 旧DeltaT变化 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        for g in ('G-S','F-S'):
            e=entries[v,g,'accepted'];m=e['metrics']
            change=[100*(m['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae']-1) for t in TARGETS]
            lines.append(f"| {NAMES[v]} | {g} | {e['step'] or '回退f0'} | {m['dev_common']['standardized_mse']:.6f} | {m['dev_common']['metrics']['tavg']['mae']:.5f} | {m['dev_common']['metrics']['delta_t']['mae']:.5f} | {change[0]:+.2f}% | {change[1]:+.2f}% |")
    lines+=['','无SP G-S合格对照是第11,500步，而非第15,500步无约束最优。无SP F-S回退f0。','',
        '## 候选、最终与接受模型的区别','',
        '| 设置 | 组 | 类型 | 步数 | 新标准化MSE | 旧两MAE均≤1.05×f0 |','|---|---|---|---:|---:|---|']
    for v in VARIANTS:
        for g in ('G-S','F-S'):
            for kind,label in (('unconstrained','无约束最佳'),('feasible','合格最佳'),('final','最终')):
                e=entries.get((v,g,kind))
                if e is None:
                    lines.append(f'| {NAMES[v]} | {g} | {label} | 无 | — | — |');continue
                m=e['metrics'];passed=all(m['old_validation']['metrics'][t]['mae']<=1.05*base['old_validation']['metrics'][t]['mae'] for t in TARGETS)
                lines.append(f"| {NAMES[v]} | {g} | {label} | {e['step']} | {m['dev_common']['standardized_mse']:.6f} | {'是' if passed else '否'} |")
    lines+=['','## 完整物理误差','',
        '下面列出最佳无约束候选；合格、最终、接受模型的同类指标及全部U/L/B/P来源见 `validation_metrics.csv`。所有MAE、RMSE、绝对误差P95单位为N·m；DeltaT为六角度max-min。','',
        '| 设置 | 组 | 验证集 | 目标 | MAE | RMSE | P95 |','|---|---|---|---|---:|---:|---:|']
    for v in VARIANTS:
        for g in ('G-S','F-S'):
            m=entries[v,g,'unconstrained']['metrics']
            for role in ROLES:
                for t in TARGETS:
                    q=m[role]['metrics'][t]
                    lines.append(f"| {NAMES[v]} | {g} | {'旧' if role=='old_validation' else '新'} | {t} | {q['mae']:.5f} | {q['rmse']:.5f} | {q['absolute_error_p95']:.5f} |")
    lines+=['','## 预算与异常处理','',
        '| 设置 | 组 | 更新次数 | 分钟 | AMP重试 | 实际旧抽取 | 实际新抽取 | 最终LR |','|---|---|---:|---:|---:|---:|---:|---:|']
    for b in budget:
        lines.append(f"| {NAMES[b['variant']]} | {b['group']} | {b['successful_updates']} | {b['minutes']:.2f} | {b['amp_skipped_attempts']} | {b['old_draws']} | {b['new_draws']} | {b['final_lr']:.2g} |")
    lines+=['',f"本轮六次新增训练合计 {sum(b['minutes'] for b in budget if b['new_run']):.2f} 分钟。均完成20,000次成功更新，无提前停止；旧40,000和新1,400覆盖率均100%。成功更新的名义抽取每组为旧1,120,000＋新160,000，实际计数额外包含AMP重试；重试撤销BN缓冲更新，但额外Dropout随机数消耗仍可能使后续序列在不同组间分化。",'',
        '同一随机初始化方向、采样计划和训练随机种子已尽量对齐。不同系数造成的AMP重试/调度器变化是记录中的训练结果，不人为补齐。CUDA自适应池化反向按既有warn_only策略存在非确定性可能；单次精确恢复检查不构成跨设备逐位复现保证。','',
        '## 判断','']
    control_score=entries['no_sp','G-S','accepted']['metrics']['dev_common']['standardized_mse']
    feasible=[(entries[v,g,'accepted']['metrics']['dev_common']['standardized_mse'],v,g) for v in VARIANTS if v!='no_sp' for g in ('G-S','F-S') if results[v,g]['success']]
    winners=[(s,v,g) for s,v,g in feasible if s<control_score]
    lines.append(f'六组SP中 {len(feasible)} 组取得合格更新；其中 {len(winners)} 组的新标准化MSE低于无SP G-S第11,500步合格对照（{control_score:.6f}）。')
    if winners:
        s,v,g=min(winners)
        lines.append(f'本轮最有希望的是 {NAMES[v]} / {g}：满足原f0的两项旧MAE约束，新标准化MSE={s:.6f}，比上述无SP合格对照低 {100*(1-s/control_score):.1f}%。这表明在本轮SmallCNN开发验证上，SP改善了可接受的新旧精度平衡。')
    elif feasible:
        s,v,g=min(feasible)
        lines.append(f'SP虽产生合格模型，但最佳 {NAMES[v]} / {g} 的新标准化MSE={s:.6f}，未超过已有无SP G-S合格表现。本轮没有证据证明SP优于现有可接受方案。')
    else:
        lines.append('本轮未找到满足条件的SP改善模型，全部回退原f0；普通热启动G-S仍保有合格改善。主要瓶颈是新结构误差降低时旧分布至少一项MAE超过5%门槛，具体见候选表及精度取舍图。')
    lines+=['','## 同组无SP比较与P来源','',
        '| SP设置 | 组 | 无约束候选新MSE相对同组无SP变化 | 接受模型新MSE相对同组无SP变化 | 候选P Tavg变化vs f0 | 候选P DeltaT变化vs f0 |','|---|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        if v=='no_sp':continue
        for g in ('G-S','F-S'):
            candidate=entries[v,g,'unconstrained']['metrics']['dev_common']
            uc=entries['no_sp',g,'unconstrained']['metrics']['dev_common']['standardized_mse']
            ac=entries['no_sp',g,'accepted']['metrics']['dev_common']['standardized_mse']
            accepted=entries[v,g,'accepted']['metrics']['dev_common']['standardized_mse']
            pchanges=[100*(candidate['sources']['P']['metrics'][t]['mae']/base['dev_common']['sources']['P']['metrics'][t]['mae']-1) for t in TARGETS]
            lines.append(f"| {NAMES[v]} | {g} | {100*(candidate['standardized_mse']/uc-1):+.2f}% | {100*(accepted/ac-1):+.2f}% | {pchanges[0]:+.2f}% | {pchanges[1]:+.2f}% |")
    if feasible:
        best_score,best_variant,best_group=min(feasible)
        chosen=entries[best_variant,best_group,'accepted']['metrics']['dev_common']
        pm=[100*(chosen['sources']['P']['metrics'][t]['mae']/base['dev_common']['sources']['P']['metrics'][t]['mae']-1) for t in TARGETS]
        lines+=['',f"最佳合格SP的P来源变化（相对原f0）：Tavg {pm[0]:+.2f}%，DeltaT {pm[1]:+.2f}%。该来源仍需单独检查，不能仅依赖总新MSE。"]
        if winners:
            near=sorted({read(VARIANTS[v]/'config.json')['sp']['alpha'] for score,v,g in winners})
            candidates='、'.join(f'{a:.2f}' for a in near)
            lines.append(f"后续优先保留本轮合格表现超过无SP合格对照的已测α：{candidates}；其中最佳候选为 {NAMES[best_variant]} / {best_group}。这只是本轮有希望的候选范围，不表示相邻未测系数已得到验证。β可按预先提出的0.003/0.01/0.03少量比较，无SP对照继续保留；本轮不追加训练。")
        else:
            lines.append('暂不建议据此固定SP最优系数或迁移架构；先确认合格模型与普通热启动的差距是否值得继续。')
    else:
        lines+=['','目前没有值得直接固定的收缩系数；应先分析旧分布退化，而不是仅凭无约束新误差最低继续缩小α。']
    lines+=['','### 本轮具体取舍与下一步优先级','',
        '按预先固定的合格新标准化MSE规则，优先α=0.80、β=0.01的G-S（13500步），下一轮若继续，应先固定这一α少量比较β，并保留无SP对照。α=0.95与其分数接近，不能用一次种子的小差距宣称稳定胜负。', '',
        'α=0.50的G-S也有价值：新Tavg/DeltaT MAE约0.01505/0.04980，低于α=0.80的0.01894/0.05033；旧Tavg MAE下降约6.33%。但其新DeltaT RMSE/P95约0.06348/0.12898，高于α=0.80的0.06190/0.12349，因此在既定平方误差指标下排名靠后。不能把“标准化MSE较差”误写成所有指标都较差，或直接排除较强收缩范围。', '',
        '三个F-S设置均没有满足约束的更新。旧DeltaT是直接瓶颈：α=.95/.80/.50在所有更新验证节点上的最低MAE增幅分别约5.21%/5.15%/6.94%，都高于5%。其中α=.80的最佳无约束候选新MSE仅0.04040，但旧DeltaT增加约5.15%，必须严格回退，不能因接近门槛就放宽。']
    lines+=['','本轮三个SP合格G模型的P来源两项目标MAE均低于原f0，没有继续退化；α=.80合格模型的P来源改善约19.42%（Tavg）和11.19%（DeltaT）。SP是小幅改善而非普遍优势：三组G有合格收益，三组F仍失败，且α排名随MAE或MSE的关注点存在取舍。','','U/L/B/P源误差与P退化情况见下图及CSV；不能以总体误差下降掩盖单一来源的退化。验证集参与系数选择，本轮是单种子开发筛选，不能宣称文献最优、普遍最优或跨架构适用。后续β比较及ResNet20/Polar90迁移均未执行，须在本轮结果讨论后再决定。','',
        '## 图表','']
    for p in sorted(out.glob('*.png')):lines += [f'### {p.stem}','',f'![{p.stem}]({p.name})','']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:verify_models()
    run()
    import subprocess
    subprocess.run([sys.executable,str(PROJECT.parent/'maintenance/collect_03_reports.py')],check=True)
