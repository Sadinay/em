"""Read saved logical-model validation results; no training/test inference; PNG only."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCHS = ('small_cnn_v2', 'mini_inception_v2', 'resnet20_v2')
NAMES = {'small_cnn_v2':'SmallCNN V2', 'mini_inception_v2':'Mini-Inception V2', 'resnet20_v2':'ResNet20 V2'}
TARGETS = ('tavg','delta_t')
ROLES = ('old_validation','dev_common')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def table(path, values):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def verify(folder, history, result, base):
    cfg=read(folder.parent.parent/'config.json')
    assert result['updates']==cfg['max_updates'] and result['status']=='complete'
    assert [h['step'] for h in history]==list(range(0,cfg['max_updates']+1,500))
    assert result['successful_update_old_draws']==cfg['max_updates']*cfg['old_per_update']
    assert result['successful_update_new_draws']==cfg['max_updates']*cfg['new_per_update']
    assert result['old_unique_covered']==40000 and result['new_unique_covered']==1400
    eligible=[h for h in history[1:] if all(h['metrics']['old_validation']['metrics'][t]['mae']<=1.05*base['old_validation']['metrics'][t]['mae'] for t in TARGETS)
              and h['metrics']['dev_common']['standardized_mse']<base['dev_common']['standardized_mse']-cfg['improvement_epsilon']]
    selected=min(eligible,key=lambda h:h['metrics']['dev_common']['standardized_mse']) if eligible else None
    assert result['success']==bool(eligible)
    assert result['selected_step']==(selected['step'] if selected else 0)
    assert (folder/'best_feasible.pt').exists()==bool(eligible)
    for name,digest in result['checkpoint_sha256'].items():assert sha(folder/name)==digest


def run(root):
    root=Path(root);version=root.parent.name.rsplit('_',1)[1];out=root/'report';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    runs={};bases={};histories={};records=[];budgets=[];registry={};entries={};verified=[]
    for arch in ARCHS:
        folder=root/arch
        if not (folder/'baseline/metrics.json').exists():continue
        base=read(folder/'baseline/metrics.json');bases[arch]=base
        entries[arch,'f0']={'folder':folder/'baseline','metrics':base,'step':0,'kind':'f0','group':'both'}
        for group in ('G-S','F-S'):
            run_folder=folder/'runs'/group
            if not (run_folder/'result.json').exists():continue
            result=read(run_folder/'result.json');history=read(run_folder/'history.json')
            verify(run_folder,history,result,base)
            runs[arch,group]=result;histories[arch,group]=history
            choices={'unconstrained':result['best']['unconstrained_step'],'final':result['updates']}
            if result['best']['feasible_step'] is not None:choices['feasible']=result['best']['feasible_step']
            for kind,step in choices.items():
                location=run_folder/f'step{step:05d}'
                entries[arch,group+'_'+kind]={'folder':location,'metrics':read(location/'metrics.json'),'step':step,'kind':kind,'group':group}
            registry[arch+'/'+group]={'original_f0':read(folder/'config.json')['initial_checkpoint'],
                'original_f0_sha256':read(folder/'config.json')['initial_checkpoint_sha256'],
                'path_base':str(root.relative_to(root.parents[2])),
                'best_unconstrained':{'path':f"{arch}/runs/{group}/best_unconstrained.pt",'step':choices['unconstrained']},
                'best_feasible':None if 'feasible' not in choices else {'path':f"{arch}/runs/{group}/best_feasible.pt",'step':choices['feasible']},
                'final':{'path':f"{arch}/runs/{group}/last_checkpoint.pt",'step':result['updates'],'full_resume_state':True},
                'accepted':{'path':result['selected_checkpoint'],'step':result['selected_step'],
                            'path_base':'03_new_spmsm_project' if result['fallback_to_f0'] else str(folder.relative_to(root.parents[2]))},
                'fallback_to_f0':result['fallback_to_f0'],'checkpoint_sha256':result['checkpoint_sha256']}
            budgets.append({'architecture':arch,'group':group,'updates':result['updates'],'minutes':result['elapsed_seconds']/60,
                            **result['sampled'],'successful_old_draws':result['successful_update_old_draws'],'successful_new_draws':result['successful_update_new_draws'],'old_unique':result['old_unique_covered'],'new_unique':result['new_unique_covered'],
                            'final_lr':result['final_learning_rate'],'stop_reason':result['stop_reason']})
            verified.append({'architecture':arch,'group':group,'selection_recomputed':True,'checkpoint_hashes_match':True})
    if not runs:
        print('No completed logical run to report yet');return
    checked=set()
    for (arch,name),entry in entries.items():
        for role in ROLES:
            if str(entry['folder']/role) not in checked:
                data=rows(entry['folder']/(role+'_predictions.csv'))
                reference=rows(root/arch/'baseline'/(role+'_predictions.csv'))
                assert [r['gene_id'] for r in data]==[r['gene_id'] for r in reference]
                for t in TARGETS:
                    y=np.array([float(r[t+'_true_nm']) for r in data]);p=np.array([float(r[t+'_pred_nm']) for r in data])
                    assert np.array_equal(y,[float(r[t+'_true_nm']) for r in reference]) and np.isfinite(p).all()
                    m=entry['metrics'][role]['metrics'][t];err=p-y
                    assert abs(np.abs(err).mean()-m['mae'])<1e-12
                    assert abs(np.sqrt(np.mean(err**2))-m['rmse'])<1e-12
                    assert abs(np.percentile(np.abs(err),95)-m['absolute_error_p95'])<1e-12
                checked.add(str(entry['folder']/role))
            metric=entry['metrics'][role]
            for source,block in {'all':metric,**metric.get('sources',{})}.items():
                reference=bases[arch][role] if source=='all' else bases[arch][role]['sources'][source]
                for target in TARGETS:
                    m=block['metrics'][target]
                    records.append({'version':version,'architecture':arch,'model':name,'group':entry['group'],'kind':entry['kind'],'step':entry['step'],
                        'role':role,'source':source,'n':block['n'],'target':target,'mae_nm':m['mae'],'rmse_nm':m['rmse'],
                        'p95_nm':m['absolute_error_p95'],'mae_change_percent_vs_own_f0':100*(m['mae']/reference['metrics'][target]['mae']-1)})
    table(out/'validation_metrics.csv',records);table(out/'training_budgets.csv',budgets);save(out/'model_registry.json',registry)
    fig,axes=plt.subplots(3,4,figsize=(18,11),constrained_layout=True)
    for row,arch in enumerate(ARCHS):
        if arch not in bases:continue
        base=bases[arch]
        for group,color in (('G-S','#2478b5'),('F-S','#e58124')):
            if (arch,group) not in histories:continue
            h=histories[arch,group];x=[p['step'] for p in h]
            axes[row,0].plot(x,[p['metrics']['dev_common']['standardized_mse'] for p in h],color=color,label=group)
            for col,t in enumerate(TARGETS,1):
                axes[row,col].plot(x,[100*(p['metrics']['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae']-1) for p in h],color=color,label=group)
            axes[row,3].step(x,[p['learning_rate'] for p in h],where='post',color=color,label=group)
        axes[row,0].set_yscale('log')
        for col,title in enumerate(('新验证标准化MSE','旧Tavg MAE变化(%)','旧DeltaT MAE变化(%)','学习率')):
            ax=axes[row,col];ax.set(title=NAMES[arch]+'\n'+title,xlabel='成功更新步数');ax.grid(alpha=.2);ax.legend(fontsize=8)
            if col in (1,2):ax.axhline(5,color='#c74440',ls=':',lw=1.2)
    fig.suptitle(f'{version} · 6×20小模型：旧误差相对各自冻结f0；红点线为+5%上限',fontsize=16)
    fig.savefig(out/'01_训练验证曲线.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,14),constrained_layout=True)
    for row,arch in enumerate(ARCHS):
        if arch not in bases:continue
        for col,target in enumerate(TARGETS):
            ax=axes[row,col];numbers=[]
            for name,color in (('f0','#989fa6'),('G-S_unconstrained','#2478b5'),('F-S_unconstrained','#e58124')):
                if (arch,name) not in entries:continue
                e=entries[arch,name];values=rows(e['folder']/'dev_common_predictions.csv')
                x=[float(r[target+'_true_nm']) for r in values];y=[float(r[target+'_pred_nm']) for r in values];numbers+=x+y
                ax.scatter(x,y,s=13,alpha=.65,c=color,label=f"{name.split('_')[0]} MAE={e['metrics']['dev_common']['metrics'][target]['mae']:.5f}")
            lo,hi=min(numbers),max(numbers);pad=(hi-lo)*.04
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'r--',lw=1)
            ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),title=NAMES[arch]+' · '+target,xlabel='FEMM真实值 (N·m)',ylabel='预测值 (N·m)')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(f'{version}：同一批200个新验证基因；展示最佳无约束候选，不等同于接受模型',fontsize=14)
    fig.savefig(out/'02_新基因预测性能.png',dpi=170);plt.close(fig)
    lines=[f'# {version}：三个6×20输入模型的补样回放更新','',f'已完成 {len(runs)}/6 个训练组。每个网络均使用自己的冻结原始 f0，G/F独立初始化；阈值也是相对该网络自身f0，不能与VGG16的阈值混用。','',
           '输入2×6×20（空气/永磁体one-hot），保留原BatchNorm与物理batch64。v1有效48旧+16新，固定10000步；v2有效56旧+8新，固定20000步。每版每组成功更新对应新样本抽取160000次，旧样本分别480000/1120000次；AMP重试会额外重复抽取当前批次，实际次数见预算CSV。此扩展预算不同于已完成的VGG16实验，不能把模型间差异单独归因于架构。','',
           'SmallCNN和Mini-Inception初始更新学习率1e−4；ResNet20为3e−5，均为各自原初始值的0.1倍。其他优化器/AMP/损失与选择规则沿用。每500步验证；无受约束早停。','',
           '| 网络 | 组 | 新验证最佳步数 | 最佳新MSE | 旧Tavg变化 | 旧DeltaT变化 | 接受步数 | 是否回退f0 |','|---|---|---:|---:|---:|---:|---:|---|']
    for (arch,group),r in runs.items():
        m=entries[arch,group+'_unconstrained']['metrics'];b=bases[arch]
        changes=[100*(m['old_validation']['metrics'][t]['mae']/b['old_validation']['metrics'][t]['mae']-1) for t in TARGETS]
        lines.append(f"| {NAMES[arch]} | {group} | {r['best']['unconstrained_step']} | {r['best']['unconstrained_score']:.5f} | {changes[0]:+.1f}% | {changes[1]:+.1f}% | {r['selected_step']} | {'是' if r['fallback_to_f0'] else '否'} |")
    lines+=['','## 实际接受模型的性能','',
            '下表按实际接受路径评价：存在合格更新则使用最佳合格模型，否则明确回退自身f0。MAE单位均为N·m；这与上表的无约束候选不同。','',
            '| 网络 | 组 | 接受模型/步数 | 旧Tavg MAE（变化） | 旧DeltaT MAE（变化） | 新Tavg MAE | 新DeltaT MAE | 新标准化MSE |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for (arch,group),r in runs.items():
        m=bases[arch] if r['fallback_to_f0'] else entries[arch,group+'_feasible']['metrics']
        old=m['old_validation']['metrics'];new=m['dev_common']['metrics'];b=bases[arch]['old_validation']['metrics']
        choice='f0（回退）' if r['fallback_to_f0'] else str(r['selected_step'])
        lines.append(f"| {NAMES[arch]} | {group} | {choice} | {old['tavg']['mae']:.5f} ({100*(old['tavg']['mae']/b['tavg']['mae']-1):+.1f}%) | {old['delta_t']['mae']:.5f} ({100*(old['delta_t']['mae']/b['delta_t']['mae']-1):+.1f}%) | {new['tavg']['mae']:.5f} | {new['delta_t']['mae']:.5f} | {m['dev_common']['standardized_mse']:.5f} |")
    lines+=['','接受条件：旧两项MAE各自≤自身f0×1.05，且新标准化MSE优于自身f0；在合格节点中择新MSE最低者。没有合格更新则明确回退f0。`best_unconstrained.pt`、`best_feasible.pt`（如存在）和`last_checkpoint.pt`分别是无约束最佳、合格最佳、最终模型。完整路径见[模型登记](model_registry.json)。','',
            '完整新旧MAE、RMSE、P95以及U/L/B/P来源误差在 [validation_metrics.csv](validation_metrics.csv)；每个来源50个新验证样本。实际抽样、时间、覆盖率和学习率在 [training_budgets.csv](training_budgets.csv)。','',
            '![验证曲线](01_训练验证曲线.png)','','![新基因预测](02_新基因预测性能.png)','',
            '## P来源：最佳无约束候选相对自身f0的MAE变化','','| 网络 | 组 | Tavg | DeltaT |','|---|---|---:|---:|']
    for (arch,group),r in runs.items():
        m=entries[arch,group+'_unconstrained']['metrics']['dev_common']['sources']['P']['metrics'];b=bases[arch]['dev_common']['sources']['P']['metrics']
        lines.append(f"| {NAMES[arch]} | {group} | {100*(m['tavg']['mae']/b['tavg']['mae']-1):+.1f}% | {100*(m['delta_t']['mae']/b['delta_t']['mae']-1):+.1f}% |")
    if version=='v2':
        comparison=[]
        for v in ('v1','v2'):
            location=root.parent.parent/f'cnn_replay_update_{v}'/'logical6x20'
            for arch in ARCHS:
                for group in ('G-S','F-S'):
                    path=location/arch/'runs'/group/'result.json'
                    if not path.exists():continue
                    r=read(path);b=read(location/arch/'baseline/metrics.json')
                    if arch in bases:assert b==bases[arch]
                    for kind,step in {'unconstrained':r['best']['unconstrained_step'],'accepted':r['selected_step'],'final':r['updates']}.items():
                        m=b if step==0 else read(path.parent/f'step{step:05d}'/'metrics.json')
                        comparison.append({'version':v,'architecture':arch,'group':group,'kind':kind,'step':step,
                            'new_standardized_mse':m['dev_common']['standardized_mse'],
                            **{role+'_'+t+'_mae':m[role]['metrics'][t]['mae'] for role in ROLES for t in TARGETS},
                            'accepted_updated_model':r['success']})
        table(out/'v1_v2_comparison.csv',comparison)
        lines+=['','两版模型与候选/接受/最终的直接比较见 [v1_v2_comparison.csv](v1_v2_comparison.csv)。相同新样本曝光量不等于相同计算预算：v2更新步数和旧样本曝光更多。']
        if len(comparison)==36:
            lines+=['','## 两版比较结论','',
                    '| 网络/组 | v1 最佳新MSE | v2 最佳新MSE | v1接受更新 | v2接受更新 |',
                    '|---|---:|---:|---|---|']
            for arch in ARCHS:
                for group in ('G-S','F-S'):
                    a,b=[next(r for r in comparison if r['version']==v and r['architecture']==arch and r['group']==group and r['kind']=='unconstrained') for v in ('v1','v2')]
                    lines.append(f"| {NAMES[arch]} / {group} | {a['new_standardized_mse']:.5f} | {b['new_standardized_mse']:.5f} | {'是' if a['accepted_updated_model'] else '否'} | {'是' if b['accepted_updated_model'] else '否'} |")
            counts={v:sum(r['accepted_updated_model'] for r in comparison if r['version']==v and r['kind']=='accepted') for v in ('v1','v2')}
            lines+=['',f"本轮v1接受{counts['v1']}/6组更新，v2接受{counts['v2']}/6组更新。更高旧回放配合更长训练，使部分网络更接近保旧要求，但没有使所有组同时达标，也没有一致降低最佳新验证误差。",'',
                    '接受门槛是相对各网络自己的原模型，而非统一相对VGG16。Mini-Inception高转矩端仍可见预测饱和；即使通过相对旧误差约束，也应结合其新验证物理误差判断用途。F-S通常有较低的新验证MSE，但本轮没有F-S更新通过全部约束。', '',
                    '## 合格更新在P来源上的表现','','| 网络/组 | 接受步数 | P的Tavg MAE变化 | P的DeltaT MAE变化 |','|---|---:|---:|---:|']
            for (arch,group),r in runs.items():
                if not r['success']:continue
                m=entries[arch,group+'_feasible']['metrics']['dev_common']['sources']['P']['metrics']
                b=bases[arch]['dev_common']['sources']['P']['metrics']
                lines.append(f"| {NAMES[arch]} / {group} | {r['selected_step']} | {100*(m['tavg']['mae']/b['tavg']['mae']-1):+.1f}% | {100*(m['delta_t']['mae']/b['delta_t']['mae']-1):+.1f}% |")
            lines+=['','5%只是预设研究容限；非常接近但超过门槛的节点仍判为不合格。模型选择使用未舍入的数值，不因显示为5.1%或5.4%而放宽。后续若比较稳定性，应另行安排多个随机种子；本轮不自动追加实验。']
    lines+=['','这是单随机种子开发实验，新验证参与了模型选择，不能作为盲测结论，不能证明某方案稳定更好或优于随机补样。新旧测试均未解封。原VGG16配置、结果和检查点保留。', '',
            '训练源码：`../update.py`；绘图与报告源码：`../report_results.py`。新数据沿用已验收划分，不复制或重算FEMM标签。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(out/'verification.json',{'checked_runs':verified,'prediction_tables_checked':len(checked),'mae_rmse_p95_recomputed':True,'test_used':False})
    save(out/'manifest.json',{'source_sha256':sha(__file__),'outputs_sha256':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.png','.csv','.md')},'pdf_generated':False})
    save(root/'STATUS.json',{'completed_groups':len(runs),'planned_groups':6,'test_used':False,'report':'report/REPORT.md'})
    print(f'Logical report updated: {root} ({len(runs)}/6)',flush=True)


def verify_saved_models(root):
    """Reload each distinct final/candidate model on GPU after training has stopped."""
    import gc
    import importlib.util
    import torch
    root=Path(root)
    spec=importlib.util.spec_from_file_location('logical_verify',root/'update.py')
    u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
    assert not (u.PROJECT/'experiments/logical6x20_training.lock').exists(), 'Training must finish before GPU verification'
    for name,digest in read(root/'original_vgg_artifacts.json').items():
        assert sha(root.parent/name)==digest, f'Original VGG artifact changed: {name}'
    data,meta,_,_=u.load_data()
    checks=[]
    for arch in ARCHS:
        u.ARCH=arch;u.HERE=root/arch
        cfg=read(u.HERE/'config.json')
        u.MODEL=(u.PROJECT/cfg['initial_checkpoint']).parent
        u.CHECKPOINT_SHA=cfg['initial_checkpoint_sha256']
        assert sha(root/'update.py')==sha(u.HERE/'source_snapshot/update.py')
        assert read(u.HERE/'audit/preflight.json')['checkpoint_next_update_bit_exact']
        model,renderer,spec,mean,std,device,opt,scheduler,scaler=u.initialize(cfg)
        baseline_bn={k:int(v) for k,v in model.state_dict().items() if k.endswith('num_batches_tracked')}
        seen={}
        for group in ('G-S','F-S'):
            folder=u.HERE/'runs'/group;r=read(folder/'result.json')
            initial=read(folder/'initialization.json')
            assert initial['weights_equal_f0'] and initial['new_optimizer_state_empty'] and initial['all_parameters_trainable']
            for name in ('last_checkpoint.pt','best_unconstrained.pt','best_feasible.pt'):
                path=folder/name
                if not path.exists():
                    assert name=='best_feasible.pt' and not r['success'];continue
                state=torch.load(path,map_location='cpu',weights_only=False)
                assert state['group']==group and state['config']==cfg
                for key,initial_count in baseline_bn.items():
                    assert int(state['model_state'][key])==initial_count+state['step'], 'BatchNorm counter includes failed/extra updates'
                if name=='last_checkpoint.pt':
                    assert state['step']==cfg['max_updates']
                    assert {int(v['step'].item()) for v in state['optimizer_state']['state'].values()}=={cfg['max_updates']}
                for key in ('target_mean','target_std'):
                    assert np.array_equal(state[key].numpy(),np.asarray(cfg[key],dtype=np.float32))
                h=hashlib.sha256()
                for key,value in state['model_state'].items():
                    h.update(key.encode());h.update(value.numpy().tobytes())
                signature=h.hexdigest();identity=(group,state['step'])
                if identity in seen:
                    assert seen[identity]==signature
                else:
                    model.load_state_dict(state['model_state'],strict=True)
                    actual=u.evaluation(model,renderer,spec,mean,std,device,data,meta)
                    expected=read(folder/f"step{state['step']:05d}"/'metrics.json')
                    assert actual==expected, f'Reloaded metrics differ: {arch} {group} {name}'
                    seen[identity]=signature
                checks.append({'architecture':arch,'group':group,'checkpoint':name,'step':state['step'],
                               'full_validation_metrics_exact_match':True,'model_state_sha256':signature})
                del state
            print(f'RELOAD verified: {root.parent.name} {arch} {group}',flush=True)
        del model,renderer,opt,scheduler,scaler
        gc.collect();torch.cuda.empty_cache()
    save(root/'final_checkpoint_verification.json',{'passed':True,'checks':checks,'original_vgg_artifacts_unchanged':True,
        'original_logical_f0_hashes_verified':True,'full_optimizer_update_counts_verified':True,'batchnorm_counters_match_successful_updates':True,'test_used':False})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true',help='Reload checkpoints on GPU; only after training stops')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.verify:verify_saved_models(root)
    run(root)
