"""Fixed selected SmallCNN test evaluation. Inference only: no backward, optimizer, SP, or model selection."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,sys,time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('frozen_sp_runner',ROOT.parent/'cnn_shrink_perturb_v1/update.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
OUT=ROOT/'report'


def evaluate_tests():
    assert not (ROOT/'metrics.json').exists(),'Completed test evaluation already exists; use --report-only'
    plan=r.read(ROOT/'evaluation_plan.json')
    assert plan['status']=='fixed_before_test_labels_read'
    for name in ('candidate','baseline'):assert r.sha(PROJECT/plan[name]['checkpoint'])==plan[name]['sha256']
    assert r.sha(r.SPLIT)==plan['split_sha256'] and r.sha(r.PILOT/'memberships.csv')==plan['membership_sha256']
    checkpoints={name:torch.load(PROJECT/plan[name]['checkpoint'],map_location='cpu',weights_only=False) for name in ('baseline','candidate')}
    ck=checkpoints['candidate'];cfg=ck['config']
    assert ck['step']==9500 and ck['group']=='F-S' and cfg['sp']['alpha']==.8 and cfg['sp']['beta']==.01 and cfg['new_per_update']==8
    with np.load(r.SPLIT) as z:split={key:np.array(z[key]) for key in ('train','validation','test')}
    assert len(split['test'])==6483 and not set(split['test']) & (set(split['train'])|set(split['validation']))
    identities=r.old_identities()
    old_meta=[{'gene_id':identities[i][1],'source':'old_GA','old_index':int(i)} for i in split['test']]
    groups={role:{identities[i][2] for i in idx} for role,idx in split.items()}
    assert not groups['test'] & (groups['train']|groups['validation'])
    grids=np.load(r.OLD/'topology_bits.npy',mmap_mode='r');targets=np.load(r.OLD/'targets_tavg_delta.npy',mmap_mode='r')
    old_ds=r.array_dataset(np.array(grids[split['test']]).swapaxes(1,2).reshape(-1,120),np.array(targets[split['test']],dtype=np.float32))
    del grids,targets
    directory=r.ACCEPTED/'data/sealed_test';checks=r.read(r.ACCEPTED/'OUTPUT_CHECKSUMS.json');hashes={}
    for name in ('gene_ids.npy','labels.csv','manifest.csv','targets_tavg_delta.npy','topology_bits.npy'):
        p=directory/name;actual=r.sha(p);assert actual==checks['data/sealed_test/'+name];hashes[str(p.relative_to(PROJECT))]=actual
    source=r.SPMSMGeneDataset(directory)
    new_ds=r.array_dataset(np.array(source.bits),np.array(source.targets));new_meta=r.csv_read(directory/'manifest.csv')
    canonical={v['gene_id']:v for v in r.csv_read(directory/'labels.csv')}
    assert len(new_ds)==len(new_meta)==len(canonical)==400
    members=r.csv_read(r.PILOT/'memberships.csv')
    test_ids={v['gene_id'] for v in members if v['split_role']=='test'}
    assert {v['gene_id'] for v in new_meta}==test_ids
    assert not test_ids & {v['gene_id'] for v in members if v['split_role']!='test'}
    assert not test_ids & {v[1] for v in identities}
    for bits,target,item in zip(new_ds.bits,new_ds.targets,new_meta):
        assert r.genotype_sha256(bits)==item['gene_id'] and item['status']=='verified' and item['condition_fingerprint']==r.CONDITION
        assert item['split_role']=='test' and all(item[k]==canonical[item['gene_id']][k] for k in ('tavg_nm','delta_t_nm','bits'))
        assert np.array_equal(target,np.array([item['tavg_nm'],item['delta_t_nm']],dtype=np.float32))
    data={'old_test':old_ds,'new_test':new_ds};meta={'old_test':old_meta,'new_test':new_meta}
    for ds in data.values():assert np.isfinite(ds.targets).all() and np.isin(ds.bits,[0,1]).all()
    r.save(ROOT/'data_audit.json',{'passed':True,'new_test_n':400,'old_test_n':6483,'no_training_validation_overlap':True,'old_repair_families_disjoint':True,
        'accepted_test_hashes':hashes,'old_test_bits_sha256':r.array_sha(old_ds.bits),'old_test_targets_sha256':r.array_sha(old_ds.targets),
        'target_policy':'unchanged accepted six-angle Tavg mean and DeltaT range, both N*m; no samples removed/relabelled','test_unsealed_with_user_authorization':True,
        'new_source_counts':{s:sum(v['source']==s for v in new_meta) for s in 'ULBP'},'plan_sha256':r.sha(ROOT/'evaluation_plan.json')})
    from dataclasses import fields
    spec=r.tr.TrainingSpec(**{f.name:cfg['baseline_config'][f.name] for f in fields(r.tr.TrainingSpec)})
    r.tr.configure_reproducibility(r.SEED);torch.set_num_threads(4)
    device=torch.device('cuda');model=r.tr.build_model(spec).to(device=device,memory_format=torch.channels_last)
    mean,std=ck['target_mean'],ck['target_std']
    assert all(torch.equal(checkpoints['baseline'][key],ck[key]) for key in ('target_mean','target_std'))
    result={};state_checks={};started=time.perf_counter()
    for name,state in checkpoints.items():
        model.load_state_dict(state['model_state'],strict=True);model.eval().requires_grad_(False)
        before=r.tensor_digest(model.state_dict().items());result[name]={}
        with torch.inference_mode():
            for role,ds in data.items():
                loss,_,indices,actual,pred=r.tr.evaluate(model,ds,np.arange(len(ds)),spec,None,mean,std,device)
                assert np.array_equal(indices,np.arange(len(ds))) and np.isfinite(pred).all()
                y=actual.astype(np.float64) if role=='old_test' else np.array([[v['tavg_nm'],v['delta_t_nm']] for v in meta[role]],dtype=np.float64)
                p=pred.astype(np.float64);metrics=r.tr.regression_metrics(y,p)
                block={'n':len(ds),'standardized_mse':loss,'metrics':metrics}
                if role=='new_test':
                    block['sources']={}
                    for s in 'ULBP':
                        mask=np.array([v['source']==s for v in meta[role]])
                        block['sources'][s]={'n':int(mask.sum()),'metrics':r.tr.regression_metrics(y[mask],p[mask])}
                result[name][role]=block
                rows=[]
                for item,yt,yp in zip(meta[role],y,p):
                    row={'gene_id':item['gene_id'],'source':item['source']}
                    for j,t in enumerate(('tavg','delta_t')):row.update({t+'_true_nm':yt[j],t+'_pred_nm':yp[j]})
                    rows.append(row)
                r.csv_write(ROOT/'predictions'/f'{name}_{role}_predictions.csv',rows)
        after=r.tensor_digest(model.state_dict().items());assert after==before and not model.training
        assert all(param.grad is None for param in model.parameters())
        assert r.sha(PROJECT/plan[name]['checkpoint'])==plan[name]['sha256']
        state_checks[name]={'state_sha256_before':before,'state_sha256_after':after,'unchanged_parameters_and_buffers':True,'eval_mode':True,'inference_mode':True,'gradients_none':True}
    r.save(ROOT/'metrics.json',result)
    r.save(ROOT/'inference_verification.json',{'passed':True,'models':state_checks,'backward_calls':0,'optimizer_steps':0,'SP_applications':0,
        'elapsed_seconds':time.perf_counter()-started,'source_sha256':r.sha(Path(__file__)),'plan_sha256':r.sha(ROOT/'evaluation_plan.json'),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    print('Test inference complete; model parameters and BN buffers unchanged.',flush=True)


def report():
    OUT.mkdir(exist_ok=True);m=r.read(ROOT/'metrics.json');plan=r.read(ROOT/'evaluation_plan.json');rows=[]
    for model in ('baseline','candidate'):
        for role in ('old_test','new_test'):
            for source,block in {'all':m[model][role],**m[model][role].get('sources',{})}.items():
                ref=m['baseline'][role] if source=='all' else m['baseline'][role]['sources'][source]
                for target in ('tavg','delta_t'):
                    metric=block['metrics'][target]
                    rows.append({'model':model,'test_set':role,'source':source,'n':block['n'],'target':target,**metric,
                        'mae_change_percent_vs_f0':100*(metric['mae']/ref['metrics'][target]['mae']-1)})
    r.csv_write(OUT/'test_metrics.csv',rows)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    for i,role in enumerate(('old_test','new_test')):
        for j,target in enumerate(('tavg','delta_t')):
            ax=axes[i,j];values=[]
            for name,color,label in [('baseline','#2478b5','原f0'),('candidate','#eb861b','F-S SP α=.80')]:
                data=r.csv_read(ROOT/'predictions'/f'{name}_{role}_predictions.csv')
                x=np.array([float(v[target+'_true_nm']) for v in data]);y=np.array([float(v[target+'_pred_nm']) for v in data]);values.extend([x.min(),x.max(),y.min(),y.max()])
                q=m[name][role]['metrics'][target]
                assert abs(np.abs(y-x).mean()-q['mae'])<1e-12 and abs(np.sqrt(np.mean((y-x)**2))-q['rmse'])<1e-12
                ax.scatter(x,y,s=5 if i==0 else 16,alpha=.25 if i==0 else .65,color=color,label=f"{label}：MAE={q['mae']:.5f}")
            lo,hi=min(values),max(values);pad=(hi-lo)*.04
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'r--',lw=1)
            ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),title=f"{'旧测试集 n=6483' if i==0 else '新测试集 n=400'} · {target}",xlabel='FEMM真实值（N·m）',ylabel='模型预测值（N·m）');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('固定模型测试：SmallCNN F-S · α=.80 β=.01 · 新样本12.5% · 9500步\n仅推理，模型不更新；蓝色原f0，橙色更新模型',fontsize=15)
    fig.savefig(OUT/'01_测试集真实值与预测值.png',dpi=180);plt.close(fig)
    lines=['# 固定SmallCNN模型测试结果','','模型在读取测试标签前已固定：F-S、α=.80、β=.01、新样本12.5%、9500步，选自6%验证容限复核。此处仅与其原始f0作同测试集比较，没有根据测试结果改模型或重新选择系数。','',
        '新测试集400个（U/L/B/P各100个），旧测试集6483个。只推理，eval模式＋inference_mode，无反向传播、无优化器更新；所有参数和BatchNorm缓冲状态前后哈希一致。checkpoint文件未改动。','',
        '## 两个模型的测试误差','','误差单位为N·m。MAE变化相对原f0在同一测试集上的误差。','',
        '| 测试集 | 目标 | 原f0 MAE | 更新模型 MAE | MAE变化 | 更新模型 RMSE | 更新模型 P95 |','|---|---|---:|---:|---:|---:|---:|']
    for role in ('old_test','new_test'):
        for t in ('tavg','delta_t'):
            b=m['baseline'][role]['metrics'][t];c=m['candidate'][role]['metrics'][t]
            lines.append(f"| {'旧' if role=='old_test' else '新'} | {t} | {b['mae']:.6f} | {c['mae']:.6f} | {100*(c['mae']/b['mae']-1):+.2f}% | {c['rmse']:.6f} | {c['absolute_error_p95']:.6f} |")
    lines+=['','| 测试集 | 原f0 标准化MSE | 更新模型标准化MSE |','|---|---:|---:|']
    for role in ('old_test','new_test'):lines.append(f"| {role} | {m['baseline'][role]['standardized_mse']:.6f} | {m['candidate'][role]['standardized_mse']:.6f} |")
    lines+=['','## 新测试集分来源','','| 来源 | 数量 | Tavg MAE | Tavg变化 | DeltaT MAE | DeltaT变化 |','|---|---:|---:|---:|---:|---:|']
    for s in 'ULBP':
        b=m['baseline']['new_test']['sources'][s];c=m['candidate']['new_test']['sources'][s]
        vals=[c['metrics'][t]['mae'] for t in ('tavg','delta_t')];changes=[100*(c['metrics'][t]['mae']/b['metrics'][t]['mae']-1) for t in ('tavg','delta_t')]
        lines.append(f"| {s} | {c['n']} | {vals[0]:.6f} | {changes[0]:+.2f}% | {vals[1]:.6f} | {changes[1]:+.2f}% |")
    old_changes=[100*(m['candidate']['old_test']['metrics'][t]['mae']/m['baseline']['old_test']['metrics'][t]['mae']-1) for t in ('tavg','delta_t')]
    new_changes=[100*(m['candidate']['new_test']['metrics'][t]['mae']/m['baseline']['new_test']['metrics'][t]['mae']-1) for t in ('tavg','delta_t')]
    lines+=['','## 结果解读','',f"在400个新测试样本上，Tavg/DeltaT MAE相对原f0变化 {new_changes[0]:+.2f}% / {new_changes[1]:+.2f}%；旧测试集对应变化 {old_changes[0]:+.2f}% / {old_changes[1]:+.2f}%。",'',
        '6%是开发阶段旧验证集上的模型选择容限；测试结果用于独立展示泛化效果，不再据此改门槛或替换检查点。原始模型在历史V3阶段可能已经报告过旧测试结果，不能把这次旧测试复核说成首次盲测；400新测试集此前在补样更新/SP筛选阶段未用于预测或选择。','',
        '测试推理不会让模型记忆这些样本；但测试结果现在已被研究者看到，未来若据此继续选参数，应说明该测试集已参与开发反馈，不能再次称完全未见的盲测。','',
        '![测试性能](01_测试集真实值与预测值.png)','',
        '完整基线/更新模型的MAE、RMSE、P95及分来源指标见 `test_metrics.csv`。源代码、模型身份、数据哈希、预测与推理不变性验证位于 `experiments/cnn_test_smallcnn_sp080_v1/`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    r.save(OUT/'report_manifest.json',{'plan_sha256':r.sha(ROOT/'evaluation_plan.json'),'inference_verification':r.read(ROOT/'inference_verification.json'),
        'files_sha256':{p.name:r.sha(p) for p in OUT.iterdir() if p.name!='report_manifest.json'}})
    print(json.dumps({'candidate_test_mse':{role:m['candidate'][role]['standardized_mse'] for role in ('old_test','new_test')},'old_mae_change_percent':old_changes,'new_mae_change_percent':new_changes},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if not args.report_only:evaluate_tests()
    report()
