"""Post-hoc tolerance sensitivity; does not modify training/selection records or checkpoints."""
from pathlib import Path
import csv,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
CONTROL=ROOT.parent/'cnn_replay_update_v2/logical6x20/small_cnn_v2'
OUT=ROOT/'report'
VARIANTS={'无SP':CONTROL,'α=0.95':ROOT/'alpha_095','α=0.80':ROOT/'alpha_080','α=0.50':ROOT/'alpha_050'}
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=read(CONTROL/'baseline/metrics.json');records=[];registry=[];metrics=[];inputs={}
for name,folder in VARIANTS.items():
    for group in ('G-S','F-S'):
        run=folder/'runs'/group;result=read(run/'result.json');history=read(run/'history.json')
        for path in (run/'result.json',run/'history.json',folder/'config.json'):
            inputs[str(path.relative_to(PROJECT))]=sha(path)
        for tolerance in (.05,.06,.075):
            options=[e for e in history if e['step']>0 and all(e['metrics']['old_validation']['metrics'][t]['mae']<=base['old_validation']['metrics'][t]['mae']*(1+tolerance) for t in ('tavg','delta_t')) and e['metrics']['dev_common']['standardized_mse']<base['dev_common']['standardized_mse']-1e-12]
            best=min(options,key=lambda e:e['metrics']['dev_common']['standardized_mse']) if options else None
            step=best['step'] if best else 0;m=best['metrics'] if best else base
            if tolerance==.05:assert step==result['selected_step']
            path=None
            if best:
                for key,s in [('best_feasible.pt',result['best']['feasible_step']),('best_unconstrained.pt',result['best']['unconstrained_step']),('last_checkpoint.pt',result['updates'])]:
                    if step==s:
                        path=run/key;assert sha(path)==result['checkpoint_sha256'][key];break
            row={'tolerance_percent':tolerance*100,'setting':name,'group':group,'qualified':bool(best),'step':step,
                'old_standardized_mse':m['old_validation']['standardized_mse'],'new_standardized_mse':m['dev_common']['standardized_mse'],
                **{t+'_old_mae_change_percent':100*(m['old_validation']['metrics'][t]['mae']/base['old_validation']['metrics'][t]['mae']-1) for t in ('tavg','delta_t')},
                'checkpoint_available':bool(path),'checkpoint':str(path.relative_to(PROJECT)) if path else ('f0' if not best else 'not_saved_for_this_step')}
            records.append(row)
            if tolerance==.06:
                registry.append({**row,'checkpoint_sha256':sha(path) if path else None})
                for role in ('old_validation','dev_common'):
                    for source,block in {'all':m[role],**m[role].get('sources',{})}.items():
                        for t in ('tavg','delta_t'):
                            metrics.append({'setting':name,'group':group,'step':step,'role':role,'source':source,'n':block['n'],'target':t,**block['metrics'][t]})
for filename,rows in [('容限敏感性.csv',records),('6pct容限_物理误差.csv',metrics)]:
    with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'6pct容限_模型索引.json').write_text(json.dumps({'analysis':'post-hoc exploratory; original 5% results unchanged','reference':'original SmallCNN f0','old_mae_tolerance':.06,'new_mse_must_improve_f0':True,'models':registry,'test_used':False,'new_training':False},ensure_ascii=False,indent=2),encoding='utf8')
lines=['# 旧精度容限放宽复核：5% → 6%','','根据本轮讨论，以6%作为小幅放宽的观察门槛，同时列5%与7.5%敏感性。所有组（含无SP、G、F）使用同一门槛；旧Tavg和DeltaT的MAE仍需分别过关，新验证标准化MSE仍需优于原f0。','','这是看过结果后的探索性复核，不改写原先5%预设实验结论，也不证明6%已经满足工程需要。未重训、未改数据/权重、未解封测试集。','',
'## 6%门槛下的结果','','| 设置 | 组 | 最佳合格步数 | 旧Tavg MAE变化 | 旧DeltaT MAE变化 | 新标准化MSE |','|---|---|---:|---:|---:|---:|']
for r in records:
    if r['tolerance_percent']!=6:continue
    lines.append(f"| {r['setting']} | {r['group']} | {r['step'] if r['qualified'] else '无，回退f0'} | {r['tavg_old_mae_change_percent']:+.2f}% | {r['delta_t_old_mae_change_percent']:+.2f}% | {r['new_standardized_mse']:.6f} |")
selected=next(r for r in registry if r['setting']=='α=0.80' and r['group']=='F-S')
control=next(r for r in registry if r['setting']=='无SP' and r['group']=='F-S')
g=next(r for r in registry if r['setting']=='α=0.80' and r['group']=='G-S')
lines+=['',f"6%门槛下，α=.80、β=.01的F-S（{selected['step']}步）成为有希望的选择：新MSE={selected['new_standardized_mse']:.6f}，比同门槛无SP F-S低 {100*(1-selected['new_standardized_mse']/control['new_standardized_mse']):.2f}%，比α=.80 G-S低 {100*(1-selected['new_standardized_mse']/g['new_standardized_mse']):.2f}%。代价是旧Tavg/DeltaT MAE增加约1.52%/5.15%。不是F全面优于G，而是接受稍大的旧误差后，F的新结构误差更低。",'',
'## 门槛敏感性','','| 门槛 | 无SP F | α=.95 F | α=.80 F | α=.50 F |','|---|---|---|---|---|']
for t in (5,6,7.5):
    cells=[]
    for name in VARIANTS:
        r=next(r for r in records if r['tolerance_percent']==t and r['setting']==name and r['group']=='F-S')
        cells.append(f"{r['step']}步 / {r['new_standardized_mse']:.6f}" if r['qualified'] else '无合格更新')
    lines.append(f"| {t}% | "+' | '.join(cells)+' |')
lines+=['','6%已能纳入表现更好的F候选，没有必要仅为接纳模型继续放宽到7.5%。7.5%下α=.50 F的13500步仅有保存的验证结果，没有对应留存权重；此处只做指标敏感性分析，不声称该模型可直接交付。6%筛选出的更新权重均已保存，其路径和哈希见 `6pct容限_模型索引.json`。','','后续若采用6%，应在下一轮开始前固定下来，并保持同一规则评价全部系数和对照。这里保留原5%训练配置/结果作为历史，不修改已有accepted字段。']
(OUT/'6pct容限复核.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(OUT/'容限复核核验.json').write_text(json.dumps({'original_5pct_selection_reproduced':True,'same_rule_for_all_groups':True,'inputs_sha256':inputs,'test_used':False,'new_training':False,'source_sha256':sha(Path(__file__))},ensure_ascii=False,indent=2),encoding='utf8')
subprocess.run([sys.executable,str(PROJECT.parent/'maintenance/collect_03_reports.py')],check=True)
print(json.dumps(selected,ensure_ascii=False))
