"""Focused old/new MSE comparisons, saved validation only. No model or dataset loading."""
from pathlib import Path
import sys as _layout_sys
_LAYOUT_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'cnn_zone').is_dir())
_layout_sys.path.insert(0, str(_LAYOUT_PROJECT.parent / 'maintenance'))
from experiment_paths import audit_dir, report_dir, artifact_sha, relocated

import importlib.util
import json
import hashlib
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sp_report',ROOT/'report_results.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
OUT=report_dir(ROOT)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
base=r.read(r.CONTROL/'baseline/metrics.json')
entries={};histories={};records=[]
for variant,folder in r.VARIANTS.items():
    for group in ('G-S','F-S'):
        run=folder/'runs'/group
        result=r.read(run/'result.json');histories[variant,group]=r.read(run/'history.json')
        kinds={'final':20000,'best_new':result['best']['unconstrained_step']}
        if result['success']:kinds['feasible']=result['selected_step']
        for kind,step in kinds.items():
            m=r.read(run/f'step{step:05d}/metrics.json')
            entries[variant,group,kind]=(step,m)
            for role in r.ROLES:
                records.append({'setting':r.NAMES[variant],'group':group,'checkpoint_kind':kind,'step':step,'validation':role,
                    'standardized_mse':m[role]['standardized_mse'],
                    'Tavg_mse_nm2':m[role]['metrics']['tavg']['rmse']**2,
                    'DeltaT_mse_nm2':m[role]['metrics']['delta_t']['rmse']**2})
r.table(OUT/'新旧基因_MSE对比.csv',records)
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for row,group in enumerate(('G-S','F-S')):
    for col,role in enumerate(r.ROLES):
        ax=axes[row,col]
        for variant in r.VARIANTS:
            h=histories[variant,group][1:]
            ax.plot([v['step'] for v in h],[v['metrics'][role]['standardized_mse'] for v in h],color=r.COLORS[variant],label=r.NAMES[variant],lw=1.6)
        ax.set(title=f"{group} · {'旧验证基因（6483个）' if col==0 else '新验证基因（200个）'}",xlabel='成功更新步数',ylabel='双目标标准化 MSE（越低越好）')
        ax.grid(alpha=.2);ax.legend(fontsize=10)
fig.suptitle('无SP与不同SP系数：更新模型在新、旧基因上的MSE\nSP的β=0.01；相同56旧+8新回放；展示500～20000步（第0步见原报告）',fontsize=14)
fig.savefig(OUT/'04_新旧基因_MSE训练过程.png',dpi=180);plt.close(fig)
for kind,index,title in [('final','05','同为20000步：训练结束时的新旧基因MSE'),('best_new','06','最佳新验证MSE候选：新旧基因误差取舍')]:
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.subplots_adjust(left=.07,right=.98,bottom=.24,top=.80,wspace=.2)
    for ax,group in zip(axes,('G-S','F-S')):
        ticks=[]
        for i,variant in enumerate(r.VARIANTS):
            step,m=entries[variant,group,kind]
            tick=r.NAMES[variant]+f'\n柱：{step}步' if kind=='best_new' else r.NAMES[variant]+f'\n{step}步'
            if kind=='best_new' and (variant,group,'feasible') in entries:
                tick+=f"\n圈：{entries[variant,group,'feasible'][0]}步"
            ticks.append(tick)
            for shift,role,color,label in [(-.19,'old_validation','#2478b5','旧验证基因'),(.19,'dev_common','#eb861b','新验证基因')]:
                value=m[role]['standardized_mse']
                ax.bar(i+shift,value,.34,color=color,label=label if i==0 else None)
                ax.text(i+shift,value+.0015,f'{value:.5f}',ha='center',va='bottom',fontsize=8)
                if kind=='best_new' and (variant,group,'feasible') in entries:
                    _,accepted=entries[variant,group,'feasible']
                    ax.scatter([i+shift],[accepted[role]['standardized_mse']],s=75,facecolors='none',edgecolors='black',linewidths=1.3,zorder=5)
        ax.set(title=group,xticks=np.arange(4),xticklabels=ticks,ylabel='双目标标准化 MSE（越低越好）')
        ax.set_ylim(0,max(entries[v,group,kind][1][role]['standardized_mse'] for v in r.VARIANTS for role in r.ROLES)*1.24)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    if kind=='best_new':
        from matplotlib.lines import Line2D
        handles.append(Line2D([],[],marker='o',markerfacecolor='none',markeredgecolor='black',linestyle='none'));labels.append('通过旧精度约束的最优模型')
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=len(labels),frameon=False)
    fig.suptitle(title,fontsize=17,y=.98)
    note='所有柱子均为训练20000步的实际模型，未用f0回退替换。' if kind=='final' else '柱：按新验证MSE选出的最佳候选。空心圈：满足原5%旧MAE约束的最优模型；G四种设置均有圈，F均无合格模型。'
    fig.text(.5,.07,note+'\n每个模型都在同一旧验证集和同一新验证集上评价；MSE使用固定的原训练目标尺度，不混合两验证集。',ha='center',fontsize=10,linespacing=1.7)
    fig.savefig(OUT/f'{index}_新旧基因_MSE_{"最终模型" if kind=="final" else "最佳候选与合格模型"}.png',dpi=180);plt.close(fig)
lines=['# 是否使用SP：新旧基因MSE直接比较','','这里比较的每个模型都从自身同一个原始f0出发，再按56旧+8新训练。无SP为普通热启动（α=1、β=0），其余β=.01、α=.95/.80/.50。图里的“旧/新”只表示评价的验证基因，不是不同模型版本。','',
       '图中MSE均为两个目标使用固定原训练标准差归一化后的平均平方误差（无量纲），数值越低越好；没有把新旧验证集合并。各目标物理MSE=RMSE²，单位(N·m)²，另存于 `新旧基因_MSE对比.csv`。','',
       '## 先看结论','','按既定新MSE＋旧两目标MAE各自不增加超过5%的规则，G组α=.80最有希望；相对无SP合格模型，新MSE降低约4.2%。α=.95也改善约3.7%。α=.50也合格，其MAE较好，但DeltaT平方误差尾部更大，新综合MSE改善约1.0%。','','F组各SP设置都没有通过旧精度约束，即使新MSE较低也不能据此认定值得替换原模型。MSE图用于直接比较预测误差；原来的MAE接受规则没有被修改。','',
       '## G组：真正通过约束的模型','','| 设置 | 步数 | 旧验证MSE | 新验证MSE | 新MSE相对无SP合格对照变化 |','|---|---:|---:|---:|---:|']
control=entries['no_sp','G-S','feasible'][1]['dev_common']['standardized_mse']
for v in r.VARIANTS:
    step,m=entries[v,'G-S','feasible']
    lines.append(f"| {r.NAMES[v]} | {step} | {m['old_validation']['standardized_mse']:.6f} | {m['dev_common']['standardized_mse']:.6f} | {100*(m['dev_common']['standardized_mse']/control-1):+.2f}% |")
for filename in ['04_新旧基因_MSE训练过程.png','05_新旧基因_MSE_最终模型.png','06_新旧基因_MSE_最佳候选与合格模型.png']:
    lines+=['',f'## {Path(filename).stem}','',f'![{Path(filename).stem}]({filename})','']
lines+=['只重绘现有验证结果，未重新训练或解封测试集。这是单种子开发比较，不能宣称SP普遍有效或系数普遍最优。']
(OUT/'MSE对比说明.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
r.save(OUT/'MSE图核验.json',{'new_training':False,'test_used':False,'values_source':'saved metrics.json and history.json','metric':'existing dual-target standardized MSE; physical target MSE = saved RMSE squared','source_sha256':r.sha(Path(__file__)),'outputs_sha256':{p.name:r.sha(p) for p in OUT.iterdir() if 'MSE' in p.name and p.name!='MSE图核验.json'}})
subprocess.run([sys.executable,str(_LAYOUT_PROJECT.parent/'maintenance/collect_03_reports.py')],check=True)
print('MSE comparisons generated and archived; no training or test access.')
