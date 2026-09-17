"""Audit frozen selections, prepare portable FEMM queues and publish concise PNG report."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import expansion as e
from expansion import HERE,PROJECT,PILOT,old,sha,save,read,require,csv_read,csv_write

def registry():
    import torch
    rows=[]
    files=[]
    for p in sorted((PROJECT/'cnn_zone/models/v3_40000_6runs').glob('*/*/*/best_checkpoint.pt')):files.append(('f0',p,None,'original_frozen'))
    for root in [PROJECT/'experiments/cnn_replay_update_v2/models/vgg16',*[PROJECT/'experiments/cnn_replay_update_v2/models'/n for n in ('small_cnn_v2','resnet20_v2')]]:
        for group in ('G-S','F-S'):
            rp=root/'runs'/group/'result.json'
            if not rp.exists():continue
            result=read(rp)
            for kind in ('best_unconstrained','best_feasible'):
                cp=rp.parent/(kind+'.pt')
                if cp.exists():files.append(('f1',cp,result,kind))
    for rp in sorted((PROJECT/'experiments/cnn_shrink_perturb_v1/models/small_cnn_v2').glob('alpha_*/runs/*/result.json')):
        for kind in ('best_unconstrained','best_feasible'):
            cp=rp.parent/(kind+'.pt')
            if cp.exists():files.append(('f2',cp,read(rp),kind))
    for gen,p,result,kind in files:
        ck=torch.load(p,map_location='cpu',weights_only=False);cfg=ck['config'];base=cfg.get('baseline_config',cfg);step=ck.get('step',0);generation=gen
        if result:
            expected=result.get('checkpoint_sha256',{}).get(p.name)
            if expected:require(sha(p)==expected,'Registry checkpoint differs from recorded result')
        metrics=ck.get('metrics',{})
        accepted=(kind=='best_feasible') or (result is not None and result.get('selected_step')==step and result.get('success',False))
        alpha=cfg.get('sp_alpha',cfg.get('alpha',1.));beta=cfg.get('sp_beta',cfg.get('beta',0.))
        if gen=='f2':
            sp=cfg['sp'];alpha=sp['alpha'];beta=sp['beta']
            require(alpha==int(next(z for z in p.parts if z.startswith('alpha_')).split('_')[1])/100,'SP identity/path mismatch')
        rows.append({'generation':generation,'architecture':cfg.get('model_class',base['architecture']),'input':base['input_mode'],'data_source':ck.get('group','old_train_only'),'new_sample_fraction':cfg.get('new_per_update',0)/cfg.get('effective_batch_size',64),'sp_alpha':alpha,'sp_beta':beta,'update_step':step,'original_epoch':ck.get('epoch',''),'checkpoint_role':kind,'checkpoint_path':p.relative_to(PROJECT).as_posix(),'sha256':sha(p),'passes_original_acceptance': 'reference_not_applicable' if gen=='f0' else accepted,'acceptance_reference':'own architecture f0; two old MAEs <=1.05 times f0 and new-dev MSE improved','selected_for_current_expansion':p==e.CHECKPOINT,'historical_only_no_new_experiments':'mini_inception' in str(p),'state_signature_sha256':e.digest({k:list(v.shape) for k,v in ck['model_state'].items()})})
        del ck
    csv_write(HERE/'model_registry.csv',rows)
    target=PROJECT/'experiments/model_registry';target.mkdir(exist_ok=True)
    shutil.copy2(HERE/'model_registry.csv',target/'registry.csv')
    text='''# 模型代次登记

- f0：仅用原始训练集训练，尚无新增样本回放更新。
- f1：新旧数据经验回放更新，未施加 SP。
- f2：先一次性 SP，再经验回放更新。

代次不代替架构、输入视图、数据组、系数、步数或检查点哈希。不同架构各有自己的 f0/f1/f2。历史 V3 报告在时间线称 v0，与这里的模型代次不同；未重命名旧文件。

当前登记覆盖七个原始 f0、后续保留候选架构的 v2 最佳候选/合格模型、SmallCNN 首轮 SP 最佳候选/合格模型；不是历史所有优化步骤的穷举。Mini-Inception 保留历史 f0 登记，不新增实验。

本轮唯一选样模型为 f1 Polar90 VGG16、F-S、12.5%、4000步 best_unconstrained；未通过当时旧分布双目标5%约束，仅用于冻结特征选样。不得用 accepted 的 f0 回退路径替代。

详见 [登记表](registry.csv)，本轮完整身份见 [model_identity.json](../input_distribution_expansion_f1_v1/model_identity.json)。f2 α=.80/F-S 在原始5%下不合格；随后6%分析及测试使用不追溯改变该判定，使用记录保留在原实验。
'''
    (target/'README.md').write_text(text,encoding='utf-8');save(HERE/'registry_audit.json',{'records':len(rows),'current_f1_matches':sum(r['selected_for_current_expansion'] for r in rows),'registry_sha256':sha(HERE/'model_registry.csv')})

def audit_and_queue():
    ref=csv_read(HERE/'reference_pool.csv');cand=csv_read(HERE/'candidates_train.csv');train=csv_read(HERE/'train_8000.csv');dev=csv_read(HERE/'validation.csv');test=csv_read(HERE/'test.csv');roles={'candidate_train':cand,'dev':dev,'test':test};known={x.tobytes() for x in np.load(HERE/'known_genes_packed.npy')};all_new=set();families={};parents={};source_stats={}
    for role,rows in roles.items():
        count=25000 if role=='candidate_train' else 250;require(Counter(r['source'] for r in rows)==Counter({s:count for s in 'ULBP'}),'Source quotas')
        bits=old.bits_of(rows);old.validate_repaired(bits);families[role]={r['family_id'] for r in rows};parents[role]={r['parent_old_index'] for r in rows if r['source']=='P'}
        for r,b in zip(rows,bits):
            packed=np.packbits(b).tobytes();require(packed not in known and packed not in all_new,'Known/duplicate repaired gene');all_new.add(packed)
            raw=np.array(list(map(int,r['raw_bits'])),dtype=np.uint8);repaired,passes=old.repair_isolated(raw.reshape(20,6).T.copy());require(np.array_equal(repaired.T.reshape(120),b) and passes==int(r['repair_passes']) and np.count_nonzero(raw!=b)==int(r['repair_changed_cells']) and old.mapping.genotype_sha256(raw)==r['raw_gene_id'],'Repair metadata mismatch')
            held=r['parameter_regime']=='parameter_holdout';require(not held or role=='test','Holdout parameter leakage');p=json.loads(r['generator_parameters']);source=r['source']
            if source=='U':require(p['m'] in [3,6,114,117] if held else 12<=p['m']<=108,'U ranges')
            elif source in 'LB':require(tuple(p['scale']) in (old.HELD_SCALES[source] if held else old.SCALES[source]) and 12<=p['m']<=108,'Field ranges')
            else:
                rs=p['rectangles_r_a_h_w'];require(len(rs)==(1 if held else 2) and all(tuple(x[2:]) in ([(4,10),(6,8)] if held else [(1,2),(2,3),(3,5)]) for x in rs),'P ranges')
        source_stats[role]=dict(Counter(r['source'] for r in rows))
    for a,b in [('candidate_train','dev'),('candidate_train','test'),('dev','test')]:require(not families[a]&families[b] and not parents[a]&parents[b],'Family/parent leakage')
    require(Counter((r['source'],r['parameter_regime']) for r in test)==Counter({(s,k):125 for s in 'ULBP' for k in ('same_range','parameter_holdout')}),'Test regime quotas')
    candidate_ids={r['gene_id'] for r in cand};require(len(train)==8000 and len({r['gene_id'] for r in train})==8000 and all(r['gene_id'] in candidate_ids for r in train),'Training selection cardinality')
    # Existing archived graph is used for physical material area only, not G features/selection.
    graph=read(PILOT/'region_graph.json');require(graph['template_sha256']==sha(old.physical.TEMPLATE_FILE) and graph['mat_sha256']==sha(old.physical.MAT_FILE),'Archived geometry graph differs');areas=np.zeros(120)
    for r in graph['regions']:
        if r['gene_index']>=0:areas[r['gene_index']]+=r['mesh_area_mm2']
    queue=[]
    for r in train+dev+test:
        b=old.bits_of([r])[0];queue.append({**r,'experiment_groups':'f1_expansion','magnet_area_sector_mm2':float(b@areas),'magnet_area_full_motor_mm2':float(4*(b@areas)),'status':'pending'})
    require(len(queue)==len({r['gene_id'] for r in queue})==10000,'Final queue uniqueness')
    csv_write(HERE/'femm_queue.csv',queue);csv_write(HERE/'membership.csv',[{k:r[k] for k in ('gene_id','split_role','source','family_id','parent_old_index','parameter_regime')} for r in queue])
    pilot=[]
    for source in 'ULBP':
        group=sorted([r for r in queue[:8000] if r['source']==source],key=lambda r:(int(r['magnet_cells']),r['gene_id']));require(len(group)>=5,'Insufficient selected source for pilot')
        pilot.extend(group[int(i)] for i in np.linspace(0,len(group)-1,5).round().astype(int))
    require(len({r['gene_id'] for r in pilot})==20,'Pilot duplicates');csv_write(HERE/'pilot20.csv',pilot)
    for name,rows,pr in [('train_dev',queue[:9000],pilot),('sealed_test',queue[9000:],[])]:
        job=PROJECT/'femm_zone/workspaces/f1e1'/('te' if name=='sealed_test' else 'td');job.mkdir(parents=True,exist_ok=True);csv_write(job/'femm_queue.csv',rows);csv_write(job/'pilot20.csv',pr,list(queue[0]))
    audit={'status':'passed','reference_count':len(ref),'candidate_count':len(cand),'train':len(train),'validation':len(dev),'new_test':len(test),'unique_final_genes':len(queue),'nominal_angle_tasks':len(queue)*6,'pilot_genes':20,'pilot_included_in_train':True,'source_counts':source_stats,'all_generated_unique':len(all_new),'all_known_excluded':True,'family_and_parent_sets_disjoint':True,'repair_recomputed_for_all_generated':True,'holdout_parameters_not_in_train':True,'test_labels_read':0,'test_features_extracted':False,'reference_scaler_manifest_matches':read(HERE/'feature_scaler.json')['fit_manifest_sha256']==sha(HERE/'reference_pool.csv'),'model_state_invariant':read(HERE/'cache/state_invariance.json')['unchanged'],'generation_function_sha256':e.digest(__import__('inspect').getsource(e.generate)),'generator_and_repair_source_sha256':sha(old.__file__),'source_code_sha256':sha(e.__file__),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()}
    require(audit['reference_scaler_manifest_matches'] and audit['model_state_invariant'],'Feature provenance')
    save(HERE/'final_audit.json',audit)
    usage_path=PROJECT/'experiments/TEST_SET_USAGE.json';usage=read(usage_path)
    entry={'count':1000,'status':'frozen_identities_no_FEMM_labels_yet','membership':'experiments/input_distribution_expansion_f1_v1/test.csv','membership_sha256':sha(HERE/'test.csv'),'identity_sha256':e.digest([r['gene_id'] for r in test]),'features_extracted':False,'labels_read':False,'predictions_generated':False,'prior_400_test_remains_exposed':True,'storage':'experiments/input_distribution_expansion_f1_v1/femm_zone/workspaces/f1e1/te'}
    sets=usage.setdefault('subsequent_test_sets',{});key='input_distribution_expansion_f1_v1'
    if key in sets:require(sets[key]['membership_sha256']==entry['membership_sha256'],'Registered new test changed')
    else:sets[key]=entry
    sets[key]['storage']='femm_zone/workspaces/f1e1/te (sealed test only)'
    save(usage_path,usage)

def manifest():
    import femm_entry as fem
    paths=[Path(old.__file__),Path(old.physical.__file__),Path(old.mapping.__file__),PROJECT/'cnn_zone/src/fem_mesh.py',old.physical.MAT_FILE,old.physical.TEMPLATE_FILE,HERE/'femm_entry.py',HERE/'model_identity.json',HERE/'membership.csv',HERE/'femm_queue.csv',HERE/'pilot20.csv']
    paths += list((PROJECT/'femm_zone/workspaces/f1e1').glob('*/*.csv'))
    files={p.relative_to(PROJECT).as_posix():sha(p) for p in paths}
    physics=[Path(old.physical.__file__),Path(old.mapping.__file__),old.physical.MAT_FILE,old.physical.TEMPLATE_FILE]
    cfg=fem.portable_physics();save(HERE/'task_manifest.json',{'files':files,'physics_sources':{p.relative_to(PROJECT).as_posix():sha(p) for p in physics},'portable_physics':cfg,'portable_physics_fingerprint':e.digest(cfg),'historical_accepted_condition_fingerprint':e.CONDITION,'historical_fingerprint_note':'Old full runtime fingerprint includes earlier runner/host; physical files and settings are byte-identical. New portable fingerprint removes paths, retains all physics values. Runtime additionally pins installed solver binary/version.','nominal_genes':10000,'nominal_angle_tasks':60000,'default_workers':6,'pilot_in_budget':20,'test_storage':'femm_zone/workspaces/f1e1/te; default status/export exclude test labels','no_femm_started':True})
    req=[]
    for package in ('numpy','scipy','matplotlib','pyfemm','pywin32'):req.append(package+'=='+importlib.metadata.version(package))
    (HERE/'requirements-femm.txt').write_text('\n'.join(req)+'\n',encoding='utf-8')

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    out=HERE/'report';out.mkdir(exist_ok=True)
    cand=csv_read(HERE/'candidates_train.csv');train=csv_read(HERE/'train_8000.csv');diag=read(HERE/'anchor_diagnostic.json');summ=read(HERE/'selection_summary.json');audit=read(HERE/'final_audit.json');initial=read(HERE/'cache/lcmd_train_initial.json');colors=['#3378b8','#e88c32']
    fig,axs=plt.subplots(1,3,figsize=(15,4.4),layout='constrained');xx=np.arange(4)
    for i,(name,rows) in enumerate([('候选100000',cand),('选中8000',train)]):
        counts=Counter(r['source'] for r in rows);axs[0].bar(xx+(i-.5)*.35,[100*counts[s]/len(rows) for s in 'ULBP'],width=.35,label=name,color=colors[i]);axs[1].hist([int(r['magnet_cells']) for r in rows],bins=np.arange(-.5,122,4),density=True,histtype='step',lw=2,label=name,color=colors[i])
    axs[0].set(xticks=xx,xticklabels=list('ULBP'),ylabel='比例 (%)',title='生成来源分布');axs[0].legend();axs[1].set(xlabel='修正后磁体格数 / 120',ylabel='概率密度',title='磁体用量分布');axs[1].legend()
    x=np.arange(3)
    for i,(name,v) in enumerate([('仅512旧anchor',initial['coverage']),('增加8000新中心后',summ['candidate_final_nearest_F'])]):axs[2].bar(x+(i-.5)*.35,[v[k] for k in ('mean','p95','max')],.35,label=name,color=colors[i])
    axs[2].set(xticks=x,xticklabels=['均值','P95','最大值'],ylabel='标准化512维F空间最近距离',title='全体候选至中心集合的距离');axs[2].legend(fontsize=8);fig.suptitle('f1 Polar90 VGG16 · F＋LCMD输入覆盖补样（尚未FEMM求解）');fig.savefig(out/'01_选样分布与覆盖.png',dpi=180);plt.close(fig)
    s=read(HERE/'reference_audit.json');mi=read(HERE/'model_identity.json');f=read(HERE/'cache/features_reference.json');fc=read(HERE/'cache/features_candidates.json');t=read(HERE/'cache/lcmd_train.json');an=read(HERE/'cache/lcmd_anchors.json');sc=read(HERE/'feature_scaler.json')
    lines=['# f1 特征覆盖补样：首轮任务包','', '**完成选样与输入准备；未训练 CNN，未启动 FEMM。**','',f'冻结 f1：Polar90 VGG16（8×224×224），F-S，12.5%新数据，4000步最佳无约束候选。未通过原始旧精度5%约束，不能写成合格预测模型。SHA256：`{mi["sha256"]}`。','', '| 项目 | 数量 |','|---|---:|',f'| 已标注允许训练参照池 | {s["reference_count"]:,} |','| f1实际训练身份 | 41,400 |','| 旧anchor（不求解） | 512 |','| 修正后训练候选 | 100,000 |','| 新训练 / 新验证 / 新测试 | 8,000 / 1,000 / 1,000 |','| 新增唯一FEMM基因 / 正常角度任务 | 10,000 / 60,000 |','| 固定运行检查（计入训练预算） | 20 |','',f'最终训练来源：'+', '.join(f'{k}={summ["source_counts"].get(k,0)}' for k in 'ULBP')+'。候选来源等额，LCMD不强制选中配额；偏斜表示这些生成来源在当前f1特征和旧anchor条件下更常落入高累计距离簇，不表示其预测误差或物理价值必然更高。','', '![选样分布与F覆盖](01_选样分布与覆盖.png)','', '| 512个旧anchor | 全参照池最近F距离均值 | P95 | 最大值 |','|---|---:|---:|---:|']
    for k,v in diag.items():
        if k in ('LCMD','random_fixed_seed'):lines.append(f'| {k} | {v["mean"]:.6f} | {v["p95"]:.6f} | {v["max"]:.6f} |')
    wins={k:diag['LCMD'][k]<diag['random_fixed_seed'][k] for k in ('mean','p95','max')};lines+=['',f'固定随机对照只抽取一次。LCMD是否更低：{wins}；不预设全部指标占优。距离包含中心自身的零距离。', '',f'特征尺度只拟合 {s["reference_count"]:,} 个允许训练参照点；两个256维分支RMS为 {sc["branch_rms"][0]:.6f}、{sc["branch_rms"][1]:.6f}，拼接后除√2。anchor首先选择距参照均值最近的实际样本，后续按簇累计最近距离平方选簇，再取簇内最远点，哈希处理并列。训练LCMD从512个旧anchor出发，逐步加入8000个新中心。FP64直接差分，分块距离，不构建N×N矩阵。','',f'特征推理：参照池 {f["elapsed_seconds"]:.1f}s、候选 {fc["elapsed_seconds"]:.1f}s；512 anchor LCMD {an["elapsed_seconds"]:.1f}s；8000训练LCMD {t["elapsed_seconds"]:.1f}s（初始512中心距离另计 {initial["seconds"]:.1f}s）。FP32推理、不启用AMP/TF32；模型参数及缓冲哈希前后不变。缓存和LCMD均有身份校验及恢复状态。', '',f'沿用原始训练资源51,838个（其中11,838未被f1使用），与G/F训练并集合并为54,537个。排除旧验证/测试12,966个和历史898个标签冲突；前轮新验证/测试身份用于去重。旧538个来源审计例外按既有政策保留，不改标签。训练参照并不等于已被f1学好的区域。已知 {s["known_unique_excluded"]:,} 个结构全部用于重复排除，包括历史MAT身份和前轮未选候选。','', '训练候选/新验证/新测试使用独立随机流。P后代沿用历史128/64/64互斥父代组，父代本身仍是旧训练资源，因此不称为父代完全未见。生成后统一执行原八邻域孤立单格修正，再按实际基因去重。新测试每来源125常规＋125原有参数留出；新验证每来源250常规。U/L/B/P是生成来源，不是互斥物理拓扑类别。','', '旧400新测试及6483旧测试已经在此前固定模型比较中使用，记录保留；本轮1000新测试为新登记、尚无标签。未读取验证/测试性能标签，未对新验证/测试提取F特征或按预测筛选。未来测试求解结果单独放femm_zone/workspaces/f1e1/te，默认统计/导出不读其标签。','', '**FEMM规格**：初始内角29°，机械行程0/3/6/9/12/15°，内角29/32/35/38/41/44°；3.5A正向正弦电流按机械角×4更新，29°不加进电流相位；Min Angle15，倍率1。Tavg为六点均值，DeltaT为六点最大减最小，单位N·m。包内物理源码、MAT及FEM模板哈希锁定；不同电脑路径从便携物理指纹中归一化，运行指纹仍记录求解器二进制哈希与版本。','', '源码：[选样入口](../expansion.py)、[FEMM入口](../femm_entry.py)、[交付审计和绘图](../deliver.py)。详见[运行说明](../FEMM运行说明.md)、[最终清单审计](../final_audit.json)、[模型登记表](../model_registry.csv)。','', '**结论边界**：这是基于更新模型特征的覆盖补样，不能保证覆盖所有120位拓扑，也不能在FEMM与训练完成前宣称精度提高。新f1特征、anchor选择和补样规模同时变化，后续收益不能全部归因于单一因素。']
    rs=summ['repair_and_magnet_statistics'];pos=lines.index('![选样分布与F覆盖](01_选样分布与覆盖.png)')
    lines[pos:pos]=[f'孤立单格修正：候选中{rs["candidates"]["fraction_changed"]:.2%}发生修改、平均改变{rs["candidates"]["changed_cells_mean"]:.3f}格；选中训练中{rs["selected"]["fraction_changed"]:.2%}发生修改、平均改变{rs["selected"]["changed_cells_mean"]:.3f}格。修正后训练用量范围{rs["selected"]["magnet_cells_min"]}–{rs["selected"]["magnet_cells_max"]}格、中位数{rs["selected"]["magnet_cells_median"]:.0f}格。参数留出指原始生成参数，修正后的实际用量可能与常规范围重叠，不称为严格不重叠的拓扑分布。','']
    if (HERE/'femm_offline_checks.json').exists():lines[-2:-2]=['离线运行核验：120份小批输入的电流、角度、网格设置及哈希全部通过；换目录解压后status/prepare通过；尚未完成pilot时全量求解入口会拒绝启动。Windows长路径已通过短工作目录处理；实验源码、队列与报告仍位于本独立实验目录。未启动FEMM，运行尝试数为0。','']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def instructions():
    command='python .\\03_new_spmsm_project\\experiments\\input_distribution_expansion_f1_v1\\femm_entry.py'
    text=f'''# FEMM任务包运行说明

本包是可移动的最小03项目子树。解压到任意目录，在包含 `03_new_spmsm_project` 的目录打开 PowerShell。计算设备需Windows、已安装并注册FEMM、Python 3.12。无需CNN、PyTorch或原模型checkpoint。

```powershell
python -m pip install -r .\\03_new_spmsm_project\\experiments\\input_distribution_expansion_f1_v1\\requirements-femm.txt
{command} status
{command} prepare --scope pilot
{command} solve --scope pilot --workers 6
{command} status
```

20个检查基因全部成功并人工查看结果后，再执行完整训练/验证队列（9000个，包含这20个，不会重复成功角度）：

```powershell
{command} solve --scope all --workers 6
{command} resume --scope all --workers 6
{command} export --scope all
```

新测试1000个单独求解、结果封存，需显式指定test；不会被默认训练/验证导出读取：

```powershell
{command} prepare --scope test
{command} solve --scope test --workers 6
{command} resume --scope test --workers 6
{command} status --include-test-status
```

`prepare`只写输入，`status`只查身份与状态，不打开FEMM；`solve/resume`会启动FEMM。默认6个独立进程，可改workers。每个基因内部六角度依次运行。全部10000基因、60000正常角度任务；20个检查样本计入预算。重试另计，每角度最多2次已记录尝试；连续3个失败或最近50个完成尝试中5个失败则暂停。失败不填零，不用CNN代替。

train/dev结果：`femm_zone/workspaces/f1e1/td/femm_runs/<运行指纹>/`；test结果：`femm_zone/workspaces/f1e1/te/femm_runs/<运行指纹>/`。每个完整基因才产生label.json。成功角度的基因、工况、输入、结果和FEM/ANS哈希全部验证后复用。不要提前删除FEM/ANS，否则恢复完整性检查会拒绝复用；完成接收验证后再按项目的清理流程归档。

Ctrl+C允许正在求解的角度保存后暂停。崩溃留下active.lock时，先确认其中PID已退出，再仅移走该锁文件；不要删角度状态或重置已耗尽次数来绕过失败阈值。达到失败阈值时先调查，不能将异常当正常完成。恢复会校验源码/清单/物理指纹；修改参数不会悄悄复用旧结果。

物理固定：内角29/32/35/38/41/44，3.5A正向电流随机械行程×4更新，初电流角0，MinAngle15，转矩倍率1，Tavg六点均值、DeltaT六点最大减最小。便携物理指纹忽略电脑安装路径但锁定所有物理值和文件字节；更换FEMM二进制版本将形成不同运行指纹，不能误用旧缓存。

本机本轮未启动FEMM。包内不携带旧标签数据集、模型或特征缓存；原始03 MAT为保持物理来源指纹完整随包携带，执行器只读取inp及MaterialPosition（不会读取MAT性能标签）。不要手改清单或用未来结果替换基因。
'''
    text += '\nWindows工作目录采用短名 f1e1/td（训练验证）和f1e1/te（封存测试），避免完整实验名加两层哈希超出FEMM路径长度限制。建议在较短目录解压；入口会检查路径长度。所有.fem输入本身合计约8.4GB，求解生成的.ans另需磁盘空间，可先用20基因检查实际用量。\n'
    (HERE/'FEMM运行说明.md').write_text(text,encoding='utf-8')
    (HERE/'README.md').write_text('# f1输入分布扩样\n\n选样源码集中在[expansion.py](expansion.py)，交付与审计在[deliver.py](deliver.py)，FEMM仅用[femm_entry.py](femm_entry.py)。\n\n[结果报告](report/REPORT.md) · [FEMM运行说明](FEMM运行说明.md) · [模型登记](../model_registry/README.md) · [状态](STATUS.json)\n\n按顺序可恢复：`python expansion.py reference`、`generate`、`features`、`select`；正式清单已冻结，日常只使用FEMM入口。阶段均写缓存；更换模型/源码/参数会拒绝混用特征或选择状态。生成按来源块恢复，特征每512个保存，LCMD每100中心保存。\n',encoding='utf-8')

def package():
    manifest=read(HERE/'task_manifest.json');paths=[PROJECT/n for n in manifest['files']]
    paths += [HERE/n for n in ('task_manifest.json','requirements-femm.txt','FEMM运行说明.md','model_registry.csv','final_audit.json')]
    # Include only prepared inputs/states for the fixed20. No solved data can enter this package.
    for p in (PROJECT/'femm_zone/workspaces/f1e1/td').rglob('*'):
        if p.is_file() and (p.suffix=='.fem' or p.name in ('state.json','contract.json','femm_entry.json')):paths.append(p)
    zip_path=HERE/'FEMM_10000_f1_v1.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(paths)):z.write(p,p.relative_to(PROJECT.parent).as_posix())
        z.writestr('README.txt','Open 03_new_spmsm_project/experiments/input_distribution_expansion_f1_v1/FEMM运行说明.md. No FEMM solve has been started.\n')
    with zipfile.ZipFile(zip_path) as z:require(z.testzip() is None,'ZIP CRC failure')
    save(HERE/'package_audit.json',{'zip':zip_path.name,'sha256':sha(zip_path),'bytes':zip_path.stat().st_size,'file_count':len(set(paths))+1,'no_checkpoints':True,'no_new_labels':True,'prepared_pilot_genes':20,'normal_angle_tasks':60000})
    e.status('complete_ready_for_manual_femm',package=zip_path.name,genes=10000,normal_angle_tasks=60000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['registry','queue','manifest','report','instructions','package']);a=parser.parse_args();{'registry':registry,'queue':audit_and_queue,'manifest':manifest,'report':report,'instructions':instructions,'package':package}[a.command]()
