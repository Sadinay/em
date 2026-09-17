"""f1 coverage expansion: reference, generate, features, select, deliver. Never solves FEMM."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time, subprocess
from pathlib import Path
from collections import Counter
from dataclasses import fields
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.input_distribution_pilot_v1 import pilot as old
sha, save, read, require, digest, csv_read, csv_write = old.sha, old.save, old.read, old.require, old.digest, old.csv_read, old.csv_write
PILOT = PROJECT / 'experiments/input_distribution_pilot_v1'
ACCEPTED = PILOT / 'post_femm_baseline_20260913'
CHECKPOINT = PROJECT / 'experiments/cnn_replay_update_v2/runs/F-S/best_unconstrained.pt'
CHECKPOINT_SHA = '9612f9892e76bc354ca6c6e885a57b29f865da8b84d0f3b67b195befc6980346'
PHYSICS_SHA = '52459cf11c89ee1589f4d3a11a7e86e92450ac14ea082037a94a6c86e8a7835f'
CONDITION = '97cd342fe5d68919dffee97fe5ff2d37bcfdbfd461eb357e9ca92f75ee05c712'
SEED = 20260916
STREAMS = {name: int.from_bytes(hashlib.sha256(f'{SEED}/f1_expansion/{name}'.encode()).digest()[:8], 'big') for name in ('pool','dev','test','parents','random_anchors')}
def status(stage, **kw):
    save(HERE/'STATUS.json', {'stage':stage, 'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'cnn_training':False,'femm_started':False,**kw})
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bits_string(b): return ''.join(np.asarray(b,dtype=np.uint8).astype(str))
def frozen(path, value):
    if path.exists(): require(digest(read(path))==digest(value), f'Frozen identity changed: {path}')
    else: save(path,value)

def model_identity():
    import torch
    require(sha(CHECKPOINT)==CHECKPOINT_SHA,'Specified f1 checkpoint missing/changed; no substitution')
    x=torch.load(CHECKPOINT,map_location='cpu',weights_only=False); c=x['config']
    require(x['step']==4000 and x['group']=='F-S' and c['model_class']=='cnn_zone.src.models_v2.Semantic224VGG16V2' and c['input_shape']==[8,224,224] and c['new_per_update']==8 and c['effective_batch_size']==64,'Wrong f1 identity')
    require(c['baseline_config']['input_mode']=='polar90_224' and c['baseline_config']['architecture']=='vgg16_v2','Wrong input/architecture')
    for k in ('target_mean','target_std'): require(np.array_equal(x[k].numpy(),np.array(c[k],dtype=np.float32)),k)
    record={'generation':'f1','architecture':c['model_class'],'input':'Polar90 8x224x224','data_source':'F-S: old train 40000 + train_F 1400','new_sample_fraction':0.125,'sp_alpha':1.0,'sp_beta':0.0,'step':4000,'checkpoint':CHECKPOINT.relative_to(PROJECT).as_posix(),'sha256':CHECKPOINT_SHA,'acceptance_passed':False,'status':'best_unconstrained candidate; failed original two-target old-MAE <=1.05*f0 rule; frozen feature selection only','f0_sha256':x['f0_sha256'],'target_mean':c['target_mean'],'target_std':c['target_std']}
    frozen(HERE/'model_identity.json',record)
    return x

def reference():
    model_identity(); require(sha(old.physical.__file__)==PHYSICS_SHA,'Physical configuration drift')
    versions={}
    for name,expected in read(PILOT/'provenance.json')['input_sha256'].items():
        normalized=name.replace('\\','/')
        if any(s in normalized for s in ('workspace_200.mat','SPMSM_discrete.fem','spmsm_mapping.py','cnn_zone/src/','lookups/')):
            require(sha(PROJECT/normalized)==expected,'Accepted physics/encoding dependency changed: '+normalized);versions[normalized]=expected
    genes,split=old.old_data(); hist=read(old.DATA/'dataset_summary.json')
    with open(old.DATA/'split_manifest.csv',encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f); next(reader); identities=[r[:3] for r in reader]
    require(len(identities)==len(genes),'Old manifest size')
    allowed=np.concatenate([split['train'],split['unused_train_pool']])
    original_split=np.load(old.DATA/'split_indices.npz')
    require(set(map(int,allowed))==set(map(int,original_split['train'])),'unused_train_pool not an original training resource')
    forbidden_components={identities[i][2] for k in ('validation','test') for i in split[k]}
    reserved=read(PILOT/'parent_families.json')
    # Parent roles constrain generated children only; these parents themselves remain old training resources.
    reserved_parents=set(reserved['dev'])|set(reserved['test'])
    hold_ids={identities[i][1] for k in ('validation','test') for i in split[k]}
    membership=csv_read(PILOT/'memberships.csv')
    hold_ids.update(r['gene_id'] for r in membership if r['split_role']!='train')
    target=np.load(old.DATA/'targets_tavg_delta.npy',mmap_mode='r')
    train_set=set(map(int,split['train'])); rows=[]; exclusions=Counter(); seen={}
    for idx in allowed:
        i=int(idx); gid=old.mapping.genotype_sha256(genes[i]); require(gid==identities[i][1],'Old identity encoding')
        if identities[i][2] in forbidden_components or gid in hold_ids: exclusions['role_or_repair_family']+=1; continue
        y=np.array(target[i],copy=True)
        if not np.isfinite(y).all() or y[1]<0: exclusions['invalid_old_label']+=1; continue
        rows.append({'gene_id':gid,'bits':bits_string(genes[i]),'source':'old_train' if i in train_set else 'old_unused_train_pool','old_index':i,'repair_component_id':identities[i][2],'f1_actually_trained':i in train_set,'tavg_nm':float(y[0]),'delta_t_nm':float(y[1])})
        seen[gid]=y
    checks=read(ACCEPTED/'OUTPUT_CHECKSUMS.json'); require(read(ACCEPTED/'data_audit.json')['status']=='passed','Unaccepted import')
    for role in ('train_G','train_F'):
        for name,h in checks.items():
            if name.startswith('data/'+role+'/'): require(sha(ACCEPTED/name)==h,'Accepted training file changed: '+name)
        m=csv_read(ACCEPTED/f'data/{role}/manifest.csv')
        for r in m:
            gid=r['gene_id']; require(gid not in hold_ids,'New training/holdout conflict')
            require(r['status']=='verified' and r['condition_fingerprint']==CONDITION and r['split_role']=='train','Unaccepted label')
            y=np.array([float(r['tavg_nm']),float(r['delta_t_nm'])],dtype=np.float32)
            require(np.isfinite(y).all() and y[1]>=0,'Invalid accepted label')
            if gid in seen:
                require(np.array_equal(seen[gid],y),'Duplicate label conflict')
                row=next(z for z in rows if z['gene_id']==gid); row['source']+=';'+role; row['f1_actually_trained']=True; exclusions['GF_training_duplicate_merged']+=1
            else:
                rows.append({'gene_id':gid,'bits':r['bits'],'source':role,'old_index':'','repair_component_id':'','f1_actually_trained':role=='train_F','tavg_nm':float(y[0]),'delta_t_nm':float(y[1])});seen[gid]=y
    rows.sort(key=lambda r:r['gene_id']); require(len(seen)==len(rows),'Reference duplicates')
    csv_write(HERE/'reference_pool.csv',rows)
    actual=[{'gene_id':identities[i][1],'source':'old_train','old_index':int(i)} for i in split['train']]
    actual += [{'gene_id':r['gene_id'],'source':'train_F','old_index':''} for r in csv_read(PILOT/'train_F.csv')]
    csv_write(HERE/'f1_actual_training_membership.csv',actual)
    known,counts=old.all_history_keys(genes)
    for name in ('candidates_train.csv','dev_common.csv','test_common.csv'):
        known.update(np.packbits(b).tobytes() for b in old.bits_of(csv_read(PILOT/name)))
    np.save(HERE/'known_genes_packed.npy',np.array([np.frombuffer(k,dtype=np.uint8) for k in sorted(known)]))
    audit={'reference_count':len(rows),'source_counts':dict(Counter(r['source'] for r in rows)),'excluded':dict(exclusions),'original_train_resource_count':len(allowed),'original_validation_test_excluded':len(split['validation'])+len(split['test']),'historical_quarantined_conflicts_excluded':hist['deduplication']['quarantined_label_conflicts'],'known_unique_excluded':len(known),'history_identity_rows':counts,'f1_actual_train_count':len(actual),'retained_old_exception_policy':hist['integrity']['pm_count_mismatch_policy'],'historical_exception_count':hist['integrity']['pm_count_volume_pm_mismatches'],'test_label_reads':0,'validation_label_reads':0,'new_test_400_previously_evaluated':True,'labels_definition':'six-angle mean Tavg and max-minus-min DeltaT in N*m; initial29, multiplier1','source_hashes':{'old_summary':sha(old.DATA/'dataset_summary.json'),'old_split':sha(old.SPLIT),'accepted_checksums':sha(ACCEPTED/'OUTPUT_CHECKSUMS.json'),'accepted_audit':sha(ACCEPTED/'data_audit.json')},'reference_sha256':sha(HERE/'reference_pool.csv'),'known_sha256':sha(HERE/'known_genes_packed.npy')}
    audit['accepted_dependencies_verified']=versions
    audit['allowed_reference_target_float32_sha256']=ah(np.array([[r['tavg_nm'],r['delta_t_nm']] for r in rows],dtype=np.float32))
    audit['allowed_reference_bits_sha256']=ah(old.bits_of(rows))
    audit['historical_P_parent_note']='Parent role reservations constrain offspring; parents themselves remain old training resources.'
    save(HERE/'reference_audit.json',audit);status('reference_ready',reference_count=len(rows)); print(json.dumps(audit,ensure_ascii=False),flush=True)

def generate():
    if (HERE/'generation_audit.json').exists():
        for n,h in read(HERE/'generation_audit.json')['file_sha256'].items(): require(sha(HERE/n)==h,'Generation cache changed')
        return
    genes,split=old.old_data(); previous=read(PILOT/'parent_families.json')
    # Accepted binary training parents are unchanged; the original repair applies to generated offspring.
    parents={k:[int(i) for i in previous[k]] for k in ('pool','dev','test')}
    require(all(i in set(split['train']) for v in parents.values() for i in v),'Parent outside old training resources')
    require(all(parents.values()),'No legal reserved parents')
    require(all(not set(parents[a])&set(parents[b]) for a,b in [('pool','dev'),('pool','test'),('dev','test')]),'P parent leakage')
    frozen(HERE/'parent_families.json',parents)
    cfg={'seed':SEED,'streams':STREAMS,'candidates_per_source':25000,'dev_per_source':250,'test_per_source_regular':125,'test_per_source_holdout':125,'magnet_count_range':[12,108],'held_U_counts':[3,6,114,117],'scales':old.SCALES,'held_scales':old.HELD_SCALES,'rectangles_regular':[[1,2],[2,3],[3,5]],'rectangles_held':[[4,10],[6,8]],'regular_P_rectangles':2,'held_P_rectangles':1,'repair':'original synchronous 8-neighbor isolated-cell flip; finite 6x20 no wrap; reject cycles','parent_policy':'same historical role-reserved parent sets; legal old-training parents only; not unseen-parent generalization','feature_model_sha256':CHECKPOINT_SHA,'femm_config_sha256':PHYSICS_SHA}
    frozen(HERE/'config.json',cfg)
    rr,aa=np.indices((6,20));rr=rr.ravel();aa=aa.ravel(); factors={}
    for scale in set(sum(old.SCALES.values(),[])+sum(old.HELD_SCALES.values(),[])):
        cov=np.exp(-.5*(((rr[:,None]-rr)/scale[0])**2+((aa[:,None]-aa)/scale[1])**2));v,q=np.linalg.eigh(cov);require(v.min()>-1e-10,'Covariance');factors[scale]=q*np.sqrt(np.maximum(v,0))[None,:]
    known={r.tobytes() for r in np.load(HERE/'known_genes_packed.npy')}; initial=len(known); rejects=Counter(); start=time.perf_counter();files=['parent_families.json']
    # Persist completed source blocks with RNG state, so generation can resume without replacing samples.
    for stream,total in [('pool',25000),('dev',250),('test',250)]:
        rng=np.random.default_rng(STREAMS[stream]);rows=[]
        for source in 'ULBP':
            chunk=HERE/'cache'/f'generated_{stream}_{source}.json'
            if chunk.exists():
                ck=read(chunk);require(ck['config_sha']==sha(HERE/'config.json'),'Generator config changed');rng.bit_generator.state=ck['rng_state'];part=ck['rows'];rejects.update(ck['rejects'])
                for r in part:
                    key=np.packbits(np.fromiter(map(int,r['bits']),dtype=np.uint8)).tobytes(); require(key not in known,'Resumed duplicate');known.add(key)
                rows.extend(part);continue
            part=[]; local=Counter()
            for n in range(total):
                held=stream=='test' and n>=125;slot=n-125 if held else n
                for attempt in range(10000):
                    m=int([3,6,114,117][slot%4] if source=='U' and held else rng.integers(12,109));parent='';params={}
                    if source=='U':
                        grid=np.zeros(120,dtype=np.uint8);grid[rng.choice(120,m,replace=False)]=1;grid=grid.reshape(6,20);params={'m':m}
                    elif source in 'LB':
                        scales=old.HELD_SCALES[source] if held else old.SCALES[source];scale=scales[slot%len(scales)];field=factors[scale]@rng.standard_normal(120);grid=np.zeros(120,dtype=np.uint8);grid[np.argsort(field,kind='stable')[-m:]]=1;grid=grid.reshape(6,20);params={'m':m,'scale':list(scale)}
                    else:
                        parent=parents[stream][n%len(parents[stream])];grid=genes[parent].reshape(20,6).T.copy();masks=[];rect=[]
                        for _ in range(1 if held else 2):
                            for trial in range(10000):
                                h,w=([(4,10),(6,8)][slot%2] if held else [(1,2),(2,3),(3,5)][int(rng.integers(3))]);r,a=int(rng.integers(7-h)),int(rng.integers(21-w));mask=np.zeros((6,20),bool);mask[r:r+h,a:a+w]=True
                                if all(not (mask&prior).any() for prior in masks):break
                            else: raise RuntimeError('Rectangle placement exhausted')
                            masks.append(mask);rect.append([r,a,h,w]);grid[mask]^=1
                        params={'rectangles_r_a_h_w':rect}
                    raw=grid.T.reshape(120).copy()
                    try: grid,passes=old.repair_isolated(grid)
                    except ValueError as e: local['repair_failed:'+str(e)]+=1;continue
                    b=grid.T.reshape(120);key=np.packbits(b).tobytes()
                    if key in known:local['repaired_duplicate_or_known']+=1;continue
                    known.add(key)
                    family=f'P/parent/{parent}' if source=='P' else f'expansion_f1_v1/{source}/{stream}/field_{n}/attempt_{attempt}'
                    part.append({'gene_id':old.mapping.genotype_sha256(b),'bits':bits_string(b),'source':source,'family_id':family,'split_role':'candidate_train' if stream=='pool' else stream,'generator_parameters':json.dumps({**params,'attempt':attempt},sort_keys=True),'seed':STREAMS[stream],'parent_old_index':parent,'parameter_regime':'parameter_holdout' if held else 'same_range','selection_method':'','selection_rank':'','selection_distance_squared':'','display_region_id':'','experiment_groups':'','magnet_cells':int(b.sum()),'raw_bits':bits_string(raw),'raw_gene_id':old.mapping.genotype_sha256(raw),'repair_changed_cells':int(np.count_nonzero(raw!=b)),'repair_passes':passes});break
                else: raise RuntimeError(f'Collision budget exhausted {stream}/{source}/{n}')
            save(chunk,{'config_sha':sha(HERE/'config.json'),'rows':part,'rng_state':rng.bit_generator.state,'rejects':dict(local)});rejects.update(local);rows.extend(part);print(f'Generated {stream}/{source}: {len(part)}; {time.perf_counter()-start:.1f}s',flush=True);status('generating',stream=stream,source=source)
        name={'pool':'candidates_train.csv','dev':'validation.csv','test':'test.csv'}[stream];old.validate_repaired(old.bits_of(rows));csv_write(HERE/name,rows,old.CSV_FIELDS);files.append(name)
    save(HERE/'generation_audit.json',{'generated_unique':len(known)-initial,'known_excluded':initial,'rejections':dict(rejects),'parent_counts':{k:len(v) for k,v in parents.items()},'elapsed_seconds':time.perf_counter()-start,'file_sha256':{n:sha(HERE/n) for n in files},'no_prediction_filter':True,'test_features_extracted':False});status('generation_complete')

def model_state_hash(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def features():
    import torch
    from cnn_zone.src.training import TrainingSpec,build_model,build_renderer,make_inputs
    x=model_identity();c=x['config']['baseline_config'];spec=TrainingSpec(**{f.name:c[f.name] for f in fields(TrainingSpec)})
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=build_model(spec).to(device=device,memory_format=torch.channels_last).eval();model.load_state_dict(x['model_state'],strict=True);model.requires_grad_(False);renderer=build_renderer(spec,PROJECT,device).eval();before=model_state_hash(model);capture={}
    def hook(name):
        def callback(module,inputs):capture[name]=inputs[0].detach()
        return callback
    for n in ('head_tavg','head_delta_t'):getattr(model.regressor,n)[-1].register_forward_pre_hook(hook(n))
    deps={str(p.relative_to(PROJECT)):sha(p) for p in (old.LOOKUP,Path(old.__file__),*sorted((PROJECT/'cnn_zone/src').glob('*.py')))}
    contract={'checkpoint_sha256':CHECKPOINT_SHA,'model_state_sha256':before,'dependencies':deps,'extractor_sha256':digest(__import__('inspect').getsource(features)),'reference_sha256':sha(HERE/'reference_pool.csv'),'candidate_sha256':sha(HERE/'candidates_train.csv'),'batch_size':16,'dtype':'float32','tf32':False,'amp':False,'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'feature_definition':'pre-final-linear Tavg256 concat DeltaT256; eval inference_mode','scale_fit':'reference_pool only'}
    frozen(HERE/'cache/feature_contract.json',contract)
    for name,file in [('reference','reference_pool.csv'),('candidates','candidates_train.csv')]:
        b=old.bits_of(csv_read(HERE/file));path=HERE/f'cache/features_{name}.npy';sp=HERE/f'cache/features_{name}.json';st=read(sp) if sp.exists() else {'completed':0,'elapsed_seconds':0,'chunks':[]};arr=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(len(b),512))
        for chunk in st['chunks']:require(ah(arr[chunk['start']:chunk['end']])==chunk['sha256'],'Feature prefix corruption')
        if st.get('sha256'):require(sha(path)==st['sha256'],'Feature cache corruption');continue
        t=time.perf_counter();elapsed=st['elapsed_seconds'];chunk_start=st['completed']
        with torch.inference_mode():
            for i in range(st['completed'],len(b),16):
                tensor=torch.from_numpy(np.array(b[i:i+16],copy=True)).to(device);model(make_inputs(tensor,spec,renderer));h=torch.cat([capture[n] for n in ('head_tavg','head_delta_t')],dim=1);require(h.shape==(len(tensor),512) and torch.isfinite(h).all().item(),'Invalid features');end=i+len(tensor);arr[i:end]=h.cpu().numpy()
                if end-chunk_start>=512 or end==len(b):
                    arr.flush();st['chunks'].append({'start':chunk_start,'end':end,'sha256':ah(arr[chunk_start:end])});chunk_start=end;st.update(completed=end,elapsed_seconds=elapsed+time.perf_counter()-t);save(sp,st)
                    if end<=512 or end%8192==0 or end==len(b):print(f'Features {name} {end}/{len(b)} {st["elapsed_seconds"]:.1f}s GPUpeak {torch.cuda.max_memory_allocated()/1024**3:.2f}GiB',flush=True);status('features',role=name,completed=end,total=len(b))
        st['sha256']=sha(path);save(sp,st)
    after=model_state_hash(model);require(before==after,'Model state changed during extraction');save(HERE/'cache/state_invariance.json',{'before':before,'after':after,'unchanged':True,'training':False,'test_inference':False})
    raw=np.load(HERE/'cache/features_reference.npy',mmap_mode='r');center=raw.astype(np.float64).mean(0);scales=[float(np.sqrt(np.mean(np.sum((raw[:,k:k+256].astype(np.float64)-center[k:k+256])**2,axis=1)))) for k in (0,256)];require(min(scales)>1e-10,'Degenerate branch scale')
    frozen(HERE/'feature_scaler.json',{'center':center.tolist(),'branch_rms':scales,'formula':'(h_branch - reference_mean_branch) / sqrt(mean_reference(sum_256_squared_deviation)); concatenate two scaled branches / sqrt(2)','fit_count':len(raw),'fit_manifest_sha256':sha(HERE/'reference_pool.csv'),'feature_contract_sha256':sha(HERE/'cache/feature_contract.json'),'reference_feature_sha256':sha(HERE/'cache/features_reference.npy'),'fit_roles':['allowed_training_reference_pool']});status('features_complete')

def normalized(name):
    sc=read(HERE/'feature_scaler.json');a=np.load(HERE/f'cache/features_{name}.npy',mmap_mode='r');return (a.astype(np.float64)-np.array(sc['center']))/np.repeat(sc['branch_rms'],256)/np.sqrt(2)

class Distance:
    def __init__(self,values,block=8192):
        import torch
        self.torch=torch;self.values=torch.from_numpy(np.ascontiguousarray(values,dtype=np.float64)).to('cuda' if torch.cuda.is_available() else 'cpu');self.block=block
    def one(self,idx,n):
        result=np.empty(n,np.float64);center=self.values[idx]
        for lo in range(0,n,self.block):
            v=self.values[lo:min(lo+self.block,n)];result[lo:lo+len(v)]=((v-center)**2).sum(1).cpu().numpy()
        return result

def lcmd(values,ids,n,count,initial,name,resume=True,stop_after=None):
    """Direct FP64 distances; cluster sum-of-nearest-d2 then farthest member. Stable hash ties."""
    d=Distance(values);ids=np.array(ids);lex=np.argsort(np.argsort(ids,kind='stable'),kind='stable');path=HERE/f'cache/lcmd_{name}.npz';tracepath=HERE/f'cache/lcmd_{name}.json'
    contract={'values_sha256':ah(values),'ids_sha256':ah(ids),'candidate_count':n,'select_count':count,'initial':list(map(int,initial)),'algorithm_sha256':digest(__import__('inspect').getsource(lcmd)),'dtype':'float64','distance':'direct squared differences; row blocks8192'}
    nearest=np.full(n,np.inf);owner=np.full(n,-1,np.int64);active=np.ones(n,bool);centers=[];chosen=[];trace=[];previous=0.
    def add(idx):
        distances=d.one(idx,n);require(np.isfinite(distances).all() and (distances>=0).all(),'Invalid distance')
        replace=distances<nearest
        if centers:replace|=(distances==nearest)&(lex[idx]<lex[np.asarray(centers)[np.maximum(owner,0)]])
        nearest[replace]=distances[replace];owner[replace]=len(centers);centers.append(int(idx))
    if resume and tracepath.exists():
        st=read(tracepath);state_path=path.parent/st['state_file'];require(st['contract']==contract and sha(state_path)==st['state_sha256'],'LCMD recovery identity mismatch');z=np.load(state_path);nearest=z['nearest'];owner=z['owner'];active=z['active'];centers=z['centers'].tolist();chosen=z['chosen'].tolist();z.close();trace=st['trace'];previous=st['elapsed_seconds']
    else:
        init_started=time.perf_counter()
        for idx in initial:
            add(idx)
            if idx<n:active[idx]=False;chosen.append(int(idx));trace.append({'rank':len(chosen),'index':int(idx),'gene_id':str(ids[idx]),'distance_squared':0.,'cluster_sum':0.,'initial_mean_nearest':True})
        if resume:
            save(HERE/f'cache/lcmd_{name}_initial.json',{'coverage':coverage(nearest),'seconds':time.perf_counter()-init_started,'centers':len(initial),'values_bytes':values.nbytes,'distance_dtype':'float64','block_rows':8192})
    start=time.perf_counter()
    while len(chosen)<count:
        sums=np.bincount(owner[active],weights=nearest[active],minlength=len(centers));occupancy=np.bincount(owner[active],minlength=len(centers));sums[occupancy==0]=-np.inf;regions=np.flatnonzero(sums==sums.max());region=int(regions[np.argmin(lex[np.array(centers)[regions]])]);members=np.flatnonzero(active&(owner==region));far=members[nearest[members]==nearest[members].max()];idx=int(far[np.argmin(lex[far])]);trace.append({'rank':len(chosen)+1,'index':idx,'gene_id':str(ids[idx]),'distance_squared':float(nearest[idx]),'cluster_sum':float(sums[region]),'region_center_gene_id':str(ids[centers[region]])});chosen.append(idx);active[idx]=False;add(idx)
        if len(chosen)%100==0 or len(chosen)==count or (stop_after and len(chosen)==stop_after):
            elapsed=previous+time.perf_counter()-start
            if resume:
                # Commit immutable state first, then atomically move the pointer; interruption cannot
                # invalidate the preceding checkpoint. Retain current+previous recovery generations.
                state_path=path.with_name(path.stem+f'_rank_{len(chosen):05d}.npz');temp=state_path.with_suffix('.tmp.npz');np.savez(temp,nearest=nearest,owner=owner,active=active,centers=centers,chosen=chosen);os.replace(temp,state_path)
                old_state=read(tracepath) if tracepath.exists() else None
                older=tracepath.with_name(tracepath.stem+'_previous.json');obsolete=read(older).get('state_file') if older.exists() else None
                if old_state:save(older,old_state)
                save(tracepath,{'contract':contract,'trace':trace,'elapsed_seconds':elapsed,'state_file':state_path.name,'state_sha256':sha(state_path)})
                if obsolete and obsolete not in (state_path.name,old_state['state_file'] if old_state else ''):
                    prior=path.parent/obsolete;require(prior.resolve().parent==path.parent.resolve() and prior.name.startswith(path.stem+'_rank_'),'Unsafe cache cleanup path');prior.unlink(missing_ok=True)
            if len(chosen)%500==0 or len(chosen)==count:print(f'LCMD {name}: {len(chosen)}/{count}; {elapsed:.1f}s',flush=True);status('lcmd',selection=name,completed=len(chosen),total=count)
            if stop_after and len(chosen)==stop_after:break
    return chosen,nearest,trace

def coverage(d2):
    v=np.sqrt(d2);return {'mean':float(v.mean()),'p95':float(np.percentile(v,95)),'max':float(v.max())}

def select():
    refs=csv_read(HERE/'reference_pool.csv');candidates=csv_read(HERE/'candidates_train.csv');r=normalized('reference');c=normalized('candidates');ids=[x['gene_id'] for x in refs]
    benchmark=HERE/'cache/distance_benchmark.json'
    if not benchmark.exists():
        import torch, gc
        available,total=torch.cuda.mem_get_info() if torch.cuda.is_available() else (0,0)
        if torch.cuda.is_available():require(available>c.nbytes+8192*512*8*3,'Insufficient GPU memory for chunked FP64 distance')
        d=Distance(c);started=time.perf_counter()
        for idx in range(10):d.one(idx,len(c))
        seconds=time.perf_counter()-started
        save(benchmark,{'candidate_rows':len(c),'dimensions':512,'ten_center_seconds':seconds,'centers_per_second':10/seconds,'gpu_free_before_bytes':available,'gpu_total_bytes':total,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,'input_array_bytes':c.nbytes,'arithmetic':'direct FP64 squared difference; blocks8192'})
        print(f'FP64 distance benchmark: {10/seconds:.2f} centers/s; input {c.nbytes/1024**2:.1f}MiB',flush=True);del d;gc.collect()
    ds=((r-r.mean(0))**2).sum(1);ties=np.flatnonzero(ds==ds.min());first=min(ties,key=lambda i:ids[i]);save(HERE/'anchor_initialization.json',{'first_index':int(first),'gene_id':ids[first],'distance_to_reference_mean_squared':float(ds[first]),'tie_count':len(ties),'tie_rule':'gene hash ascending','first_counts_toward_512':True});chosen,near,trace=lcmd(r,ids,len(r),512,[first],'anchors')
    anchors=[{**refs[i],'selection_rank':k+1,'selection_distance_squared':trace[k]['distance_squared']} for k,i in enumerate(chosen)];csv_write(HERE/'anchors_512.csv',anchors);csv_write(HERE/'anchor_trace.csv',trace)
    diagnostic=HERE/'anchor_diagnostic.json'
    if not diagnostic.exists():
        random=np.random.default_rng(STREAMS['random_anchors']).choice(len(r),512,replace=False);d=Distance(r);rn=np.full(len(r),np.inf)
        for i in random:rn=np.minimum(rn,d.one(int(i),len(r)))
        csv_write(HERE/'random_anchors_512.csv',[{'gene_id':ids[i],'reference_index':int(i)} for i in random]);save(diagnostic,{'LCMD':coverage(near),'random_fixed_seed':coverage(rn),'random_seed':STREAMS['random_anchors'],'reference_count':len(r),'distance':'sqrt standardized512D squared distance; includes center points'})
    allvalues=np.concatenate([c,r[chosen]],axis=0);allids=[x['gene_id'] for x in candidates]+[ids[i] for i in chosen];selected,newnear,tr=lcmd(allvalues,allids,len(c),8000,range(len(c),len(c)+512),'train')
    rows=[{**candidates[i],'split_role':'train','selection_method':'f1_F_LCMD','selection_rank':k+1,'selection_distance_squared':tr[k]['distance_squared']} for k,i in enumerate(selected)];csv_write(HERE/'train_8000.csv',rows);csv_write(HERE/'selection_trace.csv',tr)
    repair_stats={name:{'changed_cells_mean':float(np.mean([int(x['repair_changed_cells']) for x in data])),'fraction_changed':float(np.mean([int(x['repair_changed_cells'])>0 for x in data])),'magnet_cells_min':min(int(x['magnet_cells']) for x in data),'magnet_cells_median':float(np.median([int(x['magnet_cells']) for x in data])),'magnet_cells_max':max(int(x['magnet_cells']) for x in data)} for name,data in [('candidates',candidates),('selected',rows)]}
    save(HERE/'selection_summary.json',{'train_count':len(rows),'source_counts':dict(Counter(x['source'] for x in rows)),'candidate_final_nearest_F':coverage(newnear),'repair_and_magnet_statistics':repair_stats,'selected_changed_cells_mean':repair_stats['selected']['changed_cells_mean'],'selection_sha256':sha(HERE/'train_8000.csv'),'precision':'float64 direct difference; no NxN matrix','test_or_dev_features':False});status('selection_complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['reference','generate','features','select','status']);a=parser.parse_args();HERE.mkdir(exist_ok=True);(HERE/'cache').mkdir(exist_ok=True)
    if a.command=='status':print(json.dumps(read(HERE/'STATUS.json'),ensure_ascii=False,indent=2))
    else:globals()[a.command]()
