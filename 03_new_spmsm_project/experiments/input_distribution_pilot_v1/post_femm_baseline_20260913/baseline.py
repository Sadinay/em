"""Receive FEMM records, audit frozen roles, and evaluate f0 without training/test inference.

Commands: python baseline.py audit | evaluate | report | verify | all
No imported package scripts are executed and no FEMM calls are made.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
from dataclasses import fields
import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
PROJECT = PILOT.parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.input_distribution_pilot_v1 import pilot
from femm_zone import femm_config as current_physics
from femm_zone.scripts import spmsm_mapping as mapping

PACKAGE = PROJECT / 'femm_zone/results/FEMM_results_20260913'
MODEL = pilot.MODEL
RECEIVED = HERE / 'received'
DATA = HERE / 'data'
ALLOWED_VIEWS = ('old_validation', 'train_G', 'train_F', 'dev_common')
TOL = 1e-6
TARGETS = ('tavg', 'delta_t')
VERIFIED_PHYSICS_SHA256 = '52459cf11c89ee1589f4d3a11a7e86e92450ac14ea082037a94a6c86e8a7835f'
PHYSICS_SOURCE = 'femm_zone/femm_config.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def currents(cfg, inner):
    """Use the canonical source, checked against the advisor's independent formula."""
    travel = inner - cfg['airgap']['initial_inner_angle_deg']
    ti = math.radians(travel) / cfg['current']['omega_m_rad_s']
    theta = cfg['current']['omega_e_rad_s'] * ti
    expected = {k: cfg['current']['amplitude_a'] * math.cos(theta + shift)
                for k, shift in (('A', 0), ('B', -2*math.pi/3), ('C', -4*math.pi/3))}
    actual = current_physics.phase_currents(cfg, inner, 0)
    require(set(actual) == set(expected) and
            all(math.isclose(actual[k], expected[k], rel_tol=0, abs_tol=1e-14) for k in expected),
            'Canonical currents differ from the frozen advisor formula')
    return actual


def prepared_fem(base, cfg, inner):
    return current_physics.configure_fem(base, cfg, inner, 0).encode('utf-8')


def verified_physics_hash():
    """Require the exact previously verified source; never whitelist arbitrary drift."""
    actual = sha(Path(current_physics.__file__))
    require(actual == VERIFIED_PHYSICS_SHA256, 'Current physical source differs from verified 29-degree/1 configuration')
    require(sha(PACKAGE/'source_snapshot'/PHYSICS_SOURCE) == actual,
            'Current physical source differs from the received verified snapshot')
    return actual


def unique(rows, label):
    require(len(rows) == len({r['gene_id'] for r in rows}), f'Duplicate gene ID in {label}')
    return {r['gene_id']: r for r in rows}


def assert_evaluation_role(role):
    require(role in ALLOWED_VIEWS, f'Evaluation role is sealed or unsupported: {role}')


def load_view(role):
    """Training-compatible dataset entry. Final test is deliberately not available here."""
    assert_evaluation_role(role)
    from cnn_zone.src.dataset import SPMSMGeneDataset
    if role == 'old_validation':
        _, split = pilot.old_data()
        return SPMSMGeneDataset(pilot.DATA), split['validation']
    dataset = SPMSMGeneDataset(DATA / role)
    return dataset, np.arange(len(dataset), dtype=np.int64)


def audit():
    tick = time.perf_counter()
    verified_physics_hash()
    RECEIVED.mkdir(exist_ok=True)
    expected = read(PACKAGE / 'checksums.json')
    actual_names = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file()}
    require(actual_names == set(expected) | {'checksums.json'}, 'Package file inventory differs from checksums')
    package_hashes = {}
    for name, h in expected.items():
        path = PACKAGE / name
        require(path.resolve().is_relative_to(PACKAGE.resolve()), 'Unsafe package path')
        require(sha(path) == h, f'Package checksum mismatch: {name}')
        package_hashes[name] = h
    package_identity = digest_bytes(json.dumps(expected, sort_keys=True).encode())
    receipt_path = RECEIVED / 'package_identity.json'
    if receipt_path.exists():
        require(read(receipt_path)['identity'] == package_identity, 'Different received package; use a new output directory')
    save(receipt_path, {'identity': package_identity, 'source': str(PACKAGE), 'files': package_hashes})
    # Preserve the original compressed evidence separately. Decode records directly
    # from this copy, instead of creating 66,000 tiny filesystem objects.
    copied_zip = RECEIVED / 'full_run_records.zip'
    if not copied_zip.exists():
        shutil.copy2(PACKAGE / copied_zip.name, copied_zip)
    require(sha(copied_zip) == expected[copied_zip.name], 'Received ZIP differs')
    versions, drifts = {}, {}
    provenance = read(PILOT / 'provenance.json')
    for name, h in provenance['input_sha256'].items():
        actual = sha(PROJECT / name)
        versions[name] = {'expected': h, 'actual': actual, 'matches': actual == h}
        if actual != h:
            drifts[name] = versions[name]
    require(not drifts, f'Unexpected frozen source drift: {list(drifts)}')
    seed_validation = read(PILOT / 'seed_validation.json')
    for name, h in seed_validation['file_sha256'].items():
        require(sha(PILOT / name) == h, f'Frozen local manifest changed: {name}')
    for name in ('femm_zone/femm_config.py', 'experiments/input_distribution_pilot_v1/pilot.py'):
        remote = PACKAGE / 'source_snapshot' / name
        local_text, remote_text = (PROJECT/name).read_text(encoding='utf-8'), remote.read_text(encoding='utf-8')
        (RECEIVED / (Path(name).stem + '.diff')).write_text(''.join(difflib.unified_diff(local_text.splitlines(True), remote_text.splitlines(True), fromfile='current_local/'+name, tofile='received_snapshot/'+name)), encoding='utf-8')
    manifests = {}
    for name in ('train_G', 'train_F', 'dev_common', 'test_common', 'femm_queue', 'memberships', 'pilot20'):
        source = PACKAGE / 'manifests' / (name+'.csv')
        require(sha(source) == sha(PILOT / source.name), f'Frozen manifest differs: {name}')
        manifests[name] = read_csv(source)
        target = RECEIVED / 'manifests' / source.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        require(sha(target) == sha(source), 'Received manifest changed')
    for name in ('parent_families.json', 'region_graph.json', 'seed_validation.json', 'provenance.json'):
        require(sha(PACKAGE/'audit'/name) == sha(PILOT/name), f'Frozen audit differs: {name}')
    queue = manifests['femm_queue']
    qmap = unique(queue, 'queue')
    bits = pilot.bits_of(queue)
    pilot.validate_repaired(bits)  # Read-only validation, no second repair.
    split_ids = {name: set(unique(manifests[name], name)) for name in ('train_G','train_F','dev_common','test_common')}
    G, F, dev, test = [split_ids[n] for n in ('train_G','train_F','dev_common','test_common')]
    require([len(G),len(F),len(dev),len(test)] == [1400,1400,200,400], 'Frozen split counts differ')
    require(len(G & F) == 101 and not ((G|F)&(dev|test)) and not (dev&test), 'Role intersection differs')
    require(set(qmap) == G|F|dev|test and len(qmap)==3299, 'Queue is not the role union')
    families = [{r['family_id'] for r in queue if r['gene_id'] in ids} for ids in (G|F,dev,test)]
    require(all(not families[i]&families[j] for i in range(3) for j in range(i)), 'Family leakage')
    genes, old_split = pilot.old_data()
    history, history_counts = pilot.all_history_keys(genes)
    require(all(np.packbits(b).tobytes() not in history for b in bits), 'New/historical exact identity overlap')
    parents = read(PILOT/'parent_families.json')
    parent_sets = [set(parents[k]) for k in ('pool','dev','test')]
    require(all(not parent_sets[i]&parent_sets[j] for i in range(3) for j in range(i)), 'P parent split overlap')
    require(set.union(*parent_sets) <= set(old_split['train'].tolist()), 'P parent outside old train')
    memberships = [(r['gene_id'],r['split_role'],r['experiment_group']) for r in manifests['memberships']]
    expected_memberships = [(r['gene_id'],r['split_role'],g) for r in queue for g in r['experiment_groups'].split(';')]
    require(len(set(memberships))==len(memberships) and set(memberships)==set(expected_memberships), 'Membership table mismatch')
    for row in queue:
        if row['source']=='P':
            role = 'pool' if row['split_role']=='train' else row['split_role']
            require(int(row['parent_old_index']) in parents[role], 'P parent role mismatch')
    graph = read(PILOT/'region_graph.json')
    areas = np.zeros(120)
    for region in graph['regions']:
        if region['gene_index'] >= 0:
            areas[region['gene_index']] += region['mesh_area_mm2']
    all_labels = unique(read_csv(PACKAGE/'tables/labels.csv'), 'labels')
    dataset_table = unique(read_csv(PACKAGE/'tables/dataset_all.csv'), 'dataset_all')
    wave_rows = read_csv(PACKAGE/'tables/waveforms.csv')
    waves = {(r['gene_id'],float(r['inner_angle_deg'])):r for r in wave_rows}
    require(len(waves)==len(wave_rows), 'Duplicate waveform angle')
    require(set(all_labels)==set(qmap)==set(dataset_table), 'Queue/label/table identity differs')
    valid, failures, wave_export = {}, [], []
    aggregate_errors, prepared_count, attempts = [], 0, Counter()
    with zipfile.ZipFile(copied_zip) as z:
        names = z.namelist()
        require(len(names)==len(set(names)), 'Duplicate ZIP record names')
        checksums = json.loads(z.read('RUN_CHECKSUMS.json'))
        require(set(names)==set(checksums)|{'RUN_CHECKSUMS.json'}, 'ZIP inventory mismatch')
        raw_records = {}
        for i,(name,h) in enumerate(checksums.items()):
            require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and '\\' not in name, 'Unsafe ZIP name')
            content = z.read(name)
            require(digest_bytes(content)==h, f'ZIP record checksum mismatch: {name}')
            raw_records[name] = content
            if i and i % 15000 == 0:
                print(f'ZIP verified: {i}/{len(checksums)}', flush=True)
        contract = json.loads(raw_records['contract.json'])
        cfg = contract['physical']
        fingerprint = pilot.digest(contract)
        require(fingerprint==read(PACKAGE/'package_summary.json')['condition_fingerprint'], 'Condition fingerprint mismatch')
        prior = read(PROJECT/'femm_zone/workspaces/pilot20_cross_device_check_20260909/archive_audit.json')
        require(fingerprint==prior['remote_condition_fingerprint'], 'Full run differs from independently checked pilot condition')
        for key in ('current','problem','airgap','rotor_travel_angles_deg','inner_angles_deg','initial_phases_deg','torque_multiplier'):
            require(cfg[key]==prior['physical'][key], f'Physics mismatch: {key}')
        require(contract['template_sha256']==sha(current_physics.TEMPLATE_FILE), 'Template mismatch')
        require(contract['mat_sha256']==sha(current_physics.MAT_FILE), 'MAT mismatch')
        require(contract['mapping_source_sha256']==sha(Path(mapping.__file__)), 'Mapping mismatch')
        require(contract['physical_source_sha256']==sha(PACKAGE/'source_snapshot/femm_zone/femm_config.py'), 'Frozen physical source mismatch')
        save(RECEIVED/'contract.json',contract)
        (RECEIVED/'RUN_CHECKSUMS.json').write_bytes(z.read('RUN_CHECKSUMS.json'))
        template = current_physics.TEMPLATE_FILE.read_text(encoding='utf-8')
        positions = loadmat(current_physics.MAT_FILE,variable_names=['MaterialPosition'],squeeze_me=True)['MaterialPosition']
        mapped = mapping.match_material_positions(template,positions)
        actual_angle_keys = {(n.split('/')[0],float(n.split('/')[1][6:])) for n in names if re.fullmatch(r'[0-9a-f]{64}/angle_[0-9.]+/result.json',n)}
        require(actual_angle_keys == set(waves), 'Waveforms and angle files differ')
        for i,(row,b) in enumerate(zip(queue,bits)):
            gid = row['gene_id']
            try:
                base = mapping.replace_cell_materials(template,b)
                mapping.validate_generated_model(base,mapped,b)
                require(int(b.sum())==int(row['magnet_cells']), 'PM count mismatch')
                require(abs(float(b@areas)-float(row['magnet_area_sector_mm2']))<1e-9, 'PM area mismatch')
                label = json.loads(raw_records[f'{gid}/label.json'])
                require(label['status']=='succeeded' and label['successful_angles']==6, 'Incomplete label')
                require(len(label['angles'])==6, 'Label angle count differs')
                values, records = [], []
                for inner in cfg['inner_angles_deg']:
                    prefix=f'{gid}/angle_{inner:g}/'
                    result,state,receipt = [json.loads(raw_records[prefix+n+'.json']) for n in ('result','state','artifact_retention')]
                    require(state['status']=='succeeded','Unsuccessful angle')
                    identity={'gene_id':gid,'bits':row['bits'],'inner_angle_deg':inner,'rotor_travel_deg':inner-29,
                              'currents_a':currents(cfg,inner),'condition_fingerprint':fingerprint,
                              'prepared_sha256':digest_bytes(prepared_fem(base,cfg,inner))}
                    for key,value in identity.items():
                        require(result[key]==state[key]==value,f'Angle identity/input differs: {key}')
                    require(state['result_sha256']==digest_bytes(raw_records[prefix+'result.json']), 'Result link broken')
                    require(state['artifact_retention_sha256']==digest_bytes(raw_records[prefix+'artifact_retention.json']), 'Receipt link broken')
                    expected_receipt={'policy':'results_only_v1','result_sha256':state['result_sha256'],
                                      'condition_fingerprint':fingerprint,'gene_id':gid,'inner_angle_deg':inner,
                                      'model_settings_verified':True,'fem_sha256':result['fem_sha256'],'ans_sha256':result['ans_sha256']}
                    require(receipt==expected_receipt, 'Receipt contents differ')
                    matches=[a for a in label['angles'] if a['inner_angle_deg']==inner]
                    require(len(matches)==1 and matches[0]==result,'Gene/angle label mismatch')
                    torque=result['raw_torque_nm']
                    require(isinstance(torque,(float,int)) and math.isfinite(torque),'Invalid torque')
                    w=waves[gid,inner]
                    for key in ('raw_torque_nm','rotor_travel_deg'):
                        require(float(w[key])==result[key],f'Waveform mismatch: {key}')
                    for phase,value in currents(cfg,inner).items():
                        require(float(w[phase])==value,'Waveform current mismatch')
                    for key in ('fem_sha256','ans_sha256'):
                        require(w[key]==result[key],'Waveform hash mismatch')
                    attempts.update(a['status'] for a in state['attempts'])
                    values.append(torque)
                    records.append({**w,'record_reference':f'received/full_run_records.zip!/{prefix}result.json','prepared_sha256':identity['prepared_sha256']})
                    prepared_count+=1
                metrics={'tavg_nm':float(np.mean(values)),'delta_t_nm':max(values)-min(values)}
                for source in (label,all_labels[gid],dataset_table[gid]):
                    for key,value in metrics.items():
                        err=abs(float(source[key])-value);aggregate_errors.append(err)
                        require(err<=TOL,f'Six-point aggregation mismatch: {key}')
                    require(source['condition_fingerprint']==fingerprint,'Summary condition differs')
                for key,value in row.items():
                    if key!='status':
                        require(dataset_table[gid][key]==value,f'Export changed manifest metadata: {key}')
                valid[gid]={**row,**metrics,'status':'verified','condition_fingerprint':fingerprint,
                            'waveform_reference':f'received/full_run_records.zip!/{gid}/label.json',
                            'source_package_identity':package_identity,'source_label_reference':f'received/full_run_records.zip!/{gid}/label.json'}
                wave_export.extend(records)
            except Exception as exc:
                failures.append({'gene_id':gid,'error':str(exc),'split_role':row['split_role']})
            if (i+1)%200==0 or i+1==len(queue):
                print(f'Physics/labels: {i+1}/{len(queue)}, valid={len(valid)}, conflicts={len(failures)}',flush=True)
    # Cross-check every supplied split export, preserving frozen selection order.
    for role in ('train_G','train_F','dev_common','test_common'):
        supplied=read_csv(PACKAGE/'tables'/f'{role}_labeled.csv')
        require([r['gene_id'] for r in supplied]==[r['gene_id'] for r in manifests[role]],f'{role} export order mismatch')
        for s in supplied:
            gid=s['gene_id']
            if gid in valid:
                for key in ('tavg_nm','delta_t_nm'):
                    require(abs(float(s[key])-valid[gid][key])<=TOL,f'{role} target mismatch')
        labeled=[{**r,**{k:v for k,v in valid[r['gene_id']].items() if k not in r or k=='status'}} for r in manifests[role] if r['gene_id'] in valid]
        # Keep each view's original G/F selection rank and metadata, plus shared labels.
        target=DATA/('sealed_test' if role=='test_common' else role)
        write_csv(target/'manifest.csv',labeled)
        b=pilot.bits_of(labeled)
        np.save(target/'topology_bits.npy',b.reshape(-1,20,6).swapaxes(1,2))
        np.save(target/'targets_tavg_delta.npy',np.array([[r['tavg_nm'],r['delta_t_nm']] for r in labeled],dtype=np.float32))
        np.save(target/'gene_ids.npy',np.array([r['gene_id'] for r in labeled]))
    write_csv(DATA/'labels_train_dev.csv',[v for k,v in valid.items() if k not in test])
    write_csv(DATA/'sealed_test/labels.csv',[v for k,v in valid.items() if k in test])
    write_csv(DATA/'waveforms_train_dev.csv',[r for r in wave_export if r['gene_id'] not in test])
    write_csv(DATA/'sealed_test/waveforms.csv',[r for r in wave_export if r['gene_id'] in test])
    write_csv(HERE/'conflicts.csv',failures,['gene_id','error','split_role'])
    audit_info={'status':'passed' if not failures else 'partial','expected_genes':len(queue),'valid_genes':len(valid),
                'missing_or_conflicting_genes':len(failures),'failures':failures,'failed_genes':sum(r['status']=='failed' for r in all_labels.values()),
                'queue_missing':sorted(set(qmap)-set(all_labels)),'queue_extra':sorted(set(all_labels)-set(qmap)),
                'duplicate_summary_records':0,'complete_angles':len(wave_export),'prepared_input_hashes_verified':prepared_count,
                'max_aggregate_difference_nm':max(aggregate_errors,default=0),'absolute_tolerance_nm':TOL,
                'source_counts':{role:dict(Counter(r['source'] for r in manifests[role])) for role in split_ids},
                'role_counts':{role:len(ids) for role,ids in split_ids.items()},'training_intersection':len(G&F),
                'train_dev_test_disjoint':True,'families_disjoint':True,'P_parents_from_old_train':True,
                'history_unique_genotypes':len(history),'history_population_counts':history_counts,'new_history_exact_overlap':0,
                'angle_attempt_status_counts':dict(attempts),'original_FEM_ANS_present':False,
                'evidence':'All retained records/checksums and reconstructed prepared inputs; prior independent FEMM replay covered two pilot genes only.',
                'ready':{role:split_ids[role]<=set(valid) for role in split_ids},'local_physical_source_drift':drifts,
                'new_FEMM_solves':0,'elapsed_seconds':time.perf_counter()-tick}
    save(HERE/'data_audit.json',audit_info)
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=PROJECT,capture_output=True,text=True,check=True).stdout.strip()
    status=subprocess.run(['git','status','--short'],cwd=PROJECT,capture_output=True,text=True,encoding='utf-8',check=True).stdout
    save(HERE/'import_manifest.json',{'source_package':str(PACKAGE),'package_identity':package_identity,'package_files':package_hashes,
        'original_checksums_sha256':sha(PACKAGE/'checksums.json'),'received_zip_sha256':sha(copied_zip),
        'source_device':read(PACKAGE/'package_summary.json'),'actual_git_commit':git,'git_worktree_status':status,
        'contract':contract,'frozen_local_inputs':versions,'instruction_sha256':sha(PROJECT/'femm_zone/results/Codex_SPMSM_FEMM_Import_and_Baseline.md'),
        'prior_cross_device_audit_sha256':sha(PROJECT/'femm_zone/workspaces/pilot20_cross_device_check_20260909/archive_audit.json'),
        'raw_record_files_verified':len(checksums),'femm_executed':False,'foreign_scripts_executed':False,
        'runner_difference':'COM/results-only receipts plus bounded Windows PermissionError retry and CLI diagnostics; physics contract unchanged.',
        'record_layout':'Byte-identical received ZIP; records decompressed and verified in memory; core manifest/contract files also saved under received.'})
    print(f'AUDIT {audit_info["status"]}: {len(valid)} genes / {len(wave_export)} angles',flush=True)
    return audit_info


def metric_rows(rows, role, source='all'):
    output=[]
    for target in TARGETS:
        y=np.array([float(r[target+'_true_nm']) for r in rows]);p=np.array([float(r[target+'_pred_nm']) for r in rows]);e=p-y;a=np.abs(e)
        if not len(y):
            continue
        nonconstant=len(y)>1 and np.ptp(y)>0
        rho=float(spearmanr(y,p).statistic) if nonconstant and np.ptp(p)>0 else None
        output.append({'role':role,'source':source,'target':target,'n':len(y),'mae_nm':float(a.mean()),
            'rmse_nm':float(np.sqrt(np.mean(e**2))),'median_abs_nm':float(np.median(a)),
            'p90_abs_nm':float(np.percentile(a,90)),'p95_abs_nm':float(np.percentile(a,95)),
            'max_abs_nm':float(a.max()),'bias_nm':float(e.mean()),
            'r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if nonconstant else None,
            'spearman':rho,'na_note':'' if nonconstant and rho is not None else 'N<2 or constant truth/prediction'})
    return output


def evaluate_baseline():
    import torch
    from cnn_zone.src.training import TrainingSpec,build_model,build_renderer,make_inputs,evaluate,configure_reproducibility
    from cnn_zone.src.dataset import SPMSMGeneDataset
    require(read(HERE/'data_audit.json')['status']=='passed','Resolve import conflicts before baseline')
    imports=read(HERE/'import_manifest.json')
    for name,item in imports['frozen_local_inputs'].items():
        if name!='femm_zone\\femm_config.py':
            require(sha(PROJECT/name)==item['expected'],f'Frozen model/input changed: {name}')
    config=read(MODEL/'config.json'); result=read(MODEL/'result.json'); history=read(MODEL/'history.json')
    checkpoint_hash=sha(MODEL/'best_checkpoint.pt')
    ckpt=torch.load(MODEL/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    require(ckpt['config']==config and ckpt['epoch']==result['best_epoch']==39 and ckpt['seed']==20260903,'Wrong f0 identity')
    require(config['input_mode']=='polar90_224' and config['architecture']=='vgg16_v2' and config['train_samples']==40000 and not config['circular_angular_padding'],'Wrong f0 pipeline')
    for key in ('target_mean','target_std'):
        require(np.array_equal(ckpt[key].numpy(),np.array(config[key],dtype=np.float32)),f'{key} differs')
    spec=TrainingSpec(**{f.name:config[f.name] for f in fields(TrainingSpec)})
    torch.set_num_threads(4)
    configure_reproducibility(20260903)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(spec).to(device=device,memory_format=torch.channels_last).eval()
    model.load_state_dict(ckpt['model_state'],strict=True);model.requires_grad_(False)
    renderer=build_renderer(spec,PROJECT,device).eval()
    require(renderer.coordinate_system=='polar90','Renderer coordinate mismatch')
    bits,split=pilot.old_data()
    test_ids=set(r['gene_id'] for r in read_csv(DATA/'sealed_test/manifest.csv'))
    new=read_csv(DATA/'labels_train_dev.csv')
    require(not test_ids&{r['gene_id'] for r in new},'Sealed test entered prediction pool')
    require(all(r['split_role'] in ('train','dev') for r in new),'Unexpected role entered new inference pool')
    targets=np.load(pilot.DATA/'targets_tavg_delta.npy',mmap_mode='r')
    meta=[{'gene_id':mapping.genotype_sha256(bits[i]),'bits':''.join(map(str,bits[i].tolist())),
           'source':'old_GA','family_id':'old_GA_history','split_role':'old_validation','selection_method':'',
           'old_index':int(i),'magnet_cells':int(bits[i].sum()),'nearest_old_train_hamming':'',
           'magnet_area_full_motor_mm2':'','tavg_nm':float(targets[i,0]),'delta_t_nm':float(targets[i,1])} for i in split['validation']]
    require(not test_ids&{r['gene_id'] for r in meta},'Sealed test overlaps old validation')
    for role in ('test_common','sealed_test','test','old_test'):
        try:
            assert_evaluation_role(role)
        except ValueError:
            pass
        else:
            raise AssertionError('Test role guard failed')
    all_predictions=[];timings={}
    for role,rows in [('old_validation',meta),('new_train_dev',new)]:
        if role=='old_validation':
            dataset,indices=load_view(role)
        else:
            # A private, explicitly train+dev-only dataset object avoids a broad test-readable input directory.
            dataset=SPMSMGeneDataset.__new__(SPMSMGeneDataset)
            dataset.bits=np.array([[int(c) for c in r['bits']] for r in rows],dtype=np.uint8)
            dataset.targets=np.array([[r['tavg_nm'],r['delta_t_nm']] for r in rows],dtype=np.float32)
            indices=np.arange(len(rows),dtype=np.int64)
        start=time.perf_counter()
        print(f'F0 inference {role}: {len(rows)} samples, {device}, original batch={spec.physical_batch_size}, original AMP policy',flush=True)
        loss,old_style_metrics,returned,actual,predicted=evaluate(model,dataset,indices,spec,renderer,ckpt['target_mean'],ckpt['target_std'],device)
        require(np.array_equal(returned,indices),'Inference row order changed')
        require(np.isfinite(predicted).all(),'Nonfinite f0 prediction')
        timings[role]={'seconds':time.perf_counter()-start,'n':len(rows),'standardized_mse':loss}
        for row,truth,pred in zip(rows,actual,predicted):
            record={k:row.get(k,'') for k in ('gene_id','bits','source','family_id','split_role','selection_method','old_index','magnet_cells','magnet_area_full_motor_mm2','nearest_old_train_hamming')}
            record['member_train_G']=int('G' in row.get('selection_method','').split(';'))
            record['member_train_F']=int('F' in row.get('selection_method','').split(';'))
            for j,target in enumerate(TARGETS):
                # New labels retain original double precision; old labels use their frozen float32 array.
                true=float(truth[j]) if role=='old_validation' else float(row[target+'_nm'])
                record.update({target+'_true_nm':true,target+'_pred_nm':float(pred[j]),target+'_error_nm':float(pred[j])-true})
            all_predictions.append(record)
        if role=='old_validation':
            reference=next(h for h in history if h['epoch']==39)
            diffs={target:{k:old_style_metrics[target][k]-reference['validation_metrics'][target][k] for k in ('mae','rmse','bias','absolute_error_p95')} for target in TARGETS}
            max_difference=max(abs(v) for d in diffs.values() for v in d.values())
            consistency={'reference':'history.json epoch 39 validation (not test)','reference_metrics':reference['validation_metrics'],
                         'recomputed_metrics':old_style_metrics,'differences':diffs,'max_physical_metric_difference_nm':max_difference,
                         'reference_standardized_mse':reference['validation_standardized_mse'],'recomputed_standardized_mse':loss,
                         'metric_tolerance_nm':1e-5,'passed':max_difference<=1e-5}
            save(HERE/'old_validation_consistency.json',consistency)
            require(consistency['passed'],'Old validation pipeline differs beyond predeclared 1e-5 Nm metric tolerance')
        print(f'F0 done {role}: {timings[role]["seconds"]:.1f}s',flush=True)
    require(len(all_predictions)==len({r['gene_id'] for r in all_predictions})==9382,'Prediction count or identity differs')
    require(not test_ids&{r['gene_id'] for r in all_predictions},'Test predictions detected')
    for name,value in model.state_dict().items():
        require(torch.equal(value.detach().cpu(),ckpt['model_state'][name]),f'Model state changed: {name}')
    require(sha(MODEL/'best_checkpoint.pt')==checkpoint_hash,'Checkpoint file changed')
    write_csv(HERE/'baseline_predictions.csv',all_predictions)
    groups={'old_validation':[r for r in all_predictions if r['split_role']=='old_validation'],
            'train_G':[r for r in all_predictions if r['member_train_G']],
            'train_F':[r for r in all_predictions if r['member_train_F']],
            'dev_common':[r for r in all_predictions if r['split_role']=='dev'],
            'train_union':[r for r in all_predictions if r['split_role']=='train']}
    metrics=[]
    for name,rows in groups.items():
        metrics.extend(metric_rows(rows,name))
        if name!='old_validation':
            for source in 'ULBP':
                metrics.extend(metric_rows([r for r in rows if r['source']==source],name,source))
    write_csv(HERE/'baseline_metrics.csv',metrics)
    save(HERE/'baseline_metrics.json',metrics)
    structures=[]
    for name in ('train_G','train_F','dev_common','train_union'):
        for key in ('magnet_cells','magnet_area_full_motor_mm2','nearest_old_train_hamming'):
            values=np.array([float(r[key]) for r in groups[name]])
            structures.append({'role':name,'quantity':key,'n':len(values),'min':float(values.min()),
                               'median':float(np.median(values)),'max':float(values.max()),
                               'p10':float(np.percentile(values,10)),'p90':float(np.percentile(values,90))})
    write_csv(HERE/'structure_statistics.csv',structures)
    save(HERE/'model_manifest.json',{'checkpoint_sha256':checkpoint_hash,'checkpoint':str(MODEL/'best_checkpoint.pt'),
         'config':config,'model_state_unchanged':True,'test_evaluated':False,'predicted_unique_genes':len(all_predictions),
         'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
         'dtype':'Original training.evaluate AMP float16 on CUDA; float32 inverse scaling; original batch size 8',
         'target_mean':ckpt['target_mean'].tolist(),'target_std':ckpt['target_std'].tolist(),'timings':timings,
         'test_guard_checks_passed':True,'script_sha256':sha(__file__)})
    make_figures(all_predictions,groups,renderer,device)
    print('BASELINE complete; final test not predicted',flush=True)


def make_figures(predictions,groups,renderer,device):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def output(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,target in zip(axes,TARGETS):
        bounds=[]
        for name,color,alpha,size in [('old_validation','#64748b',.25,6),('dev_common','#e66029',.8,17)]:
            rows=groups[name]; y=[r[target+'_true_nm'] for r in rows];p=[r[target+'_pred_nm'] for r in rows]
            ax.scatter(y,p,s=size,alpha=alpha,c=color,label=f'{name} (n={len(rows)})');bounds+=y+p
        lo,hi=min(bounds),max(bounds);ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(xlabel='FEMM / frozen target (N m)',ylabel='Frozen f0 prediction (N m)',title=target);ax.legend();ax.grid(alpha=.2)
    output(fig,'truth_prediction')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for ax,target in zip(axes,TARGETS):
        values=[[abs(r[target+'_error_nm']) for r in groups['dev_common'] if r['source']==s] for s in 'ULBP']
        ax.boxplot(values,tick_labels=[s+' (50)' for s in 'ULBP'],showfliers=True)
        ax.set(title='Public dev: '+target,ylabel='Absolute error (N m)');ax.grid(axis='y',alpha=.2)
    output(fig,'dev_source_errors')
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    colors=['#4083ba','#50a78b','#e8ad43','#bd6a9e']; bottom=np.zeros(2)
    for s,c in zip('ULBP',colors):
        counts=np.array([sum(r['source']==s for r in groups[name]) for name in ('train_G','train_F')]);axes[0].bar(['G','F'],counts,bottom=bottom,color=c,label=s);bottom+=counts
    axes[0].set(title='Training source counts',ylabel='Genes');axes[0].legend()
    for ax,key,title in zip(axes[1:],['magnet_cells','magnet_area_full_motor_mm2','nearest_old_train_hamming'],['Magnet cells','Full motor PM area (mm2)','Nearest old-train Hamming']):
        bins=np.histogram_bin_edges([float(r[key]) for name in ('train_G','train_F') for r in groups[name]],bins=20)
        for name,color in [('train_G','#2878a5'),('train_F','#d8752e')]:
            ax.hist([float(r[key]) for r in groups[name]],bins=bins,histtype='step',lw=2,label=name,color=color)
        ax.set(title=title,ylabel='Genes');ax.legend()
    output(fig,'GF_structure_distributions')
    selected=[]
    for role in ('dev_common','train_union'):
        for target in TARGETS:
            for row in sorted(groups[role],key=lambda r:-abs(r[target+'_error_nm'])):
                if row['gene_id'] not in {r['gene_id'] for r in selected}:
                    selected.append(row);break
    # Exact original renderer channels, shown as a material-color composite.
    palette=np.array([[.86,.93,1],[.85,.2,.25],[1,.65,.15],[.96,.96,.96],[.4,.45,.5],[.6,.4,.75],[.1,.1,.1]])
    fig,axes=plt.subplots(len(selected),3,figsize=(12,2.8*len(selected)),gridspec_kw={'width_ratios':[1.3,1,1.8]},layout='constrained')
    with torch.inference_mode():
        for index,row in enumerate(selected):
            b=np.array([int(c) for c in row['bits']],dtype=np.uint8)
            x=renderer(torch.from_numpy(b[None]).to(device)).cpu().numpy()[0]
            rgb=np.einsum('chw,cd->hwd',x[:7],palette);rgb=np.clip(rgb,0,1)
            axes[index,0].imshow(b.reshape(20,6).T,cmap=ListedColormap(['#dbeafe','#dc524d']),vmin=0,vmax=1,aspect='equal',origin='lower')
            axes[index,0].set(title=f'{row["gene_id"][:12]} | {row["split_role"]}/{row["source"]}',xlabel='Angular index',ylabel='Radial index')
            axes[index,1].imshow(rgb,origin='lower',extent=[0,90,0,64],aspect='auto');axes[index,1].set(title='Original Polar90 material input',xlabel='Physical angle (deg)',ylabel='Radius (mm)')
            axes[index,2].axis('off')
            lines=[f'PM cells: {row["magnet_cells"]}; nearest Hamming: {row["nearest_old_train_hamming"]}', 'Target       True       Pred       Error  (N m)']
            for target in TARGETS:
                lines.append(f'{target:8s} {row[target+"_true_nm"]:9.5f} {row[target+"_pred_nm"]:9.5f} {row[target+"_error_nm"]:+9.5f}')
            axes[index,2].text(0,.72,'\n'.join(lines),va='top',fontfamily='monospace',fontsize=9)
    fig.legend(handles=[Patch(facecolor=color,label=label) for color,label in zip(palette,
               ('Design air','PM inward','PM outward','Fixed air','Iron','Winding','Other'))],
               loc='upper center',bbox_to_anchor=(.5,-.005),ncol=7,fontsize=9)
    output(fig,'large_error_structures')
    write_csv(folder/'structure_cases.csv',selected)


def report():
    verified_physics_hash()
    a=read(HERE/'data_audit.json');m=read(HERE/'baseline_metrics.json');model=read(HERE/'model_manifest.json');consistency=read(HERE/'old_validation_consistency.json')
    def metric(role,target,source='all'):
        return next(r for r in m if r['role']==role and r['source']==source and r['target']==target)
    audit_text=f'''# FEMM 数据接收核验

可信完整基因 **{a['valid_genes']}/{a['expected_genes']}**，六角度记录 **{a['complete_angles']}**，失败/缺失/冲突 **{a['missing_or_conflicting_genes']}**。队列外记录和重复汇总均为 0。

逐记录哈希、清理凭据、基因 bits/ID、六角度、电流、六点标签重算以及 **{a['prepared_input_hashes_verified']} 份重建 FEM 输入哈希**全部通过。汇总最大差 {a['max_aggregate_difference_nm']:.3g} N·m，固定容差 1e-6 N·m。

G/F 各 1400、交集 101；公共 dev=200、test=400。训练/留出互斥、生成家族互斥、P 父代属于旧 train，完整已知历史 {a['history_unique_genotypes']} 个唯一基因与新样本无精确重复。未重新修正基因或选种子。

原始 FEM/ANS 已由对方清理，本轮依据完整保留记录和输入重建核验。先前两基因/12 角度的独立 FEMM 重算证据被复用；本次新 FEMM 调用为 0。

**修复后的配置状态：**本机 `femm_zone/femm_config.py` 已恢复为与结果包逐字节一致的冻结版本：内角 29～44°、倍率 1，电流 3.5 A 从零电角按机械行程四倍推进。历史首次导入发现过 0～15°/−2 的本机版本，`received/femm_config.diff` 与原导入清单记录的是当时状态。当前入口直接使用唯一配置模块，无额外电流覆盖。修复说明与独立核验状态见 [physics_fix_20260913.md](physics_fix_20260913.md)。

`data/` 是独立导入结果。`train_G`、`train_F`、`dev_common` 可被 SPMSMGeneDataset 直接读取；`sealed_test` 单独隔离。各目录 manifest.csv 保留原始选样次序、身份与来源元数据。labels_train_dev.csv 是训练/开发的共享底层标签，G/F 视图中的交集不产生第二个物理标签来源。原包与旧数据均未覆盖。
'''
    (HERE/'data_audit.md').write_text(audit_text,encoding='utf-8')
    lines=['# 冻结旧 CNN 的新增结构基线评估','',
        f"全量接收核验通过：{a['valid_genes']} 个基因、{a['complete_angles']} 个角度；G/F 两套训练数据均完整就绪。旧模型在原验证集上的推理与历史记录一致，但在新增结构上误差明显增加。",'',
        '本次未训练更新模型、未开发路由、未调用 FEMM、未预测旧 test 或公共新 test。', '',
        '## 数据和模型身份','',
        f"f0 为 polar90_224 / vgg16_v2 / seed_20260903，旧 train=40000，最佳 epoch=39。权重 SHA-256 `{model['checkpoint_sha256']}`。复用原 training.evaluate、8 通道 Polar90 renderer、通道顺序和 checkpoint 的目标均值/标准差；eval 模式，模型状态前后逐张量完全一致。",'',
        f"实际设备：{model['gpu'] or model['device']}，{model['dtype']}。共推理 9382 个唯一基因：旧 validation 6483、新训练并集 2699、公共 dev 200。G/F 的 101 个交集只推理一次。",'',
        f"旧 validation 与 history.json 第39轮的物理误差指标最大差 {consistency['max_physical_metric_difference_nm']:.3g} N·m（预设一致性容差1e-5）。比较的是旧验证记录，没有使用旧测试指标代替。",'',
        '## 关键误差','',
        '残差统一为预测−真实；所有误差单位 N·m，DeltaT 是六点峰峰差，非百分比。', '',
        '| 数据 | N | Tavg MAE | Tavg RMSE | Tavg P95 | DeltaT MAE | DeltaT RMSE | DeltaT P95 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for role in ALLOWED_VIEWS:
        t,d=metric(role,'tavg'),metric(role,'delta_t')
        lines.append(f"| {role} | {t['n']} | {t['mae_nm']:.6f} | {t['rmse_nm']:.6f} | {t['p95_abs_nm']:.6f} | {d['mae_nm']:.6f} | {d['rmse_nm']:.6f} | {d['p95_abs_nm']:.6f} |")
    lines+=['','公共 dev 与旧 validation 的 MAE 倍数：'+', '.join(f"{target} {metric('dev_common',target)['mae_nm']/metric('old_validation',target)['mae_nm']:.1f} 倍" for target in TARGETS)+'。这是本批结构与旧模型之间的初始差距，不是补样训练后的表现。','',
        '## 公共 dev 来源差异','', '| 来源 | N | Tavg MAE | Tavg 最大误差 | Tavg 偏差 | DeltaT MAE | DeltaT 最大误差 | DeltaT 偏差 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for source in 'ULBP':
        t,d=metric('dev_common','tavg',source),metric('dev_common','delta_t',source)
        lines.append(f"| {source} | {t['n']} | {t['mae_nm']:.6f} | {t['max_abs_nm']:.6f} | {t['bias_nm']:+.6f} | {d['mae_nm']:.6f} | {d['max_abs_nm']:.6f} | {d['bias_nm']:+.6f} |")
    for target in TARGETS:
        worst=max('ULBP',key=lambda s:metric('dev_common',target,s)['mae_nm'])
        lines+=['',f"按公共 dev 的 MAE，{target} 最困难来源为 {worst}（{metric('dev_common',target,worst)['mae_nm']:.6f} N·m）。"]
    cases=read_csv(HERE/'figures/structure_cases.csv')
    lines+=['','图中可见的具体误差：']
    for case,target in zip(cases[:2],TARGETS):
        lines+=['',f"- 公共 dev 的 {case['source']} 来源 `{case['gene_id'][:12]}`：{target} 真实值 {float(case[target+'_true_nm']):.5f}，预测 {float(case[target+'_pred_nm']):.5f}，偏差 {float(case[target+'_error_nm']):+.5f} N·m。"]
    lines+=['','这些例子表现为低转矩结构被高估、较高转矩波动被低估，是当前数据上的观察；仅凭本次基线不能确定其因果机制。']
    lines+=['','U=独立位置配置，L=小尺度相关结构，B=大尺度相关结构，P=旧训练父代的空间扰动。每来源仅50个 dev 样本，尾部与小差异不宜过度解释。P父代仍属于旧train，不能称为完全未见父代。','',
        '## G/F 样本组成与就绪情况','',
        'G 来源 U/L/B/P=680/617/51/52；F=281/355/567/197。各1400，交集101；各自选样顺序和独立视图保留。磁体面积按492区域参考网格中的设计区域面积求和，再按四扇区换算；不是直接把格数叫面积。最近 Hamming 复用经过原始清单哈希核验的结果，参考集仅为旧train40000。','',
        'G-S/G-E 将共享旧40000+train_G；F-S/F-E 将共享旧40000+train_F。数据和固定f0输入链路具备进入后续公平训练对照的条件，无缺失标签或缺失模型权重。首次导入时发现的本机配置差异已修复至冻结版本，已完成标签和CNN基线不需要重新计算；当前状态见 physics_fix_20260913.md。','',
        '旧模型在G或F上误差较高，只说明对应样本对旧模型更困难，不能据此判定哪种选样方法更有效。G的固定位置WL0/1仅编码局部材料和一圈邻接，没有显式强化桥接断裂、气隙或长程连接；F可能继承旧CNN的盲区。两者均不是已经验证的物理距离，Hamming也不保证精度。','',
        '## 图表与大误差案例','',
        '![真实值与预测值](figures/truth_prediction.png)','',
        '![公共dev按来源的绝对误差](figures/dev_source_errors.png)','',
        '![G/F组成与结构分布](figures/GF_structure_distributions.png)','',
        '![训练/dev大误差结构](figures/large_error_structures.png)','',
        '例图分别取公共dev及新训练并集的Tavg/DeltaT大误差结构；未从test挑选。逻辑图采用bits.reshape(20,6).T，材料图直接取原8通道renderer输出的材料彩色合成。详细身份与数值见 figures/structure_cases.csv。','',
        '## 下一阶段','',
        '先执行原定G/F两个更新模型的公平对照：相同架构、同一f0初始化、配对随机种子、相同优化和训练预算，使用共同dev并同时监测旧validation。E组复用同一fG/fF与冻结f0，不额外训练两套有利模型。路由阈值仅在旧validation与新dev开发，最终test继续封存。','',
        '每个更新模型计划40000+1400条训练样本，新增占3.38%，这是样本比例，不代表梯度贡献比例。本轮只测出初始差距，不能证明1400足够或不足。若后续检验样本量趋势，可用冻结顺序的嵌套700/1400子集，并保持设置可比；本次未执行。不能仅因旧模型误差大就扩充几千次FEMM，也不能宣称3299个样本覆盖全部结构空间。','',
        '完整指标含N、MAE、RMSE、中位数、P90/P95、最大误差、偏差、R²和Spearman，见 baseline_metrics.csv/.json。原始预测及残差在 baseline_predictions.csv；逐条FEMM证据通过完整gene_id可追溯到received归档。']
    (HERE/'BASELINE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    readme='''# 03 FEMM 接收与冻结 f0 基线

数据源：`../../../femm_zone/results/FEMM_results_20260913/`，原包保持不变。执行依据为同级 results 中的 `Codex_SPMSM_FEMM_Import_and_Baseline.md`。

从 em 根目录只读复核已有数据和基线（不重新推理）：

```powershell
python ./03_new_spmsm_project/experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/baseline.py verify
```

`verify` 把当前核验状态写入 `physics_fix_20260913_verification.json`，保留历史 `output_verification.json` 和 `OUTPUT_CHECKSUMS.json`。`audit`、`evaluate`、`report` 及 `all` 用于重新生成对应输出，会改写已有导入证据或基线，日常复核使用 `verify`。脚本不训练、不调用FEMM、不执行结果包里的脚本。

结果入口：`BASELINE_REPORT.md`；数据核验 `data_audit.md/.json`；来源 `import_manifest.json`；模型 `model_manifest.json`；旧验证一致性 `old_validation_consistency.json`；完整预测及指标 `baseline_predictions.csv`、`baseline_metrics.csv/.json`。

`data/train_G`、`data/train_F`、`data/dev_common` 各含 `topology_bits.npy`（N,6,20）、`targets_tavg_delta.npy`（N,2）、`gene_ids.npy` 和 `manifest.csv`，与现有 SPMSMGeneDataset 兼容。目标顺序Tavg、DeltaT，单位N·m；标签CSV保存原始双精度，训练数组沿用float32。G/F共享标签来自同一已核验底层版本。

导入本脚本后，`load_view('train_G')`、`load_view('train_F')`、`load_view('dev_common')`、`load_view('old_validation')` 返回原Dataset和索引。它显式拒绝test角色。最终test位于独立 `data/sealed_test/`，本阶段没有预测或误差分析。

`received/full_run_records.zip` 是原始记录归档的逐字节副本；每条记录在内存解包核验，避免落盘66000个小文件；核心contract和冻结清单另存于received。引用格式 `received/full_run_records.zip!/<完整gene_id>/angle_29/result.json`。

当前配置已恢复为冻结的 29° 初始内角、倍率 1、电流 3.5 A 从电角 0° 同步推进。输入重建直接调用唯一的 `femm_zone/femm_config.py`，额外覆盖电流的临时补丁已删除。详见 [配置修复与历史证据说明](physics_fix_20260913.md)。

原 `import_manifest.json`、`data_audit.md/.json`、`BASELINE_REPORT.md` 与 `received/femm_config.diff` 保留首次导入时的事实，其中“当前是0°/−2、尚待修复”指修复前状态。数据、预测、指标、原始校验清单均未因修复重写；原 `OUTPUT_CHECKSUMS.json` 中 `baseline.py` 和本 README 的哈希对应历史版本，新版本哈希记录在独立修复核验文件中。
'''
    (HERE/'README.md').write_text(readme,encoding='utf-8')
    print('REPORT written',flush=True)


def verify_local_sources(manifest):
    """Keep historical import hashes intact while allowing only the verified repair."""
    physical_hash = verified_physics_hash()
    require(manifest['contract']['physical_source_sha256'] == physical_hash and
            manifest['package_files']['source_snapshot/'+PHYSICS_SOURCE] == physical_hash,
            'Verified physical source is not the frozen import source')
    physical_item = None
    for name, item in manifest['frozen_local_inputs'].items():
        if name.replace('\\', '/') == PHYSICS_SOURCE:
            require(item['expected'] == physical_hash, 'Frozen physical source expectation differs')
            physical_item = item
        else:
            require(sha(PROJECT/name) == item['actual'], f'Original local source changed during import: {name}')
    require(physical_item is not None, 'Physical source missing from the import manifest')
    return {'historical_import_actual_sha256': physical_item['actual'],
            'frozen_expected_sha256': physical_hash, 'current_sha256': physical_hash,
            'matches_frozen_expected': True,
            'restored_since_import': physical_item['actual'] != physical_hash,
            'other_frozen_local_inputs_unchanged': True}


def verify_outputs():
    """Read-only checks of prior evidence; write a separate post-repair status record."""
    manifest=read(HERE/'import_manifest.json')
    for name,h in manifest['package_files'].items():
        require(sha(PACKAGE/name)==h,f'Original package changed: {name}')
    physical_status = verify_local_sources(manifest)
    current_entry_files = {}
    historical_output_changes = {}
    for name, expected in read(HERE/'OUTPUT_CHECKSUMS.json').items():
        actual = sha(HERE/name)
        if name in ('baseline.py', 'README.md'):
            current_entry_files[name] = {'historical_sha256': expected, 'current_sha256': actual}
        elif actual != expected:
            historical_output_changes[name] = {'historical_sha256': expected, 'current_sha256': actual}
    require(not historical_output_changes,
            f'Historical baseline outputs changed: {list(historical_output_changes)}')
    canonical=unique(read_csv(DATA/'labels_train_dev.csv'),'canonical labels')
    sealed=unique(read_csv(DATA/'sealed_test/labels.csv'),'sealed labels')
    predictions=unique(read_csv(HERE/'baseline_predictions.csv'),'predictions')
    _,old_split=pilot.old_data()
    for role,count in [('train_G',1400),('train_F',1400),('dev_common',200)]:
        dataset,indices=load_view(role)
        rows=read_csv(DATA/role/'manifest.csv')
        require(len(rows)==len(dataset)==len(indices)==count,'Dataset role count differs')
        require([r['gene_id'] for r in rows]==[r['gene_id'] for r in read_csv(PILOT/(role+'.csv'))],'Selection order changed')
        for row,bits,target in zip(rows,dataset.bits,dataset.targets):
            gid=row['gene_id']
            require(gid not in sealed and mapping.genotype_sha256(bits)==gid,'Dataset genotype/role mismatch')
            require(np.array_equal(target,np.array([canonical[gid]['tavg_nm'],canonical[gid]['delta_t_nm']],dtype=np.float32)),'Dataset target/identity mismatch')
    require(len(predictions)==9382 and not set(predictions)&set(sealed),'Final test isolation failed')
    old_rows=[r for r in predictions.values() if r['split_role']=='old_validation']
    require({int(r['old_index']) for r in old_rows}==set(old_split['validation'].tolist()),'Old validation indices differ')
    require(not {int(r['old_index']) for r in old_rows}&set(old_split['test'].tolist()),'Old test evaluated')
    for forbidden in ('test_common','sealed_test','old_test','test'):
        try:
            load_view(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError('Forbidden dataset loaded')
    for row in predictions.values():
        for target in TARGETS:
            require(abs(float(row[target+'_pred_nm'])-float(row[target+'_true_nm'])-float(row[target+'_error_nm']))<1e-12,'Residual sign/value differs')
    for item in read(HERE/'baseline_metrics.json'):
        role=item['role']
        rows=[r for r in predictions.values() if (r['split_role']=='old_validation' if role=='old_validation' else
              r['split_role']=='dev' if role=='dev_common' else r['split_role']=='train' if role=='train_union' else
              r['member_'+role]=='1') and (item['source']=='all' or r['source']==item['source'])]
        errors=[float(r[item['target']+'_error_nm']) for r in rows]
        require(len(errors)==item['n'],'Metric denominator mismatch')
        require(abs(sum(abs(e) for e in errors)/len(errors)-item['mae_nm'])<1e-12,'Metric MAE mismatch')
    for case in read_csv(HERE/'figures/structure_cases.csv'):
        require(case['gene_id'] in canonical and case['gene_id'] not in sealed,'Sealed test example plotted')
    save(HERE/'physics_fix_20260913_verification.json',{'status':'passed','new_training_views':[1400,1400],
         'dev_count':200,'sealed_test_count':400,'prediction_count':9382,'metric_denominators_verified':True,
         'test_guards_passed':True,'original_package_unchanged':True,'physical_source':physical_status,
         'historical_outputs_unchanged_except_current_entry_files':not historical_output_changes,
         'historical_output_changes':historical_output_changes,
         'current_entry_files':current_entry_files,'historical_output_checksums_sha256':sha(HERE/'OUTPUT_CHECKSUMS.json'),
         'new_femm_calls':0,'new_inference_calls':0,
         'note':'This verification reads existing evidence without re-running inference or replacing historical output_verification.json/OUTPUT_CHECKSUMS.json. See physics_fix_20260913.md.'})
    print('OUTPUT VERIFICATION passed; separate status: physics_fix_20260913_verification.json',flush=True)


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['audit','evaluate','report','verify','all'],default='all',nargs='?')
    args=parser.parse_args()
    if args.command in ('audit','all'):
        audit()
    if args.command in ('evaluate','all'):
        evaluate_baseline()
    if args.command in ('report','all'):
        report()
    if args.command in ('verify','all'):
        verify_outputs()
