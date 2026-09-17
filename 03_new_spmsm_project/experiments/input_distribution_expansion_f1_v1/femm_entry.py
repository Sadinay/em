"""Portable 03 FEMM adapter; reuses pilot solver, retries, locks and six-process scheduler.
Default is status only. Test jobs/results live in a separate sealed_test tree.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
JOB_ROOT=PROJECT/'femm_zone/workspaces/f1e1'
sys.path.insert(0,str(PROJECT))
from experiments.input_distribution_pilot_v1 import pilot as core

def verify_bundle():
    manifest=core.read(HERE/'task_manifest.json')
    for n,h in manifest['files'].items():core.require(core.sha(PROJECT/n)==h,'Frozen package dependency changed: '+n)
    for n,h in manifest['physics_sources'].items():core.require(core.sha(PROJECT/n)==h,'FEMM physics source drift: '+n)
    return manifest

def portable_physics():
    cfg=core.physical_config()
    cfg['mat_file']='data_zone/raw/workspace_200.mat';cfg['template_file']='femm_zone/models/SPMSM_discrete.fem'
    cfg['input_sha256']={cfg['mat_file']:core.sha(core.physical.MAT_FILE),cfg['template_file']:core.sha(core.physical.TEMPLATE_FILE)}
    return cfg

def contract():
    import importlib.metadata
    manifest=verify_bundle();cfg=portable_physics();core.require(core.digest(cfg)==manifest['portable_physics_fingerprint'],'Frozen physical condition mismatch')
    solver=core.solver_identity();solver.pop('executable') # path differs across devices; actual binary hash/version must match for reuse
    return {'physical':cfg,'portable_physics_fingerprint':core.digest(cfg),'template_sha256':core.sha(core.physical.TEMPLATE_FILE),'mat_sha256':core.sha(core.physical.MAT_FILE),'mapping_source_sha256':core.sha(core.mapping.__file__),'physical_source_sha256':core.sha(core.physical.__file__),'runner_source_sha256':core.sha(core.__file__),'adapter_sha256':core.sha(__file__),'solver':solver,'pyfemm_version':importlib.metadata.version('pyfemm')}

def validate_manifest():
    verify_bundle()
    for name in ('femm_queue.csv','pilot20.csv'):
        path=core.HERE/name
        core.require(path.exists(),'Missing partition queue '+str(path))
    return {'status':'frozen_task_manifest_verified'}

def regression_contract():
    # Historical .ans were intentionally removed. Require exact accepted physics/template/MAT hashes,
    # then require pilot20 on this device before full work; never pretend to have rerun old solutions.
    verify_bundle();contract()

core.solve_contract=contract
core.check_seed_manifest=validate_manifest
core.regression_reuse=regression_contract

def setup(part):
    core.HERE=JOB_ROOT/('te' if part=='sealed_test' else 'td')
    # FEMM itself may retain legacy Windows MAX_PATH behavior even when Python supports long paths.
    core.require(len(str(core.HERE.resolve()))+170<260,'FEMM working path too long; extract the package under a short path such as C:\\em')
    return core.HERE

def pilot_passed():
    setup('train_dev');c=contract();root=core.HERE/'femm_runs'/core.digest(c)
    rows=core.csv_read(core.HERE/'pilot20.csv')
    result=core.aggregate_results(root,rows,c,write_tables=False)
    core.require(len(result)==20 and all(r['status']=='succeeded' for r in result),'Complete and inspect the 20-gene pilot before all/test solving')

def status(include_test=False):
    verify_bundle();out={}
    for part in (['train_dev','sealed_test'] if include_test else ['train_dev']):
        job=setup(part);entry=job/'femm_entry.json';record={'genes':9000 if part=='train_dev' else 1000,'prepared_genes':0,'succeeded_angles':0,'attempts':0,'failed_angles':0,'labels_read':0}
        if entry.exists():
            en=core.read(entry);root=job/en['run_directory'];c=core.read(root/'contract.json');core.require(core.digest(c)==en['condition_fingerprint'],'Stored contract changed')
            for r in core.csv_read(job/'femm_queue.csv'):
                prepared=False
                for a in c['physical']['inner_angles_deg']:
                    p=root/r['gene_id']/f'angle_{a:g}'/'state.json'
                    if p.exists():
                        st=core.read(p);core.require(st['gene_id']==r['gene_id'] and st['bits']==r['bits'] and st['condition_fingerprint']==en['condition_fingerprint'],'State identity mismatch');prepared=True;record['succeeded_angles']+=int(st['status']=='succeeded');record['attempts']+=len(st['attempts']);record['failed_angles']+=int(st['status']=='failed')
                record['prepared_genes']+=int(prepared)
        out[part]=record
    print(json.dumps(out,ensure_ascii=False,indent=2));return out

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','solve','resume','status','export'],nargs='?',default='status');p.add_argument('--scope',choices=['pilot','all','test'],default='pilot');p.add_argument('--workers',type=int,default=6);p.add_argument('--include-test-status',action='store_true');a=p.parse_args()
    verify_bundle()
    if a.command=='status':status(a.include_test_status);return
    part='sealed_test' if a.scope=='test' else 'train_dev';scope='all' if a.scope=='test' else a.scope
    if a.command in ('solve','resume') and a.scope in ('all','test'):pilot_passed()
    job=setup(part)
    if a.command=='prepare':core.prepare_cases(scope)
    elif a.command in ('solve','resume'):core.solve_queue(scope,workers=a.workers)
    elif a.command=='export':
        core.require(a.scope!='test','Test labels remain sealed; no default test export command')
        entry=core.read(job/'femm_entry.json');root=job/entry['run_directory'];c=core.read(root/'contract.json')
        with core.batch_lock(root,'export'):core.aggregate_results(root,core.csv_read(job/'femm_queue.csv'),c)

if __name__=='__main__':main()
