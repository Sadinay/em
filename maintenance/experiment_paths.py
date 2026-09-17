"""Explicit path-only relocation for the three frozen CNN experiments.
Historical source fingerprints remain valid only while relocated source bytes match
this migration's recorded hash. Training/config/checkpoint contents are not rewritten.
"""
from pathlib import Path
import hashlib
import json
from functools import lru_cache
PROJECT = Path(__file__).resolve().parents[1] / '03_new_spmsm_project'
NAMES = ('cnn_replay_update_v1','cnn_replay_update_v2','cnn_shrink_perturb_v1')
@lru_cache(None)
def manifest(name):
    return json.loads((PROJECT/'experiments'/name/'data/layout_migration.json').read_text(encoding='utf-8'))
def experiment(path):
    p=Path(path).resolve()
    return next(q for q in (p,*p.parents) if q.name in NAMES)
def relocated(path):
    p=Path(path)
    if not p.is_absolute(): p=p.absolute()
    for name in NAMES:
        root=PROJECT/'experiments'/name
        if not p.is_relative_to(root): continue
        rel=p.relative_to(root).as_posix()
        for old,new in manifest(name)['rules']:
            if rel==old or rel.startswith(old+'/'):
                return root/new/rel[len(old):].lstrip('/')
    return p
def audit_dir(model):
    root=experiment(model)
    return root/'data'/Path(model).relative_to(root/'models')/'audit'
def report_dir(model):
    root=experiment(model)
    suffix=Path(model).relative_to(root/'models')
    return root/'reports'/('logical6x20' if str(suffix)=='.' else suffix)
def artifact_sha(path):
    p=relocated(path)
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in NAMES:
        root=PROJECT/'experiments'/name
        if not p.is_relative_to(root):continue
        record=manifest(name)['relocated_sources'].get(p.relative_to(root).as_posix())
        if record:
            if actual!=record['current_sha256']:
                raise RuntimeError('Frozen relocated source changed: '+str(p))
            return record['original_sha256']
    return actual
