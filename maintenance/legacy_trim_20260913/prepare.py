"""Reduce retired 01 storage: exact duplicate documents/images, retired weights, zipped databases."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import sqlite3
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / '01_legacy_topopt_prototype'


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not (HERE / 'execution.json').exists()
    original = [p for p in ROOT.rglob('*') if p.is_file()]
    before = sum(p.stat().st_size for p in original)
    entries, databases = [], []
    def add(p, reason, evidence=None):
        assert p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink()
        entries.append({'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p),
                        'reason': reason, 'evidence': evidence or {}})
    # Only duplicate presentation assets/documents; preserve source and MAT/FEM paths.
    groups = defaultdict(list)
    for p in original:
        if p.is_relative_to(ROOT / 'FP') and p.suffix.lower() in {'.png','.jpg','.pdf','.doc','.docx','.pptx','.drawio','.fig'}:
            groups[sha(p)].append(p)
    for digest, paths in groups.items():
        paths.sort(key=lambda p: (len(p.parts), len(str(p)), str(p)))
        for p in paths[1:]:
            add(p, 'exact_duplicate_asset', {'retained_copy': str(paths[0].resolve()), 'sha256': digest})
    for p in original:
        if p.suffix == '.pt':
            evidence = {}
            for name in ['metrics.json','predictions.csv','training_history.csv']:
                q = p.parent / name
                assert q.is_file(), str(q)
                evidence[str(q.resolve())] = sha(q)
            add(p, 'retired_training_weights_results_retained', evidence)
        if p.suffix == '.asv' and p.with_suffix('.m').is_file():
            add(p, 'matlab_autosave_current_source_retained', {'source': str(p.with_suffix('.m').resolve())})
        if p.name == '.poly':
            add(p, 'legacy_solver_mesh_intermediate')
    for p in original:
        if p.suffix != '.sqlite':
            continue
        wal = p.with_name(p.name + '-wal')
        assert not wal.exists() or wal.stat().st_size == 0, f'Uncheckpointed database: {p}'
        db = sqlite3.connect(p.resolve().as_uri() + '?immutable=1', uri=True)
        try:
            assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        finally:
            db.close()
        digest = sha(p)
        target = p.with_suffix('.sqlite.zip')
        assert not target.exists(), f'Archive already exists: {target}'
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(p, p.name)
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None and hashlib.sha256(z.read(p.name)).hexdigest() == digest
        assert sha(p) == digest
        evidence = {'archive': str(target.resolve()), 'archive_sha256': sha(target), 'member': p.name}
        add(p, 'database_losslessly_archived', evidence)
        for suffix in ('-wal','-shm'):
            sidecar = p.with_name(p.name + suffix)
            if sidecar.exists():
                add(sidecar, 'inactive_database_sidecar', evidence)
        databases.append({'path': str(target.resolve()), 'bytes': target.stat().st_size, 'sha256': sha(target)})
    assert len({e['path'] for e in entries}) == len(entries)
    deleted = {e['path'] for e in entries}
    retained = [{'path': str(p.resolve()), 'sha256': sha(p)} for p in original if str(p.resolve()) not in deleted]
    plan = {'root': str(ROOT.resolve()), 'before_bytes': before, 'delete': entries, 'archives': databases,
            'retained': retained, 'deleted_bytes': sum(e['bytes'] for e in entries),
            'archive_bytes': sum(e['bytes'] for e in databases)}
    (HERE / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'delete_files':len(entries), 'categories':dict(Counter(e['reason'] for e in entries)),
                      'before_mib':before/1024**2, 'after_estimated_mib':(before-plan['deleted_bytes']+plan['archive_bytes'])/1024**2,
                      'archive_mib':plan['archive_bytes']/1024**2}, indent=2))


if __name__ == '__main__':
    main()
