"""Inventory an explicitly scoped legacy cleanup; does not delete files."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = '01_legacy_topopt_prototype/'
OBSOLETE = OLD + 'early_experiments/cnn_flow_validation/'
files = json.loads((OUT / 'inventory_before.json').read_text(encoding='utf-8'))['files']
delete, retained = [], []
cache_names = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}

for item in files:
    relative = item['path']
    path = ROOT / relative
    parts = Path(relative).parts
    if '.git' in parts or relative.startswith('maintenance/'):
        continue
    suffix = path.suffix.lower()
    reason = None
    if relative.startswith(OBSOLETE):
        reason = 'explicitly_deprecated_conv1d_component'
    elif cache_names.intersection(parts) or suffix in ('.pyc', '.pyo'):
        reason = 'regenerable_python_cache'
    elif relative.startswith(OLD):
        if suffix == '.ans':
            reason = 'legacy_femm_solution'
        elif suffix == '.fem':
            generated_area = relative.startswith((OLD+'python_topopt/runs/', OLD+'python_topopt/outputs/'))
            generated_name = path.name.lower().startswith(('case_', 'best_', 'tmp_', 'temp_'))
            worker_copy = any(p.startswith('w') and p[1:].isdigit() for p in parts)
            if generated_area or generated_name or worker_copy:
                reason = 'legacy_generated_femm_model'
        elif suffix in ('.node', '.ele', '.edge', '.pbc', '.poly'):
            reason = 'legacy_solver_mesh_intermediate'
        elif suffix == '.bak' and path.with_suffix('').is_file() and '/python_topopt/' in relative:
            reason = 'superseded_run_backup_with_current_file_present'
        elif suffix == '.log' and '/python_topopt/runs/' in relative:
            reason = 'legacy_worker_log'
        elif path.name == 'active.lock' and '/python_topopt/' in relative:
            reason = 'inactive_run_lock'
    if reason:
        assert path.is_file() and path.stat().st_size == item['bytes']
        delete.append({**item, 'reason': reason})
    elif relative.startswith(OLD):
        # Preserve actual results, original templates, and all remaining source assets.
        retained.append({**item, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

counts = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
for item in delete:
    counts[item['reason']]['files'] += 1
    counts[item['reason']]['bytes'] += item['bytes']
for name, data in [('delete_manifest.json', delete), ('retained_hashes.json', retained),
                   ('cleanup_plan.json', {'categories': dict(counts), 'retained_files': len(retained)})]:
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'categories': dict(counts), 'retained_files': len(retained),
                  'delete_bytes': sum(x['bytes'] for x in delete)}, indent=2))
