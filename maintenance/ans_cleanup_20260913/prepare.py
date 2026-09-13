"""Prepare a bounded ANS-only cleanup after verifying and backing up scalar results."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECTS = [ROOT / '02_new_ipmsm_project', ROOT / '03_new_spmsm_project']
PROTECTED = {
    '02_new_ipmsm_project/femm_6point_validation/corrected_pm_direction/angles_3deg/angle_000/model.ans': '02 CNN build_lookup.py reference mesh',
    '03_new_spmsm_project/femm_zone/results/history_replay_validation_minangle25/final_high_tavg/angle_0/model.ans': '03 CNN lookup and HD export reference mesh',
}
BASELINE = '03_new_spmsm_project/femm_zone/workspaces/teacher_angle29_5genes_20260908/'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    require(not (HERE / 'execution.json').exists(), 'Cleanup already executed; preserve original audit')
    all_files = [p for root in PROJECTS for p in root.rglob('*') if p.is_file()]
    answers = sorted(p for p in all_files if p.suffix.lower() == '.ans')
    plan, keep, errors, scalar_rows = [], [], [], []
    backup = HERE / 'saved_angle_results.zip'
    before = {root.name: sum(p.stat().st_size for p in all_files if p.is_relative_to(root)) for root in PROJECTS}
    # Fingerprint every retained file by path/length/time. Hash all result JSONs separately below.
    snapshot = [{'path': relative(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                for p in all_files if p.suffix.lower() != '.ans']
    (HERE / 'retained_files_before.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    with zipfile.ZipFile(backup, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in answers:
            rel = relative(path)
            resolved = path.resolve(strict=True)
            require(resolved == path and any(resolved.is_relative_to(root.resolve()) for root in PROJECTS), f'Unsafe target: {path}')
            if rel in PROTECTED or rel.startswith(BASELINE):
                keep.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path),
                             'reason': PROTECTED.get(rel, '03 historical G1-G5 report/hash validation; G2 also supplies pilot graph')})
                continue
            result = path.with_name('result.json')
            model = path.with_suffix('.fem')
            try:
                data = json.loads(result.read_text(encoding='utf-8-sig'))
                require(data.get('status', 'complete') == 'complete', 'Unsuccessful result')
                fields = [k for k in ('raw_torque_nm', 'raw_gap_torque_nm', 'torque_nm_raw', 'torque_nm') if k in data]
                require(len(fields) == 1, 'Missing or ambiguous raw torque')
                torque = data[fields[0]]
                require(isinstance(torque, (int, float)) and math.isfinite(torque), 'Nonfinite torque')
                require(model.is_file(), 'Missing retained FEM input')
                if 'ans_path' in data:
                    require(Path(data['ans_path']).resolve() == resolved, 'Result belongs to another ANS path')
                for key in ('ans_size_bytes', 'ans_size_bytes_before_optional_cleanup'):
                    if key in data:
                        require(data[key] == path.stat().st_size, 'ANS size differs from result')
                digest = sha(path)
                if 'ans_sha256' in data:
                    require(data['ans_sha256'] == digest, 'ANS hash differs from saved result')
                model_hash = sha(model)
                # verified_input.fem_sha256 is PRE-solve in the old runners;
                # FEMM normalizes the file during analyze. Only top-level hashes
                # describe the completed model. Independently check the actual
                # solution header against the retained FEM for every deletion.
                expected_model_hash = data.get('fem_sha256')
                if expected_model_hash:
                    require(model_hash == expected_model_hash, 'FEM hash differs from saved result')
                header = []
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        if line.strip() == '[Solution]':
                            break
                        header.append(line)
                    else:
                        raise ValueError('ANS has no solution section')
                require(''.join(header).strip() == model.read_text(encoding='utf-8').strip(), 'ANS model header differs from retained FEM')
                entry = {'path': rel, 'bytes': path.stat().st_size, 'sha256': digest,
                         'result_path': relative(result), 'result_sha256': sha(result),
                         'model_path': relative(model), 'model_sha256': model_hash,
                         'torque_field': fields[0], 'raw_torque_nm': torque,
                         'ans_header_matches_retained_fem': True}
                plan.append(entry)
                archive.write(result, relative(result))
                scalar_rows.append({'ans_path': rel, 'result_path': relative(result), 'raw_torque_field': fields[0],
                                    'raw_torque_nm': torque, 'ans_sha256': digest, 'result_sha256': entry['result_sha256']})
            except Exception as exc:
                errors.append({'path': rel, 'reason': str(exc)})
    with zipfile.ZipFile(backup) as archive:
        require(archive.testzip() is None, 'Backup CRC mismatch')
        for entry in plan:
            require(hashlib.sha256(archive.read(entry['result_path'])).hexdigest() == entry['result_sha256'], 'Backup result hash mismatch')
    with (HERE / 'saved_torques.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    info = {'workspace': str(ROOT), 'allowed_roots': [str(p) for p in PROJECTS], 'before_bytes': before,
            'delete': plan, 'keep': keep, 'unverified_keep': errors,
            'backup_path': str(backup), 'backup_sha256': sha(backup),
            'saved_torques_sha256': sha(HERE / 'saved_torques.csv'),
            'total_delete_bytes': sum(e['bytes'] for e in plan)}
    (HERE / 'plan.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps({'ans_found': len(answers), 'delete_count': len(plan), 'delete_gib': info['total_delete_bytes']/1024**3,
                      'protected_count': len(keep), 'unverified_count': len(errors), 'unverified_examples': errors[:3],
                      'counts_by_project': dict(Counter(e['path'].split('/')[0] for e in plan)), 'before_bytes': before}, indent=2))


if __name__ == '__main__':
    main()
