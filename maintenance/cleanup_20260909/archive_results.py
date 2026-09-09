"""Export legacy scalar results and consolidate small run records before deletion."""
import collections
import csv
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / '01_legacy_topopt_prototype/python_topopt/runs'
original = json.loads((OUT / 'inventory_before.json').read_text(encoding='utf-8'))['files']
groups = collections.defaultdict(list)
for item in original:
    parts = Path(item['path']).parts
    if len(parts) > 4 and parts[:3] == ('01_legacy_topopt_prototype', 'python_topopt', 'runs'):
        if parts[4] in ('samples', 'checkpoints') and Path(item['path']).suffix.lower() not in ('.fem', '.ans', '.bak', '.log', '.pyc', '.pyo'):
            groups[parts[3]].append(item)

members, summary = [], []
for run in sorted(RUNS.iterdir()):
    if not run.is_dir():
        continue
    database = run / 'dataset.sqlite'
    record = {'run': run.name, 'tables': {}, 'archive_files': 0}
    if database.is_file():
        connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        exports = run / 'result_exports'
        exports.mkdir(exist_ok=True)
        for table in ('samples', 'angle_evaluations', 'candidates'):
            cursor = connection.execute('SELECT * FROM ' + table)
            columns = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            with (exports / (table + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(rows)
            record['tables'][table] = len(rows)
        record['complete_angles'] = connection.execute("SELECT COUNT(*) FROM angle_evaluations WHERE torque IS NOT NULL").fetchone()[0]
        record['sample_statuses'] = dict(connection.execute('SELECT status, COUNT(*) FROM samples GROUP BY status'))
        connection.close()
    if groups[run.name]:
        archive = run / 'run_details.zip'
        assert not archive.exists(), archive
        checks = []
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for item in groups[run.name]:
                source = ROOT / item['path']
                data = source.read_bytes()
                assert len(data) == item['bytes']
                name = source.relative_to(run).as_posix()
                z.writestr(name, data)
                checks.append({**item, 'sha256': hashlib.sha256(data).hexdigest(),
                               'archive': archive.relative_to(ROOT).as_posix(), 'member': name})
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for item in checks:
                assert hashlib.sha256(z.read(item['member'])).hexdigest() == item['sha256']
        members.extend(checks)
        record.update(archive_files=len(checks), archive_bytes=archive.stat().st_size,
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    summary.append(record)
    print(json.dumps(record), flush=True)

(OUT / 'archive_members.json').write_text(json.dumps(members, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'result_archives.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
