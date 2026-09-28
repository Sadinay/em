from pathlib import Path
import json, statistics
from datetime import datetime, timezone, timedelta
root=Path(r'C:\Users\26096\Desktop\em\03_new_spmsm_project\ga_zone\data\nsga2_100gen_manual_20260922\femm_results')
rows=[]
for p in root.rglob('state.json'):
    try:
        d=json.loads(p.read_text(encoding='utf-8'))
        for a in d.get('attempts',[]):
            if a.get('status')=='succeeded' and a.get('started_utc') and a.get('elapsed_seconds') is not None:
                st=datetime.strptime(a['started_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                rows.append((st,float(a['elapsed_seconds']),p))
    except Exception:
        pass
starts=[r[0] for r in rows]; ends=[r[0]+timedelta(seconds=r[1]) for r in rows]; el=[r[1] for r in rows]
genes=len(list(root.rglob('label.json')))
wall=(max(ends)-min(starts)).total_seconds()
print('angles',len(rows),'genes',genes)
print('start_utc',min(starts).isoformat(),'end_utc',max(ends).isoformat(),'wall_seconds',wall)
print('sum_solver_seconds',sum(el),'mean_angle_sec',statistics.mean(el),'median_angle_sec',statistics.median(el),'p95',sorted(el)[int(.95*(len(el)-1))])
print('mean_gene_solver_sec',sum(el)/genes)
print('direct_5540_sum_solver_hours',sum(el)/genes*5540/3600)
print('observed_wall_hours',wall/3600)
print('estimated_direct_wall_hours',wall/3600*5540/601)
print('saved_pct',100*(1-601/5540))
