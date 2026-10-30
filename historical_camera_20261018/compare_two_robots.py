"""Compare two user-confirmed Luna V2.0 units using identical fixed-head algorithms."""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from compensation import predict,folds,run_group,SPECS
from dynamic_handeye.geometry import Assets,digest
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.calibration import metrics

EXACT=('affine_rbf',.02,.01)


def fixed_spec(rows,a):
    rows=sorted(rows,key=lambda r:r['waypoint_id'])
    c=np.array([a.triangulate(r['corrected_pixels']) for r in rows])
    t=np.array([r['target_s_m'] for r in rows]);q=np.array([r['q7_deg'] for r in rows]);ids=np.arange(len(c))
    cv={}
    for protocol in ('trajectory','spatial','configuration'):
        errors=[]
        for train,test in folds(ids,protocol,t,q,5):
            errors.extend(predict(c[train],t[train],c[test],EXACT)-t[test])
        cv[protocol]=metrics(errors)
    return dict(spec=EXACT,training=metrics(predict(c,t,c,EXACT)-t),fixed_spec_cv=cv,
                depth_range_m=[float(c[:,2].min()),float(c[:,2].max())],
                camera_span_mm=(np.ptp(c,axis=0)*1000).tolist(),fk_span_mm=(np.ptp(t,axis=0)*1000).tolist(),
                joint_span_deg=np.ptp(q,axis=0).tolist())


def main():
    output=ROOT/'reports/two_luna_comparison_20261030';output.mkdir(exist_ok=True)
    a=Assets()
    groups=[]
    sources=[('trajectory_A',p) for p in [
        ROOT/'datasets/raw/luna_v2_transferred_marvin360_fixed25_20261028.jsonl',
        ROOT/'datasets/raw/luna_v2_transferred_marvin360_fixed20_20261028.jsonl',
        ROOT/'datasets/raw/luna_v2_transferred_marvin360_yaw10_pitchm20_20261028.jsonl',
        ROOT/'datasets/raw/luna_v2_transferred_marvin360_yaw10_pitchm25_20261028.jsonl']]
    sources += [('trajectory_B',p) for p in sorted((ROOT/'datasets/raw').glob('luna_v2_validationB_yaw*_pitchm*_20261029.jsonl'))]
    sources += [('trajectory_C_partial_grid',p) for p in sorted((ROOT/'datasets/new_capture_20261029/shared_fast_grid_20261029').glob('head_0?.jsonl'))]
    report=dict(machine_identity_basis='User confirmed 2026-10-30: old data/parameters from another same-model Luna V2.0. No hardware serial numbers supplied.',
        same_model='V2.0FSK_M6SLC-615-B-V1.0',pf_m=[0,0,.101],joint_zero_correction_deg=[0]*7,
        exact_comparison_spec=EXACT,candidate_grid=SPECS,old_robot=[],current_robot=[],
        note='Same algorithm/parameters, different trajectories, support and acquisition conditions; not a controlled hardware comparison.')
    archive=json.loads((HERE/'RESTORE_MANIFEST.json').read_text())
    for entry in archive['files']:
        if digest(HERE/'snapshot'/entry['path'])!=entry['sha256']:raise ValueError('Archive hash mismatch')
    old=Assets(HERE/'snapshot')
    assert old.hashes['mdh']==a.hashes['mdh']
    assert np.array_equal(a.pf,old.pf)
    d=load_capture([HERE/'snapshot/datasets/raw/ball_point_head_20261018_v1.jsonl',HERE/'snapshot/datasets/raw/ball_auto_20261018_visible_v1.jsonl'],old)
    grouped={}
    for row in d['rows']:grouped.setdefault((row['session_id'],row['head_key']),[]).append(row)
    previous=json.loads((HERE/'compensation_results.json').read_text())
    for _,rows in sorted(grouped.items()):
        cached=next(g for g in previous['groups'] if g['session']==rows[0]['session_id'] and g['head']==rows[0]['head_key'])
        report['old_robot'].append(dict(assets=old.hashes,analysis=cached,exact=fixed_spec(rows,old)))
    for batch,path in sources:
        d=load_capture([path],a)
        for head in sorted({r['head_key'] for r in d['rows']}):
            rows=[r for r in d['rows'] if r['head_key']==head]
            print('CURRENT',batch,path.name,head,len(rows),flush=True)
            result=run_group(rows,a)
            exact=fixed_spec(rows,a)
            report['current_robot'].append(dict(batch=batch,source=str(path.relative_to(ROOT)),sha256=digest(path),
                assets=a.hashes,rejected=d['rejected'],analysis=result,exact=exact))
            print('EXACT',head,'fit',exact['training']['p95_mm'],'CV',exact['fixed_spec_cv']['trajectory']['p95_mm'],flush=True)
        (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if len(report['current_robot'])!=12:raise ValueError('Expected four heads each from A, B and new partial grid')


if __name__=='__main__':main()
