"""Freeze joint-zero hypotheses per source trajectory; align each target camera pose diagnostically."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reports.acquisition_error_audit_20261029.joint_zero_probe import corrected_fk
from dynamic_handeye.geometry import Assets
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.calibration import metrics

def main():
    a=Assets();out=Path(__file__).resolve().parent
    probes=json.loads((out/'joint_zero_probe.json').read_text());data={}
    for name,path in [('A','datasets/raw/luna_v2_transferred_marvin360_fixed25_20261028.jsonl'),('B','datasets/raw/luna_v2_validationB_yaw0_pitchm25_20261029.jsonl'),('new','datasets/new_capture_20261029/shared_fast_grid_20261029/head_00.jsonl')]:
        rows=sorted(load_capture([ROOT/path],a)['rows'],key=lambda r:r['waypoint_id'])
        data[name]=(np.array([a.triangulate(r['corrected_pixels']) for r in rows]),np.array([r['q7_deg'] for r in rows]))
    result={}
    for source,probe in probes['datasets'].items():
        offsets=np.array(probe['effective_offsets_deg']);x=np.r_[np.zeros(6),np.arctanh(offsets[1:]/2)]
        for target,(c,q) in data.items():
            t=corrected_fk(q,x,a);r,v=rigid_fit(c,t);ids=np.arange(len(c));errors=[]
            for held in np.array_split(ids,5):
                train=ids[(ids<held[0]-2)|(ids>held[-1]+2)];R,T=rigid_fit(c[train],t[train]);errors.extend(c[held]@R.T+T-t[held])
            result[source+'->'+target]=dict(target_camera_pose_fit=metrics(c@r.T+v-t),target_pose_blocked_cv=metrics(errors),source_offsets_deg=offsets.tolist(),source_equals_target=source==target)
    report=dict(approved=False,results=result,note='Joint offsets frozen from named source. Camera rigid pose re-estimated within target; blocked CV prevents target pose fit leakage but does not validate physical offsets, common camera bias or original FK. Diagonal uses full target-derived offsets and is diagnostic only.')
    (out/'transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    for k,v in result.items():print(k,round(v['target_pose_blocked_cv']['p95_mm'],3))

if __name__=='__main__':main()
