"""Diagnostic chain closure with bounded joint-zero perturbations; not physical accuracy validation."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,apply,pose,parameters,torso_from_arm
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.calibration import metrics
from dynamic_handeye.optimizer import robust_optimize

OUT=Path(__file__).resolve().parent


def corrected_fk(q,x,assets):
    offsets=np.r_[0.,2*np.tanh(x[6:])]
    return apply(torso_from_arm(assets)@assets.flange(q+offsets),assets.pf)


def fit(c,q,assets):
    target=apply(torso_from_arm(assets)@assets.flange(q),assets.pf)
    r,t=rigid_fit(c,target);T=np.eye(4);T[:3,:3]=r;T[:3,3]=t
    initial=np.r_[parameters(T),np.zeros(6)]
    def residual(x):
        data=(apply(pose(x[:6]),c)-corrected_fk(q,x,assets))*1000
        # Fix joint0 to remove its base-rotation gauge; regularize remaining zeros.
        return np.r_[data.ravel(),2*np.tanh(x[6:])/.5]
    return robust_optimize(residual,initial,iterations=180)


def main():
    assets=Assets();report=dict(approved=False,pf_m=assets.pf.tolist(),
        note='Both external pose and six bounded joint zeros estimated. Corrected-FK residual changes the reference and is not improved vision accuracy against original FK. Offsets are effective hypotheses, not measured corrections.',datasets={})
    for name,path in [('A',ROOT/'datasets/raw/luna_v2_transferred_marvin360_fixed25_20261028.jsonl'),
                      ('B',ROOT/'datasets/raw/luna_v2_validationB_yaw0_pitchm25_20261029.jsonl'),
                      ('new',ROOT/'datasets/new_capture_20261029/shared_fast_grid_20261029/head_00.jsonl')]:
        data=load_capture([path],assets);rows=sorted(data['rows'],key=lambda r:r['waypoint_id'])
        c=np.array([assets.triangulate(r['corrected_pixels']) for r in rows]);q=np.array([r['q7_deg'] for r in rows]);nominal=np.array([r['target_s_m'] for r in rows])
        x=fit(c,q,assets);ids=np.arange(len(rows));closure=[];original=[];fold_offsets=[]
        for held in np.array_split(ids,5):
            train=ids[(ids<held[0]-2)|(ids>held[-1]+2)];xf=fit(c[train],q[train],assets)
            pred=apply(pose(xf[:6]),c[held]);closure.extend(pred-corrected_fk(q[held],xf,assets));original.extend(pred-nominal[held])
            fold_offsets.append(np.r_[0.,2*np.tanh(xf[6:])].tolist())
        report['datasets'][name]=dict(source=data['sources'],effective_offsets_deg=np.r_[0.,2*np.tanh(x[6:])].tolist(),fold_offsets_deg=fold_offsets,
            corrected_fk_training=metrics(apply(pose(x[:6]),c)-corrected_fk(q,x,assets)),
            corrected_fk_blocked_closure=metrics(closure),original_frozen_fk_blocked_error=metrics(original))
        print(name,json.dumps(report['datasets'][name]),flush=True)
    (OUT/'joint_zero_probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
