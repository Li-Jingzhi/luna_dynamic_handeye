"""Matched fixed-pF static diagnostics; keeps original Pro/Luna references separate."""
from pathlib import Path
import configparser
import gzip
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,mdh_poses,digest
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.calibration import metrics

OUT=Path(__file__).resolve().parent


def predict(train_camera,train_target,test_camera,method):
    if method=='rigid':
        r,t=rigid_fit(train_camera,train_target)
        return test_camera@r.T+t,{}
    design=np.c_[train_target,np.ones(len(train_target))]
    seed=np.linalg.lstsq(design,train_camera,rcond=None)[0]
    if method=='affine':mapping=seed;weights=None
    else:
        def kernel(a,b):return np.exp(-np.sum((a[:,None]-b[None])**2,axis=2)/(2*.06**2))
        k=kernel(train_camera,train_camera)
        inv=np.linalg.solve(k+3*np.eye(len(k)),np.c_[design,train_camera])
        mapping=np.linalg.solve(design.T@inv[:,:4],design.T@inv[:,4:])
        weights=np.linalg.solve(k+3*np.eye(len(k)),design@mapping-train_camera)
    corrected=test_camera if weights is None else test_camera+kernel(test_camera,train_camera)@weights
    m=mapping[:3].T;sv=np.linalg.svd(m,compute_uv=False)
    return np.linalg.solve(m,(corrected-mapping[3]).T).T,dict(singular_values=sv.tolist(),
        within_Pro_scale_bounds=bool(sv.min()>=.9 and sv.max()<=1.1),
        within_Pro_element_seed_bounds=bool(np.max(abs(mapping[:3]-seed[:3]))<=.03))


def evaluate(c,t):
    result={}
    # Ordered valid positions, five contiguous blocks, purge two neighbouring valid positions.
    ids=np.arange(len(c))
    for method in ('rigid','affine','joint_camera_gaussian_l60_a3'):
        fit,diagnostic=predict(c,t,c,method);errors=[];bounds=[]
        for test in np.array_split(ids,5):
            train=ids[(ids<test[0]-2)|(ids>test[-1]+2)]
            assert len(train)>=18 and not set(train)&set(test)
            pred,info=predict(c[train],t[train],c[test],method)
            errors.extend(pred-t[test]);bounds.append(info)
        result[method]=dict(training=metrics(fit-t),blocked_cv=metrics(errors),training_map=diagnostic,fold_maps=bounds)
    return result


def main():
    a=Assets();report=dict(pf_m=[0,0,.101],note='Fixed-pF single-arm restriction of Pro camera-domain variable projection, not its full free-pF dual-arm model. Pro101 is a sensitivity assumption, not a precision measurement. No residual-based filtering; original quality rejects excluded.',datasets={})
    pro=ROOT.parent/'head_eye_pro';source=pro/'datasets/drag_replay_20261020_091535_901340/replay_capture.jsonl.gz'
    raw=gzip.decompress(source.read_bytes());reg=json.loads((pro/'configs/orin_pro_ball_registration_20261020_288.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==reg['capture_uncompressed_sha256']
    rows=sorted([json.loads(x) for x in raw.splitlines() if json.loads(x).get('capture_valid')],key=lambda r:r['waypoint_index'])
    ini=configparser.ConfigParser();ini.read(pro/'configs/1004.ini');sections=[f'R.A1.L{i}.DH' for i in range(7)]+['R.A1.FLANGE']
    chain=np.array([[float(ini[s][k]) for k in ('A','Alpha','D','Theta')] for s in sections]);poses=mdh_poses(chain,np.array([r['q7_deg'] for r in rows]))
    assert digest(pro/'configs/1004.ini')==reg['mdh_sha256']
    for p,r in zip(poses,rows):
        assert np.allclose(p[:3,:3],r['flange_rotation'],atol=1e-12)
        assert np.allclose(p[:3,3],r['flange_position_m'],atol=1e-12)
    c=np.array([r['sphere_center_left_camera_m'] for r in rows]);t=poses[:,:3,3]+np.einsum('nij,j->ni',poses[:,:3,:3],a.pf)
    report['datasets']['Pro_right_288_fixed101_diagnostic']=dict(n=len(c),depth_range_m=[float(c[:,2].min()),float(c[:,2].max())],source_sha256=hashlib.sha256(raw).hexdigest(),scores=evaluate(c,t))
    paths=dict(Luna_A_0m25=ROOT/'datasets/raw/luna_v2_transferred_marvin360_fixed25_20261028.jsonl',
               Luna_B_0m25=ROOT/'datasets/raw/luna_v2_validationB_yaw0_pitchm25_20261029.jsonl',
               Luna_new_0m25=ROOT/'datasets/new_capture_20261029/shared_fast_grid_20261029/head_00.jsonl')
    for name,path in paths.items():
        data=load_capture([path],a);rows=sorted(data['rows'],key=lambda r:r['waypoint_id'])
        c=np.array([a.triangulate(r['corrected_pixels']) for r in rows]);t=np.array([r['target_s_m'] for r in rows])
        report['datasets'][name]=dict(n=len(c),depth_range_m=[float(c[:,2].min()),float(c[:,2].max())],sources=data['sources'],scores=evaluate(c,t))
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,d in report['datasets'].items():
        print(name,d['n'],[(k,round(s['training']['p95_mm'],3),round(s['blocked_cv']['p95_mm'],3)) for k,s in d['scores'].items()])


if __name__=='__main__':main()
