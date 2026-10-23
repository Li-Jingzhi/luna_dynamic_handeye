"""Grouped optimization at observed discrete heads, not unseen-head interpolation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .fit_yellow_xyz_consistency import metrics, rbf_kernel
from .fit_yellow_xyz_v2 import support_summary, write_new
from .geometry import arm_mdh, flange_pose, rigid_fit
from .validate_pf_rt import load_capture, make_splits

SCHEMA = 'luna_discrete_multi_head_compensation_v1'


def register(camera, target, heads, train):
    transforms = {}
    for h in sorted(set(heads)):
        ids = train[heads[train] == h]
        if len(ids) < 12:
            raise ValueError('Need at least 12 training positions at each head')
        R, t = rigid_fit(target[ids], camera[ids])
        transforms[h] = dict(R=R.tolist(), t_m=t.tolist())
    return transforms


def baseline(transforms, camera, heads):
    result = np.empty_like(camera)
    for h in set(heads):
        if h not in transforms:
            raise ValueError('Unobserved discrete head')
        R, t = np.asarray(transforms[h]['R']), np.asarray(transforms[h]['t_m'])
        result[heads == h] = (camera[heads == h]-t)@R
    return result


def kernel(spec, camera, base, heads, known_camera, known_base, known_heads):
    kind, length, _ = spec
    shared = rbf_kernel(base, known_base, length) if kind in ('shared','additive') else 0.
    local = (rbf_kernel(camera, known_camera, length)*(heads[:,None] == known_heads[None,:])
             if kind in ('per_head','additive') else 0.)
    return shared+local


def fit_residual(spec, camera, target, heads, base, train):
    if spec[0] == 'rigid':
        return dict(kind='rigid')
    K = kernel(spec,camera[train],base[train],heads[train],camera[train],base[train],heads[train])
    alpha = np.linalg.solve(K+spec[2]*np.eye(len(train)),(target[train]-base[train])*1000.)
    return dict(kind=spec[0],spec=list(spec),centers_camera_m=camera[train].tolist(),
                centers_base_m=base[train].tolist(),heads=heads[train].tolist(),alpha_mm=alpha.tolist())


def residual(model, camera, base, heads):
    if model['kind'] == 'rigid':
        return np.zeros_like(camera)
    K = kernel(model['spec'],camera,base,heads,np.asarray(model['centers_camera_m']),
               np.asarray(model['centers_base_m']),np.asarray(model['heads']))
    return K@np.asarray(model['alpha_mm'])/1000.


def predict(model, camera_xyz, head_angles, *, head_velocity_deg_s=None):
    """Only certified discrete-head models; no q7 input or head interpolation."""
    if model.get('schema') != SCHEMA or not model.get('approved') or model.get('do_not_deploy',True):
        raise ValueError('Discrete-head model is not approved')
    velocity=np.asarray(head_velocity_deg_s,float)
    if velocity.shape!=(2,) or not np.isfinite(velocity).all() or np.max(abs(velocity))>.02:
        raise ValueError('Discrete-head inference requires measured stopped-head velocity')
    camera, head = np.asarray(camera_xyz,float), np.asarray(head_angles,float)
    if (camera.shape != (3,) or head.shape != (2,) or not np.isfinite(camera).all()
            or not np.isfinite(head).all() or camera[2] <= 0):
        raise ValueError('Expected finite camera XYZ and head angles')
    matches = [h for h in model['transforms'] if np.max(abs(head-np.array(h.split(','),float))) <= .05]
    if len(matches) != 1:
        raise ValueError('Unobserved discrete head; interpolation is not validated')
    h = matches[0]
    known = np.array(model['support_camera_by_head'][h])
    if np.min(np.linalg.norm(known-camera,axis=1))*1000 > model['support_radius_mm']:
        raise ValueError('Outside calibrated space at this head')
    H = np.array([h]); X=camera[None,:]
    base = baseline(model['transforms'],X,H)
    return (base+residual(model['residual'],X,base,H))[0]


def pair_diagnostics(rows, camera, heads):
    ids={h:np.where(heads==h)[0] for h in sorted(set(heads))}
    out=[]
    for a,b in itertools.combinations(ids,2):
        assert [rows[i]['group_id'] for i in ids[a]] == [rows[i]['group_id'] for i in ids[b]]
        A,B=camera[ids[a]],camera[ids[b]]
        error=np.empty_like(A)
        for test in np.array_split(np.arange(len(A)),5):
            train=np.setdiff1d(np.arange(len(A)),test)
            R,t=rigid_fit(A[train],B[train])
            error[test]=A[test]@R.T+t-B[test]
        out.append(dict(heads=[a,b],visual_only_trajectory_oof=metrics(error)))
    return out


def evaluate(capture, mdh, stereo, reference):
    rows, provenance = load_capture(capture,mdh,stereo,multi=True)
    manifest_path=Path(capture).with_suffix('.completed.json')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('complete') is not True or manifest['sha256'] != hashlib.sha256(Path(capture).read_bytes()).hexdigest():
            raise ValueError('Completed capture manifest mismatch')
        provenance['completed_manifest_verified']=True
        provenance['snapshot_is_final']=True
    ref=json.loads(Path(reference).read_text())
    for key in ('mdh_sha256','stereo_model_sha256'):
        if provenance[key] != ref[key]:
            raise ValueError('Frozen label reference mismatch: '+key)
    pf=np.asarray(ref['flange_to_ball_effective_m'],float)
    if pf.shape!=(3,) or not np.isfinite(pf).all() or np.linalg.norm(pf)>.2:
        raise ValueError('Invalid frozen effective pF')
    chain=arm_mdh(mdh); poses=np.array([flange_pose(r['q7_deg'],chain) for r in rows])
    camera=np.array([r['sphere_center_left_camera_m'] for r in rows]);heads=np.array([r['head_key'] for r in rows])
    target=poses[:,:3,3]+np.einsum('nij,j->ni',poses[:,:3,:3],pf)
    groups=list(dict.fromkeys(r['group_id'] for r in rows));cut=int(.8*len(groups))
    train_groups=set(groups[:cut]);train=np.array([i for i,r in enumerate(rows) if r['group_id'] in train_groups]);held=np.setdiff1d(np.arange(len(rows)),train)
    selection_rows=[rows[i] for i in train]
    splits=make_splits(selection_rows,poses[train])
    cached=[]
    for method,fold,tr,te in splits:
        transforms=register(camera,target,heads,train[tr]);base=baseline(transforms,camera,heads)
        cached.append((method,fold,train[tr],train[te],base))
    specs=[('rigid',0.,0.)]+[(kind,length,ridge) for kind in ('shared','per_head','additive')
                            for length in (.02,.05,.10) for ridge in (.1,1.,10.)]
    candidates=[]
    for spec in specs:
        scores={}
        for method,fold,tr,te,base in cached:
            model=fit_residual(spec,camera,target,heads,base,tr)
            error=base[te]+residual(model,camera[te],base[te],heads[te])-target[te]
            scores.setdefault(method,[]).append(metrics(error)['xyz_p95_mm'])
        selected_scores=sum([scores[k] for k in ('trajectory_block','spatial_block','configuration_block')],[])
        candidates.append(dict(spec=list(spec),fold_p95_mm=scores,worst_p95_mm=max(selected_scores),
                               mean_p95_mm=float(np.mean(selected_scores))))
    minimax=min(candidates,key=lambda c:(c['worst_p95_mm'],c['mean_p95_mm']))
    # Near-equal worst blocks are not meaningful evidence to sacrifice every
    # other region. Use training CV only; never choose from heldout labels.
    pool=[c for c in candidates if c['worst_p95_mm'] <= 1.05*minimax['worst_p95_mm']]
    winner=min(pool,key=lambda c:(c['mean_p95_mm'],c['worst_p95_mm']))
    spec=winner['spec']; transforms=register(camera,target,heads,train);base=baseline(transforms,camera,heads)
    correction=fit_residual(spec,camera,target,heads,base,train)
    error=base+residual(correction,camera,base,heads)-target
    support={h:support_summary(camera[train[heads[train]==h]],camera[held[heads[held]==h]]) for h in sorted(set(heads))}
    diagnostic=dict(selected=winner,train_groups=cut,heldout_groups=len(groups)-cut,
                    strict_minimax= minimax,
                    selection_rule='Minimum mean blocked-CV P95 within 5% of minimum worst block; training groups only',
                    train=metrics(error[train]),heldout_baseline=metrics((base-target)[held]),
                    heldout_corrected=metrics(error[held]),heldout_by_head={h:metrics(error[held[heads[held]==h]]) for h in sorted(set(heads))},
                    heldout_support_by_head=support,archived_data_only=True,
                    validation_scope='New arm positions at three observed discrete heads; no unseen-head interpolation')
    # All-point OOF is diagnostic after selection; each head of a point stays grouped.
    oof=[];oof_summary={}
    for method,fold,tr,te in make_splits(rows,poses):
        tf=register(camera,target,heads,tr);b=baseline(tf,camera,heads)
        m=fit_residual(spec,camera,target,heads,b,tr);e=b[te]+residual(m,camera[te],b[te],heads[te])-target[te]
        for i,d in zip(te,e):oof.append(dict(method=method,fold=fold,waypoint_index=rows[i]['waypoint_index'],head=heads[i],error_mm=(d*1000).tolist(),baseline_error_mm=((b[i]-target[i])*1000).tolist()))
    for method in sorted(set(r['method'] for r in oof)):
        e=np.array([r['error_mm'] for r in oof if r['method']==method])/1000
        oof_summary[method]=metrics(e)
    diagnostic['all_groups_oof_after_selection']=oof_summary
    diagnostic['matched_baseline_oof']={method:metrics(np.array([r['baseline_error_mm'] for r in oof if r['method']==method])/1000)
                                       for method in oof_summary}
    diagnostic['all_groups_oof_by_head']={method:{h:metrics(np.array([r['error_mm'] for r in oof if r['method']==method and r['head']==h])/1000)
                                                for h in sorted(set(heads))} for method in oof_summary}
    all_ids=np.arange(len(rows));tf=register(camera,target,heads,all_ids);b=baseline(tf,camera,heads)
    corr=fit_residual(spec,camera,target,heads,b,all_ids)
    model=dict(schema=SCHEMA,approved=False,do_not_deploy=True,transforms=tf,residual=corr,
               flange_to_ball_effective_m=pf.tolist(),support_radius_mm=20.,
               support_camera_by_head={h:camera[heads==h].tolist() for h in sorted(set(heads))},
               mdh_sha256=provenance['mdh_sha256'],stereo_model_sha256=provenance['stereo_model_sha256'],
               frame_id='right_arm_mdh_parent',units='m',head_tolerance_deg=.05,
               capture_sha256=provenance['snapshot_sha256'],reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
               diagnostic=diagnostic,approval_reason='Historical grouped diagnostics only; not independently certified')
    report=dict(provenance=provenance,samples=len(rows),groups=len(groups),frozen_pf_mm=(pf*1000).tolist(),
                candidates=candidates,diagnostic=diagnostic,visual_head_pair_diagnostics=pair_diagnostics(rows,camera,heads),
                final_in_sample=metrics(b+residual(corr,camera,b,heads)-target),oof=oof,
                selection_train_waypoints=list(dict.fromkeys(rows[i]['waypoint_index'] for i in train)),
                heldout_waypoints=list(dict.fromkeys(rows[i]['waypoint_index'] for i in held)),
                heldout_predictions=[dict(waypoint_index=rows[i]['waypoint_index'],head=heads[i],
                    error_mm=(error[i]*1000).tolist(),baseline_error_mm=((base[i]-target[i])*1000).tolist()) for i in held],
                label_reference_has_prior_historical_exposure=True,
                rejected_unseen_head_interpolation=True)
    return model,report


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',default=str(root/'data/ball_point_head_20261018_v1.jsonl'))
    parser.add_argument('--reference',default=str(root/'models/candidate_yellow_xyz_spatial_v2_final.json'))
    parser.add_argument('--mdh',default=str(root/'configs/luna_new.ini'))
    parser.add_argument('--stereo',default=str(root/'configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json'))
    parser.add_argument('--candidate',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();model,report=evaluate(args.capture,args.mdh,args.stereo,args.reference)
    write_new(args.candidate,model);write_new(args.report,report)
    print(json.dumps(report['diagnostic'],indent=2))


if __name__=='__main__':
    main()
