"""Fixed-view disparity + residual optimization; archived data are diagnostic only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .depth_compensation import correct, fit as fit_depth
from .fit_yellow_xyz_consistency import metrics, rbf_kernel
from .fit_yellow_xyz_v2 import (SCHEMA, correction_m, spatial_folds, support_summary,
                              write_new, _distance2, fit_correction)
from .geometry import arm_mdh, flange_pose, rigid_fit, validate_stereo
from .validate_pf_rt import load_capture


def fit_stage(camera, target, ids, spec, fb):
    if spec is None:
        R, t = rigid_fit(target[ids], camera[ids])
        return None, R, t, (camera-t)@R
    depth, R, t = fit_depth(camera[ids], target[ids], fb, *spec)
    return depth, R, t, (correct(depth, camera)-t)@R


def fit_local(camera, error, ids, spec, *, quality=None):
    if spec is None:
        return {'kind': 'rigid'}
    if spec[0] == 'gp':
        from .gaussian_process import fit
        return fit(camera[ids], error[ids],
                   quality=quality[ids] if spec[1] else None)
    if spec[0] == 'poly':
        return fit_correction(camera, error, np.zeros_like(error), ids,
                              ('poly', spec[1], (), 0, spec[2]))
    length, ridge = spec
    K = rbf_kernel(camera[ids], camera[ids], length)
    coef = np.linalg.solve(K+ridge*np.eye(len(ids)), error[ids]*1000.)
    return dict(kind='local_rbf', length_m=length, ridge=ridge,
                centers_m=camera[ids].tolist(), coefficients_mm=coef.tolist())


def evaluate(capture, mdh, stereo, reference, *, include_gp=False):
    # Effective point is a versioned pre-existing label definition, never refitted
    # for a candidate or a test fold. Historical label provenance is disclosed.
    frozen = json.loads(Path(reference).read_text())
    rows, provenance = load_capture(capture, mdh, stereo)
    if len(rows) < 80:
        raise ValueError('Need >=80 valid independent positions')
    for key in ('mdh_sha256', 'stereo_model_sha256'):
        if frozen[key] != provenance[key]:
            raise ValueError('Frozen reference mismatch: '+key)
    raw = [json.loads(s) for s in Path(capture).read_text().splitlines() if s.strip()]
    metadata = next(r for r in raw if r.get('capture_valid'))
    if np.max(abs(np.array([r['head_angles_deg'] for r in rows])-frozen['head_angles_deg'])) > .05:
        raise ValueError('Frozen reference head pose mismatch')
    pf = np.asarray(frozen['flange_to_ball_effective_m'], float)
    if pf.shape != (3,) or not np.isfinite(pf).all() or np.linalg.norm(pf) > .2:
        raise ValueError('Invalid frozen pF')
    chain = arm_mdh(mdh)
    poses = np.array([flange_pose(r['q7_deg'], chain) for r in rows])
    target = poses[:, :3, 3]+np.einsum('nij,j->ni', poses[:, :3, :3], pf)
    camera = np.array([r['sphere_center_left_camera_m'] for r in rows])
    quality = None
    if include_gp:
        quality = np.array([raw[r['source_line']-1].get('sphere_center_p95_jitter_mm', float('nan'))
                            for r in rows], float)
        if not np.isfinite(quality).all() or np.any(quality <= 0):
            raise ValueError('GP quality comparison requires positive point jitter proxies')
    n = len(rows); cut = int(.8*n); train = np.arange(cut); held = np.arange(cut,n)
    stereo_model = json.loads(Path(stereo).read_text())
    Kl, Kr, Rrl, trl = validate_stereo(stereo_model)
    if not (np.allclose(Kl, Kr) and np.allclose(Rrl, np.eye(3)) and
            np.allclose(trl[1:], 0) and trl[0] < 0):
        raise ValueError('Disparity compensation requires horizontal rectified stereo')
    fb = stereo_model['left_K'][0][0]*abs(stereo_model['right_from_left_T_m'][0])
    folds = [('temporal', x) for x in np.array_split(train,5)]
    folds += [('spatial', x) for x in spatial_folds(camera[train],5)]
    depth_specs = [None]+[(order,ridge) for order in (0,1,2) for ridge in (.1,10.)]
    local_specs = [None]+[(length,ridge) for length in (.02,.04,.08,.16) for ridge in (.1,1.,10.)]
    local_specs += [('poly',order,ridge) for order in (1,2) for ridge in (.1,10.,100.)]
    if include_gp:
        local_specs += [('gp',False),('gp',True)]
    candidates = []
    for depth_spec in depth_specs:
        cached=[]
        try:
            for typ,test in folds:
                tr=np.setdiff1d(train,test)
                depth,R,t,base=fit_stage(camera,target,tr,depth_spec,fb)
                cached.append((typ,test,tr,base))
        except (ValueError,np.linalg.LinAlgError) as exc:
            candidates.append(dict(depth_spec=depth_spec,failed=str(exc)))
            continue
        for local_spec in local_specs:
            if local_spec is not None and local_spec[0] == 'gp' and depth_spec is not None:
                continue
            scores=[]
            for typ,test,tr,base in cached:
                residual=fit_local(camera,target-base,tr,local_spec,quality=quality)
                error=base[test]+correction_m(residual,camera[test])-target[test]
                scores.append(metrics(error)['xyz_p95_mm'])
            candidates.append(dict(depth_spec=depth_spec,local_spec=local_spec,
                                   fold_p95_mm=scores,worst_p95_mm=max(scores),
                                   mean_p95_mm=float(np.mean(scores))))
    usable=[c for c in candidates if 'failed' not in c]
    if not usable:
        raise ValueError('No model completed validation')
    # Frozen rule: worst spatial/time fold, then average. Never tune on holdout.
    winner=min(usable,key=lambda c:(c['worst_p95_mm'],c['mean_p95_mm']))
    depth_spec,local_spec=winner['depth_spec'],winner['local_spec']
    depth,R,t,base=fit_stage(camera,target,train,depth_spec,fb)
    local=fit_local(camera,target-base,train,local_spec,quality=quality)
    prediction=base+correction_m(local,camera)
    _,_,_,baseline=fit_stage(camera,target,train,None,fb)
    distances=np.sqrt(_distance2(camera[held],camera[train]).min(1))*1000
    inside=distances<=20
    random_oof=np.empty((cut,3))
    for test in np.array_split(np.random.default_rng(20261018).permutation(train),5):
        tr=np.setdiff1d(train,test)
        _,_,_,b=fit_stage(camera,target,tr,depth_spec,fb)
        local_fold=fit_local(camera,target-b,tr,local_spec,quality=quality)
        random_oof[test]=b[test]+correction_m(local_fold,camera[test])-target[test]
    diagnostic=dict(selected_spec=winner,cv_temporal_fold_p95_mm=winner['fold_p95_mm'][:5],
                    cv_spatial_fold_p95_mm=winner['fold_p95_mm'][5:],
                    fixed_chronological_train=metrics(prediction[train]-target[train]),
                    fixed_chronological_heldout_baseline=metrics(baseline[held]-target[held]),
                    fixed_chronological_heldout_corrected=metrics(prediction[held]-target[held]),
                    fixed_chronological_holdout_support=support_summary(camera[train],camera[held]),
                    fixed_chronological_heldout_inside_support=metrics((prediction-target)[held][inside]) if inside.any() else None,
                    random_point_oof_diagnostic=metrics(random_oof),
                    fixed_chronological_holdout_has_prior_exposure=True)
    all_ids=np.arange(n)
    depth,R,t,base=fit_stage(camera,target,all_ids,depth_spec,fb)
    local=fit_local(camera,target-base,all_ids,local_spec,quality=quality)
    residual=(dict(kind='depth_then_residual',depth=depth,camera_from_mdh_R=R.tolist(),residual=local)
              if depth is not None else local)
    diagnostic['final_candidate_in_sample_after_refit_all']=metrics(base+correction_m(local,camera)-target)
    if local_spec is not None and local_spec[0] == 'gp':
        from .gaussian_process import predict
        selection_local=fit_local(camera,target-fit_stage(camera,target,train,depth_spec,fb)[3],
                                  train,local_spec,quality=quality)
        _, std=predict(selection_local,camera[held],with_std=True)
        diagnostic['heldout_predictive_std_mm_median_axes']=np.median(std,axis=0).tolist()
        diagnostic['uncertainty_is_model_dependent_not_accuracy_guarantee']=True
    model=dict(schema=SCHEMA,approved=False,do_not_deploy=True,
               algorithm='fixed_view_ard_gp_comparison_v1' if include_gp else 'fixed_view_disparity_residual_v1',
               approval_reason='Historical capture only; independent new trajectory required',
               capture=str(Path(capture).resolve()),capture_sha256=hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
               capture_valid_samples=n,fit_samples=n,selection_fit_samples=cut,
               candidate_refit_all_capture_points=True,frame_id='right_arm_mdh_parent',units='m',
               mdh_sha256=provenance['mdh_sha256'],stereo_model_sha256=provenance['stereo_model_sha256'],
               source_calibration_sha256=metadata['source_calibration_sha256'],
               pixel_domain=metadata['pixel_domain'],image_size=metadata['image_size'],sensor_ids=metadata['sensor_ids'],
               sphere_center_method=metadata['sphere_center_method'],head_angles_deg=frozen['head_angles_deg'],
               flange_to_ball_effective_m=pf.tolist(),camera_from_mdh_R=R.tolist(),camera_from_mdh_t_m=t.tolist(),
               residual=residual,support_radius_mm=20.,support_points_camera_m=camera.tolist(),
               training_q7_deg_validation_only=[r['q7_deg'] for r in rows],diagnostic=diagnostic,
               label_reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
               label_reference=str(reference),label_reference_has_prior_capture_exposure=True)
    report=dict(provenance=provenance,frozen_pf_mm=(pf*1000).tolist(),
                candidates=candidates,diagnostic=diagnostic,model_approved=False,
                validation_note='Historical diagnostic, not independent certification; effective pF frozen from existing historical model',
                heldout_predictions=[dict(waypoint_index=rows[i]['waypoint_index'],
                    error_mm=float(np.linalg.norm(prediction[i]-target[i])*1000),
                    distance_to_training_mm=float(distances[j])) for j,i in enumerate(held)])
    return model,report


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--mdh',default=str(root/'configs/luna_new.ini'))
    parser.add_argument('--stereo',default=str(root/'configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json'))
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--include-gp',action='store_true',help='Compare ARD GP with uniform and point-jitter proxy noise')
    args=parser.parse_args()
    model,report=evaluate(args.capture,args.mdh,args.stereo,args.reference,include_gp=args.include_gp)
    write_new(args.report,report);write_new(args.candidate,model)
    print(json.dumps(report['diagnostic'],indent=2))


if __name__=='__main__':
    main()
