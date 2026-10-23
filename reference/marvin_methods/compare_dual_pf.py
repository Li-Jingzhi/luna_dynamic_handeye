"""Compare independent/shared/near-equal tool offsets with purged dual-arm validation."""
import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .compare_pro_calibration import (
    METHODS, PARAMETERS, camera_to_arm, describe_model, fit_model, model_predict,
    prepare, project, splits,
)
from .pro_capture import ROOT, digest
from .pro_fit import metrics

SIDES = ('right', 'left')
POLICIES = ('independent', 'shared', 'near_0p5mm')
RIGID = METHODS[:4]


def dataset(path, side):
    rows, poses, points, pixels, clouds, weights, P, manifest = prepare(path, arm=side)
    return dict(rows=rows, poses=poses, points=points, pixels=pixels, clouds=clouds,
                weights=weights, P=P, manifest=manifest, path=str(path.relative_to(ROOT)))


def subset(data, indices):
    return {**data, **{name: data[name][indices] for name in ('poses', 'points', 'pixels', 'weights')}}


def native_residual(data, model):
    predicted = model_predict(data['poses'], model)
    if model['method'] == 'rigid_pixel':
        return (project(predicted, data['P']) - data['pixels']).ravel()
    delta = predicted - data['points']
    if model['method'] == 'rigid_covariance':
        delta = np.einsum('nij,nj->ni', data['weights'], delta)
    return delta.ravel()


def fit_pair(data, method, policy, independent=None):
    """Each arm has its own camera map; shared/near policies couple only native pF."""
    if method not in METHODS or policy not in POLICIES:
        raise ValueError('Unsupported method or pF policy')
    if independent is None:
        independent = {side: fit_model(method, data[side]['poses'], data[side]['points'],
                       data[side]['pixels'], data[side]['weights'], data[side]['P']) for side in SIDES}
    if policy == 'independent':
        return dict(method=method, policy=policy, models=independent,
                    parameter_count=2*PARAMETERS[method])
    # Split per-arm geometric parameters from pF; retain log-scale / depth bias.
    external = 12 if method == 'affine' else 6
    extra = int(method in ('similarity', 'depth_bias'))
    block = external + extra
    def pack_external(model):
        x = np.asarray(model['parameters'])
        return np.r_[x[:external], x[9:10] if extra else []]
    mean_pf = np.mean([independent[s]['flange_to_ball_center_m'] for s in SIDES], axis=0)
    seed = np.r_[*(pack_external(independent[s]) for s in SIDES), mean_pf]
    if policy == 'near_0p5mm':
        seed = np.r_[seed, np.zeros(3)]
    lower, upper = np.full(len(seed), -np.inf), np.full(len(seed), np.inf)
    lower[2*block:2*block+3], upper[2*block:2*block+3] = -.2, .2
    if extra:
        for index in (block-1, 2*block-1):
            lower[index], upper[index] = ((np.log(.95), np.log(1.05)) if method == 'similarity' else (-.2, .2))
    if policy == 'near_0p5mm':
        # Each difference component <=0.5/sqrt(3) mm guarantees vector difference <=0.5mm.
        lower[-3:], upper[-3:] = -.0005/np.sqrt(3), .0005/np.sqrt(3)
    def unpack(x):
        center = x[2*block:2*block+3]
        delta = x[-3:] if policy == 'near_0p5mm' else np.zeros(3)
        models = {}
        for i, side in enumerate(SIDES):
            ext = x[i*block:(i+1)*block]
            pf = center + (.5 if side == 'right' else -.5)*delta
            parameters = np.r_[ext[:external], pf, ext[-1:] if extra else []]
            models[side] = dict(method=method, parameters=parameters.tolist(), flange_to_ball_center_m=pf.tolist())
        return models
    def residual(x):
        models = unpack(x)
        return np.concatenate([native_residual(data[s], models[s]) for s in SIDES])
    result = least_squares(residual, seed, bounds=(lower, upper), x_scale='jac',
                           loss='huber' if method == 'rigid_huber' else 'linear',
                           f_scale=.0007, max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'{method}/{policy}: {result.message}')
    models = unpack(result.x)
    for model in models.values():
        if np.linalg.norm(model['flange_to_ball_center_m']) > .2:
            raise ValueError('pF exceeds physical 200mm bound')
        describe_model(model)
    rank = int(np.linalg.matrix_rank(result.jac))
    if rank != len(seed):
        raise ValueError(f'Unidentifiable fit: rank {rank}/{len(seed)}')
    if np.any(result.active_mask[:2*block+3]):
        raise ValueError('Camera/scale/depth/pF magnitude bound reached')
    return dict(method=method, policy=policy, models=models, parameter_count=len(seed),
                optimizer_evaluations=int(result.nfev), jacobian_rank=rank,
                near_difference_bound_active=bool(np.any(result.active_mask[-3:])) if policy == 'near_0p5mm' else False,
                difference_bound_note='Near policy constrains each component to 0.5/sqrt(3)mm, a box inside a 0.5mm Euclidean ball.')


def flange_delta(data, model):
    """Score actual inverse-map flange errors, also for non-rigid candidates."""
    vision_ball = camera_to_arm(data['points'], model)
    poses = data['poses']
    return vision_ball - np.einsum('nij,j->ni', poses[:, :3, :3], model['flange_to_ball_center_m']) - poses[:, :3, 3]


def scores(data, candidate):
    return {side: metrics(flange_delta(data[side], candidate['models'][side])) for side in SIDES}


def key(candidate_id):
    return candidate_id.split('/')


def eligible(candidate_id, group):
    method, policy = key(candidate_id)
    return group == 'all' or (method in RIGID and (group == 'rigid' or policy == 'shared'))


def select(records, group='all'):
    allowed = [name for name in records if eligible(name, group)]
    # User priority: 1mm tail error in BOTH arms; avoid pooling hiding the worse arm.
    return min(allowed, key=lambda name: (max(records[name][s]['p95_mm'] for s in SIDES),
                                         max(records[name][s]['rms_mm'] for s in SIDES), name))


def fit_all(data):
    result = {}
    for method in METHODS:
        independent = fit_pair(data, method, 'independent')
        for policy in POLICIES:
            result[f'{method}/{policy}'] = independent if policy == 'independent' else fit_pair(
                data, method, policy, independent['models'])
    return result


def compare(data):
    ids = [f'{m}/{p}' for m in METHODS for p in POLICIES]
    groups = ('all', 'rigid', 'shared_rigid')
    indices = {s: np.arange(len(data[s]['points'])) for s in SIDES}
    outer = {s: list(splits(indices[s])) for s in SIDES}
    cv = {name: {s: np.empty_like(data[s]['points']) for s in SIDES} for name in ids}
    nested = {g: {s: np.empty_like(data[s]['points']) for s in SIDES} for g in groups}
    folds, fold_ids = [], {s: np.empty(len(indices[s]), int) for s in SIDES}
    for fold in range(5):
        train = {s: outer[s][fold][0] for s in SIDES}
        held = {s: outer[s][fold][1] for s in SIDES}
        inner = {s: list(splits(train[s], folds=4)) for s in SIDES}
        inner_errors = {name: {s: [] for s in SIDES} for name in ids}
        for j in range(4):
            inner_train = {s: subset(data[s], inner[s][j][0]) for s in SIDES}
            candidates = fit_all(inner_train)
            for name, candidate in candidates.items():
                for s in SIDES:
                    inner_errors[name][s].extend(flange_delta(subset(data[s], inner[s][j][1]), candidate['models'][s]))
            print(f'Outer {fold+1}/5 inner {j+1}/4 fitted 21 candidates', flush=True)
        inner_scores = {name: {s: metrics(np.array(inner_errors[name][s])) for s in SIDES} for name in ids}
        choices = {g: select(inner_scores, g) for g in groups}
        candidates = fit_all({s: subset(data[s], train[s]) for s in SIDES})
        fold_record = dict(fold=fold, selected=choices, inner_scores=inner_scores,
                           split={s:dict(train=train[s].tolist(), held=held[s].tolist()) for s in SIDES}, candidates={})
        for name, candidate in candidates.items():
            for s in SIDES:
                cv[name][s][held[s]] = flange_delta(subset(data[s], held[s]), candidate['models'][s])
            fold_record['candidates'][name] = candidate
        for s in SIDES:
            fold_ids[s][held[s]] = fold
            for g in groups:
                nested[g][s][held[s]] = cv[choices[g]][s][held[s]]
        folds.append(fold_record)
        print(f'Outer {fold+1}/5 selections: {choices}', flush=True)
    final = fit_all(data)
    results = {name: dict(candidate=candidate, fit=scores(data,candidate),
                          cv={s:metrics(cv[name][s]) for s in SIDES}) for name,candidate in final.items()}
    return dict(candidates=results, selected_by_full_cv={g:select({k:v['cv'] for k,v in results.items()},g) for g in groups},
                nested_selection={g:{s:metrics(nested[g][s]) for s in SIDES} for g in groups}, folds=folds), cv, fold_ids


def export_candidate(output, name, candidate, data, cv_stats):
    """Preserve both native maps; only truly rigid models get a geometric alignment."""
    directory = output / name.replace('/', '__'); directory.mkdir()
    registrations = {}
    for side in SIDES:
        model = dict(candidate['models'][side])
        model.update(schema='head_eye_pro_candidate_observation_model_v1', units='m',
                     input=f'FK flange pose in {side}-arm MDH parent',
                     output='archived rectified left-camera sphere-center measurement domain',
                     independently_verified=False, cv_metrics=cv_stats[side],
                     bounds=dict(camera_min_m=data[side]['points'].min(0).tolist(), camera_max_m=data[side]['points'].max(0).tolist()),
                     source_capture_sha256=data[side]['manifest']['files'][0]['uncompressed_sha256'],
                     mdh_sha256=digest(ROOT/'configs/1004.ini'),
                     source_calibration_sha256=digest(ROOT/'configs/orin_pro_raw_fisheye_stereo_20261019.json'),
                     stereo_model_sha256=digest(ROOT/'configs/orin_pro_rectified_stereo_20261019.json'),
                     pf_policy=candidate['policy'], right_anchor_fixed=False,
                     validation_note='Per-candidate CV uses no held-out points; selected candidate score has selection bias, see nested_selection.')
        (directory/f'{side}.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
        registrations[side]=model
    if candidate['method'] in RIGID:
        matrices={s:np.asarray(registrations[s]['arm_to_camera_T_before_depth_mapping']) for s in SIDES}
        T=np.linalg.inv(matrices['right'])@matrices['left']
        result=dict(right_mdh_from_left_mdh_T=T.tolist(), right_anchor_fixed=False, independently_verified=False,
                    note='Both extrinsics re-estimated jointly; this is a candidate, not the deployed fixed-right-anchor alignment.')
        (directory/'alignment.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--right',type=Path,default=ROOT/'datasets/drag_replay_20261020_091535_901340')
    parser.add_argument('--left',type=Path,default=ROOT/'datasets/left_drag_replay_20261020_110645_673779')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output directory')
    data={s:dataset(getattr(args,s),s) for s in SIDES}
    args.output.mkdir(parents=True)
    result, cv, fold_ids=compare(data)
    result.update(schema='dual_arm_shared_pf_comparison_v1',
                  data={s:dict(path=data[s]['path'],manifest=data[s]['manifest']) for s in SIDES},
                  source_script_sha256=digest(Path(__file__)),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  metric='Inverse camera map to native MDH, subtract Q*pF, compare FK flange origin; mm.',
                  selection='Minimum worst-arm P95, then worst-arm RMS; all positions retained.',
                  validation='5 paired outer trajectory blocks, purge 2 neighboring valid positions; 4 inner blocks in each outer training set; both arms held out simultaneously.',
                  assumptions=['Same tool geometry and local flange mounting orientation, confirmed by user.',
                               'Equal pF remains a physical hypothesis; identical MDH rows do not prove mounting/zero-point equality.',
                               'Camera intrinsics/stereo/MDH are fixed; no new image calibration or absolute ground truth.',
                               'Right/left extrinsics re-estimated as candidates; deployment fixed-right anchor is not overwritten.',
                               'Latest sessions only. No extra residual filtering, no joint-angle-dependent runtime correction.'])
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.output/'heldout_flange_errors.csv').open('w',newline='') as stream:
        fields=['candidate','arm','waypoint_index','fold','dx_mm','dy_mm','dz_mm','error_mm']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for name in cv:
            for s in SIDES:
                for i,delta in enumerate(cv[name][s]*1000):
                    writer.writerow(dict(candidate=name,arm=s,waypoint_index=data[s]['rows'][i]['waypoint_index'],fold=int(fold_ids[s][i]),
                                         dx_mm=delta[0],dy_mm=delta[1],dz_mm=delta[2],error_mm=np.linalg.norm(delta)))
    for name,record in result['candidates'].items():
        export_candidate(args.output,name,record['candidate'],data,record['cv'])
    print(json.dumps(dict(selected=result['selected_by_full_cv'],nested=result['nested_selection']),indent=2),flush=True)


if __name__=='__main__': main()
