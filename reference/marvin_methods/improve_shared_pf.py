"""Strictly shared tool offset with a jointly fitted, common smooth observation field.

Experimental: fit both arm maps, one pF, and one Gaussian field. This does not
identify camera distortion separately from robot/model/detector errors.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from .compare_dual_pf import SIDES, dataset, fit_pair, flange_delta, subset
from .compare_pro_calibration import model_predict, splits
from .pro_capture import ROOT, digest
from .pro_fit import metrics


# Fixed before the nested run; earlier exploratory reuse is recorded in the report.
VARIANTS = {
    **{f'affine_l{int(length*1000)}_a{alpha:g}': ('affine', length, alpha)
       for length in (.03, .04, .06, .1) for alpha in (.3, 1., 3., 10.)},
    **{f'rigid_l{int(length*1000)}_a{alpha:g}': ('rigid_l2', length, alpha)
       for length in (.04, .1) for alpha in (1., 10.)},
}


def kernel(points, centers, length):
    if not np.isfinite(length) or length <= 0:
        raise ValueError('Kernel length must be positive and finite')
    return np.exp(-cdist(points, centers, 'sqeuclidean')/(2*length**2))


def _points(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError('Expected finite Nx3 positive-depth camera observations')
    return points


def camera_to_arm(points, model, side):
    """Generic target inference: no joint angles or tool offset are used."""
    if side not in SIDES:
        raise ValueError('Unknown arm')
    points = _points(points)
    T = np.asarray(model['observation_from_mdh_T'][side], float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('Invalid arm map')
    singular = np.linalg.svd(T[:3, :3], compute_uv=False)
    if singular.min() < .9 or singular.max() > 1.1:
        raise ValueError('Invalid arm-map scale')
    if model['correction_kind'] == 'gaussian':
        centers = _points(model['kernel_centers_m'])
        coefficients = np.asarray(model['kernel_coefficients_m'], float)
        if coefficients.shape != centers.shape or not np.isfinite(coefficients).all():
            raise ValueError('Invalid common-field coefficients')
        corrected = points + kernel(points, centers, model['kernel_length_m']) @ coefficients
    elif model['correction_kind'] == 'none':
        corrected = points
    else:
        raise ValueError('Unknown observation correction')
    if not np.isfinite(corrected).all() or np.any(corrected[:, 2] <= 0):
        raise ValueError('Correction produced invalid depth')
    return np.linalg.solve(T[:3, :3], (corrected-T[:3, 3]).T).T


def flange_errors(data, model, side):
    poses = data['poses']
    return (camera_to_arm(data['points'], model, side)-poses[:, :3, 3]
            -np.einsum('nij,j->ni', poses[:, :3, :3], model['shared_pf_m']))


def _maps(x, method):
    block = 12 if method == 'affine' else 6
    maps = {}
    for i, side in enumerate(SIDES):
        T = np.eye(4)
        if method == 'affine':
            T[:3, :3] = x[i*block:i*block+9].reshape(3, 3)
            T[:3, 3] = x[i*block+9:i*block+12]
        else:
            T[:3, :3] = Rotation.from_rotvec(x[i*block:i*block+3]).as_matrix()
            T[:3, 3] = x[i*block+3:i*block+6]
        maps[side] = T
    return maps


def plain_model(pair):
    method = pair['method']
    block = 12 if method == 'affine' else 6
    x = np.concatenate([pair['models'][s]['parameters'][:block] for s in SIDES])
    return dict(correction_kind='none', map_kind=method,
                observation_from_mdh_T={s:T.tolist() for s,T in _maps(x, method).items()},
                shared_pf_m=pair['models']['right']['flange_to_ball_center_m'])


def fit(data, variant, seed=None):
    method, length, alpha = VARIANTS[variant]
    if seed is None:
        seed = fit_pair(data, method, 'shared')
    block = 12 if method == 'affine' else 6
    start = 2*block
    centers = np.concatenate([_points(data[s]['points']) for s in SIDES])
    K = kernel(centers, centers, length)
    L = cholesky(K+alpha*np.eye(len(K)), lower=True)
    x0 = np.r_[*(seed['models'][s]['parameters'][:block] for s in SIDES),
               seed['models']['right']['flange_to_ball_center_m']]
    lower, upper = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    lower[start:], upper[start:] = -.2, .2
    if method == 'affine':
        for i in range(2):
            lower[i*block:i*block+9] = x0[i*block:i*block+9]-.03
            upper[i*block:i*block+9] = x0[i*block:i*block+9]+.03

    def raw(x):
        maps = _maps(x, method)
        errors = []
        for s in SIDES:
            poses = data[s]['poses']
            ball = poses[:, :3, 3]+np.einsum('nij,j->ni', poses[:, :3, :3], x[start:])
            T = maps[s]
            errors.append(ball@T[:3, :3].T+T[:3, 3]-data[s]['points'])
        return np.concatenate(errors)

    def residual(x):
        # Variable projection: eliminates all field coefficients exactly.
        return (solve_triangular(L, raw(x), lower=True)/.001).ravel()

    def jacobian(x):
        maps = _maps(x, method)
        J = np.zeros((len(K), 3, len(x)))
        offset = 0
        for i, s in enumerate(SIDES):
            poses = data[s]['poses']
            Q = poses[:, :3, :3]
            ball = poses[:, :3, 3]+np.einsum('nij,j->ni', Q, x[start:])
            n = len(ball)
            for axis in range(3):
                J[offset:offset+n, axis, i*block+axis*3:i*block+axis*3+3] = ball
                J[offset:offset+n, axis, i*block+9+axis] = 1
            J[offset:offset+n, :, start:] = np.einsum('ab,nbc->nac', maps[s][:3, :3], Q)
            offset += n
        return (solve_triangular(L, J.reshape(len(K), -1), lower=True)
                .reshape(len(K)*3, len(x))/.001)

    optimized = least_squares(residual, x0, jac=jacobian if method == 'affine' else '2-point',
                              bounds=(lower, upper), x_scale='jac', max_nfev=150,
                              ftol=1e-9, xtol=1e-9, gtol=1e-9)
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise RuntimeError(f'{variant}: {optimized.message}')
    if np.linalg.norm(optimized.x[start:]) > .2:
        raise ValueError('Tool offset exceeds 200mm')
    maps = _maps(optimized.x, method)
    for T in maps.values():
        values = np.linalg.svd(T[:3, :3], compute_uv=False)
        if values.min() < .9 or values.max() > 1.1:
            raise ValueError('Unphysical arm-map scale')
    model = dict(schema='shared_pf_joint_gaussian_experiment_v1', units='m', variant=variant,
                 map_kind=method, correction_kind='gaussian',
                 observation_from_mdh_T={s:T.tolist() for s,T in maps.items()},
                 shared_pf_m=optimized.x[start:].tolist(), kernel_length_m=length,
                 kernel_regularization=alpha, kernel_centers_m=centers.tolist(),
                 kernel_coefficients_m=cho_solve((L, True), raw(optimized.x)).tolist(),
                 geometric_parameter_count=len(x0), field_coefficient_count=3*len(K),
                 optimizer_evaluations=int(optimized.nfev), bound_active=bool(np.any(optimized.active_mask)),
                 training_camera_bounds_m={s:dict(min=data[s]['points'].min(0).tolist(),
                                                 max=data[s]['points'].max(0).tolist()) for s in SIDES},
                 independently_verified=False, right_anchor_fixed=False,
                 note='One empirical observation field shared across arms; not independently identified camera or robot physics. Non-rigid candidates are not pure Rt.')
    for s in SIDES:
        camera_to_arm(data[s]['points'], model, s)
    return model


def relative_score(scores, baseline):
    """All six requested metrics matter; a value <=1 satisfies the baseline."""
    return max(value for s in SIDES for value in (
        scores[s]['rms_mm']/max(baseline[s]['rms_mm'], 1e-12),
        scores[s]['p95_mm']/max(baseline[s]['p95_mm'], 1e-12),
        baseline[s]['fraction_within_1mm']/max(scores[s]['fraction_within_1mm'], 1e-12)))


def choose(scores, baseline):
    return min(scores, key=lambda v:(relative_score(scores[v], baseline),
                                    max(scores[v][s]['p95_mm'] for s in SIDES), v))


def fit_all(data):
    independent = fit_pair(data, 'rigid_l2', 'independent')
    seeds = {'rigid_l2':fit_pair(data, 'rigid_l2', 'shared', independent['models']),
             'affine':fit_pair(data, 'affine', 'shared')}
    models = {'shared_rigid':plain_model(seeds['rigid_l2']),
              'shared_affine':plain_model(seeds['affine'])}
    models.update({v:fit(data, v, seeds[VARIANTS[v][0]]) for v in VARIANTS})
    return models, independent


def diagnose(data):
    """Project out each arm's six external DOF before inspecting pF sensitivity."""
    independent = fit_pair(data, 'rigid_l2', 'independent')
    shared = fit_pair(data, 'rigid_l2', 'shared', independent['models'])
    result, information, offsets = {}, [], []
    for s in SIDES:
        d, model = data[s], independent['models'][s]
        x = np.asarray(model['parameters'])
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        Q, a, p = d['poses'][:, :3, :3], d['poses'][:, :3, 3], x[6:9]
        b = (a+np.einsum('nij,j->ni', Q, p))@R.T
        J = np.zeros((len(b), 3, 6))
        for i, (u,v,w) in enumerate(b):
            J[i, :, :3] = [[0,w,-v],[-w,0,u],[v,-u,0]]
            J[i, :, 3:] = np.eye(3)
        J = J.reshape(-1, 6)
        G = np.einsum('ab,nbc->nac', R, Q).reshape(-1, 3)
        G -= J@np.linalg.lstsq(J, G, rcond=None)[0]
        H = G.T@G
        information.append(H)
        offsets.append(p)
        e = model_predict(d['poses'], model)-d['points']
        es = model_predict(d['poses'], shared['models'][s])-d['points']
        def explained(features):
            F = np.c_[np.ones(len(features)), features]
            fitted = F@np.linalg.lstsq(F, es, rcond=None)[0]
            return float(1-np.sum((es-fitted)**2)/np.sum((es-es.mean(0))**2))
        result[s] = dict(independent_pf_mm=(p*1000).tolist(),
                         shared_pf_mm=(np.asarray(shared['models'][s]['flange_to_ball_center_m'])*1000).tolist(),
                         independent_fit=metrics(flange_delta(d, model)),
                         shared_fit=metrics(flange_delta(d, shared['models'][s])),
                         increase_sse_mm2=float((np.sum(es**2)-np.sum(e**2))*1e6),
                         projected_pf_singular_values=np.linalg.svd(G, compute_uv=False).tolist(),
                         pf_information_matrix=H.tolist(),
                         spatial_linear_explained_fraction=explained(d['points']),
                         pose_rotation_explained_fraction=explained(Q.reshape(-1, 9)),
                         camera_depth_error_correlation=[float(np.corrcoef(d['points'][:,2], es[:,axis])[0,1]) for axis in range(3)])
    p = np.linalg.solve(sum(information), sum(H@p for H,p in zip(information, offsets)))
    result['linearized_shared_pf_mm'] = (p*1000).tolist()
    for s,H,pi in zip(SIDES, information, offsets):
        result[s]['linearized_constraint_sse_mm2'] = float((p-pi)@H@(p-pi)*1e6)
    result['note'] = 'Full-data training diagnostics, not holdout scores or independent confidence intervals. Correlation does not identify physical cause.'
    return result


def compare(data, output):
    names = ['shared_rigid', 'shared_affine', *VARIANTS]
    outer = {s:list(splits(np.arange(len(data[s]['points'])))) for s in SIDES}
    cv = {v:{s:np.empty_like(data[s]['points']) for s in SIDES} for v in names}
    baseline_cv = {s:np.empty_like(data[s]['points']) for s in SIDES}
    nested = {s:np.empty_like(data[s]['points']) for s in SIDES}
    fold_records = []
    for f in range(5):
        train = {s:outer[s][f][0] for s in SIDES}
        held = {s:outer[s][f][1] for s in SIDES}
        inner = {s:list(splits(train[s], folds=4)) for s in SIDES}
        errors = {v:{s:[] for s in SIDES} for v in names}
        reference = {s:[] for s in SIDES}
        for j in range(4):
            models, baseline = fit_all({s:subset(data[s], inner[s][j][0]) for s in SIDES})
            for s in SIDES:
                validation = subset(data[s], inner[s][j][1])
                reference[s].extend(flange_delta(validation, baseline['models'][s]))
                for v in names:
                    errors[v][s].extend(flange_errors(validation, models[v], s))
            print(f'Outer {f+1}/5 inner {j+1}/4 complete', flush=True)
        reference_scores = {s:metrics(np.asarray(reference[s])) for s in SIDES}
        inner_scores = {v:{s:metrics(np.asarray(errors[v][s])) for s in SIDES} for v in names}
        selected = choose(inner_scores, reference_scores)
        models, baseline = fit_all({s:subset(data[s], train[s]) for s in SIDES})
        for s in SIDES:
            validation = subset(data[s], held[s])
            baseline_cv[s][held[s]] = flange_delta(validation, baseline['models'][s])
            for v in names:
                cv[v][s][held[s]] = flange_errors(validation, models[v], s)
            nested[s][held[s]] = cv[selected][s][held[s]]
        model_file = f'fold_{f}_models.json.gz'
        with gzip.open(output/model_file, 'wt', encoding='utf-8') as stream:
            json.dump(dict(models=models, baseline=baseline), stream, allow_nan=False)
        fold_records.append(dict(fold=f, selected=selected, inner_scores=inner_scores,
                                 inner_baseline=reference_scores, model_file=model_file,
                                 splits={s:dict(train=train[s].tolist(), held=held[s].tolist(),
                                                inner=[dict(train=t.tolist(), held=h.tolist()) for t,h in inner[s]]) for s in SIDES}))
        print(f'Outer {f+1}/5 selected {selected}', flush=True)
    models, baseline = fit_all(data)
    baseline_scores = {s:metrics(baseline_cv[s]) for s in SIDES}
    records = {v:dict(cv={s:metrics(cv[v][s]) for s in SIDES},
                      fit={s:metrics(flange_errors(data[s], models[v], s)) for s in SIDES},
                      model_file=f'{v}.json') for v in names}
    selected = choose({v:r['cv'] for v,r in records.items()}, baseline_scores)
    for v,model in models.items():
        (output/f'{v}.json').write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
    result = dict(baseline=baseline_scores, candidates=records, selected=selected,
                  nested={s:metrics(nested[s]) for s in SIDES}, folds=fold_records)
    with (output/'heldout_errors.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['variant','arm','waypoint_index','dx_mm','dy_mm','dz_mm','error_mm'])
        for v,errors in {**cv, 'independent_rigid':baseline_cv, 'nested':nested}.items():
            for s in SIDES:
                for row,delta in zip(data[s]['rows'], errors[s]*1000):
                    writer.writerow([v,s,row['waypoint_index'],*delta,float(np.linalg.norm(delta))])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    data = {s:dataset(ROOT/('datasets/drag_replay_20261020_091535_901340' if s=='right' else 'datasets/left_drag_replay_20261020_110645_673779'), s) for s in SIDES}
    args.output.mkdir(parents=True)
    result = compare(data, args.output)
    result.update(source_sha256=digest(Path(__file__)),
                  data={s:dict(path=data[s]['path'], manifest=data[s]['manifest']) for s in SIDES},
                  selection='Minimize worst baseline-relative ratio over both arms RMS/P95/within-1mm; baseline is independently refit within each inner training split.',
                  warnings=['Same sessions repeatedly reused for exploratory model research; nested validation cannot remove cross-round adaptive bias.',
                            'No residual exclusion; both arms purged by 2 positions; no heldout joints or FK used in target inference.',
                            'Empirical field and affine maps are not pure Rt or independently identified physical camera parameters.',
                            'Deployment unchanged; right anchor is re-estimated for these experiments.'])
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output/'diagnostics.json').write_text(json.dumps(diagnose(data), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(selected=result['selected'], baseline=result['baseline'], nested=result['nested']), indent=2), flush=True)


if __name__ == '__main__':
    main()
