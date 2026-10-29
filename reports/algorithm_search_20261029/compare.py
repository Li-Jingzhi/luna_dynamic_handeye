"""Fixed-pF exploratory selection; A/B are development data, trajectory C remains required."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dynamic_handeye.geometry import Assets, apply, pose, parameters, torso_from_arm, fixed_geometry_hashes, fixed_head_root
from dynamic_handeye.calibration import fit_geometry, metrics
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.optimizer import robust_optimize
from reports.new_data_refit_20261029.analyze import CAPTURES, clamp_encoder_quantization

OUT = Path(__file__).resolve().parent
VARIANTS = [('geometry', 0, 0., 0.)]
VARIANTS += [('legacy_two_stage', 1, .05, 1.)]
VARIANTS += [('affine', 1, 0., r) for r in (.001, .1)]
VARIANTS += [('quadratic', 2, 0., r) for r in (.001, .1)]
VARIANTS += [('joint_affine_gaussian', 1, l, r) for l in (.03, .06, .10, .20) for r in (.1, 1., 10.)]
VARIANTS += [('joint_quadratic_gaussian', 2, l, r) for l in (.06, .10) for r in (1., 10.)]


def name(spec):
    kind, degree, length, ridge = spec
    return f'{kind}_l{int(length*1000)}_r{ridge:g}'


def features(x, center, degree):
    z = (x - center) / .2
    f = [np.ones(len(x)), *z.T]
    if degree == 2:
        f += [z[:, i] * z[:, j] for i in range(3) for j in range(i, 3)]
    return np.column_stack(f)


def kernel(x, centers, length):
    return np.exp(-np.sum((x[:, None] - centers[None])**2, axis=2) / (2*length**2))


def fit_base(c, q, target, assets):
    seed = fit_geometry(c, apply(np.linalg.inv(torso_from_arm(assets)), target), q, assets)
    x0 = np.r_[parameters(seed['torso_from_head_root_T']), parameters(seed['head_tip_from_rectified_camera_T'])]
    h = assets.head(q)
    def residual(x):
        return ((apply(pose(x[:6]) @ h @ pose(x[6:]), c) - target)*1000).ravel()
    x = robust_optimize(residual, x0, iterations=250)
    return dict(kind='effective12', x=x.tolist())


def base_predict(model, c, q, assets):
    x = np.array(model['x'])
    return apply(pose(x[:6]) @ assets.head(q) @ pose(x[6:]), c)


def fit_relative(c, q, group, seed, assets):
    _, inverse, count = np.unique(group, return_inverse=True, return_counts=True)
    head = assets.head(q)
    def residual(x):
        z = apply(head @ pose(x), c)
        means = np.array([np.bincount(inverse, weights=z[:, axis]) / count for axis in range(3)]).T
        return ((z - means[inverse])*1000).ravel()
    x = robust_optimize(residual, np.array(seed['x'][6:]), iterations=250)
    return dict(kind='camera_relative_head', x=np.r_[parameters(fixed_head_root(assets)), x].tolist())


def fit_correction(base, target, spec):
    kind, degree, length, ridge = spec
    if kind == 'geometry':
        return dict(kind=kind)
    center = base.mean(0)
    f = features(base, center, degree)
    residual = target - base
    if kind == 'legacy_two_stage':
        b = np.zeros((f.shape[1],3))
        w = np.linalg.solve(kernel(base,base,length)+ridge*np.eye(len(base)),residual)
    elif length:
        k = kernel(base, base, length)
        inv = np.linalg.solve(k + ridge*np.eye(len(base)), np.c_[f, residual])
        # Variable projection jointly estimates the global trend and Gaussian field.
        b = np.linalg.solve(f.T @ inv[:, :f.shape[1]] + .01*np.eye(f.shape[1]), f.T @ inv[:, f.shape[1]:])
        w = inv[:, f.shape[1]:] - inv[:, :f.shape[1]] @ b
    else:
        b = np.linalg.solve(f.T @ f + ridge*np.eye(f.shape[1]), f.T @ residual)
        w = np.empty((0, 3))
    return dict(kind=kind, degree=degree, length_m=length, ridge=ridge, center_m=center.tolist(),
                trend_weights_m=b.tolist(), centers_m=base.tolist() if length else [], weights_m=w.tolist())


def correct(model, base):
    if model['kind'] == 'geometry':
        return base
    result = base + features(base, np.array(model['center_m']), model['degree']) @ np.array(model['trend_weights_m'])
    if model['length_m']:
        result += kernel(base, np.array(model['centers_m']), model['length_m']) @ np.array(model['weights_m'])
    return result


def predict_pixels(model, assets, pixels, head_angles, *, diagnostic=False):
    if not diagnostic:
        raise ValueError('Development model requires untouched trajectory C acceptance')
    if model['assets'] != assets.hashes or model['fixed_geometry_hashes'] != fixed_geometry_hashes(assets):
        raise ValueError('Calibration/reference assets changed')
    camera = assets.triangulate(pixels)[None]
    q = clamp_encoder_quantization(np.asarray(head_angles, float)[None])
    return correct(model['correction'], base_predict(model['baseline'], camera, q, assets))[0]


def main():
    assets = Assets()
    bpaths = [ROOT/f'datasets/raw/luna_v2_validationB_yaw{y}_pitchm{abs(p)}_20261029.jsonl'
              for y, p in [(0,-25),(0,-20),(10,-25),(10,-20)]]
    data = load_capture(CAPTURES + bpaths, assets)
    rows = data['rows']
    c = np.array([assets.triangulate(r['corrected_pixels']) for r in rows])
    q = clamp_encoder_quantization([r['head_angles_deg'] for r in rows])
    target = apply(torso_from_arm(assets), apply(assets.flange([r['q7_deg'] for r in rows]), [0,0,.101]))
    trajectory = np.array(['B' if 'validationB' in r['taught_path'] else 'A' for r in rows])
    group = np.array([f'{t}:{r["waypoint_id"]}' for t,r in zip(trajectory,rows)])
    heads = np.array([r['head_key'] for r in rows])
    ug = sorted(set(group))
    centroid = np.array([target[group == g].mean(0) for g in ug])
    axis = int(np.argmax(np.ptp(centroid, axis=0)))
    ordered = np.array(ug)[np.argsort(centroid[:, axis])]
    folds = [('A_to_B', trajectory == 'A', trajectory == 'B'), ('B_to_A', trajectory == 'B', trajectory == 'A')]
    for i, block in enumerate(np.array_split(ordered, 5)):
        te = np.isin(group, block)
        folds.append((f'spatial_{i}', ~te, te))
    group_q = np.array([np.mean([r['q7_deg'] for j,r in enumerate(rows) if group[j] == g], axis=0) for g in ug])
    pc = np.linalg.svd(group_q-group_q.mean(0),full_matrices=False)[2][0]
    ordered_q = np.array(ug)[np.argsort((group_q-group_q.mean(0))@pc)]
    for i, block in enumerate(np.array_split(ordered_q,5)):
        te = np.isin(group,block)
        folds.append((f'configuration_{i}',~te,te))
    for h in sorted(set(heads)):
        te = heads == h
        folds.append((f'head_{h}', ~te, te))
    report = dict(approved=False, fixed_pf_m=[0,0,.101], captures=data['sources'], assets=assets.hashes,
        variants=[name(s) for s in VARIANTS], folds={},
        note='Exploratory reuse of A and B; no candidate is certified without untouched trajectory C.')
    for fold, tr, te in folds:
        if not fold.startswith('head_'):
            assert not set(group[tr]) & set(group[te])
        fk_baseline = fit_base(c[tr], q[tr], target[tr], assets)
        baselines = dict(fk=fk_baseline, relative=fit_relative(c[tr], q[tr], group[tr], fk_baseline, assets))
        scores = {}
        for family, baseline in baselines.items():
            trainbase = base_predict(baseline, c[tr], q[tr], assets)
            testbase = base_predict(baseline, c[te], q[te], assets)
            for spec in VARIANTS:
                correction = fit_correction(trainbase, target[tr], spec)
                scores[family+':'+name(spec)] = metrics(correct(correction, testbase)-target[te])
        report['folds'][fold] = dict(train=int(tr.sum()), test=int(te.sum()), scores=scores)
        print(fold, min(scores, key=lambda k:scores[k]['p95_mm']), round(min(v['p95_mm'] for v in scores.values()),3), flush=True)
    report['variants'] = list(scores)
    report['summary'] = {key:dict(worst_p95_mm=max(f['scores'][key]['p95_mm'] for f in report['folds'].values()),
                           mean_p95_mm=float(np.mean([f['scores'][key]['p95_mm'] for f in report['folds'].values()])))
                         for key in report['variants']}
    selected = min(report['summary'], key=lambda key:(report['summary'][key]['worst_p95_mm'], report['summary'][key]['mean_p95_mm']))
    report['selected'] = selected
    report['all_folds_pass_1mm'] = report['summary'][selected]['worst_p95_mm'] <= 1
    baseline = fit_base(c, q, target, assets)
    family, selected_name = selected.split(':', 1)
    if family == 'relative':
        baseline = fit_relative(c, q, group, baseline, assets)
    b = base_predict(baseline, c, q, assets)
    correction = fit_correction(b, target, next(s for s in VARIANTS if name(s)==selected_name))
    model = dict(approved=False, input='corrected stereo pixels and measured yaw/pitch only', frame_id='S_MODEL',
                 assets=assets.hashes, fixed_geometry_hashes=fixed_geometry_hashes(assets), fixed_pf_m=[0,0,.101],
                 baseline=baseline, correction=correction, candidate=selected,
                 source_capture_hashes=[s['sha256'] for s in data['sources']])
    report['full_development_fit'] = metrics(correct(correction, b)-target)
    (OUT/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (OUT/'development_model.json').write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
    print('SELECTED',selected,report['summary'][selected], 'full_fit', report['full_development_fit']['p95_mm'],flush=True)


if __name__ == '__main__':
    main()
