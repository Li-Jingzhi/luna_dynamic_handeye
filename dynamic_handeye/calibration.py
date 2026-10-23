"""Variable projection: two fixed SE(3) mounts, fixed encoder-zero gauge."""
import numpy as np
from .geometry import apply, parameters, pose, torso_from_arm
from .optimizer import robust_optimize
from .registration import rigid_fit
from .dataset import arrays


def metrics(delta):
    mm = np.asarray(delta)*1000
    distance = np.linalg.norm(mm, axis=1)
    return dict(samples=len(mm), mean_mm=float(distance.mean()), rms_mm=float(np.sqrt(np.mean(distance**2))),
                p95_mm=float(np.percentile(distance, 95)), max_mm=float(distance.max()),
                axis_bias_mm=mm.mean(0).tolist(), axis_rms_mm=np.sqrt(np.mean(mm**2, axis=0)).tolist())


def predict(model, assets, camera, angles, *, output_frame="S_MODEL"):
    h = assets.head(angles, model['chain'], model.get('mjcf_mapping'))
    t = np.array(model['mdh_from_head_root_T']) @ h @ np.array(model['head_tip_from_rectified_camera_T'])
    if output_frame == 'S_MODEL':
        t = torso_from_arm(assets) @ t
    elif output_frame != 'right_arm_mdh_parent':
        raise ValueError('Unknown output frame')
    return apply(t, camera)


def fit_geometry(camera, target, angles, assets, *, chain='controller_mdh', mapping=None):
    camera, target, angles = np.asarray(camera), np.asarray(target), np.asarray(angles)
    if (camera.shape != target.shape or camera.shape != (len(angles), 3) or len(camera) < 36 or
            not np.isfinite(camera).all() or not np.isfinite(target).all()):
        raise ValueError('Need at least 36 finite paired points')
    if np.linalg.matrix_rank(angles-angles.mean(0), tol=1e-5) < 2:
        raise ValueError('Two-dimensional head excitation required')
    if np.linalg.matrix_rank(target-target.mean(0), tol=1e-5) < 3:
        raise ValueError('Spatially distributed non-coplanar sphere positions required')
    h = assets.head(angles, chain, mapping)

    def eliminate(x):
        head_points = apply(h @ pose(x), camera)
        r, t = rigid_fit(head_points, target)
        a = np.eye(4); a[:3, :3] = r; a[:3, 3] = t
        return (apply(a, head_points)-target)*1000, a

    best = None
    for rotation in ([0, 0, 0], [np.pi/2, 0, 0], [-np.pi/2, 0, 0],
                     [0, np.pi/2, 0], [0, -np.pi/2, 0], [0, 0, np.pi]):
        x = robust_optimize(lambda v: eliminate(v)[0].ravel(), np.r_[rotation, np.zeros(3)], iterations=100)
        error, a = eliminate(x)
        absolute = np.abs(error)
        cost = np.sum(np.where(absolute <= 3, .5*error**2, 3*(absolute-1.5)))
        if best is None or cost < best[0]:
            best = (cost, x, a)
    _, x, a = best
    v = np.r_[parameters(a), x]

    def residual(p):
        return (apply(pose(p[:6]) @ h @ pose(p[6:]), camera)-target).ravel()*1000

    # Joint refinement makes both mounts minimize the same Huber objective.
    v = robust_optimize(residual, v, iterations=150)
    a, x = pose(v[:6]), v[6:]
    step = np.eye(12)*1e-5
    jac = np.column_stack([(residual(v+d)-residual(v-d))/2e-5 for d in step])
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-7))
    if rank < 12:
        raise ValueError(f'Unobservable two-mount geometry: rank {rank}/12')
    if np.linalg.norm(x[3:]) > 1:
        raise ValueError('Camera mount exceeds 1m; check chain and frame convention')
    r = residual(v)
    weights = np.minimum(1., 3/np.maximum(abs(r), 1e-12))
    gradient = jac.T @ (weights*r)
    normalized_gradient = float(np.max(abs(gradient)/np.maximum(np.linalg.norm(jac, axis=0), 1)))
    if normalized_gradient > .01:
        raise ValueError('Joint mount optimization did not converge')
    return dict(schema='luna_dynamic_head_model_v1', robot='gento_luna', chain=chain, mjcf_mapping=mapping,
                assets=assets.hashes, frame_id='S_MODEL', calibration_reference_frame='right_arm_mdh_parent',
                mdh_from_head_root_T=a.tolist(), head_tip_from_rectified_camera_T=pose(x).tolist(),
                encoder_zero_offsets_deg=[0., 0.], parameter_count=12,
                jacobian_rank=rank, jacobian_singular_values=singular.tolist(),
                normalized_mount_gradient=normalized_gradient,
                fit_metrics=metrics(residual(v).reshape(-1, 3)/1000),
                approved=False, independently_verified=False, moving_head_validated=False)


def evaluate(dataset, assets, *, chain='controller_mdh', mapping=None):
    camera, target, angles = arrays(dataset)
    rows = dataset['rows']
    groups = np.array([r['group_id'] for r in rows])
    heads = np.array([r['head_key'] for r in rows])
    sessions = np.array([r['session_id'] for r in rows])

    def fold(train, test):
        try:
            model = fit_geometry(camera[train], target[train], angles[train], assets, chain=chain, mapping=mapping)
        except ValueError as exc:
            return dict(status='unavailable', reason=str(exc), train_count=int(sum(train)), test_count=int(sum(test)))
        error = predict(model, assets, camera[test], angles[test], output_frame="right_arm_mdh_parent")-target[test]
        return dict(status='diagnostic', train_groups=sorted(set(groups[train])),
                    test_groups=sorted(set(groups[test])), metrics=metrics(error))

    group_list = list(dict.fromkeys(groups))
    cut = max(1, int(.8*len(group_list)))
    train = np.isin(groups, group_list[:cut]); test = ~train
    if not test.any():
        raise ValueError('Need multiple independent waypoints')
    trajectory = fold(train, test)
    # Entire physical locations stay together across all heads in each spatial fold.
    centers = np.array([target[groups == g].mean(0) for g in group_list])
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    ordered = np.array(group_list)[np.argsort(centers[:, axis])]
    spatial = []
    for held in np.array_split(ordered, 5):
        mask = np.isin(groups, held)
        spatial.append(fold(~mask, mask))
    unseen = {h: fold(heads != h, heads == h) for h in sorted(set(heads))}
    session_scores = {s: fold(sessions != s, sessions == s) for s in sorted(set(sessions))} if len(set(sessions)) > 1 else {}
    model = fit_geometry(camera, target, angles, assets, chain=chain, mapping=mapping)
    unique = np.unique(np.array([list(map(float, h.split(','))) for h in heads]), axis=0)
    coverage = bool(len(unique) >= 6 and len(np.unique(unique[:, 0])) >= 3 and len(np.unique(unique[:, 1])) >= 3)
    by_head = {h: metrics(predict(model, assets, camera[heads == h], angles[heads == h], output_frame="right_arm_mdh_parent")-target[heads == h]) for h in sorted(set(heads))}
    estimates = predict(model, assets, camera, angles, output_frame="right_arm_mdh_parent")
    drifts = []
    for g in group_list:
        ids = np.flatnonzero(groups == g)
        if len(ids) > 1:
            drifts.extend(estimates[ids[1:]]-estimates[ids[0]])
    model.update(head_support_deg=unique.tolist(), camera_support_m=camera.tolist(),
                 source_capture_hashes=[s['sha256'] for s in dataset['sources']],
                 head_coverage_qualified=coverage)
    report = dict(schema='luna_dynamic_validation_v1', robot='gento_luna', chain=chain,
                  valid_observations=len(rows), independent_waypoints=len(group_list), head_settings=unique.tolist(),
                  source_rejections=dataset['rejected'], fit_metrics=model['fit_metrics'],
                  trajectory_holdout=trajectory, spatial_holdouts=spatial, leave_one_head_out=unseen,
                  leave_one_session_out=session_scores, fitted_error_by_head=by_head,
                  fitted_cross_head_drift=metrics(drifts) if drifts else None,
                  head_coverage_qualified=coverage, approved=False,
                  limitations=['Internal FK consistency only; no external absolute metrology.',
                      'Cross-head drift uses fitted data and is diagnostic only.',
                      'Legacy fused logs lack complete per-frame stationarity evidence.',
                      'No verified exposure synchronization or moving-head validation.',
                      'Independent unseen-head/new-session acceptance is required for deployment.'])
    return model, report
