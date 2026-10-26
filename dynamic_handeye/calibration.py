"""Six-parameter camera mounting calibration with fixed nominal head geometry."""
import numpy as np
from .geometry import apply, parameters, pose, torso_from_arm, fixed_head_root, fixed_geometry_hashes, torso_camera_poses
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
    t = torso_camera_poses(model, assets, angles)
    if output_frame == 'right_arm_mdh_parent':
        t = np.linalg.inv(torso_from_arm(assets)) @ t
    elif output_frame != 'S_MODEL':
        raise ValueError('Unknown output frame')
    return apply(t, camera)


def fit_geometry(camera, target, angles, assets, *, chain='luna_v2_hybrid', mapping=None):
    """Fit only X; target remains in the dataset's right-arm MDH reference."""
    camera, target, angles = np.asarray(camera), np.asarray(target), np.asarray(angles)
    if (camera.shape != target.shape or camera.shape != (len(angles), 3) or len(camera) < 36 or
            not np.isfinite(camera).all() or not np.isfinite(target).all()):
        raise ValueError('Need at least 36 finite paired points')
    if np.linalg.matrix_rank(target-target.mean(0), tol=1e-5) < 3:
        raise ValueError('Spatially distributed non-coplanar sphere positions required')
    if chain != 'luna_v2_hybrid' or mapping is not None:
        raise ValueError('Six-parameter fit requires the fixed V2 hybrid head chain')
    base = fixed_head_root(assets)
    head = base @ assets.head(angles, chain, mapping)
    target_s = apply(torso_from_arm(assets), target)
    local_target = apply(np.linalg.inv(head), target_s)
    r, t = rigid_fit(camera, local_target)
    initial = np.eye(4); initial[:3, :3] = r; initial[:3, 3] = t

    def residual(x):
        return (apply(head @ pose(x), camera)-target_s).ravel()*1000

    x = robust_optimize(residual, parameters(initial), iterations=150)
    step = np.eye(6)*1e-5
    jac = np.column_stack([(residual(x+d)-residual(x-d))/2e-5 for d in step])
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-7))
    if rank < 6:
        raise ValueError(f'Unobservable camera mounting: rank {rank}/6')
    if np.linalg.norm(x[3:]) > 1:
        raise ValueError('Camera mount exceeds 1m; check fixed geometry')
    r = residual(x)
    weights = np.minimum(1., 3/np.maximum(abs(r), 1e-12))
    gradient = jac.T @ (weights*r)
    normalized_gradient = float(np.max(abs(gradient)/np.maximum(np.linalg.norm(jac, axis=0), 1)))
    if normalized_gradient > .01:
        raise ValueError('Camera mount optimization did not converge')
    return dict(schema='luna_dynamic_head_model_v2', robot='gento_luna', robot_version='luna_v2_0',
                chain=chain, mjcf_mapping=mapping, assets=assets.hashes, frame_id='S_MODEL',
                calibration_reference_frame='right_arm_mdh_parent',
                fixed_geometry_hashes=fixed_geometry_hashes(assets),
                fixed_geometry_status='nominal_not_independently_measured',
                torso_from_head_root_T=base.tolist(),
                head_tip_from_rectified_camera_T=pose(x).tolist(),
                encoder_zero_offsets_deg=[0., 0.], parameter_count=6,
                jacobian_rank=rank, jacobian_singular_values=singular.tolist(),
                normalized_mount_gradient=normalized_gradient,
                fit_error_axes_frame='S_MODEL',
                fit_metrics=metrics(residual(x).reshape(-1, 3)/1000),
                approved=False, independently_verified=False, moving_head_validated=False)


def evaluate(dataset, assets, *, chain='luna_v2_hybrid', mapping=None):
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
    arms = np.array([r.get('arm_side', 'right') for r in rows])
    model['calibration_arms'] = sorted(set(arms))
    arm_metrics = {arm: metrics(predict(model, assets, camera[arms == arm], angles[arms == arm],
                         output_frame='right_arm_mdh_parent')-target[arms == arm]) for arm in sorted(set(arms))}
    report = dict(calibration_arms=model['calibration_arms'], fitted_error_by_arm=arm_metrics, schema='luna_dynamic_validation_v1', robot='gento_luna', robot_version='luna_v2_0', chain=chain,
                  valid_observations=len(rows), independent_waypoints=len(group_list), head_settings=unique.tolist(),
                  source_rejections=dataset['rejected'], fit_metrics=model['fit_metrics'],
                  trajectory_holdout=trajectory, spatial_holdouts=spatial, leave_one_head_out=unseen,
                  leave_one_session_out=session_scores, fitted_error_by_head=by_head,
                  fitted_cross_head_drift=metrics(drifts) if drifts else None,
                  head_coverage_qualified=coverage, approved=False,
                  parameter_count=6, fixed_geometry_status=model['fixed_geometry_status'],
                  error_axes_frame='right_arm_mdh_parent', fit_error_axes_frame='S_MODEL',
                  limitations=['Fixed head mounting uses nominal geometry, not independently measured axes.',
                      'Internal FK consistency only; no external absolute metrology.',
                      'Cross-head drift uses fitted data and is diagnostic only.',
                      'Legacy fused logs lack complete per-frame stationarity evidence.',
                      'No verified exposure synchronization or moving-head validation.',
                      'Independent unseen-head/new-session acceptance is required for deployment.'])
    return model, report
