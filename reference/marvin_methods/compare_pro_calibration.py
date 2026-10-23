"""Compare fixed-camera calibration models with purged, nested trajectory validation.

Run with python -m interactive_ball_calibration.compare_pro_calibration --output DIR.
Requires scipy and matplotlib in addition to the acquisition requirements.
All errors use the same archived, unmodified sphere-center measurements as targets.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .pro_capture import ROOT, digest, flange_pose, mdh_rows
from .pro_fit import load_capture, metrics, predict
from .registration import rigid_fit, solve_joint


METHODS = ('rigid_l2', 'rigid_huber', 'rigid_covariance', 'rigid_pixel',
           'similarity', 'depth_bias', 'affine')
PARAMETERS = dict(zip(METHODS, (9, 9, 9, 9, 10, 10, 15)))


def splits(indices, folds=5, purge=2):
    """Use original trajectory indices, including when training has gaps."""
    indices = np.asarray(indices, int)
    for held in np.array_split(indices, folds):
        distance = np.min(abs(indices[:, None] - held), axis=1)
        train = indices[distance > purge]
        if len(train) < 18:
            raise ValueError('Insufficient training positions after purging')
        yield train, held


def project(points, projections):
    homogeneous = np.c_[points, np.ones(len(points))]
    pixels = []
    for projection in projections:
        image = homogeneous @ projection.T
        if np.any(image[:, 2] <= 0):
            raise ValueError('Prediction behind camera')
        pixels.append(image[:, :2] / image[:, 2:])
    return np.hstack(pixels)


def model_predict(poses, model):
    x = np.asarray(model['parameters'])
    if model['method'] == 'affine':
        M, t, offset = x[:9].reshape(3, 3), x[9:12], x[12:15]
        return predict(poses, offset, M, t)
    R, t, offset = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6], x[6:9]
    base = poses[:, :3, 3] + np.einsum('nij,j->ni', poses[:, :3, :3], offset)
    if model['method'] == 'similarity':
        base = base * np.exp(x[9])
    camera = base @ R.T + t
    if model['method'] == 'depth_bias':
        camera = camera / (1 + x[9] * camera[:, 2:3])
    return camera


def camera_to_arm(points, model):
    """Invert the fitted observation model; output is the MDH parent frame."""
    x = np.asarray(model['parameters'])
    points = np.asarray(points, float)
    if model['method'] == 'affine':
        return (points - x[9:12]) @ np.linalg.inv(x[:9].reshape(3, 3)).T
    if model['method'] == 'depth_bias':
        points = points / (1 - x[9] * points[:, 2:3])
    result = (points - x[3:6]) @ Rotation.from_rotvec(x[:3]).as_matrix()
    return result / np.exp(x[9]) if model['method'] == 'similarity' else result


def describe_model(model):
    """Export usable matrices without labelling an affine matrix as a rotation."""
    x = np.asarray(model['parameters'])
    if model['method'] == 'affine':
        M, t = x[:9].reshape(3, 3), x[9:12]
        model['linear_map_singular_values'] = np.linalg.svd(M, compute_uv=False).tolist()
        model['arm_to_observation_linear_M'] = M.tolist()
        model['translation_m'] = t.tolist()
    else:
        R, t = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]
        model['left_camera_from_arm_mdh_R'] = R.tolist()
        model['left_camera_from_arm_mdh_t_m'] = t.tolist()
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        if model['method'] == 'similarity':
            model['scale'] = float(np.exp(x[9])); T[:3, :3] *= model['scale']
        if model['method'] == 'depth_bias':
            model['inverse_depth_bias_per_m'] = float(x[9])
            model['observation_equation'] = 'observed_xyz = physical_camera_xyz / (1 + bias * physical_camera_z)'
        model['arm_to_camera_T_before_depth_mapping'] = T.tolist()
        model['camera_to_arm_T_after_undoing_depth_mapping'] = np.linalg.inv(T).tolist()
    return model


def fit_model(method, poses, points, pixels, whitening, projections):
    offset = np.array([0., 0., .1])
    base = poses[:, :3, 3] + np.einsum('nij,j->ni', poses[:, :3, :3], offset)
    R, t = rigid_fit(base, points)
    seed = np.r_[Rotation.from_matrix(R).as_rotvec(), t, offset]
    if method == 'affine':
        seed = np.r_[R.ravel(), t, offset]
    elif method in ('similarity', 'depth_bias'):
        seed = np.r_[seed, 0.]

    def residual(x):
        predicted = model_predict(poses, dict(method=method, parameters=x))
        if method == 'rigid_pixel':
            return (project(predicted, projections) - pixels).ravel()
        delta = predicted - points
        if method == 'rigid_covariance':
            # Fixed 0.2mm covariance floor prevents quiet windows dominating the fit.
            delta = np.einsum('nij,nj->ni', whitening, delta)
        return delta.ravel()

    lower, upper = np.full(len(seed), -np.inf), np.full(len(seed), np.inf)
    offset_slice = slice(12, 15) if method == 'affine' else slice(6, 9)
    lower[offset_slice], upper[offset_slice] = -.2, .2
    if method == 'similarity':
        lower[-1], upper[-1] = np.log(.95), np.log(1.05)
    if method == 'depth_bias':
        lower[-1], upper[-1] = -.2, .2
    result = least_squares(residual, seed, bounds=(lower, upper), x_scale='jac',
                           loss='huber' if method == 'rigid_huber' else 'linear',
                           f_scale=.0007, max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'{method} optimization failed: {result.message}')
    offset = result.x[offset_slice]
    if np.linalg.norm(offset) > .2 or np.any(result.active_mask):
        raise ValueError(f'{method} reached physical parameter bounds')
    singular = np.linalg.svd(result.jac, compute_uv=False)
    return describe_model(dict(method=method, parameters=result.x.tolist(), parameter_count=len(seed),
                flange_to_ball_center_m=offset.tolist(), optimizer_evaluations=result.nfev,
                jacobian_condition=float(singular[0] / singular[-1]),
                jacobian_condition_note='Parameter units differ; diagnostic only, not a covariance estimate.'))


def verify_archive(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    for item in manifest['files']:
        data = (directory / item['file']).read_bytes()
        decoded = gzip.decompress(data)
        if (hashlib.sha256(data).hexdigest() != item['compressed_sha256'] or
                hashlib.sha256(decoded).hexdigest() != item['uncompressed_sha256'] or
                len(data) != item['compressed_bytes'] or
                len(decoded) != item['uncompressed_bytes']):
            raise ValueError(f'Archive hash/size mismatch: {item["file"]}')
    return manifest


def prepare(directory,arm='right'):
    manifest = verify_archive(directory)
    stereo_path = ROOT / 'configs/orin_pro_rectified_stereo_20261019.json'
    mdh_path = ROOT / 'configs/1004.ini'
    raw_path = ROOT / 'configs/orin_pro_raw_fisheye_stereo_20261019.json'
    rows, rejected = load_capture(directory / 'replay_capture.jsonl.gz', stereo_path, mdh_path, raw_path,arm=arm)
    if len(rows) != manifest['valid_points'] or len(rows) + rejected != manifest['captured_points']:
        raise ValueError('Capture valid/rejected counts disagree with the archive manifest')
    chain = mdh_rows(mdh_path,0 if arm=='left' else 1)
    poses = np.array([flange_pose(r['q7_deg'], chain) for r in rows])
    points = np.array([r['sphere_center_left_camera_m'] for r in rows])
    pixels = np.array([r['sphere_center_corrected_stereo_pixels'] for r in rows])
    clouds = np.array([r['sphere_center_framewise_m'] for r in rows])
    covariance = np.array([np.cov(cloud.T) for cloud in clouds])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = np.array([V @ np.diag(1 / np.sqrt(e + .0002**2)) @ V.T
                          for e, V in zip(eigenvalues, eigenvectors)])
    stereo = json.loads(stereo_path.read_text())
    projections = np.array([stereo[f'{eye}_P'] for eye in ('left', 'right')])
    return rows, poses, points, pixels, clouds, whitening, projections, manifest


def compare(poses, points, pixels, whitening, projections):
    count = len(points)
    indices = np.arange(count)
    out_of_fold = {method: np.empty_like(points) for method in METHODS}
    nested = np.empty_like(points)
    nested_folds, folds = [], []
    fold_ids = np.empty(count, int)
    baseline = np.empty_like(points)

    def fit_on(method, train):
        return fit_model(method, poses[train], points[train], pixels[train], whitening[train], projections)

    for fold, (train, held) in enumerate(splits(indices)):
        inner_error = {method: [] for method in METHODS}
        for inner_train, inner_held in splits(train, folds=4):
            for method in METHODS:
                model = fit_on(method, inner_train)
                inner_error[method].extend(model_predict(poses[inner_held], model) - points[inner_held])
        inner_metrics = {method: metrics(np.array(delta)) for method, delta in inner_error.items()}
        selected = min(METHODS, key=lambda method: (inner_metrics[method]['rms_mm'], PARAMETERS[method]))
        fold_models = {}
        for method in METHODS:
            model = fit_on(method, train)
            out_of_fold[method][held] = model_predict(poses[held], model)
            fold_models[method] = dict(model=model, metrics=metrics(out_of_fold[method][held] - points[held]))
        nested[held] = out_of_fold[selected][held]
        nested_folds.append(dict(fold=fold, selected=selected, inner_metrics=inner_metrics,
                                 metrics=metrics(nested[held] - points[held])))
        offset, R, t, _ = solve_joint(poses[train], points[train], iterations=2000)
        baseline[held] = predict(poses[held], offset, R, t)
        fold_ids[held] = fold
        folds.append(dict(fold=fold, train_indices=train.tolist(), held_indices=held.tolist(), models=fold_models))
        print(f'Fold {fold + 1}/5: inner-selected {selected}, outer RMS '
              f'{nested_folds[-1]["metrics"]["rms_mm"]:.3f}mm', flush=True)

    results = {}
    for method in METHODS:
        model = fit_on(method, indices)
        results[method] = dict(model=model, fit_metrics=metrics(model_predict(poses, model) - points),
                               cv_metrics=metrics(out_of_fold[method] - points))
    winner = min(METHODS, key=lambda method: (results[method]['cv_metrics']['rms_mm'], PARAMETERS[method]))
    offset, R, t, _ = solve_joint(poses, points, iterations=2000)
    reference = dict(fit_metrics=metrics(predict(poses, offset, R, t) - points),
                     cv_metrics=metrics(baseline - points), flange_to_ball_center_m=offset.tolist())
    return dict(methods=results, selected_method=winner, baseline_alternating=reference,
                nested_selection_metrics=metrics(nested - points), nested_folds=nested_folds,
                folds=folds), out_of_fold, nested, fold_ids


def diagnostics(rows, points, clouds, cv_delta):
    frames = [f for row in rows for f in row['camera_window_frames']]
    error = np.linalg.norm(cv_delta, axis=1) * 1000
    jitter = np.array([np.percentile(np.linalg.norm(cloud - np.median(cloud, axis=0), axis=1), 95)
                       * 1000 for cloud in clouds])
    half_shift = np.linalg.norm(np.median(clouds[:, :25], axis=1) - np.median(clouds[:, 25:], axis=1), axis=1) * 1000
    features = {f'camera_{axis}_m': points[:, i] for i, axis in enumerate('xyz')}
    features.update(window_jitter_p95_mm=jitter, first_last_half_median_shift_mm=half_shift,
                    source_skew_ms=np.array([np.mean([f['pair_source_clock_skew_ms'] for f in r['camera_window_frames']]) for r in rows]),
                    epipolar_abs_px=np.array([np.mean([abs(f['epipolar_dv_px']) for f in r['camera_window_frames']]) for r in rows]))
    features.update({f'joint_{i+1}_deg': np.array([r['q7_deg'][i] for r in rows]) for i in range(7)})
    correlations = {name: dict(error_norm=float(np.corrcoef(value, error)[0, 1]),
                               signed_z=float(np.corrcoef(value, cv_delta[:, 2])[0, 1]))
                    for name, value in features.items() if np.std(value) > 1e-12}
    depth_bins = []
    for indices in np.array_split(np.argsort(points[:, 2]), 3):
        depth_bins.append(dict(z_range_m=[float(points[indices, 2].min()), float(points[indices, 2].max())],
                               metrics=metrics(cv_delta[indices])))
    def summary(values):
        return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)), max=float(np.max(values)))
    return dict(frame_count=len(frames), window_jitter_p95_mm=summary(jitter),
                first_last_half_median_shift_mm=summary(half_shift),
                source_skew_ms=summary([f['pair_source_clock_skew_ms'] for f in frames]),
                arrival_skew_ms=summary([f['pair_dequeue_skew_ms'] for f in frames]),
                epipolar_abs_px=summary([abs(f['epipolar_dv_px']) for f in frames]),
                joint_window_max_span_deg=max(max(r['robot_window_q7_range_deg']) for r in rows),
                correlations=correlations, depth_bins=depth_bins,
                note='Correlations are descriptive, confounded by trajectory; they do not identify causal error sources.')


def sphere_points(pixels, projections, diameter):
    """Recompute the archived silhouette correction, without using any fitted FK."""
    def triangulate(uv):
        homogeneous = cv2.triangulatePoints(projections[0], projections[1], uv[:, :2].T, uv[:, 2:].T).T
        return homogeneous[:, :3] / homogeneous[:, 3:]
    points = triangulate(pixels)
    corrected = pixels.copy()
    for _ in range(6):
        factor = 1 - (diameter / 2 / points[:, 2]) ** 2
        for i, P in enumerate(projections):
            K = P[:, :3]
            rays = np.c_[pixels[:, 2*i:2*i+2], np.ones(len(pixels))] @ np.linalg.inv(K).T
            rays[:, :2] *= factor[:, None]
            image = rays @ K.T
            corrected[:, 2*i:2*i+2] = image[:, :2] / image[:, 2:]
        points = triangulate(corrected)
    return points


def sensitivity(rows, clouds, projections, result):
    pixels = np.array([[f['raw_stereo_pixels'] for f in r['camera_window_frames']] for r in rows])
    nominal = sphere_points(pixels.reshape(-1, 4), projections, .04).reshape(clouds.shape)
    if not np.allclose(nominal, clouds, rtol=0, atol=1e-10):
        raise ValueError('Archived 40mm sphere geometry cannot be reproduced')
    diameter_effect = {}
    for diameter in (0., .038, .042):
        alternative = sphere_points(pixels.reshape(-1, 4), projections, diameter).reshape(clouds.shape)
        delta = np.median(alternative, axis=1) - np.median(nominal, axis=1)
        diameter_effect[str(diameter * 1000)] = metrics(delta)
    offsets = np.array([f['models']['rigid_l2']['model']['flange_to_ball_center_m'] for f in result['folds']])
    nominal_R = np.array(result['methods']['rigid_l2']['model']['left_camera_from_arm_mdh_R'])
    angle = [float(np.linalg.norm(Rotation.from_matrix(np.array(f['models']['rigid_l2']['model']['left_camera_from_arm_mdh_R']) @ nominal_R.T).as_rotvec()) * 180 / np.pi)
             for f in result['folds']]
    depth = np.median(nominal, axis=1)[:, 2]
    fb = abs(projections[1][0, 3] - projections[0][0, 3])
    return dict(sphere_diameter_alternatives_mm=diameter_effect,
                sphere_geometry_reproduction_max_difference_m=float(np.max(abs(nominal - clouds))),
                depth_error_mm_for_0_1px_disparity_bias_range=(np.array([depth.min(), depth.max()])**2 / fb * .1 * 1000).tolist(),
                fusion_mean_minus_median=metrics(np.mean(clouds, axis=1) - np.median(clouds, axis=1)),
                rigid_fold_offset_min_mm=(offsets.min(0) * 1000).tolist(),
                rigid_fold_offset_max_mm=(offsets.max(0) * 1000).tolist(),
                rigid_fold_rotation_difference_deg=angle,
                note='Sensitivity probes, not new calibration targets, ground truth or causal attribution.')


def cross_session_validation(reference, current, current_models):
    """Train on exactly one session, predict only the other, without pooling frames."""
    ref_rows, ref_poses, ref_points, ref_pixels, _, ref_weights, ref_projections, ref_manifest = reference
    rows, poses, points, _, _, _, _, manifest = current
    result, per_position = {}, []
    for method in METHODS:
        ref_model = fit_model(method, ref_poses, ref_points, ref_pixels, ref_weights, ref_projections)
        new_model = current_models[method]['model']
        old_to_new = model_predict(poses, ref_model) - points
        new_to_old = model_predict(ref_poses, new_model) - ref_points
        a, b = metrics(old_to_new), metrics(new_to_old)
        result[method] = dict(reference_model=ref_model, reference_to_current=a, current_to_reference=b,
                              pooled_metrics=metrics(np.vstack([old_to_new, new_to_old])),
                              session_balanced_rms_mm=float(np.sqrt((a['rms_mm']**2 + b['rms_mm']**2) / 2)))
        for direction, samples, delta in (('reference_to_current', rows, old_to_new),
                                           ('current_to_reference', ref_rows, new_to_old)):
            for row, error in zip(samples, delta):
                per_position.append(dict(method=method, direction=direction, waypoint_index=row['waypoint_index'],
                                         error_mm=float(np.linalg.norm(error) * 1000)))
    return dict(reference_manifest=ref_manifest, current_manifest=manifest, methods=result,
                note='Separate-session transfer of fixed model families. Comparing candidates on these scores is exploratory selection, not independent validation of a selected final model.'), per_position


def save_plots(output, result, points, predictions, nested, fold_ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = sorted(METHODS, key=lambda m: result['methods'][m]['cv_metrics']['rms_mm'])
    winner = result['selected_method']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    x = np.arange(len(order))
    axes[0, 0].bar(x - .2, [result['methods'][m]['fit_metrics']['rms_mm'] for m in order], .4, label='Training RMS')
    axes[0, 0].bar(x + .2, [result['methods'][m]['cv_metrics']['rms_mm'] for m in order], .4, label='Blocked CV RMS')
    axes[0, 0].set_xticks(x, order, rotation=30, ha='right'); axes[0, 0].legend(); axes[0, 0].set_ylabel('mm')
    for method in dict.fromkeys(('rigid_l2', winner)):
        errors = np.sort(np.linalg.norm(predictions[method] - points, axis=1) * 1000)
        axes[0, 1].plot(errors, np.arange(1, len(errors) + 1) / len(errors), label=method)
    errors = np.sort(np.linalg.norm(nested - points, axis=1) * 1000)
    axes[0, 1].plot(errors, np.arange(1, len(errors) + 1) / len(errors), '--', label='Nested selection')
    axes[0, 1].axvline(1, color='grey', ls=':'); axes[0, 1].set_xlabel('Held-out 3D error (mm)'); axes[0, 1].legend()
    delta = (predictions[winner] - points) * 1000
    for i, axis in enumerate('XYZ'):
        axes[1, 0].plot(delta[:, i], label=axis)
    for boundary in np.where(np.diff(fold_ids))[0]: axes[1, 0].axvline(boundary + .5, color='grey', ls=':')
    axes[1, 0].set_xlabel('Sorted trajectory position'); axes[1, 0].set_ylabel('Held-out signed error (mm)'); axes[1, 0].legend()
    scatter = axes[1, 1].scatter(points[:, 2] * 1000, np.linalg.norm(delta, axis=1),
                                c=fold_ids + 1, cmap=plt.get_cmap('tab10', 5), vmin=.5, vmax=5.5)
    axes[1, 1].set_xlabel('Measured depth (mm)'); axes[1, 1].set_ylabel('Held-out 3D error (mm)')
    fig.colorbar(scatter, ax=axes[1, 1], label='Trajectory fold', ticks=np.arange(1, 6))
    fig.suptitle('Pro calibration: single-session internal consistency (not absolute accuracy)')
    fig.savefig(output / 'comparison.svg'); fig.savefig(output / 'comparison.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/drag_replay_20261019_191653_275606')
    parser.add_argument('--arm',choices=('left','right'),default='right')
    parser.add_argument('--reference-dataset', type=Path, help='Optional separate session for bidirectional transfer validation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve previous analyses')
    rows, poses, points, pixels, clouds, whitening, projections, manifest = prepare(args.dataset,arm=args.arm)
    result, predictions, nested, fold_ids = compare(poses, points, pixels, whitening, projections)
    winner = result['selected_method']
    transfer_rows = None
    if args.reference_dataset:
        reference = prepare(args.reference_dataset,arm=args.arm)
        current = (rows, poses, points, pixels, clouds, whitening, projections, manifest)
        result['cross_session'], transfer_rows = cross_session_validation(reference, current, result['methods'])
        result['cross_session']['reference_dataset'] = str(args.reference_dataset)
    result.update(schema='head_eye_pro_multimethod_comparison_v1',
                  arm_side=args.arm,
                  data_manifest=manifest, dataset=str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
                  captured_waypoints=manifest['captured_points'], valid_waypoints=len(rows),
                  rejected_waypoints=manifest['captured_points']-len(rows),
                  stereo_observations=len(rows)*50,
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  source_script_sha256=digest(Path(__file__)),
                  selection_criterion='Minimum pooled blocked-CV RMS; parameter count breaks exact ties.',
                  validation='5 outer trajectory blocks in valid-waypoint order, purge 2 adjacent valid positions; 4 inner blocks for model selection; 50 frames never split.',
                  diagnostics=diagnostics(rows, points, clouds, predictions[winner] - points),
                  sensitivity=sensitivity(rows, clouds, projections, result),
                  independently_verified=False,
                  bounds=dict(camera_min_m=points.min(0).tolist(), camera_max_m=points.max(0).tolist(),
                              joint_min_deg=np.min([r['q7_deg'] for r in rows], axis=0).tolist(),
                              joint_max_deg=np.max([r['q7_deg'] for r in rows], axis=0).tolist()),
                  warnings=['No images: intrinsics were not recalibrated.',
                            'Winner CV metrics share data with model selection; use nested selection metrics for a less biased procedure estimate.',
                            'Single session with correlated poses; no external ground truth or absolute accuracy claim.',
                            'Similarity/depth/affine models fit systematic observation discrepancies; parameters are not verified physical corrections.',
                            '40mm sphere diameter assumed; hardware exposure synchronization unverified.'])
    args.output.mkdir(parents=True)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    selected = dict(result['methods'][winner]['model'], schema='head_eye_pro_candidate_observation_model_v1',
                    units='m', input=f'FK flange pose in {args.arm}-arm MDH parent', output='archived left-camera sphere-center measurement domain',
                    inverse_function='interactive_ball_calibration.compare_pro_calibration.camera_to_arm',
                    independently_verified=False, cv_metrics=result['methods'][winner]['cv_metrics'],
                    nested_selection_metrics=result['nested_selection_metrics'], bounds=result['bounds'],
                    source_capture_sha256=manifest['files'][0]['uncompressed_sha256'],
                    mdh_sha256=digest(ROOT / 'configs/1004.ini'),
                    stereo_model_sha256=digest(ROOT / 'configs/orin_pro_rectified_stereo_20261019.json'))
    (args.output / 'best_candidate.json').write_text(json.dumps(selected, indent=2, allow_nan=False) + '\n')
    errors = []
    for i, row in enumerate(rows):
        item = dict(waypoint_index=row['waypoint_index'], fold=int(fold_ids[i]), z_m=points[i, 2])
        for method in METHODS:
            item[f'{method}_cv_mm'] = np.linalg.norm(predictions[method][i] - points[i]) * 1000
        item['nested_selection_cv_mm'] = np.linalg.norm(nested[i] - points[i]) * 1000
        for axis, value in zip('xyz', (predictions[winner][i] - points[i]) * 1000): item[f'best_delta_{axis}_mm'] = value
        errors.append(item)
    with (args.output / 'per_position_errors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(errors[0])); writer.writeheader(); writer.writerows(errors)
    if transfer_rows:
        with (args.output / 'cross_session_errors.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(transfer_rows[0])); writer.writeheader(); writer.writerows(transfer_rows)
    save_plots(args.output, result, points, predictions, nested, fold_ids)
    print(json.dumps(dict(selected=winner, cv=result['methods'][winner]['cv_metrics'],
                         nested=result['nested_selection_metrics']), indent=2))


if __name__ == '__main__':
    main()
