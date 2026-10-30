"""Audit old-camera fixed-head residuals with frozen MDH, pF and zero offsets."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from dynamic_handeye.geometry import Assets, digest
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.calibration import metrics


def score(camera, target, indices):
    r, t = rigid_fit(camera[indices], target[indices])
    return camera @ r.T + t, r, t


def evaluate(rows, assets):
    rows = sorted(rows, key=lambda r: r['waypoint_id'])
    c = np.array([assets.triangulate(r['corrected_pixels']) for r in rows])
    fused = np.array([r['camera_xyz_m'] for r in rows])
    target = np.array([r['target_s_m'] for r in rows])
    q = np.array([r['q7_deg'] for r in rows])
    ids = np.arange(len(rows))
    fitted, r, t = score(c, target, ids)
    fit_fused, _, _ = score(fused, target, ids)
    trajectory_cv = np.empty_like(target)
    protocols = {}
    spatial_axis = int(np.argmax(np.ptp(target, axis=0)))
    centered = q - q.mean(0)
    configuration = centered @ np.linalg.svd(centered, full_matrices=False)[2][0]
    for protocol, order in [('trajectory', ids), ('spatial', np.argsort(target[:, spatial_axis])),
                            ('configuration', np.argsort(configuration))]:
        errors = []; folds = []
        for fold, test in enumerate(np.array_split(order, 5)):
            # Trajectory protocol purges neighbouring VALID positions, as in the new-data comparison.
            train = ids[(ids < test[0]-2) | (ids > test[-1]+2)] if protocol == 'trajectory' else np.setdiff1d(ids, test)
            if len(train) < 18: raise ValueError('insufficient training positions')
            assert not set(train) & set(test)
            predicted, _, _ = score(c, target, train)
            delta = predicted[test]-target[test]
            errors.extend(delta)
            if protocol == 'trajectory': trajectory_cv[test] = predicted[test]
            folds.append(dict(fold=fold, train_waypoints=[rows[i]['waypoint_id'] for i in train],
                              test_waypoints=[rows[i]['waypoint_id'] for i in test], metrics=metrics(delta)))
        protocols[protocol] = dict(pooled=metrics(errors), folds=folds)
    upper = np.triu_indices(len(c), 1)
    distances = np.abs(np.linalg.norm(c[:, None]-c[None], axis=2)[upper]
                       - np.linalg.norm(target[:, None]-target[None], axis=2)[upper])*1000
    result = dict(n=len(c), head_deg=rows[0]['head_key'], session=rows[0]['session_id'],
        training=metrics(fitted-target), fused_xyz_training=metrics(fit_fused-target),
        cv=protocols, fixed_camera_to_torso_R=r.tolist(), fixed_camera_to_torso_t_m=t.tolist(),
        depth_range_m=[float(c[:, 2].min()), float(c[:, 2].max())],
        pair_distance_absolute_error_mm=dict(p95=float(np.percentile(distances,95)), max=float(distances.max())),
        rigid_max_error_lower_bound_mm=float(distances.max()/2),
        fused_vs_pixel_triangulation=metrics(fused-c),
        training_below_1mm_fraction=float(np.mean(np.linalg.norm(fitted-target,axis=1)<.001)),
        cv_below_1mm_fraction=float(np.mean(np.linalg.norm(trajectory_cv-target,axis=1)<.001)))
    predictions = [dict(session=row['session_id'], head=row['head_key'], waypoint=row['waypoint_id'],
        fit_error_mm=float(np.linalg.norm(fitted[i]-target[i])*1000),
        trajectory_cv_error_mm=float(np.linalg.norm(trajectory_cv[i]-target[i])*1000)) for i,row in enumerate(rows)]
    return result, predictions


def main():
    snapshot = HERE/'snapshot'
    restored = json.loads((HERE/'RESTORE_MANIFEST.json').read_text())
    for entry in restored['files']:
        if digest(snapshot/entry['path']) != entry['sha256']: raise ValueError('Archive modified: '+entry['path'])
    a = Assets(snapshot)
    paths = [snapshot/'datasets/raw/ball_point_head_20261018_v1.jsonl',
             snapshot/'datasets/raw/ball_auto_20261018_visible_v1.jsonl']
    data = load_capture(paths, a)
    assert np.array_equal(a.pf, [0,0,.101])
    groups = {}
    for row in data['rows']: groups.setdefault((row['session_id'], row['head_key']), []).append(row)
    report = dict(source_commit=restored['source_commit'], assets=a.hashes,
        pf_m=a.pf.tolist(), joint_zero_correction_deg=[0]*7, method='SE3 Kabsch; each fixed head/session separately',
        pixels='archived sphere-projection-corrected pixels; retriangulated using archived camera parameters',
        rejected=data['rejected'], valid_rows=len(data['rows']), groups=[],
        note='Development diagnostics, not external physical truth or arbitrary-head certification. No error-based filtering.')
    predictions = []
    for key, rows in sorted(groups.items()):
        result, points = evaluate(rows, a); report['groups'].append(result); predictions.extend(points)
        print(key, result['n'], 'fit',round(result['training']['p95_mm'],3),
              'trajectory CV',round(result['cv']['trajectory']['pooled']['p95_mm'],3))
    (HERE/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (HERE/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)


if __name__ == '__main__': main()
