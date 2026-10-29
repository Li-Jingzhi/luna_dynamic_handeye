"""Evaluate frozen models on trajectory B; never fit or select models on B."""
from collections import Counter
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dynamic_handeye.calibration import metrics
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.geometry import Assets, apply, digest, fixed_geometry_hashes, torso_from_arm
from dynamic_handeye.registration import rigid_fit
from reports.new_data_refit_20261029.analyze import clamp_encoder_quantization, predict_selected

OUT = Path(__file__).resolve().parent
HEADS = ([0, -25], [0, -20], [10, -25], [10, -20])
MODELS = ROOT / 'reports/new_data_refit_20261029'


def quantiles(values):
    return dict(zip(('median', 'p95', 'max'), map(float, np.percentile(values, [50, 95, 100]))))


def main():
    assets = Assets()
    paths = [ROOT / f'datasets/raw/luna_v2_validationB_yaw{y}_pitchm{abs(p)}_20261029.jsonl' for y, p in HEADS]
    dataset = load_capture(paths, assets)
    summaries = {}
    quality = []
    raw = []
    identities = [set(), set()]
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        manifest = json.loads(path.with_suffix('.completed.json').read_text())
        assert manifest['complete'] and manifest['sha256'] == digest(path)
        assert len(rows) == manifest['observations'] == 110
        assert len({r['waypoint_index'] for r in rows}) == len(rows)
        assert sum(r.get('capture_valid') is True for r in rows) == manifest['valid']
        summaries[','.join(map(str, manifest['head_target_deg']))] = manifest
        for row in rows:
            if row.get('capture_valid') is not True:
                continue
            for frame in row['camera_window_frames']:
                for eye in (0, 1):
                    identity = (str(path), frame['source_timestamp_ns'][eye])
                    assert identity not in identities[eye], 'Reused source frame within capture'
                    identities[eye].add(identity)
            quality.append(row)
        raw.extend(rows)
    rows = dataset['rows']
    camera = np.array([assets.triangulate(r['corrected_pixels']) for r in rows])
    head = clamp_encoder_quantization([r['head_angles_deg'] for r in rows])
    labels = apply(torso_from_arm(assets), apply(assets.flange([r['q7_deg'] for r in rows]), [0, 0, .101]))
    keys = np.array([r['head_key'] for r in rows])
    report = dict(approved=False, protocol='Frozen models on new arm trajectory at four previously observed head settings',
                  measured_pf_m=[0, 0, .101], assets=assets.hashes, captures=dataset['sources'],
                  capture_summaries=summaries, valid=len(rows), total=len(raw),
                  rejected=dataset['rejected'], acceptance_p95_mm=1., models={})
    predictions = {}
    model_files = dict(fixed6='fixed6_model.json', learned12='learned12_model.json',
                       learned12_torso_rbf='selected_model.json')
    for name, filename in model_files.items():
        model = json.loads((MODELS / filename).read_text())
        assert model['camera_assets'] == assets.hashes
        assert model['fixed_geometry_hashes'] == fixed_geometry_hashes(assets)
        assert model['flange_to_ball_center_m'] == [0, 0, .101]
        assert not set(model['source_capture_hashes']) & {s['sha256'] for s in dataset['sources']}
        prediction = predict_selected(model, assets, camera, head)
        predictions[name] = prediction
        error = prediction - labels
        report['models'][name] = dict(model_file=str(MODELS / filename), sha256=digest(MODELS / filename),
            overall=metrics(error), by_head={h: metrics(error[keys == h]) for h in sorted(set(keys))},
            fraction_within_1mm=float(np.mean(np.linalg.norm(error, axis=1) <= .001)))
    selected = json.loads((MODELS / 'selected_model.json').read_text())
    baseline = predictions['learned12']
    nearest = np.linalg.norm(baseline[:, None] - np.array(selected['centers_m'])[None], axis=2).min(axis=1) * 1000
    report['rbf_support'] = dict(nearest_training_center_mm=quantiles(nearest),
        fraction_farther_than_50mm=float(np.mean(nearest > 50)),
        fraction_farther_than_100mm=float(np.mean(nearest > 100)),
        correction_size_mm=quantiles(np.linalg.norm(predictions['learned12_torso_rbf'] - baseline, axis=1) * 1000))
    report['rbf_support']['error_by_nearest_center'] = {}
    for label, mask in (('within_50mm', nearest <= 50), ('50_to_100mm', (nearest > 50) & (nearest <= 100)),
                        ('beyond_100mm', nearest > 100)):
        if mask.any():
            report['rbf_support']['error_by_nearest_center'][label] = metrics((predictions['learned12_torso_rbf'] - labels)[mask])
    report['quality'] = dict(unique_frames_per_eye=[len(s) for s in identities],
        frame_3d_p95_jitter_mm=quantiles([r['sphere_center_p95_jitter_mm'] for r in quality]),
        corrected_epipolar_abs_px=quantiles([abs(r['sphere_center_corrected_stereo_pixels'][1] - r['sphere_center_corrected_stereo_pixels'][3]) for r in quality]),
        frame_disparity_p95_jitter_px=quantiles([r['disparity_jitter_p95_px'] for r in quality]),
        half_window_shift_mm=quantiles([r['sphere_center_half_window_shift_mm'] for r in quality]))
    groups = {h: {r['waypoint_id']: i for i, r in enumerate(rows) if r['head_key'] == h} for h in sorted(set(keys))}
    multiplicities = Counter(sum(w in g for g in groups.values()) for w in set().union(*(set(g) for g in groups.values())))
    report['waypoint_head_coverage'] = dict(sorted(multiplicities.items()))
    report['diagnostic_camera_pair_rigid_fit'] = {}
    report['frozen_model_cross_head_drift'] = {}
    for h1, h2 in combinations(groups, 2):
        shared = sorted(groups[h1].keys() & groups[h2].keys())
        i = np.array([groups[h1][w] for w in shared]); j = np.array([groups[h2][w] for w in shared])
        r, t = rigid_fit(camera[i], camera[j])
        report['diagnostic_camera_pair_rigid_fit'][h1 + ' / ' + h2] = metrics(camera[i] @ r.T + t - camera[j])
        report['frozen_model_cross_head_drift'][h1 + ' / ' + h2] = metrics(predictions['learned12_torso_rbf'][i] - predictions['learned12_torso_rbf'][j])
    report['diagnostic_per_head_rigid_fit_on_B'] = {}
    for h in sorted(groups):
        i = keys == h
        r, t = rigid_fit(camera[i], labels[i])
        report['diagnostic_per_head_rigid_fit_on_B'][h] = metrics(camera[i] @ r.T + t - labels[i])
    report['limitations'] = ['B uses known head settings, not unseen-head acceptance.',
        'Camera-pair and per-head rigid transforms were fitted on B for diagnosis only.',
        'No independently measured absolute torso or stereo scale reference.',
        'Incomplete attempt1 is explicitly excluded; no residual-based sample exclusion.']
    report['frozen_model_passed_1mm'] = report['models']['learned12_torso_rbf']['overall']['p95_mm'] <= 1
    with (OUT / 'predictions.csv').open('w') as f:
        fields = ['head', 'waypoint', 'model', 'x_s_m', 'y_s_m', 'z_s_m', 'dx_mm', 'dy_mm', 'dz_mm', 'error_mm']
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for name, prediction in predictions.items():
            for row, point, target in zip(rows, prediction, labels):
                delta = (point - target) * 1000
                writer.writerow(dict(head=row['head_key'], waypoint=row['waypoint_id'], model=name,
                    x_s_m=point[0], y_s_m=point[1], z_s_m=point[2], dx_mm=delta[0], dy_mm=delta[1],
                    dz_mm=delta[2], error_mm=np.linalg.norm(delta)))
    (OUT / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(valid=report['valid'], models=report['models'], support=report['rbf_support'],
                         quality=report['quality'], waypoint_heads=report['waypoint_head_coverage']), indent=2))


if __name__ == '__main__':
    main()
