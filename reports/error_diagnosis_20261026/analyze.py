"""Offline ablation only: fixed-head fits and a Pro-style Gaussian residual field."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dynamic_handeye.geometry import Assets, apply, torso_from_arm, torso_camera_poses
from dynamic_handeye.dataset import load_capture, arrays
from dynamic_handeye.calibration import fit_geometry, predict, metrics


def kernel(x, centers, length=.06):
    return np.exp(-np.sum((x[:, None]-centers[None])**2, axis=2)/(2*length**2))


def main():
    assets = Assets()
    data = load_capture([ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl'], assets)
    camera, target, angles = arrays(data)
    groups = np.array([r['group_id'] for r in data['rows']])
    heads = np.array([r['head_key'] for r in data['rows']])
    unique_groups = list(dict.fromkeys(groups))
    train = np.isin(groups, unique_groups[:int(.8*len(unique_groups))])
    test = ~train
    assert not (set(groups[train]) & set(groups[test])), 'Ball groups leaked across trajectory split'
    probe = kernel(camera[:5], camera[:5])
    np.testing.assert_allclose(probe, probe.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(probe), 1., atol=1e-12)
    sm = torso_from_arm(assets)
    out = dict(scope='diagnostic only; no production parameters replaced',
               gaussian_ablation='Pro kernel length 60 mm, alpha 3 fixed without Luna tuning; not full Pro joint affine/pF algorithm',
               error_axes='right_arm_mdh_parent', trajectory={})
    model = fit_geometry(camera[train], target[train], angles[train], assets)
    base = predict(model, assets, camera[test], angles[test], output_frame='right_arm_mdh_parent')
    out['trajectory']['common_six_parameter'] = metrics(base-target[test])
    separate = []
    for h in sorted(set(heads)):
        tr, te = train & (heads == h), test & (heads == h)
        fitted = fit_geometry(camera[tr], target[tr], angles[tr], assets)
        error = predict(fitted, assets, camera[te], angles[te], output_frame='right_arm_mdh_parent')-target[te]
        separate.extend(error)
        out['trajectory'][f'independent_fixed_head_{h}'] = metrics(error)
    out['trajectory']['independent_fixed_heads_combined'] = metrics(separate)

    def compensated(tr, te):
        m = fit_geometry(camera[tr], target[tr], angles[tr], assets)
        pose = torso_camera_poses(m, assets, angles[tr])
        desired = apply(np.linalg.inv(pose), apply(sm, target[tr]))
        coefficients = np.linalg.solve(kernel(camera[tr], camera[tr])+3*np.eye(sum(tr)), desired-camera[tr])
        delta = kernel(camera[te], camera[tr]) @ coefficients
        corrected = predict(m, assets, camera[te]+delta, angles[te], output_frame='right_arm_mdh_parent')
        plain = predict(m, assets, camera[te], angles[te], output_frame='right_arm_mdh_parent')
        return dict(baseline=metrics(plain-target[te]), corrected=metrics(corrected-target[te]),
                    correction_norm_p95_mm=float(np.percentile(np.linalg.norm(delta, axis=1)*1000,95)),
                    train_count=int(sum(tr)), test_count=int(sum(te)))
    out['trajectory']['gaussian_ablation'] = compensated(train, test)
    out['leave_one_head_out'] = {h: compensated(heads != h, heads == h) for h in sorted(set(heads))}
    out['limitations'] = ['Single session/right arm only.',
        'Head poses occupy different camera regions: a spatial field can encode head effects indirectly.',
        'No nested model selection or untouched external session.',
        'Independent fixed-head maps cannot be used as one dynamic rigid extrinsic.']
    path = Path(__file__).with_name('comparison.json')
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
