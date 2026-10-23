"""Yellow-only fixed-head camera-XYZ to numerical FK-XYZ calibration.

This intentionally uses exactly ONE capture and does not use marker orientation or
q7 at runtime. q7 only constructs calibration FK labels. The terminal contiguous
20% is reserved as an untouched validation segment; model selection uses temporal
block CV inside the first 80% only. Never report the training fit as generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .geometry import arm_mdh, flange_pose, rigid_fit, solve_joint


def features(camera_xyz, order=1):
    x, y, z = ((np.asarray(camera_xyz) - [0.08, 0.08, 0.38]) / 0.2).T
    terms = [np.ones(len(x)), x, y, z]
    if order >= 2:
        terms += [x*x, y*y, z*z, x*y, x*z, y*z]
    return np.column_stack(terms)


def rbf_kernel(x, y, length_m):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d2 = np.maximum(0., (x*x).sum(1)[:, None] +
                    (y*y).sum(1)[None, :] - 2*x@y.T)
    return np.exp(-d2/(2.*length_m**2))


def metrics(residual_m):
    residual = np.asarray(residual_m, float)*1000.
    if (residual.ndim != 2 or residual.shape[1] != 3 or len(residual) == 0
            or not np.isfinite(residual).all()):
        raise ValueError("Expected nonempty finite Nx3 residuals")
    d = np.linalg.norm(residual, axis=1)
    return {
        "samples": len(d),
        "xyz_mean_mm": float(d.mean()),
        "xyz_rms_mm": float(np.sqrt(np.mean(d**2))),
        "xyz_p50_mm": float(np.percentile(d, 50)),
        "xyz_p95_mm": float(np.percentile(d, 95)),
        "xyz_max_mm": float(d.max()),
        "distance_std_mm": float(d.std()),
        "axis_bias_mm": residual.mean(0).tolist(),
        "axis_std_mm": residual.std(0).tolist(),
    }


def _load(capture, mdh):
    rows = [json.loads(s) for s in Path(capture).read_text(encoding="utf-8").splitlines()
            if s.strip()]
    valid = [r for r in rows if r.get("capture_valid", True)]
    if len(valid) < 80:
        raise ValueError("Yellow-only position consistency requires >=80 unique waypoints")
    expected = mdh_provenance(mdh)
    ids = [r["waypoint_index"] for r in valid]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate waypoint labels; do not treat frames as independent points")
    for row in valid:
        if (row.get("capture_kind") != "interactive_ball_auto" or
                row.get("mdh_sha256") != expected or
                row.get("pixel_domain") != "ideal_pinhole" or
                row.get("image_size") != [1280, 992] or
                row.get("sphere_center_method") != "sphere_center_projection_stereo_v1" or
                int(row.get("valid_frames", 0)) < 50):
            raise ValueError("Incompatible/incomplete formal yellow sphere capture")
    for key in ("source_calibration_sha256", "stereo_model_sha256", "sensor_ids"):
        if len({json.dumps(r.get(key), sort_keys=True) for r in valid}) != 1:
            raise ValueError(f"Mixed calibration/IDs: {key}")
    h = np.asarray([r["head_angles_deg"] for r in valid], float)
    body = np.asarray([r["body_q_deg"] for r in valid], float)
    if (h.shape != (len(valid), 2) or body.shape != (len(valid), 6)
            or not np.isfinite(h).all() or not np.isfinite(body).all()):
        raise ValueError("Invalid head/body feedback")
    if np.ptp(h, axis=0).max() > .25 or np.ptp(body, axis=0).max() > .3:
        raise ValueError("Moving head/body violates fixed-head contract")
    Q = np.asarray([r["q7_deg"] for r in valid], float)
    camera = np.asarray([r["sphere_center_left_camera_m"] for r in valid], float)
    if (Q.shape != (len(valid), 7) or camera.shape != (len(valid), 3)
            or not np.isfinite(Q).all() or not np.isfinite(camera).all()
            or np.any(camera[:, 2] <= 0)):
        raise ValueError("Invalid q7 or sphere observations")
    return rows, valid, Q, camera


def _base_fit(poses, camera, train_indices):
    pf, R_cam_from_mdh, t_cam_from_mdh, _ = solve_joint(
        poses[train_indices], camera[train_indices], iterations=120)
    if not np.isfinite(pf).all() or np.linalg.norm(pf) > .2:
        raise ValueError("Unphysical effective flange offset")
    target = poses[:, :3, 3] + np.einsum("nij,j->ni", poses[:, :3, :3], pf)
    # row-vector inverse mapping: camera XYZ to MDH parent XYZ
    base_pred = (camera - t_cam_from_mdh) @ R_cam_from_mdh
    return pf, R_cam_from_mdh, t_cam_from_mdh, target, base_pred


def _fit_residual(camera, target, base, ids, spec):
    kind, size, ridge = spec
    if kind == "rigid":
        return {"kind": kind}
    errors_mm = 1000. * (target[ids] - base[ids])
    if kind == "poly":
        design = features(camera[ids], int(size))
        penalty = np.ones(design.shape[1])
        penalty[0] = 0
        coef = np.linalg.solve(design.T @ design + ridge*np.diag(penalty),
                               design.T @ errors_mm)
        return {"kind": kind, "order": int(size), "coef_mm": coef.tolist()}
    if kind == "rbf":
        K = rbf_kernel(camera[ids], camera[ids], float(size))
        coef = np.linalg.solve(K + ridge*np.eye(len(ids)), errors_mm)
        return {"kind": kind, "length_m": float(size), "alpha": float(ridge),
                "centers_m": camera[ids].tolist(), "coef_mm": coef.tolist()}
    raise ValueError(f"Unknown residual: {kind}")


def _apply_residual(model, camera):
    kind = model["kind"]
    if kind == "rigid":
        return np.zeros((len(camera), 3), float)
    if kind == "poly":
        return features(camera, model["order"]) @ np.asarray(model["coef_mm"])/1000.
    if kind == "rbf":
        return (rbf_kernel(camera, np.asarray(model["centers_m"]),
                           model["length_m"]) @ np.asarray(model["coef_mm"]))/1000.
    raise ValueError(f"Unknown residual kind: {kind}")


def candidate_specs():
    yield ("rigid", 0, 0.)
    for degree in (1, 2, 3):
        for alpha in (.01, 1., 100.):
            yield ("poly", degree, alpha)
    for length in (.015, .035, .07, .14, .30):
        for alpha in (.01, 1., 100.):
            yield ("rbf", length, alpha)


def fit(capture, mdh, *, train_fraction=.8):
    if not .65 <= train_fraction <= .85:
        raise ValueError("Keep >=15% completely untouched terminal validation points")
    raw, rows, q, camera = _load(capture, mdh)
    chain = arm_mdh(mdh)
    poses = np.asarray([flange_pose(row, chain) for row in q])
    n = len(rows)
    cut = int(n*train_fraction)
    train = np.arange(cut, dtype=int)
    untouched = np.arange(cut, n, dtype=int)
    # Temporal blocks: nearby arm positions cannot leak across tuning folds.
    folds = np.array_split(train, 5)
    validation = []
    for spec in candidate_specs():
        fold_p95 = []
        for held in folds:
            inner_train = np.setdiff1d(train, held)
            try:
                pf, R, t, target, base = _base_fit(poses, camera, inner_train)
                model = _fit_residual(camera, target, base, inner_train, spec)
                pred = base[held] + _apply_residual(model, camera[held])
                fold_p95.append(metrics(pred-target[held])["xyz_p95_mm"])
            except (ValueError, np.linalg.LinAlgError):
                fold_p95 = [float("inf")]
                break
        validation.append({"spec": spec, "fold_p95_mm": fold_p95,
                           "worst_fold_p95_mm": max(fold_p95)})
    best = min(validation, key=lambda r: (r["worst_fold_p95_mm"],
                                          sum(r["fold_p95_mm"])))
    selected = tuple(best["spec"])
    pf, R, t, target, base = _base_fit(poses, camera, train)
    correction = _fit_residual(camera, target, base, train, selected)
    train_prediction = base[train] + _apply_residual(correction, camera[train])
    test_prediction = base[untouched] + _apply_residual(correction, camera[untouched])

    baseline = metrics(base[untouched]-target[untouched])
    train_score = metrics(train_prediction-target[train])
    held_score = metrics(test_prediction-target[untouched])

    near_mm = np.sqrt(np.maximum(0.,
        np.min((camera[untouched]**2).sum(axis=1)[:,None] +
               (camera[train]**2).sum(axis=1)[None,:] -
               2*camera[untouched]@camera[train].T, axis=1)))*1000.
    nearest = {
        "median_mm": float(np.median(near_mm)),
        "p95_mm": float(np.percentile(near_mm,95)),
        "max_mm": float(near_mm.max()),
        "fraction_over_20mm": float(np.mean(near_mm>20.)),
    }

    # The report never calls in-sample performance an acceptance result.
    passed = bool(held_score["xyz_p95_mm"] < 1.0)
    return {
        "schema": "luna_single_yellow_xyz_consistency_v1",
        "capture_file": str(capture),
        "capture_sha256": hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
        "capture_total_rows": len(raw),
        "capture_valid_rows": n,
        "train_samples": cut,
        "untouched_samples": n-cut,
        "split": (f"chronological_{cut}_{n-cut}; hyperparameters selected "
                  "only by 5 temporal folds within train"),
        "input": "sphere_center_left_camera_m (XYZ only; no four-ball, no q7 at runtime)",
        "target": "FK(q7,pF_effective) for calibration labels only",
        "frame_id": "right_arm_mdh_parent",
        "head_angles_deg": np.median([row["head_angles_deg"] for row in rows],axis=0).tolist(),
        "mdh_sha256": mdh_provenance(mdh),
        "flange_to_ball_effective_mm": (1000*pf).tolist(),
        "camera_from_mdh_R": R.tolist(),
        "camera_from_mdh_t_m": t.tolist(),
        "residual": correction,
        "model_selection": {
            "selected_spec": list(selected),
            "selected_inner_fold_p95_mm": best["fold_p95_mm"],
            "selected_worst_fold_p95_mm": best["worst_fold_p95_mm"],
        },
        "training_xyz": train_score,
        "untouched_baseline_xyz": baseline,
        "untouched_corrected_xyz": held_score,
        "untouched_nearest_training_distance": nearest,
        "untouched_xyz_p95_under_1mm": passed,
        "independent_new_trajectory_verified": False,
        "do_not_deploy": True,
        "deployment_block_reason": (
            "Untouched holdout P95 >=1mm" if not passed else
            "One-capture holdout alone cannot certify new-trajectory generalization"),
        "warning": (
            "This defines numerical agreement to an effective FK point, not verified true "
            "world coordinates. A single yellow sphere has no observable orientation. "
            "Even a passing chronological heldout would require an independent new "
            "trajectory before deployment. Never use robot q7/FK at runtime to create "
            "a circular self-confirming vision output."),
    }


def apply(model, camera_xyz):
    if model.get("do_not_deploy", True):
        raise ValueError("Model has not passed yellow-only independent validation")
    camera = np.asarray(camera_xyz, float)
    if camera.shape != (3,) or not np.isfinite(camera).all():
        raise ValueError("Expected a finite left-camera XYZ")
    R = np.asarray(model["camera_from_mdh_R"], float)
    t = np.asarray(model["camera_from_mdh_t_m"], float)
    base = (camera-t)@R
    return base + _apply_residual(model["residual"],camera[None,:])[0]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--mdh", default=str(root/"configs/luna_new.ini"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fit(args.capture, args.mdh)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Saved {path}; untouched P95 = {result['untouched_corrected_xyz']['xyz_p95_mm']:.3f} mm; "
          f"passed={result['untouched_xyz_p95_under_1mm']}")


if __name__ == "__main__":
    main()
