"""Fit a bounded joint-zero correction using physical sphere-centre captures.

This is a second-stage fixed-head calibration. Run the rigid sphere registration first.
Only if a fixed rigid registration cannot reach the target across independent trajectories
should this module be used. The original MDH file is never modified.

The fitted model defines a corrected FK convention:
    q_corrected = q_feedback + joint_zero_correction_deg

A fixed rigid transform then maps the left-camera physical sphere centre into that
corrected FK frame. Acceptance requires both ordinary heldout performance and, when
multiple capture trajectories are supplied, leave-one-capture-out generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .fit_projection import grouped_holdout
from .geometry import arm_mdh, flange_pose, rigid_fit


def _load(captures, expected_mdh):
    rows, groups, capture_ids = [], [], []
    for capture_id, capture in enumerate(captures):
        path = Path(capture)
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        items = [row for row in items if row.get("capture_valid", True)]
        for index, row in enumerate(items):
            if (row.get("capture_kind") != "interactive_ball_auto" or
                    row.get("mdh_sha256") != expected_mdh or
                    row.get("pixel_domain") != "ideal_pinhole"):
                raise ValueError(f"Incompatible observation in {capture}:{index + 1}")
            if row.get("sphere_center_method") != "sphere_center_projection_stereo_v1":
                raise ValueError(
                    "Kinematic correction requires new physical sphere-centre captures; "
                    "legacy ellipse-centre data are not valid for this fit")
            center = np.asarray(row.get("sphere_center_left_camera_m"), float)
            q = np.asarray(row.get("q7_deg"), float)
            if (center.shape != (3,) or q.shape != (7,) or
                    not np.isfinite(center).all() or not np.isfinite(q).all()):
                raise ValueError(f"Invalid q7/sphere centre in {capture}:{index + 1}")
            rows.append(row)
            groups.append(f"{path.name}:{index // 5}")
            capture_ids.append(capture_id)
    if len(rows) < 40:
        raise ValueError("Need at least 40 valid physical sphere-centre observations")
    return rows, groups, np.asarray(capture_ids, int)


def _fk_points(q, rows_mdh, pf, dq_deg):
    result = []
    for qi in np.asarray(q, float):
        pose = flange_pose(qi + dq_deg, rows_mdh)
        result.append(pose[:3, 3] + pose[:3, :3] @ pf)
    return np.asarray(result)


def _rigid_residual(q, camera_xyz, rows_mdh, pf, dq_deg, fit_mask):
    fk = _fk_points(q, rows_mdh, pf, dq_deg)
    R, t = rigid_fit(camera_xyz[fit_mask], fk[fit_mask])
    predicted = (R @ camera_xyz.T).T + t
    return R, t, 1000. * (predicted - fk)


def _huber_cost(residual_mm, delta_mm=1.0):
    norm = np.linalg.norm(residual_mm, axis=1)
    return float(np.sum(np.where(norm <= delta_mm,
                                 .5 * norm**2,
                                 delta_mm * (norm - .5 * delta_mm))))


def _optimize_zero(q, camera_xyz, rows_mdh, pf, fit_mask, *,
                   max_zero_deg=2.0, max_iterations=60):
    """Bounded LM-like optimizer, NumPy-only.

    q1 is kept at zero because a constant q1 offset is largely gauge-equivalent to
    the fitted camera/MDH rigid rotation. Optimizing it would make the result less
    identifiable without improving the intended geometry.
    """
    dq = np.zeros(7, float)
    active = np.arange(1, 7)
    damping = 1e-2
    step_eps_deg = 0.002

    for _ in range(max_iterations):
        R, t, residual = _rigid_residual(q, camera_xyz, rows_mdh, pf, dq, fit_mask)
        current = residual[fit_mask]
        cost = _huber_cost(current)
        point_norm = np.linalg.norm(current, axis=1)
        weights = np.where(point_norm <= 1.0, 1.0, 1.0 / np.maximum(point_norm, 1e-9))
        sqrt_w = np.repeat(np.sqrt(weights), 3)
        base = current.ravel()
        J = np.zeros((base.size, len(active)), float)

        for column, joint in enumerate(active):
            plus = dq.copy()
            minus = dq.copy()
            plus[joint] += step_eps_deg
            minus[joint] -= step_eps_deg
            _, _, r_plus = _rigid_residual(
                q, camera_xyz, rows_mdh, pf, plus, fit_mask)
            _, _, r_minus = _rigid_residual(
                q, camera_xyz, rows_mdh, pf, minus, fit_mask)
            J[:, column] = (
                (r_plus[fit_mask] - r_minus[fit_mask]) /
                (2. * step_eps_deg)).ravel()

        JW = J * sqrt_w[:, None]
        rw = base * sqrt_w
        system = JW.T @ JW + damping * np.eye(len(active))
        gradient = JW.T @ rw
        try:
            step = -np.linalg.solve(system, gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(system) @ gradient
        step = np.clip(step, -.25, .25)

        candidate = dq.copy()
        candidate[active] = np.clip(
            candidate[active] + step, -max_zero_deg, max_zero_deg)
        _, _, candidate_residual = _rigid_residual(
            q, camera_xyz, rows_mdh, pf, candidate, fit_mask)
        candidate_cost = _huber_cost(candidate_residual[fit_mask])

        if candidate_cost < cost:
            dq = candidate
            damping = max(damping / 2., 1e-6)
            if np.linalg.norm(step) < 1e-5:
                break
        else:
            damping = min(damping * 5., 1e6)

    R, t, residual = _rigid_residual(
        q, camera_xyz, rows_mdh, pf, dq, fit_mask)
    return dq, R, t, residual


def _stats(residual_mm, mask):
    distance = np.linalg.norm(residual_mm[mask], axis=1)
    axis = residual_mm[mask]
    return {
        "samples": int(mask.sum()),
        "xyz_mean_mm": float(distance.mean()),
        "xyz_rms_mm": float(np.sqrt(np.mean(distance**2))),
        "xyz_p50_mm": float(np.percentile(distance, 50)),
        "xyz_p95_mm": float(np.percentile(distance, 95)),
        "xyz_max_mm": float(distance.max()),
        "axis_bias_mm": axis.mean(axis=0).tolist(),
        "axis_std_mm": axis.std(axis=0).tolist(),
    }


def fit(captures, ball_model, mdh, *, max_zero_deg=2.0):
    model = json.loads(Path(ball_model).read_text(encoding="utf-8"))
    pf = np.asarray(model["flange_to_ball_center_m"], float)
    expected_mdh = mdh_provenance(mdh)
    if pf.shape != (3,) or model.get("mdh_sha256") != expected_mdh:
        raise ValueError("Ball model does not match the current right-arm MDH")
    rows_mdh = arm_mdh(mdh)
    rows, groups, capture_ids = _load(captures, expected_mdh)

    q = np.asarray([row["q7_deg"] for row in rows], float)
    camera_xyz = np.asarray(
        [row["sphere_center_left_camera_m"] for row in rows], float)
    head = np.asarray([row["head_angles_deg"] for row in rows], float)
    body = np.asarray([row["body_q_deg"] for row in rows], float)
    if np.ptp(head, axis=0).max() > .25 or np.ptp(body, axis=0).max() > .3:
        raise ValueError("Head or body moved across fixed-head captures")

    held = grouped_holdout(groups)
    train = ~held

    # Baseline: no kinematic correction, only one rigid camera registration.
    _, _, baseline_residual = _rigid_residual(
        q, camera_xyz, rows_mdh, pf, np.zeros(7), train)

    dq, R, t, corrected_residual = _optimize_zero(
        q, camera_xyz, rows_mdh, pf, train, max_zero_deg=max_zero_deg)

    cross_capture = []
    unique_captures = np.unique(capture_ids)
    if len(unique_captures) >= 2:
        for test_id in unique_captures:
            fit_mask = capture_ids != test_id
            test_mask = capture_ids == test_id
            dq_fold, R_fold, t_fold, fold_residual = _optimize_zero(
                q, camera_xyz, rows_mdh, pf, fit_mask,
                max_zero_deg=max_zero_deg)
            cross_capture.append({
                "test_capture": str(captures[int(test_id)]),
                "train_samples": int(fit_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "joint_zero_correction_deg": dq_fold.tolist(),
                **_stats(fold_residual, test_mask),
            })

    heldout_stats = _stats(corrected_residual, held)
    baseline_heldout = _stats(baseline_residual, held)
    cross_p95 = max(
        (fold["xyz_p95_mm"] for fold in cross_capture), default=None)

    passed = heldout_stats["xyz_p95_mm"] < 1.0
    if cross_p95 is not None:
        passed = passed and cross_p95 < 1.0

    return {
        "schema": "luna_fixed_head_kinematic_correction_v1",
        "frame_id": "right_arm_mdh_parent",
        "source_frame": "head_left_camera",
        "observation_kind": "sphere_center_projection_stereo_v1",
        "units": "m",
        "head_angles_deg": np.median(head, axis=0).tolist(),
        "flange_to_ball_center_m": pf.tolist(),
        "joint_zero_correction_deg": dq.tolist(),
        "joint_zero_gauge": "q1 fixed at 0 deg; optimize q2..q7",
        "max_zero_correction_deg": float(max_zero_deg),
        "R_mdh_from_left_camera": R.tolist(),
        "t_mdh_from_left_camera_m": t.tolist(),
        "mdh_sha256": expected_mdh,
        "ball_model": str(ball_model),
        "capture_files": [str(path) for path in captures],
        "samples_total": len(rows),
        "train_samples": int(train.sum()),
        "heldout_samples": int(held.sum()),
        "baseline_heldout": baseline_heldout,
        "corrected_heldout": heldout_stats,
        "cross_capture_validation": cross_capture,
        "acceptance_criteria": {
            "heldout_xyz_p95_mm_lt": 1.0,
            "cross_capture_each_xyz_p95_mm_lt": 1.0,
        },
        "heldout_internal_consistency_passed": bool(passed),
        "independently_verified": False,
        "warning": (
            "Internal FK/vision consistency only. Do not overwrite the robot MDH from "
            "this file. A correction is accepted only if independent trajectory "
            "generalization also passes; physical whole-robot-base accuracy remains "
            "unverified."),
    }


def apply_camera_to_fk(model, camera_center_m):
    R = np.asarray(model["R_mdh_from_left_camera"], float)
    t = np.asarray(model["t_mdh_from_left_camera_m"], float)
    point = np.asarray(camera_center_m, float)
    if R.shape != (3, 3) or t.shape != (3,) or point.shape != (3,):
        raise ValueError("Invalid kinematic correction model")
    return R @ point + t


def corrected_fk_point(model, q7_deg, mdh):
    rows_mdh = arm_mdh(mdh)
    q = np.asarray(q7_deg, float)
    dq = np.asarray(model["joint_zero_correction_deg"], float)
    pf = np.asarray(model["flange_to_ball_center_m"], float)
    if q.shape != (7,) or dq.shape != (7,) or pf.shape != (3,):
        raise ValueError("Invalid corrected FK input/model")
    pose = flange_pose(q + dq, rows_mdh)
    return pose[:3, 3] + pose[:3, :3] @ pf


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", nargs="+", required=True)
    parser.add_argument("--ball-model", required=True)
    parser.add_argument("--mdh", default=str(root / "configs/luna_new.ini"))
    parser.add_argument("--max-zero-deg", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fit(
        args.capture, args.ball_model, args.mdh,
        max_zero_deg=args.max_zero_deg)
    with Path(args.output).open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(
        f"Saved {args.output}; corrected heldout P95="
        f"{result['corrected_heldout']['xyz_p95_mm']:.3f} mm; "
        f"pass={result['heldout_internal_consistency_passed']}")


if __name__ == "__main__":
    main()
