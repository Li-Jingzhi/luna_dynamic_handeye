"""Fit a visual-9D to FK-position consistency model.

Runtime input is visual pose only:
    XYZ in the left-camera frame + Rotation6D of the flange/marker frame.

Robot q7 is used only while producing calibration labels.  The fitted model is

    X_FK = R_MC @ (X_visual + R_visual @ delta_visual) + t_MC

where delta_visual is a fixed point offset expressed in the visual flange/marker
frame.  This explicitly models orientation-dependent point-definition mismatch
without using q7 at runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .fit_projection import grouped_holdout
from .geometry import arm_mdh, flange_pose, rigid_fit


def rotation6d_to_matrix(value):
    value = np.asarray(value, float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("Rotation6D must contain six finite values")
    a = value[:3]
    b = value[3:]
    na = np.linalg.norm(a)
    if na < 1e-8:
        raise ValueError("Degenerate Rotation6D first axis")
    x = a / na
    y = b - x * np.dot(x, b)
    ny = np.linalg.norm(y)
    if ny < 1e-8:
        raise ValueError("Degenerate Rotation6D second axis")
    y /= ny
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def matrix_to_rotation6d(R):
    R = np.asarray(R, float)
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")
    return np.r_[R[:, 0], R[:, 1]]


def _load(captures, expected_mdh, chain, pf):
    rows, groups, capture_ids = [], [], []
    visual_xyz, visual_R, target_xyz = [], [], []
    for capture_id, capture in enumerate(captures):
        path = Path(capture)
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        items = [row for row in items if row.get("capture_valid", True)]
        for index, row in enumerate(items):
            if (row.get("mdh_sha256") != expected_mdh or
                    row.get("pixel_domain") != "ideal_pinhole"):
                raise ValueError(f"Incompatible observation in {capture}:{index + 1}")
            xyz = np.asarray(row.get("visual_xyz_camera_m"), float)
            rot6 = row.get("visual_rotation6d_camera")
            q = np.asarray(row.get("q7_deg"), float)
            if (xyz.shape != (3,) or q.shape != (7,) or
                    not np.isfinite(xyz).all() or not np.isfinite(q).all()):
                raise ValueError(f"Invalid 9D/FK observation in {capture}:{index + 1}")
            Rv = rotation6d_to_matrix(rot6)
            pose = flange_pose(q, chain)
            target = pose[:3, 3] + pose[:3, :3] @ pf
            rows.append(row)
            visual_xyz.append(xyz)
            visual_R.append(Rv)
            target_xyz.append(target)
            groups.append(f"{path.name}:{index // 5}")
            capture_ids.append(capture_id)
    if len(rows) < 40:
        raise ValueError("Need at least 40 visual-9D/FK observations")
    return (rows, np.asarray(visual_xyz), np.asarray(visual_R), np.asarray(target_xyz),
            groups, np.asarray(capture_ids, int))


def _solve(visual_xyz, visual_R, target_xyz, fit_mask, iterations=100):
    delta = np.zeros(3, float)
    previous = None
    for _ in range(iterations):
        adjusted = visual_xyz + np.einsum("nij,j->ni", visual_R, delta)
        R, t = rigid_fit(adjusted[fit_mask], target_xyz[fit_mask])
        A = np.einsum("ij,njk->nik", R, visual_R[fit_mask]).reshape(-1, 3)
        rhs = (target_xyz[fit_mask] -
               (R @ visual_xyz[fit_mask].T).T - t).reshape(-1)
        updated = np.linalg.lstsq(A, rhs, rcond=None)[0]
        if np.linalg.norm(updated) > .20:
            raise ValueError("Visual reference offset exceeds 200 mm")
        if previous is not None and np.linalg.norm(updated-delta) < 1e-12:
            delta = updated
            break
        previous = delta
        delta = updated
    adjusted = visual_xyz + np.einsum("nij,j->ni", visual_R, delta)
    R, t = rigid_fit(adjusted[fit_mask], target_xyz[fit_mask])
    predicted = (R @ adjusted.T).T + t
    residual_mm = 1000. * (predicted-target_xyz)
    return delta, R, t, residual_mm


def _stats(residual_mm, mask):
    error = residual_mm[mask]
    distance = np.linalg.norm(error, axis=1)
    return {
        "samples": int(mask.sum()),
        "xyz_mean_mm": float(distance.mean()),
        "xyz_rms_mm": float(np.sqrt(np.mean(distance**2))),
        "xyz_p50_mm": float(np.percentile(distance, 50)),
        "xyz_p95_mm": float(np.percentile(distance, 95)),
        "xyz_max_mm": float(distance.max()),
        "axis_bias_mm": error.mean(axis=0).tolist(),
        "axis_std_mm": error.std(axis=0).tolist(),
    }


def fit(captures, ball_model, mdh):
    ball = json.loads(Path(ball_model).read_text(encoding="utf-8"))
    pf = np.asarray(ball["flange_to_ball_center_m"], float)
    expected_mdh = mdh_provenance(mdh)
    if pf.shape != (3,) or ball.get("mdh_sha256") != expected_mdh:
        raise ValueError("Ball/FK label model does not match current MDH")
    chain = arm_mdh(mdh)
    rows, xyz, rotations, target, groups, capture_ids = _load(
        captures, expected_mdh, chain, pf)
    head = np.asarray([row["head_angles_deg"] for row in rows], float)
    body = np.asarray([row["body_q_deg"] for row in rows], float)
    if np.ptp(head, axis=0).max() > .25 or np.ptp(body, axis=0).max() > .3:
        raise ValueError("Head or body moved across fixed-head 9D captures")

    held = grouped_holdout(groups)
    train = ~held
    delta, R, t, residual = _solve(xyz, rotations, target, train)

    cross = []
    for test_id in np.unique(capture_ids):
        if len(np.unique(capture_ids)) < 2:
            break
        fit_mask = capture_ids != test_id
        test_mask = capture_ids == test_id
        fold_delta, fold_R, fold_t, fold_residual = _solve(
            xyz, rotations, target, fit_mask)
        cross.append({
            "test_capture": str(captures[int(test_id)]),
            "visual_reference_offset_m": fold_delta.tolist(),
            **_stats(fold_residual, test_mask),
        })

    held_stats = _stats(residual, held)
    cross_p95 = max((item["xyz_p95_mm"] for item in cross), default=None)
    passed = held_stats["xyz_p95_mm"] < 1.0
    if cross_p95 is not None:
        passed = passed and cross_p95 < 1.0

    return {
        "schema": "luna_visual9d_fk_consistency_v1",
        "frame_id": "right_arm_mdh_parent",
        "source_frame": "head_left_camera",
        "runtime_input": "visual_xyz_camera_m + visual_rotation6d_camera",
        "uses_q7_at_runtime": False,
        "head_angles_deg": np.median(head, axis=0).tolist(),
        "flange_to_ball_center_m_for_labels": pf.tolist(),
        "visual_reference_offset_m": delta.tolist(),
        "R_mdh_from_left_camera": R.tolist(),
        "t_mdh_from_left_camera_m": t.tolist(),
        "mdh_sha256": expected_mdh,
        "capture_files": [str(path) for path in captures],
        "samples_total": len(rows),
        "train_samples": int(train.sum()),
        "heldout_samples": int(held.sum()),
        "heldout": held_stats,
        "cross_capture_validation": cross,
        "acceptance_criteria": {
            "heldout_xyz_p95_mm_lt": 1.0,
            "cross_capture_each_xyz_p95_mm_lt": 1.0,
        },
        "heldout_internal_consistency_passed": bool(passed),
        "independently_verified": False,
        "warning": (
            "Numerical FK/vision consistency model. q7 is a calibration label only; "
            "runtime mapping is visual-9D only. Validate on an untouched third trajectory "
            "before claiming submillimeter generalization."),
    }


def apply(model, visual_xyz_camera_m, visual_rotation6d_camera):
    xyz = np.asarray(visual_xyz_camera_m, float)
    Rv = rotation6d_to_matrix(visual_rotation6d_camera)
    delta = np.asarray(model["visual_reference_offset_m"], float)
    R = np.asarray(model["R_mdh_from_left_camera"], float)
    t = np.asarray(model["t_mdh_from_left_camera_m"], float)
    if xyz.shape != (3,) or delta.shape != (3,) or R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("Invalid visual-9D consistency model/input")
    return R @ (xyz + Rv @ delta) + t


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", nargs="+", required=True)
    parser.add_argument("--ball-model", required=True)
    parser.add_argument("--mdh", default=str(root / "configs/luna_new.ini"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fit(args.capture, args.ball_model, args.mdh)
    with Path(args.output).open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Saved {args.output}; heldout P95={result['heldout']['xyz_p95_mm']:.3f} mm; "
          f"pass={result['heldout_internal_consistency_passed']}")


if __name__ == "__main__":
    main()
