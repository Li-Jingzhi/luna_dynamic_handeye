"""Jointly fit flange-to-ball offset and left-camera-to-arm-MDH registration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .geometry import arm_mdh, flange_pose, solve_joint, triangulate, validate_stereo


def load_rows(path):
    all_rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in all_rows if row.get("capture_valid", True)]
    rejected = len(all_rows) - len(rows)
    if rejected:
        print(f"Ignoring {rejected} waypoints marked capture_valid=false", flush=True)
    if len(rows) < 18:
        raise ValueError("Need at least 18 distinct automatically captured waypoints")
    ids = [row.get("waypoint_index") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Each waypoint must occur once; correlated frames are not separate samples")
    return rows


def fit(capture, stereo, mdh):
    model = json.loads(Path(stereo).read_text(encoding="utf-8"))
    validate_stereo(model)
    rows = load_rows(capture)
    rows_mdh = arm_mdh(mdh)
    expected_hash = mdh_provenance(mdh)
    expected_stereo_hash = hashlib.sha256(Path(stereo).read_bytes()).hexdigest()
    poses, points, stereo_errors = [], [], []
    head_ref = body_ref = None
    for i, row in enumerate(rows):
        if (row.get("capture_kind") != "interactive_ball_auto" or row.get("mdh_sha256") != expected_hash or
                row.get("stereo_model_sha256") != expected_stereo_hash or
                row.get("sensor_ids") != model["sensor_ids"] or row.get("image_size") != model["image_size"] or
                row.get("pixel_domain") != model["pixel_domain"]):
            raise ValueError(f"Row {i} has incompatible robot or camera contract")
        head = np.asarray(row["head_angles_deg"], float)
        body = np.asarray(row["body_q_deg"], float)
        if head.shape != (2,) or body.shape != (6,) or not np.isfinite(head).all() or not np.isfinite(body).all():
            raise ValueError(f"Invalid head/body feedback in row {i}")
        if head_ref is None:
            head_ref, body_ref = head, body
        if np.max(np.abs(head - head_ref)) > .25 or np.max(np.abs(body - body_ref)) > .3:
            raise ValueError("Head/body moved during collection; fixed camera/arm registration is invalid")
        poses.append(flange_pose(row["q7_deg"], rows_mdh))
        legacy_point, error = triangulate(row["stereo_pixels"], model)
        if row.get("sphere_center_left_camera_m") is not None:
            point = np.asarray(row["sphere_center_left_camera_m"], float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(f"Invalid sphere centre in row {i}")
        else:
            point = legacy_point
        points.append(point)
        stereo_errors.append(error)
    poses, points = np.asarray(poses), np.asarray(points)
    # Deterministic split by waypoint, including the whole pose in exactly one split.
    held = np.array([int(hashlib.sha256(str(r["waypoint_index"]).encode()).hexdigest()[:8], 16) % 5 == 0
                     for r in rows])
    if held.sum() < 3 or (~held).sum() < 12:
        held = np.arange(len(rows)) % 5 == 0
    train = ~held
    offset, R, t, residual = solve_joint(poses[train], points[train])
    median = np.median(residual)
    inlier = residual <= max(.01, median + 3 * np.median(np.abs(residual - median)))
    if inlier.sum() < max(12, int(.7 * train.sum())):
        raise ValueError("Too many inconsistent stereo/FK observations")
    train_ids = np.where(train)[0][inlier]
    offset, R, t, train_error = solve_joint(poses[train_ids], points[train_ids])
    if np.linalg.norm(offset) > .2:
        raise ValueError("Estimated ball offset exceeds 200 mm; check MDH, camera scale, and detections")
    predicted = (R @ (poses[:, :3, 3] + np.einsum("nij,j->ni", poses[:, :3, :3], offset)).T).T + t
    errors_mm = np.linalg.norm(predicted - points, axis=1) * 1000
    heldout_p95 = float(np.percentile(errors_mm[held], 95))
    stereo_p95 = float(np.percentile(stereo_errors, 95))
    result = {
        "schema": "luna_sh5_interactive_ball_v1", "frame_definition": "left_camera_from_right_arm_mdh_parent",
        "flange_to_ball_center_m": offset.tolist(),
        "left_camera_from_arm_mdh_R": R.tolist(), "left_camera_from_arm_mdh_t_m": t.tolist(),
        "mdh_sha256": expected_hash, "stereo_sha256": expected_stereo_hash,
        "pixel_domain": model["pixel_domain"], "image_size": model["image_size"], "sensor_ids": model["sensor_ids"],
        "head_angles_deg": head_ref.tolist(), "body_q_deg": body_ref.tolist(),
        "train_waypoints": len(train_ids), "heldout_waypoints": int(held.sum()),
        "train_xyz_p95_mm": float(np.percentile(errors_mm[train_ids], 95)),
        "heldout_xyz_p95_mm": heldout_p95,
        "heldout_xyz_rms_mm": float(np.sqrt(np.mean(errors_mm[held] ** 2))),
        "acceptance_criteria": {"heldout_xyz_p95_mm_lt": 1.0, "stereo_reprojection_p95_px_lt": 0.5},
        "heldout_internal_consistency_passed": heldout_p95 < 1 and stereo_p95 < .5,
        "stereo_reprojection_p95_px": stereo_p95,
        "heldout_errors_mm": errors_mm[held].tolist(),
        "observation_kind": ("sphere_center_projection_stereo_v1" if
                             all(row.get("sphere_center_left_camera_m") is not None for row in rows)
                             else "legacy_ellipse_center_triangulation"),
        "independently_verified": False,
        "warning": "MDH parent frame is not yet verified as whole-robot base; verify p_F and heldout 3D points physically.",
    }
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--stereo", required=True)
    parser.add_argument("--mdh", default=str(root / "configs/luna_new.ini"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fit(args.capture, args.stereo, args.mdh)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(f"p_F = {np.asarray(result['flange_to_ball_center_m']) * 1000} mm; "
          f"heldout p95 = {result['heldout_xyz_p95_mm']:.1f} mm; saved {args.output}")


if __name__ == "__main__":
    main()
