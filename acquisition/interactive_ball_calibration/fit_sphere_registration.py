"""Fit fixed-head left-camera sphere-centre coordinates directly to FK labels.

This is the preferred fixed-head position calibration for the 40 mm flange ball.
It does not assume that the centre of the observed silhouette ellipse is the
perspective projection of the physical sphere centre.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .fit_projection import grouped_holdout
from .geometry import arm_mdh, flange_pose, rigid_fit


def fit(captures, ball_model, mdh):
    model = json.loads(Path(ball_model).read_text(encoding="utf-8"))
    pf = np.asarray(model["flange_to_ball_center_m"], float)
    expected_mdh = mdh_provenance(mdh)
    if pf.shape != (3,) or model.get("mdh_sha256") != expected_mdh:
        raise ValueError("Ball model does not match the current right-arm MDH")
    chain = arm_mdh(mdh)

    rows, groups = [], []
    for capture in captures:
        items = [json.loads(line) for line in Path(capture).read_text().splitlines() if line.strip()]
        items = [row for row in items if row.get("capture_valid", True)]
        for index, row in enumerate(items):
            if (row.get("capture_kind") != "interactive_ball_auto" or
                    row.get("mdh_sha256") != expected_mdh or
                    row.get("pixel_domain") != "ideal_pinhole"):
                raise ValueError(f"Incompatible observation in {capture}:{index + 1}")
            if row.get("sphere_center_method") != "sphere_center_projection_stereo_v1":
                raise ValueError("Sphere-aware fixed-head registration requires corrected-centre captures")
            center = np.asarray(row.get("sphere_center_left_camera_m"), float)
            if center.shape != (3,) or not np.isfinite(center).all() or center[2] <= 0:
                raise ValueError("Invalid physical sphere centre in capture")
            rows.append(row)
            groups.append(f"{Path(capture).name}:{index // 5}")

    if len(rows) < 40:
        raise ValueError("Need at least 40 fixed-head sphere-centre observations")
    head = np.asarray([row["head_angles_deg"] for row in rows], float)
    body = np.asarray([row["body_q_deg"] for row in rows], float)
    if np.ptp(head, axis=0).max() > .25 or np.ptp(body, axis=0).max() > .3:
        raise ValueError("Head or body moved across fixed-head captures")

    fk_xyz = []
    camera_xyz = []
    for row in rows:
        pose = flange_pose(row["q7_deg"], chain)
        fk_xyz.append(pose[:3, 3] + pose[:3, :3] @ pf)
        camera_xyz.append(row["sphere_center_left_camera_m"])
    fk_xyz = np.asarray(fk_xyz, float)
    camera_xyz = np.asarray(camera_xyz, float)

    held = grouped_holdout(groups)
    train = ~held
    R, t = rigid_fit(camera_xyz[train], fk_xyz[train])

    predicted = (R @ camera_xyz.T).T + t
    delta_mm = 1000. * (predicted - fk_xyz)
    distance_mm = np.linalg.norm(delta_mm, axis=1)
    held_distance = distance_mm[held]

    disparity_values = np.asarray([
        float(row.get("sphere_corrected_disparity_jitter_p95_px", np.nan)) for row in rows
    ])
    jitter_values = np.asarray([
        float(row.get("sphere_center_p95_jitter_mm", np.nan)) for row in rows
    ])

    result = {
        "schema": "luna_fixed_head_sphere_registration_v1",
        "frame_id": "right_arm_mdh_parent",
        "source_frame": "head_left_camera",
        "observation_kind": "sphere_center_projection_stereo_v1",
        "units": "m",
        "head_angles_deg": np.median(head, axis=0).tolist(),
        "R_mdh_from_left_camera": R.tolist(),
        "t_mdh_from_left_camera_m": t.tolist(),
        "flange_to_ball_center_m": pf.tolist(),
        "mdh_sha256": expected_mdh,
        "ball_model": str(ball_model),
        "capture_files": [str(path) for path in captures],
        "samples_total": len(rows),
        "train_samples": int(train.sum()),
        "heldout_samples": int(held.sum()),
        "heldout_xyz_mean_mm": float(held_distance.mean()),
        "heldout_xyz_rms_mm": float(np.sqrt(np.mean(held_distance**2))),
        "heldout_xyz_p95_mm": float(np.percentile(held_distance, 95)),
        "heldout_axis_bias_mm": delta_mm[held].mean(axis=0).tolist(),
        "heldout_axis_std_mm": delta_mm[held].std(axis=0).tolist(),
        "sphere_corrected_disparity_jitter_p95_px": (
            float(np.nanpercentile(disparity_values, 95))
            if np.isfinite(disparity_values).any() else None),
        "sphere_center_point_jitter_p95_mm": (
            float(np.nanpercentile(jitter_values, 95)) if np.isfinite(jitter_values).any() else None),
        "acceptance_criteria": {
            "heldout_xyz_p95_mm_lt": 1.0,
            "sphere_corrected_disparity_jitter_p95_px_lte": 0.1,
        },
        "xyz_bounds_m": np.stack((fk_xyz.min(axis=0), fk_xyz.max(axis=0))).tolist(),
        "independently_verified": False,
    }
    disparity_ok = (result["sphere_corrected_disparity_jitter_p95_px"] is None or
                    result["sphere_corrected_disparity_jitter_p95_px"] <= .1)
    result["heldout_internal_consistency_passed"] = bool(
        result["heldout_xyz_p95_mm"] < 1.0 and disparity_ok)
    result["warning"] = (
        "Internal FK/vision consistency only. The corrected sphere-centre projection "
        "uses the known 40 mm diameter and stereo depth; whole-robot-base physical "
        "accuracy remains independently unverified.")
    return result


def apply(model, camera_center_m):
    """Map one recovered sphere centre in C_L to the configured FK coordinate frame."""
    R = np.asarray(model["R_mdh_from_left_camera"], float)
    t = np.asarray(model["t_mdh_from_left_camera_m"], float)
    point = np.asarray(camera_center_m, float)
    if R.shape != (3, 3) or t.shape != (3,) or point.shape != (3,):
        raise ValueError("Invalid fixed-head sphere registration")
    return R @ point + t


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
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Saved {args.output}; heldout XYZ p95={result['heldout_xyz_p95_mm']:.3f} mm; "
          f"pass={result['heldout_internal_consistency_passed']}")


if __name__ == "__main__":
    main()
