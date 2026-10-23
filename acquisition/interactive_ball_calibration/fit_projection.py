"""Fit fixed-head stereo projection matrices in the right-arm MDH parent frame."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capt_head.geometry import dlt, project, triangulate
from capt_head.drag_capture import mdh_provenance
from .geometry import arm_mdh, flange_pose


def fit_matrices(xyz, pixels, held):
    train = ~held
    mask = train.copy()
    for _ in range(12):
        left = dlt(xyz[mask], pixels[mask, :2])
        right = dlt(xyz[mask], pixels[mask, 2:])
        error = np.maximum(np.linalg.norm(project(left, xyz) - pixels[:, :2], axis=1),
                           np.linalg.norm(project(right, xyz) - pixels[:, 2:], axis=1))
        updated = train & (error <= 4.)
        if updated.sum() < 20:
            raise ValueError("Projection fit has fewer than 20 training inliers")
        if np.array_equal(updated, mask):
            break
        mask = updated
    return dlt(xyz[mask], pixels[mask, :2]), dlt(xyz[mask], pixels[mask, 2:]), mask


def grouped_holdout(groups):
    groups = np.asarray(groups)
    held = np.array([int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 5 == 0 for g in groups])
    if held.sum() < 12:
        ranked = sorted(set(groups), key=lambda g: hashlib.sha256(g.encode()).hexdigest())
        for group in ranked:
            held |= groups == group
            if held.sum() >= 12:
                break
    if held.sum() < 12 or (~held).sum() < 30:
        raise ValueError("Insufficient grouped train/heldout observations")
    return held


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
                    row.get("stereo_model_sha256") != model.get("stereo_sha256") or
                    row.get("pixel_domain") != "ideal_pinhole"):
                raise ValueError(f"Incompatible observation in {capture}:{index + 1}")
            rows.append(row)
            groups.append(f"{Path(capture).name}:{index // 5}")
    if len(rows) < 40:
        raise ValueError("Need at least 40 fixed-head observations")
    head = np.asarray([r["head_angles_deg"] for r in rows], float)
    body = np.asarray([r["body_q_deg"] for r in rows], float)
    if np.ptp(head, axis=0).max() > .25 or np.ptp(body, axis=0).max() > .3:
        raise ValueError("Head or body moved across fixed-head captures")
    xyz = []
    for row in rows:
        pose = flange_pose(row["q7_deg"], chain)
        xyz.append(pose[:3, 3] + pose[:3, :3] @ pf)
    xyz = np.asarray(xyz)
    pixels = np.asarray([r["stereo_pixels"] for r in rows], float)
    held = grouped_holdout(groups)
    left, right, inliers = fit_matrices(xyz, pixels, held)
    recovered = np.asarray([triangulate(left, right, uv) for uv in pixels[held]])
    delta_mm = (recovered - xyz[held]) * 1000
    distance = np.linalg.norm(delta_mm, axis=1)
    reprojection = np.maximum(
        np.linalg.norm(project(left, xyz[held]) - pixels[held, :2], axis=1),
        np.linalg.norm(project(right, xyz[held]) - pixels[held, 2:], axis=1))
    xyz_p95 = float(np.percentile(distance, 95))
    reprojection_p95 = float(np.percentile(reprojection, 95))
    return {
        "schema": "luna_fixed_head_projection_v1", "frame_id": "right_arm_mdh_parent",
        "image_size": model["image_size"], "pixel_domain": model["pixel_domain"],
        "sensor_ids": model["sensor_ids"],
        "units": "m", "head_angles_deg": np.median(head, axis=0).tolist(),
        "P_left": left.tolist(), "P_right": right.tolist(),
        "flange_to_ball_center_m": pf.tolist(), "mdh_sha256": expected_mdh,
        "ball_model": str(ball_model), "capture_files": [str(p) for p in captures],
        "samples_total": len(rows), "train_samples": int((~held).sum()),
        "train_inliers": int(inliers.sum()), "heldout_samples": int(held.sum()),
        "heldout_xyz_rms_mm": float(np.sqrt(np.mean(distance ** 2))),
        "heldout_xyz_mean_mm": float(distance.mean()),
        "heldout_xyz_p95_mm": xyz_p95,
        "heldout_axis_bias_mm": delta_mm.mean(axis=0).tolist(),
        "heldout_axis_std_mm": delta_mm.std(axis=0).tolist(),
        "heldout_reprojection_p95_px": reprojection_p95,
        "acceptance_criteria": {"heldout_xyz_p95_mm_lt": 1.0, "heldout_reprojection_p95_px_lt": 0.5},
        "heldout_internal_consistency_passed": xyz_p95 < 1 and reprojection_p95 < .5,
        "xyz_bounds_m": np.stack((xyz.min(axis=0), xyz.max(axis=0))).tolist(),
        "independently_verified": False,
        "warning": "Errors are against FK labels in the right-arm MDH parent frame, not independent whole-robot-base metrology."
    }


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
    print(f"Saved {args.output}; heldout XYZ p95={result['heldout_xyz_p95_mm']:.2f} mm, "
          f"reprojection p95={result['heldout_reprojection_p95_px']:.2f} px")


if __name__ == "__main__":
    main()
