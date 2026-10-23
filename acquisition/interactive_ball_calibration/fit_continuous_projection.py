"""Fit fixed-head projection matrices from a continuously dragged ball trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from capt_head.geometry import project, triangulate
from .fit_projection import fit_matrices
from .geometry import arm_mdh, flange_pose, triangulate as stereo_triangulate, validate_stereo


def _metrics(left, right, xyz, pixels, mask):
    delta = []
    for point, uv in zip(xyz[mask], pixels[mask]):
        delta.append((triangulate(left, right, uv) - point) * 1000)
    delta = np.asarray(delta)
    distance = np.linalg.norm(delta, axis=1)
    reprojection = np.maximum(
        np.linalg.norm(project(left, xyz[mask]) - pixels[mask, :2], axis=1),
        np.linalg.norm(project(right, xyz[mask]) - pixels[mask, 2:], axis=1))
    return {
        "samples": int(mask.sum()),
        "xyz_mean_mm": float(distance.mean()),
        "xyz_rms_mm": float(np.sqrt(np.mean(distance ** 2))),
        "xyz_median_mm": float(np.median(distance)),
        "xyz_p95_mm": float(np.percentile(distance, 95)),
        "xyz_max_mm": float(distance.max()),
        "axis_bias_mm": delta.mean(axis=0).tolist(),
        "axis_std_mm": delta.std(axis=0).tolist(),
        "reprojection_mean_px": float(reprojection.mean()),
        "reprojection_p95_px": float(np.percentile(reprojection, 95)),
        "reprojection_max_px": float(reprojection.max()),
    }


def fit(capture, ball_model, stereo_model, mdh, assumed_head_deg=(0., -25.)):
    capture = Path(capture)
    rows = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 100:
        raise ValueError("Need at least 100 continuous stereo/q7 samples")
    ball = json.loads(Path(ball_model).read_text(encoding="utf-8"))
    stereo = json.loads(Path(stereo_model).read_text(encoding="utf-8"))
    validate_stereo(stereo)
    expected_mdh = mdh_provenance(mdh)
    expected_stereo = hashlib.sha256(Path(stereo_model).read_bytes()).hexdigest()
    if ball.get("mdh_sha256") != expected_mdh or ball.get("stereo_sha256") != expected_stereo:
        raise ValueError("Ball offset model does not match the current MDH/stereo calibration")
    for index, row in enumerate(rows):
        if (row.get("capture_policy") != "continuous" or row.get("mdh_sha256") != expected_mdh or
                row.get("pixel_domain") != stereo["pixel_domain"] or
                row.get("image_size") != stereo["image_size"] or row.get("sensor_ids") != stereo["sensor_ids"]):
            raise ValueError(f"Incompatible continuous observation at row {index + 1}")
    body = np.asarray([row["body_q_deg"] for row in rows], float)
    if np.ptp(body, axis=0).max() > .3:
        raise ValueError("Body moved during fixed-head capture")
    times = np.asarray([row["camera_timestamp_monotonic_ns"] for row in rows], float) / 1e9
    if np.any(np.diff(times) <= 0):
        raise ValueError("Camera timestamps must increase strictly")
    q7 = np.asarray([row["q7_deg"] for row in rows], float)
    pixels = np.asarray([row["stereo_pixels"] for row in rows], float)
    if q7.shape != (len(rows), 7) or pixels.shape != (len(rows), 4):
        raise ValueError("Invalid continuous q7/pixel vectors")
    precompensated = np.asarray([row.get("camera_pipeline_delay_ms", 0.) for row in rows], float)
    if not np.isfinite(precompensated).all() or np.ptp(precompensated) > 1e-6:
        raise ValueError("Continuous rows mix different camera delay compensation")
    precompensated_ms = float(np.median(precompensated))

    selected = []
    rejected_stereo = rejected_timing = 0
    last_time = -np.inf
    for index, row in enumerate(rows):
        if row["camera_age_ms"] > 10 or row["pair_dequeue_skew_ms"] > 10:
            rejected_timing += 1
            continue
        try:
            stereo_triangulate(row["stereo_pixels"], stereo)
        except ValueError:
            rejected_stereo += 1
            continue
        if times[index] - last_time < .1:
            continue
        selected.append(index)
        last_time = times[index]
    selected = np.asarray(selected, int)
    if len(selected) < 100:
        raise ValueError("Fewer than 100 samples remain after timing/stereo filtering")
    blocks = ((times[selected] - times[0]) // 10).astype(int)
    held0 = blocks % 5 == 4
    if held0.sum() < 30 or (~held0).sum() < 60:
        raise ValueError("Insufficient isolated trajectory blocks for validation")

    chain = arm_mdh(mdh)
    offset = np.asarray(ball["flange_to_ball_center_m"], float)
    candidates = []
    delay_candidates = [0] if precompensated_ms > 0 else range(0, 201, 5)
    for delay_ms in delay_candidates:
        target = times[selected] - delay_ms / 1000
        usable = target >= times[0]
        source = selected[usable]
        held = held0[usable]
        q_interp = np.column_stack([np.interp(target[usable], times, q7[:, joint]) for joint in range(7)])
        xyz = []
        for q in q_interp:
            pose = flange_pose(q, chain)
            xyz.append(pose[:3, 3] + pose[:3, :3] @ offset)
        xyz = np.asarray(xyz)
        try:
            left, right, inliers = fit_matrices(xyz, pixels[source], held)
        except ValueError:
            continue
        reprojection = np.maximum(
            np.linalg.norm(project(left, xyz) - pixels[source, :2], axis=1),
            np.linalg.norm(project(right, xyz) - pixels[source, 2:], axis=1))
        candidates.append((int(inliers.sum()), -float(np.median(reprojection[~held])), delay_ms,
                           source, held, q_interp, xyz, left, right, inliers, reprojection))
    if not candidates:
        raise ValueError("No camera-delay candidate produced a valid projection fit")
    best = max(candidates, key=lambda item: item[:2])
    inlier_count, _, delay_ms, source, held, q_interp, xyz, left, right, inliers, reprojection = best
    processed = []
    for local, source_index in enumerate(source):
        processed.append({
            "source_row": int(source_index + 1),
            "trajectory_block": int(((times[source_index] - times[0]) // 10)),
            "split": "heldout" if held[local] else "train",
            "train_inlier": bool(inliers[local]),
            "camera_delay_ms": precompensated_ms + delay_ms,
            "interpolated_q7_deg": q_interp[local].tolist(),
            "xyz_right_arm_mdh_parent_m": xyz[local].tolist(),
            "stereo_pixels": pixels[source_index].tolist(),
            "source_camera_timestamp_monotonic_ns": rows[source_index]["camera_timestamp_monotonic_ns"],
        })
    result = {
        "schema": "luna_fixed_head_continuous_projection_v1",
        "frame_id": "right_arm_mdh_parent",
        "units": "m",
        "capture_file": str(capture),
        "ball_model": str(ball_model),
        "stereo_model": str(stereo_model),
        "mdh_sha256": expected_mdh,
        "stereo_sha256": expected_stereo,
        "assumed_head_angles_deg": list(map(float, assumed_head_deg)),
        "head_angles_verified": False,
        "camera_delay_ms": precompensated_ms + delay_ms,
        "capture_precompensated_delay_ms": precompensated_ms,
        "additional_fitted_delay_ms": delay_ms,
        "delay_selection": ("capture already compensated; no additional delay fitted" if precompensated_ms > 0 else
                            "0..200 ms in 5 ms steps; maximize <=4 px training inliers, tie-break training median"),
        "raw_samples": len(rows),
        "processed_samples": len(source),
        "rejected_timing": rejected_timing,
        "rejected_stereo_geometry": rejected_stereo,
        "train_samples": int((~held).sum()),
        "train_inliers": inlier_count,
        "heldout_samples": int(held.sum()),
        "heldout_blocks": sorted(set(int(x) for x in ((times[source] - times[0]) // 10)[held])),
        "P_left": left.tolist(),
        "P_right": right.tolist(),
        "xyz_bounds_m": np.stack((xyz.min(axis=0), xyz.max(axis=0))).tolist(),
        "train_all": _metrics(left, right, xyz, pixels[source], ~held),
        "train_inlier": _metrics(left, right, xyz, pixels[source], inliers),
        "heldout": _metrics(left, right, xyz, pixels[source], held),
        "independently_verified": False,
        "warning": "Errors use FK+pF labels in the right-arm MDH parent frame. Head [0,-25] is assumed because the raw capture did not record head feedback.",
    }
    return result, processed


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--ball-model", required=True)
    parser.add_argument("--stereo", required=True)
    parser.add_argument("--mdh", default=str(root / "configs/luna_new.ini"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--processed-output", required=True)
    args = parser.parse_args()
    result, processed = fit(args.capture, args.ball_model, args.stereo, args.mdh)
    for target in (Path(args.output), Path(args.processed_output)):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with Path(args.processed_output).open("x", encoding="utf-8") as stream:
        for row in processed:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved {args.output}; delay={result['camera_delay_ms']} ms, "
          f"heldout XYZ p95={result['heldout']['xyz_p95_mm']:.2f} mm")


if __name__ == "__main__":
    main()
