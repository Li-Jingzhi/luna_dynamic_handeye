"""Fuse formal stopped-arm frames into one FK-labelled stereo point per pose."""
import argparse
import json
from pathlib import Path

import numpy as np

from .measurement_contract import DEFAULT_CONTRACT, load_contract, valid_measurement
from .drag_capture import mdh_provenance
from interactive_ball_calibration.geometry import arm_mdh, flange_pose


def fuse(rows, config, ball_model, mdh):
    model = json.loads(Path(ball_model).read_text(encoding="utf-8"))
    mdh_hash = mdh_provenance(mdh)
    pf = np.asarray(model["flange_to_ball_center_m"], float)
    if pf.shape != (3,) or model.get("mdh_sha256") != mdh_hash:
        raise ValueError("Ball offset model does not match the selected MDH")
    chain = arm_mdh(mdh)
    points, group = [], []
    seen_frames = set()

    def finish():
        if len(group) < config["min_valid_frames_per_point"]:
            return
        selected = group[:config["valid_frames_per_point"]]
        pixels = np.asarray([row["stereo_pixels"] for row in selected], float)
        q = np.median([row["q7_deg"] for row in selected], axis=0)
        if (np.max(np.ptp([row["q7_deg"] for row in selected], axis=0)) >
                config["max_stationary_q_range_deg"] or
                np.max(np.ptp([row["head_angles_deg"] for row in selected], axis=0)) > .25 or
                np.max(np.ptp([row["body_q_deg"] for row in selected], axis=0)) > .3):
            return
        disparity = pixels[:, 0]-pixels[:, 2]
        jitter = float(np.percentile(np.abs(disparity-np.median(disparity)), 95))
        if jitter > config["max_point_disparity_jitter_p95_px"]:
            return
        pose = flange_pose(q, chain)
        xyz = pose[:3, 3]+pose[:3, :3]@pf
        points.append({"point_id": len(points)+1, "q7_deg": q.tolist(),
                       "head_angles_deg": selected[0]["head_angles_deg"],
                       "fk_xyz_m": xyz.tolist(), "stereo_pixels": np.median(pixels, axis=0).tolist(),
                       "valid_frames": len(selected), "pixel_std_px": pixels.std(axis=0).tolist(),
                       "pixel_mad_px": np.median(np.abs(pixels-np.median(pixels, axis=0)), axis=0).tolist(),
                       "pixel_p95_jitter_px": np.percentile(
                           np.abs(pixels-np.median(pixels, axis=0)), 95, axis=0).tolist(),
                       "disparity_jitter_p95_px": jitter,
                       "contract_sha256": config["contract_sha256"],
                       "source_calibration_sha256": config["calibration_sha256"]})

    for row in rows:
        if (row.get("capture_policy") != "stationary" or
                "head_angles_deg" not in row or
                np.max(np.abs(np.asarray(row["head_angles_deg"])-config["fixed_head_deg"])) > .25):
            continue
        try:
            valid_measurement(row, config)
        except ValueError:
            continue
        if row.get("mdh_sha256") != mdh_hash:
            continue
        q = np.asarray(row["q7_deg"], float)
        if q.shape != (7,) or not np.isfinite(q).all():
            continue
        if group and np.max(np.abs(q-np.asarray(group[0]["q7_deg"], float))) > \
                config["max_stationary_q_range_deg"]:
            finish()
            group = []
            seen_frames = set()
        frame_number = tuple(row.get("frame_number", ()))
        if frame_number and frame_number not in seen_frames:
            group.append(row)
            seen_frames.add(frame_number)
    finish()
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", nargs="+", required=True)
    parser.add_argument("--ball-model", required=True)
    parser.add_argument("--mdh", default=str(Path(__file__).resolve().parents[1]/"configs/luna_new.ini"))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _ = load_contract(args.contract)
    points = []
    for path in args.capture:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for point in fuse(rows, config, args.ball_model, args.mdh):
            point["point_id"] = len(points)+1
            points.append(point)
    with Path(args.output).open("x", encoding="utf-8") as out:
        for point in points:
            out.write(json.dumps(point, allow_nan=False)+"\n")
    print(f"Saved {len(points)} stopped-arm points to {args.output}; target {config['fixed_head_points']}")


if __name__ == "__main__":
    main()
