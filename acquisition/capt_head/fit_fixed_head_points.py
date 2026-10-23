"""Fit FK-consistent fixed-head stereo projections from fused independent poses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import project, triangulate
from .measurement_contract import DEFAULT_CONTRACT, load_contract
from interactive_ball_calibration.fit_projection import fit_matrices


def fit(points, config):
    if len(points) < config["fixed_head_points"]:
        raise ValueError(f"Need {config['fixed_head_points']} independent stopped-arm poses")
    if any(point.get("contract_sha256") != config["contract_sha256"] or
           point.get("source_calibration_sha256") != config["calibration_sha256"] or
           point.get("valid_frames", 0) < config["min_valid_frames_per_point"] or
           point.get("disparity_jitter_p95_px", 1e9) > config["max_point_disparity_jitter_p95_px"]
           for point in points):
        raise ValueError("Fused points violate the current measurement contract")
    xyz = np.asarray([point["fk_xyz_m"] for point in points], float)
    pixels = np.asarray([point["stereo_pixels"] for point in points], float)
    if xyz.shape != (len(points), 3) or pixels.shape != (len(points), 4) or not np.isfinite(xyz).all() or not np.isfinite(pixels).all():
        raise ValueError("Invalid FK or stereo point matrix")
    held = np.asarray([int(hashlib.sha256(str(point["point_id"]).encode()).hexdigest()[:8], 16)
                       % 100 < round(100*config["heldout_fraction"]) for point in points])
    if held.sum() < 30 or (~held).sum() < 100:
        raise ValueError("Insufficient independent train and heldout poses")
    left, right, inliers = fit_matrices(xyz, pixels, held)
    recovered = np.asarray([triangulate(left, right, uv) for uv in pixels[held]])
    delta_mm = 1000*(recovered-xyz[held])
    distance = np.linalg.norm(delta_mm, axis=1)
    reprojection = np.maximum(np.linalg.norm(project(left, xyz[held])-pixels[held, :2], axis=1),
                              np.linalg.norm(project(right, xyz[held])-pixels[held, 2:], axis=1))
    result = {"schema": "luna_fixed_head_fk_consistency_v1", "frame_id": "right_arm_mdh_parent",
              "pixel_domain": config["pixel_domain"], "image_size": config["image_size"],
              "sensor_ids": config["sensor_ids"], "head_angles_deg": config["fixed_head_deg"],
              "contract_sha256": config["contract_sha256"],
              "source_calibration_sha256": config["calibration_sha256"],
              "P_left": left.tolist(), "P_right": right.tolist(),
              "points": len(points), "training_points": int((~held).sum()),
              "training_inliers": int(inliers.sum()), "heldout_points": int(held.sum()),
              "heldout_xyz_mean_mm": float(distance.mean()),
              "heldout_xyz_rms_mm": float(np.sqrt(np.mean(distance**2))),
              "heldout_xyz_p95_mm": float(np.percentile(distance, 95)),
              "heldout_axis_bias_mm": delta_mm.mean(axis=0).tolist(),
              "heldout_axis_std_mm": delta_mm.std(axis=0).tolist(),
              "heldout_reprojection_p95_px": float(np.percentile(reprojection, 95))}
    result["acceptance_pass"] = (result["heldout_xyz_p95_mm"] < config["max_heldout_xyz_p95_mm"] and
                                 result["heldout_reprojection_p95_px"] < config["max_heldout_reprojection_p95_px"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _ = load_contract(args.contract)
    points = [json.loads(line) for line in Path(args.points).read_text().splitlines() if line.strip()]
    result = fit(points, config)
    with Path(args.output).open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"heldout P95={result['heldout_xyz_p95_mm']:.3f} mm; pass={result['acceptance_pass']}")


if __name__ == "__main__":
    main()
