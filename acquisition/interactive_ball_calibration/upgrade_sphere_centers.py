"""Upgrade legacy stopped-ball captures with sphere-centre projection correction.

The legacy files store the subpixel centres of the left/right sphere silhouettes.
For a finite-radius sphere those ellipse centres are not exactly the perspective
projections of the physical 3-D centre.  This tool keeps the original pixels and
adds the stable known-radius stereo correction used by the current capture path.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import stereo_sphere_center, validate_stereo


def upgrade(rows, stereo, diameter_m=.04):
    validate_stereo(stereo)
    result = []
    for index, source in enumerate(rows):
        row = dict(source)
        if not row.get("capture_valid", True):
            result.append(row)
            continue
        pixels = np.asarray(row.get("stereo_pixels"), float)
        if pixels.shape != (4,) or not np.isfinite(pixels).all():
            raise ValueError(f"Invalid stereo pixels at row {index + 1}")
        sphere = stereo_sphere_center(
            {"center": pixels[:2].tolist()},
            {"center": pixels[2:].tolist()},
            stereo, diameter_m)
        row.update(
            sphere_center_method=sphere["method"],
            yellow_ball_diameter_m=float(diameter_m),
            sphere_center_left_camera_m=sphere["center_left_camera_m"],
            sphere_center_corrected_stereo_pixels=sphere["corrected_stereo_pixels"],
            sphere_projection_correction_px=sphere["projection_correction_px"],
            sphere_center_stereo_reprojection_p95_px=sphere["stereo_reprojection_error_px"],
            legacy_sphere_center_upgrade=True,
        )
        # A legacy row is already a 50-frame fused observation.  Its original
        # disparity-jitter metric remains the best available per-point stability
        # measurement and is copied to the corrected field for compatibility.
        if row.get("disparity_jitter_p95_px") is not None:
            row["sphere_corrected_disparity_jitter_p95_px"] = float(
                row["disparity_jitter_p95_px"])
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--stereo", required=True)
    parser.add_argument("--diameter-mm", type=float, default=40.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stereo = json.loads(Path(args.stereo).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(args.capture).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    upgraded = upgrade(rows, stereo, args.diameter_mm / 1000.)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", encoding="utf-8") as out:
        for row in upgraded:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    valid = sum(bool(row.get("capture_valid", True)) for row in upgraded)
    print(f"Saved {len(upgraded)} rows ({valid} valid) to {output}")


if __name__ == "__main__":
    main()
