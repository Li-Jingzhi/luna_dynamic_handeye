"""Create a ball model from an independently measured flange offset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .geometry import validate_stereo


def build_model(stereo_path, mdh_path, pf_mm):
    stereo_path = Path(stereo_path)
    stereo = json.loads(stereo_path.read_text(encoding="utf-8"))
    validate_stereo(stereo)
    pf_mm = np.asarray(pf_mm, float)
    if pf_mm.shape != (3,) or not np.isfinite(pf_mm).all() or np.linalg.norm(pf_mm) > 200:
        raise ValueError("Measured flange offset must be three finite millimetre values within 200 mm")
    return {
        "schema": "luna_measured_flange_ball_offset_v1",
        "flange_to_ball_center_m": (pf_mm / 1000).tolist(),
        "measurement_method": "precision_instrument",
        "mdh_sha256": mdh_provenance(mdh_path),
        "stereo_sha256": hashlib.sha256(stereo_path.read_bytes()).hexdigest(),
        "pixel_domain": stereo["pixel_domain"],
        "image_size": stereo["image_size"],
        "sensor_ids": stereo["sensor_ids"],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo", required=True)
    parser.add_argument("--mdh", default=str(root / "configs/luna_new.ini"))
    parser.add_argument("--pf-mm", nargs=3, type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_model(args.stereo, args.mdh, args.pf_mm)
    with Path(args.output).open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Saved measured p_F={args.pf_mm} mm to {args.output}")


if __name__ == "__main__":
    main()
