"""Fit q0 projection and G(q) from independently labelled capture rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capt_head.calibrate import fit_head, fit_reference
from capt_head.records import load_rows, write_new


def prepare(rows):
    if not rows:
        raise ValueError("No observations")
    centers = {}
    for row in rows:
        if row.get("capture_kind") != "gq_head_sweep":
            raise ValueError("Expected gq_head_sweep capture")
        xyz = np.asarray(row.get("xyz_base_m"), float)
        pf = np.asarray(row.get("yellow_center_flange_m"), float)
        if xyz.shape != (3,) or pf.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(pf).all():
            raise ValueError("Each row needs measured xyz_base_m and yellow_center_flange_m; raw q7 alone is not base XYZ")
        if row.get("fk_frame_independently_verified") is not True:
            raise ValueError("Base XYZ/FK frame must be independently verified")
        q = np.asarray(row["q7_deg"], float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Invalid right-arm q7")
        # Cluster stationary arm locations so neighboring video frames share one split.
        pose = int(row["head_pose_index"])
        matches = [(name, np.max(np.abs(q - center))) for name, center in centers.items()
                   if name[0] == pose]
        nearest = min(matches, key=lambda item: item[1]) if matches else None
        if nearest and nearest[1] <= 1.0:
            group = f"{pose}:{nearest[0][1]}"
        else:
            index = sum(key[0] == pose for key in centers)
            centers[(pose, index)] = q
            group = f"{pose}:{index}"
        row["group_id"] = group
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
        row["split"] = "train" if bucket < 7 else ("validation" if bucket < 9 else "test")
        row["frame_id"] = "robot_base"
        row["units"] = "m"
        row["calibration_sample_valid"] = True
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True,
                        help="JSONL with verified xyz_base_m and measured yellow_center_flange_m on each row")
    parser.add_argument("--reference-output", required=True)
    parser.add_argument("--head-output", required=True)
    args = parser.parse_args()
    rows = prepare(load_rows(args.data))
    ref = fit_reference(rows)
    head = fit_head(rows, ref)
    write_new(args.reference_output, ref)
    write_new(args.head_output, head)
    print(json.dumps({"reference": args.reference_output, "head": args.head_output,
                      "reference_fit": ref["reference_fit"], "dynamic_fit": head["dynamic_fit"]}, indent=2))


if __name__ == "__main__":
    main()
