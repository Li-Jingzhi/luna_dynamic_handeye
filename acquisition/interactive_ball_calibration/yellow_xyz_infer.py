"""Read fresh fused yellow-sphere records from stdin and emit FK-consistent XYZ.

Usage:
  python3 -B -m interactive_ball_calibration.yellow_xyz_infer \
      --approved-model models/approved_yellow_xyz_v2.json \
      < live_fused_50_frame_jsonl_stream

Important: accepts only an EXTERNALLY CERTIFIED approved model, live/monotonic
camera and robot feedback timestamps, quality-qualified fused 50-frame records,
matching stereo calibration and the calibrated fixed head pose. No q7 is read
by the online inference calculation. Does not actuate any hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .yellow_xyz_release import inference_from_fused_observation


def process(model, record, mdh):
    point = inference_from_fused_observation(model, record, mdh=mdh)
    return {
        "ok": True,
        "frame_id": "right_arm_mdh_parent",
        "units": "m",
        "xyz_right_arm_mdh_parent_m": np.asarray(point, float).tolist(),
        "model_schema": model["schema"],
        "source_waypoint_index": record.get("waypoint_index"),
        "source_camera_timestamp_monotonic_ns": record["camera_timestamp_monotonic_ns"],
    }


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-model", required=True)
    parser.add_argument("--mdh", default=str(root/"configs/luna_new.ini"))
    args=parser.parse_args()
    model=json.loads(Path(args.approved_model).read_text(encoding="utf-8"))
    cert=model.get("external_validation", {})
    if (not model.get("approved") or model.get("do_not_deploy",True) or
            cert.get("approved_for_fixed_head_runtime") is not True):
        parser.error("Model is not externally approved. Refusing online inference.")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            observation=json.loads(line)
            result=process(model, observation, args.mdh)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            result={"ok": False, "reason": str(exc)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__=="__main__":
    main()
