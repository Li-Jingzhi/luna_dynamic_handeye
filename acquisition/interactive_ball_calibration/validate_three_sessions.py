"""Strict independent third-session validation for fixed-head FK/vision consistency.

The first TWO capture files are the ONLY sources permitted to train parameters.
The third capture file is NEVER used to fit or select a calibration.
All three files must store sphere_center_projection_stereo_v1; capture quality
must be from real PASS-frame fusion, not synthetic placeholders.

This validator cannot prove submillimeter consistency from unavailable visual
Rotation6D.  It reports rigid and bounded joint-zero diagnostics and fails
closed when the untouched third session exceeds 1 mm XYZ P95.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from capt_head.drag_capture import mdh_provenance
from .fit_kinematic_correction import _fk_points, _optimize_zero
from .geometry import arm_mdh, rigid_fit


def load_session(path, expected_mdh):
    source = Path(path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"Empty capture: {path}")
    valid = [row for row in rows if row.get("capture_valid", True)]
    if len(valid) < 40:
        raise ValueError(f"Too few valid calibration points: {path}")
    for i, row in enumerate(valid):
        if (row.get("capture_kind") != "interactive_ball_auto"
                or row.get("mdh_sha256") != expected_mdh
                or row.get("pixel_domain") != "ideal_pinhole"
                or row.get("sphere_center_method") != "sphere_center_projection_stereo_v1"
                or int(row.get("valid_frames", 0)) < 50):
            raise ValueError(
                f"Not a compatible real 50-frame sphere-center observation "
                f"in {path}, valid row {i}")
        xyz = np.asarray(row.get("sphere_center_left_camera_m"), float)
        q = np.asarray(row.get("q7_deg"), float)
        if (xyz.shape != (3,) or q.shape != (7,)
                or not np.isfinite(xyz).all() or not np.isfinite(q).all()):
            raise ValueError(f"Malformed XYZ/q7 in {path}: {i}")
    head = np.asarray([row["head_angles_deg"] for row in valid], float)
    body = np.asarray([row["body_q_deg"] for row in valid], float)
    if np.ptp(head, axis=0).max() > .25 or np.ptp(body, axis=0).max() > .3:
        raise ValueError(f"Head/body moved within {path}")
    return {
        "file": str(source),
        "total": len(rows),
        "valid": len(valid),
        "rejected": len(rows)-len(valid),
        "q": np.asarray([row["q7_deg"] for row in valid], float),
        "xyz": np.asarray([row["sphere_center_left_camera_m"] for row in valid], float),
        "head": np.median(head, axis=0),
        "body": np.median(body, axis=0),
        "visual_pose9_available": int(sum(
            row.get("visual_rotation6d_camera") is not None for row in valid)),
        "disparity_p95_px": float(np.percentile([
            float(row["sphere_corrected_disparity_jitter_p95_px"])
            for row in valid], 95)),
        "point_jitter_p95_mm": float(np.percentile([
            float(row.get("sphere_center_p95_jitter_mm", float("nan")))
            for row in valid
            if row.get("sphere_center_p95_jitter_mm") is not None],95))
            if any(row.get("sphere_center_p95_jitter_mm") is not None
                   for row in valid) else None,
    }


def residual_stats(predicted_m, fk_target_m):
    error_mm = (np.asarray(predicted_m) - np.asarray(fk_target_m))*1000.
    distance_mm = np.linalg.norm(error_mm,axis=1)
    return {
        "samples":len(distance_mm),
        "mean_mm":float(np.mean(distance_mm)),
        "rms_mm":float(np.sqrt(np.mean(distance_mm**2))),
        "p50_mm":float(np.percentile(distance_mm,50)),
        "p95_mm":float(np.percentile(distance_mm,95)),
        "max_mm":float(np.max(distance_mm)),
        "axis_bias_mm":np.mean(error_mm,axis=0).tolist(),
        "axis_std_mm":np.std(error_mm,axis=0).tolist()
    }


def evaluate(train_paths, test_path, ball_model, mdh):
    if len(train_paths) != 2:
        raise ValueError("Exactly two training sessions A/B are required")
    inputs = [Path(path).resolve() for path in [*train_paths,test_path]]
    if len(set(inputs)) != 3:
        raise ValueError("Training and test capture paths must be disjoint")
    ball = json.loads(Path(ball_model).read_text(encoding="utf-8"))
    expected_mdh = mdh_provenance(mdh)
    if ball.get("mdh_sha256") != expected_mdh:
        raise ValueError("Ball model and MDH checksum mismatch")
    pf=np.asarray(ball["flange_to_ball_center_m"],float)
    if pf.shape!=(3,) or not np.isfinite(pf).all():
        raise ValueError("Invalid ball offset")
    sessions = [load_session(path,expected_mdh) for path in inputs]
    chain=arm_mdh(mdh)
    train_q=np.vstack([s["q"] for s in sessions[:2]])
    train_cam=np.vstack([s["xyz"] for s in sessions[:2]])
    test_q=sessions[2]["q"]
    test_cam=sessions[2]["xyz"]

    # Ensure same nominal fixed-head domain; a feedback-angle match cannot
    # guarantee true head extrinsic repeatability, so don't claim it does.
    reference_head=sessions[0]["head"]
    if any(np.max(np.abs(s["head"]-reference_head))>.25 for s in sessions):
        raise ValueError("Capture sessions have different head reference poses")
    zeros=np.zeros(7,float)
    train_fk=_fk_points(train_q,chain,pf,zeros)
    test_fk=_fk_points(test_q,chain,pf,zeros)
    R0,t0=rigid_fit(train_cam,train_fk)
    rigid_pred=lambda camera:(R0@camera.T).T+t0
    rigid_train=residual_stats(rigid_pred(train_cam),train_fk)
    rigid_test=residual_stats(rigid_pred(test_cam),test_fk)

    dq,R,t,_=_optimize_zero(
        train_q, train_cam, chain, pf,
        np.ones(len(train_q),bool), max_zero_deg=2.)
    corrected_train_fk=_fk_points(train_q,chain,pf,dq)
    corrected_test_fk=_fk_points(test_q,chain,pf,dq)
    zero_pred=lambda camera:(R@camera.T).T+t
    zero_train=residual_stats(zero_pred(train_cam),corrected_train_fk)
    zero_test=residual_stats(zero_pred(test_cam),corrected_test_fk)

    # Important: zero-corrected labels are a *different* FK convention.
    # Report them independently; do not silently declare success against
    # the original unmodified robot FK definition.
    zero_model_against_original=residual_stats(zero_pred(test_cam),test_fk)
    physical_9d_available=all(s["visual_pose9_available"]==s["valid"]
                              for s in sessions)
    passed=bool(rigid_test["p95_mm"]<1.0)
    return {
        "schema":"luna_three_session_locked_holdout_v1",
        "training_captures":[s["file"] for s in sessions[:2]],
        "untouched_test_capture":sessions[2]["file"],
        "data_counts":[{
            "file":s["file"],"total":s["total"],"valid":s["valid"],
            "rejected":s["rejected"],
            "pose9_samples":s["visual_pose9_available"],
            "disparity_jitter_p95_px":s["disparity_p95_px"],
            "point_jitter_p95_mm":s["point_jitter_p95_mm"],
        } for s in sessions],
        "head_nominal_deg":reference_head.tolist(),
        "fixed_pf_m":pf.tolist(),
        "mdh_sha256":expected_mdh,
        "rigid_unmodified_fk":{
            "train":rigid_train,"untouched_test":rigid_test,
        },
        "diagnostic_bounded_joint_zero":{
            "joint_zero_correction_deg":dq.tolist(),
            "train_vs_corrected_fk":zero_train,
            "untouched_test_vs_corrected_fk":zero_test,
            "untouched_test_vs_original_fk":zero_model_against_original,
            "warning":"The corrected FK convention is not the original unmodified FK. "
                      "This model MUST NOT be approved from training errors."
        },
        "vision_pose9_data_complete":physical_9d_available,
        "pose9_validation_status":("ready_to_fit" if physical_9d_available
                                   else "blocked_missing_real_rotation6d"),
        "strict_unmodified_fk_p95_lt_1mm":passed,
        "final_submillimeter_achieved":passed,
        "warning":"C was never used for parameter fitting. This is an internal "
                  "FK numerical-consistency test, not independent robot-base metrology.",
    }


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",nargs=2,required=True,metavar=("A","B"))
    parser.add_argument("--test",required=True,help="Untouched third trajectory C")
    parser.add_argument("--ball-model",required=True)
    parser.add_argument("--mdh",default=str(root/"configs/luna_new.ini"))
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    output=evaluate(args.train,args.test,args.ball_model,args.mdh)
    with Path(args.output).open("x",encoding="utf-8") as f:
        json.dump(output,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    result=output["rigid_unmodified_fk"]["untouched_test"]["p95_mm"]
    print(f"Untouched C vs original FK P95={result:.3f} mm; "
          f"PASS={output['final_submillimeter_achieved']}; "
          f"Pose9={output['pose9_validation_status']}")


if __name__=="__main__":
    main()
