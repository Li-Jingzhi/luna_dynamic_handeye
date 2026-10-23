"""Promote an independently validated yellow-sphere XYZ calibration, fail closed.

No robot movement and no side effect on the live preview or SDK. Model promotion
requires matching stereo/head/MDH and a NEW capture never used for training/tuning.
The candidate itself is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .fit_yellow_xyz_consistency import _load, metrics
from .fit_yellow_xyz_v2 import (
    SCHEMA, CAMERA_CORRECTION_METHOD,
    correction_m, write_new, support_summary, predict_points)
from .geometry import arm_mdh, flange_pose


def _verify_schema(model):
    if model.get("schema") != SCHEMA:
        raise ValueError("Wrong calibration schema")
    if model.get("approved") is not False or model.get("do_not_deploy") is not True:
        raise ValueError("Expected original unapproved candidate model")
    if not np.isfinite(np.asarray(model.get("flange_to_ball_effective_m"), float)).all():
        raise ValueError("Invalid effective pF")
    if not np.isfinite(np.asarray(model.get("camera_from_mdh_R"), float)).all():
        raise ValueError("Invalid SE3")
    if model.get("frame_id") != "right_arm_mdh_parent":
        raise ValueError("Unexpected coordinate frame")


def _predict_candidate(model, points):
    points = np.asarray(points, float)
    R = np.asarray(model["camera_from_mdh_R"], float)
    t = np.asarray(model["camera_from_mdh_t_m"], float)
    return (points-t)@R + correction_m(model["residual"], points)


def evaluate_external(candidate_path, capture, mdh):
    model = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    _verify_schema(model)
    if model["mdh_sha256"] != mdh_provenance(mdh):
        raise ValueError("Robot MDH hash changed")
    rows_all, rows, q, camera = _load(capture, mdh)
    if hashlib.sha256(Path(capture).read_bytes()).hexdigest() == model["capture_sha256"]:
        raise ValueError("External capture is identical to training capture")
    if Path(capture).resolve() == Path(model["capture"]).resolve():
        raise ValueError("External test cannot reuse the training file path")
    for key in ("source_calibration_sha256", "stereo_model_sha256",
                "pixel_domain", "image_size", "sensor_ids",
                "sphere_center_method"):
        if key == "sphere_center_method":
            correct = CAMERA_CORRECTION_METHOD
        else:
            correct = model[key]
        if any(row.get(key) != correct for row in rows):
            raise ValueError(f"External capture contract mismatch: {key}")
    head = np.asarray([row["head_angles_deg"] for row in rows], float)
    if np.max(np.abs(head-np.asarray(model["head_angles_deg"]))) > .05:
        raise ValueError("Head pose changed; fixed-head model is invalid")
    trained_q=np.asarray(model.get("training_q7_deg_validation_only"), float)
    if trained_q.ndim != 2 or trained_q.shape[1] != 7:
        raise ValueError("Candidate lacks offline training joint provenance")
    # New trajectory must differ in joint pose, not merely be a renamed or
    # recaptured copy of the original path.
    min_joint_distance=np.min(np.max(np.abs(
        q[:, None, :] - trained_q[None, :, :]), axis=2), axis=1)
    novel_fraction=float(np.mean(min_joint_distance > .5))
    if novel_fraction < .7:
        raise ValueError("External trajectory repeats too many calibration joint poses")

    chain=arm_mdh(mdh)
    pf=np.asarray(model["flange_to_ball_effective_m"], float)
    targets=[]
    for angles in q:
        T=flange_pose(angles, chain)
        targets.append(T[:3, 3]+T[:3, :3]@pf)
    targets=np.asarray(targets)
    predictions=_predict_candidate(model, camera)
    quality=metrics(predictions-targets)
    coverage=support_summary(
        np.asarray(model["support_points_camera_m"], float), camera)
    diag=model.get("diagnostic", {})
    internal=diag.get("fixed_chronological_heldout_corrected", {}).get("xyz_p95_mm", float("inf"))
    spatial_folds=diag.get("cv_spatial_fold_p95_mm", [])
    temporal_folds=diag.get("cv_temporal_fold_p95_mm", [])
    complete_cv=(len(spatial_folds)==5 and len(temporal_folds)==5)
    internal_pass=internal < 1.0
    cv_pass=complete_cv and max(spatial_folds+temporal_folds) < 1.0
    ext_pass=quality["xyz_p95_mm"]<1.0
    domain_pass=coverage["inside_fraction"]>=.95
    approved=bool(internal_pass and cv_pass and ext_pass and domain_pass)
    return {
        "schema":"luna_yellow_xyz_external_release_v1",
        "candidate_file":str(candidate_path),
        "candidate_sha256":hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest(),
        "external_capture_file":str(capture),
        "external_capture_sha256":hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
        "external_valid_rows":len(rows),
        "external_invalid_rows":len(rows_all)-len(rows),
        "external_novel_joint_pose_fraction_gt_0_5deg":novel_fraction,
        "external_xyz_error":quality,
        "external_spatial_support":coverage,
        "train_chronological_heldout_pass":internal_pass,
        "train_spatial_temporal_cv_pass":bool(cv_pass),
        "external_p95_under_1mm":ext_pass,
        "external_support_at_least_95_percent":domain_pass,
        "approved_for_fixed_head_runtime":approved,
        "warning": "Validated ONLY for numerical FK agreement at this reference head pose "
                   "and inside camera XYZ training spatial support, NOT absolute world accuracy.",
    }, model


def promote(candidate_path, external_capture, mdh, report_path, approved_path,
            *, new_acquisition_confirmed=False):
    if not new_acquisition_confirmed:
        raise ValueError("Operator must confirm that the external trajectory is a genuinely new acquisition")
    outcome, model=evaluate_external(candidate_path, external_capture, mdh)
    write_new(report_path, outcome)
    if outcome["approved_for_fixed_head_runtime"]:
        certified=dict(model)
        certified.update(
            approved=True, do_not_deploy=False,
            approval_reason="Independent new-trajectory P95<1mm and spatial+temporal CV passed",
            external_validation=outcome)
        write_new(approved_path, certified)
        return True
    return False


def inference_from_fused_observation(approved_model, observed, *, mdh):
    """Use a fused 50-frame sphere + current robot head feedback, never q7.

    The input must be a trusted live/captured fused observation; bare stereo
    sidecar frames lacking actual head angle must be refused.
    """
    model=approved_model
    if not model.get("approved") or model.get("do_not_deploy"):
        raise ValueError("Model has not been independently certified")
    if model.get("mdh_sha256") != mdh_provenance(mdh):
        raise ValueError("MDH changed since certification")
    for key in ("source_calibration_sha256", "stereo_model_sha256",
                "image_size", "sensor_ids", "pixel_domain"):
        if observed.get(key) != model[key]:
            raise ValueError(f"Calibration provenance mismatch: {key}")
    if (observed.get("sphere_center_method") != CAMERA_CORRECTION_METHOD or
            int(observed.get("valid_frames", 0)) < 50 or
            observed.get("capture_valid") is not True):
        raise ValueError("Only fresh quality-qualified fused sphere samples accepted")
    # Do not accept archive rows or outdated robot poses masquerading as a
    # fresh visual measurement. The fused capture timestamps must originate
    # from the same machine's monotonic clock as the prediction call.
    camera_stamp=observed.get("camera_timestamp_monotonic_ns")
    robot_stamp=observed.get("robot_feedback_timestamp_monotonic_ns")
    if not isinstance(camera_stamp, int) or not isinstance(robot_stamp, int):
        raise ValueError("Fresh local camera and robot monotonic timestamps required")
    now=time.monotonic_ns()
    if (camera_stamp > now or robot_stamp > now or
            (now-camera_stamp)/1e6 > 500. or
            (now-robot_stamp)/1e6 > 500. or
            abs(robot_stamp-camera_stamp)/1e6 > 250.):
        raise ValueError("Camera/robot timestamps stale or asynchronous")
    jitter=observed.get("sphere_corrected_disparity_jitter_p95_px")
    jitter3d=observed.get("sphere_center_p95_jitter_mm")
    if (jitter is None or jitter3d is None or
            not np.isfinite([jitter,jitter3d]).all() or
            float(jitter) > .1 or float(jitter3d) > 1.):
        raise ValueError("Fused sphere measurement jitter exceeds acceptance threshold")
    if "head_angles_deg" not in observed:
        raise ValueError("Current measured robot head feedback is mandatory")
    head=np.asarray(observed["head_angles_deg"], float)
    if (head.shape != (2,) or not np.isfinite(head).all() or
            np.max(np.abs(head-np.asarray(model["head_angles_deg"]))) > .05):
        raise ValueError("Outside calibrated head pose")
    return predict_points(model, observed["sphere_center_left_camera_m"])


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--external-capture", required=True)
    parser.add_argument("--mdh", default=str(root/"configs/luna_new.ini"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--approved-output", required=True)
    parser.add_argument("--confirm-new-acquisition", action="store_true",
                        help="Attest that this external path is a genuinely new post-calibration capture")
    args=parser.parse_args()
    approved=promote(args.candidate, args.external_capture,
                     args.mdh, args.report, args.approved_output,
                     new_acquisition_confirmed=args.confirm_new_acquisition)
    if not approved:
        print("REJECTED: mandatory external P95/support/CV criteria not met; "
              "no production model was written.")
    else:
        print("APPROVED: model saved", args.approved_output)


if __name__ == "__main__":
    main()
