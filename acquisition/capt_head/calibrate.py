"""Two-stage calibration: fresh reference DLT, then rigid two-axis head fit.

No calibration outputs are installed or approved by this module. Original
CAPT/stereo artifacts are deliberately never opened for writing.
"""
from __future__ import annotations
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .geometry import (dlt, project, triangulate, ideal_pixels,
                       robust_optimize, inverse_head_points, dynamic_projections)
from .records import valid_rows, camera_contract

DEG = np.pi/180


def _rms(a):
    x = np.asarray(a, float)
    return float(np.sqrt(np.mean(x*x)))


def _percentile(a):
    return float(np.percentile(np.asarray(a, float), 95))


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def _stats(rows, Pl, Pr, lens, params=None, reference=None):
    if not rows:
        return None
    pairs = ideal_pixels([r["stereo_pixels"] for r in rows], lens)
    xyz = np.asarray([r["xyz_base_m"] for r in rows], float)
    angles = np.asarray([r["head_angles_rad"] for r in rows], float)
    xyzerr, pxerr = [], []
    for i, X in enumerate(xyz):
        L, R = (Pl, Pr) if params is None else dynamic_projections(Pl, Pr, params, angles[i], reference)
        try:
            recovered = triangulate(L, R, pairs[i])
            xyzerr.append(np.linalg.norm(recovered-X)*1000.)
            pxerr.append(max(
                np.linalg.norm(project(L, [X])[0]-pairs[i, :2]),
                np.linalg.norm(project(R, [X])[0]-pairs[i, 2:])))
        except ValueError:
            xyzerr.append(1.e6)
            pxerr.append(1.e6)
    return dict(samples=len(rows), xyz_rms_mm=_rms(xyzerr),
                xyz_p95_mm=_percentile(xyzerr),
                reprojection_rms_px=_rms(pxerr),
                reprojection_p95_px=_percentile(pxerr))


def fit_reference(rows, *, lens=None, allow_raw_research=False):
    rows, rejected = valid_rows(rows)
    contract = camera_contract(rows, lens)
    if contract["input_pixel_domain"] == "raw" and lens is None and not allow_raw_research:
        raise ValueError("Raw pixels need measured lens K/D, or explicit raw-research opt-in")
    if lens is not None and lens.get("independently_verified") is not True:
        # Usable for research calibration; don't silently elevate it.
        pass
    q0 = np.asarray(contract["reference_head_angles_rad"], float)
    ref_rows = [r for r in rows if np.max(np.abs(np.asarray(r["head_angles_rad"])-q0)) <= .25*DEG]
    train = [r for r in ref_rows if r["split"] == "train"]
    held = [r for r in ref_rows if r["split"] in ("validation", "test")]
    if len(train) < 30 or len({r["group_id"] for r in train}) < 20 or len(held) < 8:
        raise ValueError("Need >=30 reference training points, >=20 distinct groups, >=8 reference heldout points")
    X = np.asarray([r["xyz_base_m"] for r in train], float)
    pair = ideal_pixels([r["stereo_pixels"] for r in train], lens)
    mask = np.ones(len(X), bool)
    for _ in range(12):
        Pl = dlt(X[mask], pair[mask, :2])
        Pr = dlt(X[mask], pair[mask, 2:])
        error = np.maximum(np.linalg.norm(project(Pl, X)-pair[:, :2], axis=1),
                           np.linalg.norm(project(Pr, X)-pair[:, 2:], axis=1))
        updated = error <= 4.
        if updated.sum() < max(20, int(np.ceil(.6*len(train)))):
            raise ValueError("Reference fit has <60% <=4px inliers; check timestamps, labels, fisheye and camera order")
        if np.array_equal(updated, mask):
            break
        mask = updated
    # Explicit final refit on final mask, even if the last update changed it.
    Pl, Pr = dlt(X[mask], pair[mask, :2]), dlt(X[mask], pair[mask, 2:])
    model = dict(
        schema="head_2dof_capt_v1", stage="reference", units="m", frame_id="robot_base",
        **contract, P_left_reference=Pl.tolist(), P_right_reference=Pr.tolist(),
        head_parameters=None, head_angle_hull_rad=None, lens=copy.deepcopy(lens),
        xyz_bounds_m=np.stack((X.min(0)-.03, X.max(0)+.03)).tolist(),
        max_reprojection_error_px=4., physical_accuracy_independently_validated=False,
        independently_verified=False, research_only=True, motion_enabled=False,
        source_kind="stationary_recorded_robot_labels",
        lens_independently_verified=bool(lens and lens.get("independently_verified") is True),
        fk_labels_independently_verified=all(r.get("fk_frame_independently_verified") is True for r in rows),
        reference_inliers=int(mask.sum()), reference_samples=len(train),
        reference_rejected=len(rejected),
        reference_fit={"train":_stats([r for j,r in enumerate(train) if mask[j]],Pl,Pr,lens),
                       "heldout":_stats(held,Pl,Pr,lens)},
        made_at=datetime.now(timezone.utc).isoformat(),
        warning="Research-only calibration. Independently verify physical XYZ and lens before robot control.")
    model["fit_quality_passed"] = bool(
        model["reference_fit"]["heldout"]["xyz_p95_mm"] <= 10 and
        model["reference_fit"]["heldout"]["reprojection_p95_px"] <= 3)
    model["calibration_id"] = "ref_"+_digest(model)
    return model


def fit_head(rows, reference_model):
    rows, _ = valid_rows(rows)
    ref = reference_model
    if ref.get("schema") != "head_2dof_capt_v1" or ref.get("stage") != "reference":
        raise ValueError("First provide a freshly fitted reference model")
    contract = camera_contract(rows, ref.get("lens"))
    for key in ("sensor_ids", "image_size", "input_pixel_domain", "reference_head_angles_rad", "yellow_offset_calibration_id"):
        if contract[key] != ref[key]:
            raise ValueError(f"Reference model/observations mismatch: {key}")
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] != "train"]
    if len(train) < 60 or len(held) < 12:
        raise ValueError("Need >=60 multihead train and >=12 independent heldout records")
    if len({r["group_id"] for r in train}) < 40:
        raise ValueError("Insufficient distinct head/ball stationary groups")
    angles = np.asarray([r["head_angles_rad"] for r in train])
    q0 = np.asarray(ref["reference_head_angles_rad"], float)
    if len(np.unique(np.round(angles[:, 0]/DEG, 1))) < 3 or len(np.unique(np.round(angles[:, 1]/DEG, 1))) < 3:
        raise ValueError("Need at least three yaw and three pitch settings")
    hull = cv2.convexHull(angles.astype(np.float32))
    if cv2.contourArea(hull) < (2*DEG)**2:
        raise ValueError("Head calibration lacks two-dimensional angle coverage")
    XYZ = np.asarray([r["xyz_base_m"] for r in train], float)
    uv = ideal_pixels([r["stereo_pixels"] for r in train], ref.get("lens"))
    Pl, Pr = np.asarray(ref["P_left_reference"]), np.asarray(ref["P_right_reference"])
    def residual(params):
        mapped = inverse_head_points(params, angles, q0, XYZ)
        px = np.hstack((project(Pl, mapped)-uv[:, :2],
                        project(Pr, mapped)-uv[:, 2:])).ravel()
        # Axis length and axial pivot redundancy are gauges, not physical observables.
        gauge = []
        for prm in np.asarray(params).reshape(2,6):
            a, c = prm[:3], prm[3:]
            gauge.extend((10*(np.linalg.norm(a)-1), 10*np.dot(a, c)))
        return np.r_[px, gauge]
    # Two physical starting conventions for yaw/tilt, only research fit is claimed.
    centre = np.median(XYZ, axis=0)
    seeds = [
        np.r_[0,0,1,centre,0,1,0,centre].astype(float),
        np.r_[0,1,0,centre,0,0,1,centre].astype(float)
    ]
    options = [robust_optimize(residual, seed) for seed in seeds]
    solution = min(options, key=lambda p: np.median(np.abs(residual(p))))
    if not np.isfinite(solution).all():
        raise ValueError("Failed to fit finite two-axis head model")
    model = copy.deepcopy(ref)
    model.update(stage="dynamic", head_parameters=solution.tolist(),
                 head_angle_hull_rad=hull[:, 0].tolist(),
                 xyz_bounds_m=np.stack((XYZ.min(0)-.03, XYZ.max(0)+.03)).tolist(),
                 max_reprojection_error_px=4.,
                 dynamic_fit={
                     "train":_stats(train, Pl, Pr, model["lens"], solution, q0),
                     "validation":_stats([r for r in held if r["split"]=="validation"], Pl, Pr, model["lens"], solution, q0),
                     "test":_stats([r for r in held if r["split"]=="test"], Pl, Pr, model["lens"], solution, q0)
                 },
                 physical_accuracy_independently_validated=False, independently_verified=False,
                 research_only=True, motion_enabled=False,
                 warning="NOT independently verified: do not use for actuation; moving-head frames require exposure/encoder time sync.")
    reports = [v for k, v in model["dynamic_fit"].items() if k != "train" and v]
    model["fit_quality_passed"] = bool(reports and all(
        v["xyz_p95_mm"] <= 10 and v["reprojection_p95_px"] <= 3 for v in reports))
    model["calibration_id"] = "head_"+_digest(model)
    return model
