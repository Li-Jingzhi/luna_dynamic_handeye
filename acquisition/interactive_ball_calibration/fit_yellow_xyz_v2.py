"""Yellow-only FK-consistency calibration v2: spatially blocked + temporal CV.

Only the third session is used to fit. Joint readings generate offline FK labels;
runtime inference consumes ONLY corrected stereo yellow-sphere XYZ.
This is an experimental candidate: no unattended activation, and no claim of
<1 mm until both chronological holdout AND an external new trajectory pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capt_head.drag_capture import mdh_provenance
from .fit_yellow_xyz_consistency import _base_fit, _load, metrics, features
from .geometry import arm_mdh, flange_pose

SCHEMA = "luna_yellow_xyz_spatial_cv_v2"
CAMERA_CORRECTION_METHOD = "sphere_center_projection_stereo_v1"
SUPPORT_RADIUS_MM = 20.0


def _distance2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.maximum(0., np.sum(x*x, axis=1)[:, None] +
                      np.sum(y*y, axis=1)[None, :] - 2*x@y.T)


def _farthest_centers(points, count):
    """Deterministic farthest-point sampling, no third-party solver dependency."""
    points = np.asarray(points, float)
    count = min(len(points), int(count))
    first = int(np.argmin(np.sum(points, axis=1)))
    ids = [first]
    closest2 = _distance2(points, points[[first]]).ravel()
    for _ in range(1, count):
        index = int(np.argmax(closest2))
        ids.append(index)
        closest2 = np.minimum(closest2, _distance2(points, points[[index]]).ravel())
    return points[ids]


def spatial_folds(points, k=5):
    """Leave whole spatial clusters out; do not split individual frames/points."""
    points = np.asarray(points, float)
    if len(points) < k*12:
        raise ValueError("Insufficient points for spatial block CV")
    centres = _farthest_centers(points, k)
    for _ in range(12):
        groups = np.argmin(_distance2(points, centres), axis=1)
        next_centres = np.array([
            points[groups == i].mean(axis=0) if np.any(groups == i)
            else centres[i] for i in range(k)])
        if np.allclose(next_centres, centres, atol=1e-8):
            break
        centres = next_centres
    groups = np.argmin(_distance2(points, centres), axis=1)
    fold = [np.where(groups == i)[0] for i in range(k)]
    if any(len(part) < 8 for part in fold):
        # If k-means yields tiny outer clusters, use equal-count spatial
        # slabs along the dominant position axis instead of silent skipping.
        centered = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projections = centered @ vt[0]
        ids = np.argsort(projections, kind="stable")
        fold = [np.asarray(part, dtype=int) for part in np.array_split(ids, k)]
    return fold


def _specs():
    yield ("rigid", 0, (), 0, 0.)
    for degree in (1, 2):
        for ridge in (0.1, 10., 100.):
            yield ("poly", degree, (), 0, ridge)
    for degree in (1, 2):
        for scales in ((.025, .08), (.06, .16), (.06, .16, .32)):
            for anchors in (24, 48):
                for ridge in (1., 30.):
                    yield ("hybrid", degree, scales, anchors, ridge)


def _basis(x, order, scales, anchors):
    parts = [features(x, order) if order else np.zeros((len(x), 0))]
    for scale in scales:
        parts.append(np.exp(-_distance2(x, anchors)/(2*scale*scale)))
    return np.concatenate(parts, axis=1)


def fit_correction(camera, fk, base, ids, spec):
    kind, order, scales, anchor_count, ridge = spec
    if kind == "rigid":
        return {"kind": "rigid"}
    anchors = (_farthest_centers(camera[ids], anchor_count)
               if scales else np.empty((0, 3)))
    basis = _basis(camera[ids], order, scales, anchors)
    # Normalize each column on the fitting subset; consistent in runtime.
    norms = np.maximum(np.sqrt(np.mean(basis*basis, axis=0)), .05)
    design = basis / norms
    penalty = np.ones(design.shape[1])
    if order:
        penalty[0] = .02
    coef = np.linalg.solve(
        design.T@design + float(ridge)*np.diag(penalty),
        design.T@((fk[ids]-base[ids])*1000.))
    return {
        "kind": kind, "polynomial_order": int(order),
        "rbf_lengthscales_m": list(scales),
        "anchor_positions_m": anchors.tolist(),
        "column_normalizers": norms.tolist(), "coefficients_mm": coef.tolist(),
        "ridge": float(ridge),
    }


def correction_m(model, camera):
    camera = np.asarray(camera, float)
    if model["kind"] == "ard_gp":
        from .gaussian_process import predict
        return predict(model, camera)
    if model["kind"] == "depth_then_residual":
        from .depth_compensation import correct
        R = np.asarray(model["camera_from_mdh_R"], float)
        depth_delta = (correct(model["depth"], camera)-camera)@R
        return depth_delta + correction_m(model["residual"], camera)
    if model["kind"] == "local_rbf":
        from .fit_yellow_xyz_consistency import rbf_kernel
        return rbf_kernel(camera, np.asarray(model["centers_m"]),
                          model["length_m"])@np.asarray(model["coefficients_mm"])/1000.
    if model["kind"] == "rigid":
        return np.zeros((len(camera), 3))
    b = _basis(camera, model["polynomial_order"],
               model["rbf_lengthscales_m"],
               np.asarray(model["anchor_positions_m"], float).reshape(-1, 3))
    norms = np.asarray(model["column_normalizers"], float)
    return (b/norms)@np.asarray(model["coefficients_mm"], float)/1000.


def predict_points(model, camera, *, check_support=True):
    """Fail closed on unapproved models and points outside the trained domain."""
    if model.get("schema") != SCHEMA or model.get("approved") is not True:
        raise ValueError("Model is not approved for production inference")
    camera = np.asarray(camera, float)
    single = camera.shape == (3,)
    camera = camera[None, :] if single else camera
    if camera.ndim != 2 or camera.shape[1] != 3 or not np.isfinite(camera).all():
        raise ValueError("Expected finite camera XYZ in metres")
    if check_support:
        distance = np.sqrt(_distance2(
            camera, np.asarray(model["support_points_camera_m"], float)).min(axis=1))*1000
        if np.any(distance > model["support_radius_mm"]):
            raise ValueError("Outside calibrated yellow-sphere XYZ work volume")
    R = np.asarray(model["camera_from_mdh_R"], float)
    t = np.asarray(model["camera_from_mdh_t_m"], float)
    points = (camera-t)@R + correction_m(model["residual"], camera)
    return points[0] if single else points


def support_summary(known, unknown, radius_mm=SUPPORT_RADIUS_MM):
    near = np.sqrt(_distance2(unknown, known).min(axis=1))*1000
    return {
        "threshold_mm": float(radius_mm),
        "total": len(unknown),
        "inside": int(np.sum(near <= radius_mm)),
        "inside_fraction": float(np.mean(near <= radius_mm)),
        "nearby_mm_median": float(np.median(near)),
        "nearby_mm_p95": float(np.percentile(near, 95)),
        "nearby_mm_max": float(np.max(near)),
    }


def uniqueness(camera, poses, effective_pf_m, q):
    camera = np.asarray(camera, float)
    q = np.asarray(q, float)
    fk = poses[:, :3, 3] + np.einsum(
        "nij,j->ni", poses[:, :3, :3], effective_pf_m)
    ii, jj = np.triu_indices(len(camera), 1)
    cam_dist = np.sqrt(np.sum((camera[ii]-camera[jj])**2, axis=1))*1000
    q_diff = np.max(np.abs(q[ii]-q[jj]), axis=1)
    f_dist = np.sqrt(np.sum((fk[ii]-fk[jj])**2, axis=1))*1000
    near = (cam_dist <= 1.) & (q_diff >= 10.)
    medium = (cam_dist <= 3.) & (q_diff >= 10.)
    pairs = np.where(near)[0]
    worst = pairs[np.argsort(f_dist[pairs])[-5:][::-1]]
    return {
        "nearby_distance_limit_mm": 1.,
        "different_joint_configuration_deg": 10.,
        "near_1mm_pair_count": int(near.sum()),
        "near_1mm_fk_delta_max_mm": float(np.max(f_dist[near])) if near.any() else None,
        "near_1mm_fk_delta_gt_2mm": int(np.sum(f_dist[near] > 2.)),
        "within_3mm_pair_count": int(medium.sum()),
        "within_3mm_fk_delta_gt_2mm": int(np.sum(f_dist[medium] > 2.)),
        "worst_pairs": [{
            "valid_index_a": int(ii[z]), "valid_index_b": int(jj[z]),
            "camera_separation_mm": float(cam_dist[z]),
            "fk_separation_mm": float(f_dist[z]),
            "max_joint_separation_deg": float(q_diff[z]),
        } for z in worst],
        "warning": "Only a diagnostic; sparse nearby pairs cannot prove global single-valuedness.",
    }


def evaluate(capture, mdh, *, fraction=.8):
    if not .75 <= fraction <= .85:
        raise ValueError("Reserve 15-25% terminal points")
    raw, rows, q, camera = _load(capture, mdh)
    chain = arm_mdh(mdh)
    poses = np.asarray([flange_pose(angle, chain) for angle in q])
    cut = int(len(rows)*fraction)
    train = np.arange(cut, dtype=int)
    held = np.arange(cut, len(rows), dtype=int)
    # Even our fixed original chronological holdout has already been viewed in
    # earlier experiments. It is retained for an apples-to-apples comparison,
    # NOT an independently untouched future certification dataset.
    folds = []
    for block in np.array_split(train, 5):
        folds.append(("temporal", block))
    for block in spatial_folds(camera[train], 5):
        folds.append(("spatial", block))
    cached = []
    for typ, block in folds:
        tr = np.setdiff1d(train, block)
        pf, R, t, labels, base = _base_fit(poses, camera, tr)
        cached.append((typ, block, tr, labels, base))
    candidates = []
    for spec in _specs():
        scores = {"temporal": [], "spatial": []}
        for typ, block, tr, labels, base in cached:
            model = fit_correction(camera, labels, base, tr, spec)
            pred = base[block] + correction_m(model, camera[block])
            scores[typ].append(metrics(pred-labels[block])["xyz_p95_mm"])
        candidates.append({
            "spec": [spec[0], spec[1], list(spec[2]), spec[3], spec[4]],
            "temporal_fold_p95_mm": scores["temporal"],
            "spatial_fold_p95_mm": scores["spatial"],
            "worst_cv_p95_mm": max(scores["temporal"] + scores["spatial"]),
            "mean_cv_p95_mm": float(np.mean(scores["temporal"] + scores["spatial"])),
        })
    raw_winner = min(candidates, key=lambda x:(
        x["worst_cv_p95_mm"], x["mean_cv_p95_mm"]))
    # Within 10% of the most optimistic worst fold, favor fewer parameters.
    # A more complex model is accepted only if BOTH the spatial+temporal
    # worst-fold result and average result meaningfully beat a simpler fit.
    # This prevents RBF from being selected solely due to a single lucky block.
    def complexity(c):
        kind, order, scales, count, ridge = c["spec"]
        return ({"rigid":0, "poly":1, "hybrid":2}[kind], order,
                count, len(scales))
    pool = [c for c in candidates if
            c["worst_cv_p95_mm"] <= 1.10*raw_winner["worst_cv_p95_mm"]]
    winner = min(pool, key=lambda c:(complexity(c), c["mean_cv_p95_mm"]))
    kind, order, scales, count, ridge = winner["spec"]
    chosen = (kind, order, tuple(scales), count, ridge)
    pf, R, t, labels, base = _base_fit(poses, camera, train)
    correction = fit_correction(camera, labels, base, train, chosen)
    predicted = base+correction_m(correction, camera)
    baseline_heldout = metrics(base[held]-labels[held])
    final_train = metrics(predicted[train]-labels[train])
    fixed_holdout = metrics(predicted[held]-labels[held])
    coverage = support_summary(camera[train], camera[held])
    coverage_inside = np.sqrt(_distance2(camera[held], camera[train]).min(axis=1))*1000 <= SUPPORT_RADIUS_MM
    within = metrics((predicted-labels)[held][coverage_inside]) if coverage_inside.sum()>=5 else None

    # The heldout is an archived diagnostic already inspected by prior runs;
    # once model selection is frozen, refit the candidate on ALL 318 measured
    # positions for best intended operating-volume coverage. Only a brand new,
    # never-consulted capture may certify this final candidate.
    all_ids = np.arange(len(rows), dtype=int)
    pf_final, R_final, t_final, targets_final, base_final = _base_fit(
        poses, camera, all_ids)
    correction_final = fit_correction(
        camera, targets_final, base_final, all_ids, chosen)
    fitted_all = metrics(
        base_final + correction_m(correction_final, camera) - targets_final)
    meta = rows[0]
    model = {
        "schema": SCHEMA, "approved": False, "do_not_deploy": True,
        "approval_reason": "One old capture and previously inspected holdout; no unobserved external trajectory passed",
        "capture": str(capture),
        "capture_sha256": hashlib.sha256(Path(capture).read_bytes()).hexdigest(),
        "capture_valid_samples": len(rows), "fit_samples": len(rows),
        "selection_fit_samples": len(train),
        "candidate_refit_all_capture_points": True,
        "fixed_chronological_diagnostic_samples": len(held),
        "frame_id": "right_arm_mdh_parent",
        "units": "m",
        "mdh_sha256": mdh_provenance(mdh),
        "source_calibration_sha256": meta["source_calibration_sha256"],
        "stereo_model_sha256": meta["stereo_model_sha256"],
        "pixel_domain": meta["pixel_domain"],
        "image_size": meta["image_size"],
        "sensor_ids": meta["sensor_ids"],
        "sphere_center_method": CAMERA_CORRECTION_METHOD,
        "head_angles_deg": np.median(
            np.asarray([r["head_angles_deg"] for r in rows]), axis=0).tolist(),
        "flange_to_ball_effective_m": pf_final.tolist(),
        "camera_from_mdh_R": R_final.tolist(),
        "camera_from_mdh_t_m": t_final.tolist(),
        "residual": correction_final,
        "support_radius_mm": SUPPORT_RADIUS_MM,
        "support_points_camera_m": camera.tolist(),
        "training_q7_deg_validation_only": q.tolist(),
        "diagnostic": {
            "uniqueness": uniqueness(camera, poses, pf, q),
            "selected_spec": winner["spec"],
            "selection_rule": "simplest model within 10% of minimum worst 5+5 block-CV P95",
            "raw_lowest_worst_cv_spec": raw_winner["spec"],
            "cv_best_worst_p95_mm": winner["worst_cv_p95_mm"],
            "cv_temporal_fold_p95_mm": winner["temporal_fold_p95_mm"],
            "cv_spatial_fold_p95_mm": winner["spatial_fold_p95_mm"],
            "final_candidate_in_sample_after_refit_all": fitted_all,
            "fixed_chronological_train": final_train,
            "fixed_chronological_heldout_baseline": baseline_heldout,
            "fixed_chronological_heldout_corrected": fixed_holdout,
            "fixed_chronological_holdout_support": coverage,
            "fixed_chronological_heldout_inside_support": within,
            "fixed_chronological_holdout_has_prior_exposure": True,
            "meets_p95_target_on_fixed_holdout": bool(fixed_holdout["xyz_p95_mm"]<1.),
        },
        "warning": "Do not deploy. Only numerical agreement with FK labels; "
                   "not independent physical accuracy. Position-only marker has "
                   "no observed orientation, so mapping may not be single-valued.",
    }
    summary = {
        "schema": "luna_yellow_xyz_spatial_validation_report_v2",
        "candidate_model": model,
        "model_candidates_by_cv": sorted(candidates, key=lambda a:a["worst_cv_p95_mm"]),
        "strict_approval_required": "New independent acquisition + P95 <1 mm + >=95% spatial support + matching provenance",
        "model_approved": False,
    }
    return summary


def write_new(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--capture", required=True)
    parser.add_argument("--mdh", default=str(root/"configs/luna_new.ini"))
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = evaluate(args.capture, args.mdh)
    write_new(args.report, {k:v for k,v in result.items() if k != "candidate_model"})
    write_new(args.candidate, result["candidate_model"])
    d = result["candidate_model"]["diagnostic"]
    print("model", args.candidate, "approved=False")
    print("spec", d["selected_spec"], "worst_space_time_CV_P95_mm",
          round(d["cv_best_worst_p95_mm"],3))
    print("chronological_heldout_P95_mm",
          round(d["fixed_chronological_heldout_corrected"]["xyz_p95_mm"],3),
          "coverage", round(d["fixed_chronological_holdout_support"]["inside_fraction"],3))


if __name__ == "__main__":
    main()
