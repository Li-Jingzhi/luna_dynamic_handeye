"""Grouped cross-validation of a shared effective pF and discrete-head R/t."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from capt_head.drag_capture import mdh_provenance
from capt_head.measurement_contract import load_contract
from .capture_point_head import HEAD_POSES
from .fit_yellow_xyz_v2 import spatial_folds
from .geometry import arm_mdh, flange_pose, rigid_fit
from .validate_three_sessions import residual_stats

ROOT = Path(__file__).resolve().parents[1]
FIXED_CAPTURES = [ROOT / "data" / name for name in (
    "ball_auto_20261018_replay_v4_sphere.jsonl",
    "ball_auto_20261018_generalization_094544_sphere.jsonl",
    "ball_auto_20261018_visible_v1.jsonl")]


def head_key(angles):
    return ",".join(f"{x:g}" for x in angles)


def load_capture(path, mdh, stereo, *, multi=False):
    """Read one complete-line snapshot, preserving source row and waypoint IDs."""
    path = Path(path)
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    incomplete_tail = bool(lines and not lines[-1].endswith(b"\n"))
    if incomplete_tail:
        lines.pop()
    raw = [json.loads(line) for line in lines if line.strip()]
    stereo_path = Path(stereo)
    calibration = json.loads(stereo_path.read_text(encoding="utf-8"))
    expected_mdh = mdh_provenance(mdh)
    expected_stereo = hashlib.sha256(stereo_path.read_bytes()).hexdigest()
    contract, _ = load_contract()
    expected_heads = {head_key(p) for p in HEAD_POSES}
    rows, seen, rejected = [], set(), Counter()
    for line_number, row in enumerate(raw, 1):
        if row.get("capture_valid") is not True:
            rejected[row.get("rejection_reason", "capture_valid_false")] += 1
            continue
        if (row.get("capture_kind") != ("interactive_ball_point_head" if multi else "interactive_ball_auto")
                or row.get("mdh_sha256") != expected_mdh
                or row.get("stereo_model_sha256") != expected_stereo
                or row.get("source_calibration_sha256") != calibration["source_calibration_sha256"]
                or row.get("sensor_ids") != calibration["sensor_ids"]
                or row.get("image_size") != calibration["image_size"]
                or row.get("pixel_domain") != "ideal_pinhole"
                or row.get("sphere_center_method") != "sphere_center_projection_stereo_v1"
                or int(row.get("valid_frames", 0)) < contract["valid_frames_per_point"]):
            raise ValueError(f"Incompatible capture contract at {path}:{line_number}")
        q = np.asarray(row.get("q7_deg"), float)
        xyz = np.asarray(row.get("sphere_center_left_camera_m"), float)
        head = np.asarray(row.get("head_angles_deg"), float)
        body = np.asarray(row.get("body_q_deg"), float)
        if (q.shape != (7,) or xyz.shape != (3,) or head.shape != (2,) or body.shape != (6,)
                or not all(np.isfinite(a).all() for a in (q, xyz, head, body)) or xyz[2] <= 0):
            raise ValueError(f"Invalid numeric data at {path}:{line_number}")
        target = np.asarray(row["target_head_deg"] if multi else [0., -25.], float)
        key = head_key(target)
        if target.shape != (2,) or np.max(abs(head-target)) > .25 or (multi and key not in expected_heads):
            raise ValueError(f"Unexpected head pose at {path}:{line_number}")
        identity = (int(row["waypoint_index"]), key)
        if identity in seen:
            raise ValueError(f"Duplicate waypoint/head at {path}:{line_number}")
        seen.add(identity)
        jitter = float(row.get("disparity_jitter_p95_px", float("nan")))
        if not np.isfinite(jitter) or jitter > contract["max_point_disparity_jitter_p95_px"]:
            rejected["point_disparity_quality"] += 1
            continue
        rows.append(dict(source_file=str(path), source_line=line_number,
                         session=path.stem, waypoint_index=identity[0],
                         group_id=f"{path.stem}:{identity[0]}", head_key=key,
                         head_angles_deg=head.tolist(), q7_deg=q.tolist(),
                         sphere_center_left_camera_m=xyz.tolist(), body_q_deg=body.tolist()))
    incomplete_groups = 0
    if multi:
        heads = {}
        for row in rows:
            heads.setdefault(row["group_id"], set()).add(row["head_key"])
        complete = {g for g, h in heads.items() if h == expected_heads}
        incomplete_groups = len(heads)-len(complete)
        rows = [row for row in rows if row["group_id"] in complete]
    if not rows or np.ptp(np.array([r["body_q_deg"] for r in rows]), axis=0).max() > .3:
        raise ValueError(f"No usable rows or body moved in {path}")
    return rows, dict(file=str(path), snapshot_sha256=hashlib.sha256(b"".join(lines)).hexdigest(),
                      complete_source_rows=len(raw), incomplete_tail_ignored=incomplete_tail,
                      processed_rows=len(rows), processed_groups=len({r["group_id"] for r in rows}),
                      incomplete_or_invalid_head_groups_excluded=incomplete_groups,
                      rejected=dict(rejected), snapshot_is_final=not multi,
                      mdh_sha256=expected_mdh, stereo_model_sha256=expected_stereo)


def fit_model(poses, xyz, heads):
    """Optimize three pF coordinates, eliminating each head's R/t by Kabsch."""
    poses, xyz, heads = np.asarray(poses), np.asarray(xyz), np.asarray(heads)
    rotations, origins = poses[:, :3, :3], poses[:, :3, 3]
    keys = sorted(set(heads))
    if any(np.sum(heads == h) < 12 for h in keys):
        raise ValueError("Each training head needs at least 12 arm points")
    diversity = np.linalg.svd((rotations-rotations.mean(0)).reshape(-1, 3), compute_uv=False)
    if diversity[-1] < .12:
        raise ValueError("Insufficient flange orientation diversity")

    def evaluate(pf):
        fk = origins + np.einsum("nij,j->ni", rotations, pf)
        error, transforms = np.empty_like(xyz), {}
        for h in keys:
            ids = heads == h
            R, t = rigid_fit(fk[ids], xyz[ids])
            error[ids] = fk[ids] @ R.T + t-xyz[ids]
            transforms[h] = dict(R=R.tolist(), t_m=t.tolist())
        return error.ravel(), transforms

    best = None
    for seed in ([0., 0., 0.], [0., 0., .1]):
        pf = np.array(seed)
        converged = False
        for iteration in range(100):
            error, transforms = evaluate(pf)
            step = np.eye(3)*1e-5
            J = np.column_stack([(evaluate(pf+d)[0]-evaluate(pf-d)[0])/2e-5 for d in step])
            delta = np.linalg.lstsq(J, -error, rcond=None)[0]
            if np.linalg.norm(delta) < 1e-9:
                converged = True
                break
            cost = error@error
            for scale in 2.**-np.arange(20):
                proposal = pf+scale*delta
                candidate_error = evaluate(proposal)[0]
                if candidate_error@candidate_error < cost:
                    pf = proposal
                    break
            else:
                converged = np.linalg.norm(J.T@error) < 1e-8
                break
        error, transforms = evaluate(pf)
        candidate = (float(error@error), pf, transforms, converged, iteration+1, J)
        if best is None or candidate[0] < best[0]:
            best = candidate
    cost, pf, transforms, converged, iterations, J = best
    singular = np.linalg.svd(J, compute_uv=False)
    if np.linalg.norm(pf) > .2 or singular[-1] < 1e-8 or not converged:
        raise ValueError("pF optimization unconverged, unobservable or outside 200 mm")
    return dict(pf_m=pf.tolist(), camera_from_mdh_by_head=transforms,
                optimizer="variable_projection_pF_exact_Kabsch_per_head",
                iterations=iterations, converged=converged,
                pf_projected_jacobian_singular_values=singular.tolist(),
                pf_projected_jacobian_condition=float(singular[0]/singular[-1]))


def residuals(model, poses, xyz, heads):
    pf = np.asarray(model["pf_m"])
    fk = poses[:, :3, 3]+np.einsum("nij,j->ni", poses[:, :3, :3], pf)
    error = np.empty_like(xyz)
    for h in set(heads):
        transform = model["camera_from_mdh_by_head"][h]
        R, t = np.asarray(transform["R"]), np.asarray(transform["t_m"])
        ids = heads == h
        error[ids] = (xyz[ids]-t)@R-fk[ids]
    return error


def make_splits(rows, poses, *, k=5):
    groups = list(dict.fromkeys(r["group_id"] for r in rows))
    first = np.array([next(i for i, r in enumerate(rows) if r["group_id"] == g) for g in groups])
    if len(groups) < k*12:
        raise ValueError("Need at least 60 arm groups for five-fold validation")
    rng = np.random.default_rng(20261018)
    configs = np.array([rows[i]["q7_deg"] for i in first])
    centered = configs-configs.mean(0)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    layouts = dict(random_grouped=np.array_split(rng.permutation(len(groups)), k),
                   trajectory_block=np.array_split(np.arange(len(groups)), k),
                   spatial_block=spatial_folds(poses[first, :3, 3], k=k),
                   configuration_block=np.array_split(np.argsort(centered@axis, kind="stable"), k))
    result = []
    for name, folds in layouts.items():
        for fold, group_ids in enumerate(folds):
            test_groups = {groups[i] for i in group_ids}
            test = np.array([i for i, r in enumerate(rows) if r["group_id"] in test_groups], int)
            train = np.setdiff1d(np.arange(len(rows)), test)
            result.append((name, fold, train, test))
    return result


def evaluate_dataset(rows, mdh, *, sessions=False):
    chain = arm_mdh(mdh)
    poses = np.array([flange_pose(r["q7_deg"], chain) for r in rows])
    xyz = np.array([r["sphere_center_left_camera_m"] for r in rows])
    heads = np.array([r["head_key"] for r in rows])
    body = np.array([r["body_q_deg"] for r in rows])
    if np.ptp(body, axis=0).max() > .3:
        raise ValueError("Body changed across the dataset")
    full = fit_model(poses, xyz, heads)
    full_error = residuals(full, poses, xyz, heads)
    splits = make_splits(rows, poses) if not sessions else []
    if sessions:
        for fold, session in enumerate(dict.fromkeys(r["session"] for r in rows)):
            test = np.array([i for i, r in enumerate(rows) if r["session"] == session])
            splits.append(("leave_session_out", fold, np.setdiff1d(np.arange(len(rows)), test), test))
    results, predictions = [], []
    for method, fold, train, test in splits:
        train_groups = {rows[i]["group_id"] for i in train}
        test_groups = {rows[i]["group_id"] for i in test}
        assert train_groups.isdisjoint(test_groups)
        split = dict(method=method, fold=fold, train_indices=train.tolist(), test_indices=test.tolist(),
                     train_groups=len(train_groups), test_groups=len(test_groups))
        try:
            model = fit_model(poses[train], xyz[train], heads[train])
            error = residuals(model, poses[test], xyz[test], heads[test])
            split.update(model=model,
                         train=residual_stats(residuals(model, poses[train], xyz[train], heads[train]), np.zeros((len(train), 3))),
                         test=residual_stats(error, np.zeros_like(error)))
            split["test_by_head"] = {h: residual_stats(error[heads[test] == h], np.zeros_like(error[heads[test] == h]))
                                     for h in sorted(set(heads[test]))}
            split["pf_delta_to_full_fit_mm"] = ((np.asarray(model["pf_m"])-full["pf_m"])*1000).tolist()
            split["Rt_delta_to_full_fit"] = {}
            for h, transform in model["camera_from_mdh_by_head"].items():
                reference = full["camera_from_mdh_by_head"][h]
                rotation_delta = np.asarray(transform["R"])@np.asarray(reference["R"]).T
                angle = np.linalg.norm(cv2.Rodrigues(rotation_delta)[0])*180/np.pi
                translation = np.linalg.norm(np.asarray(transform["t_m"])-reference["t_m"])*1000
                split["Rt_delta_to_full_fit"][h] = dict(rotation_deg=float(angle), translation_mm=float(translation))
            for i, delta in zip(test, error):
                predictions.append(dict(method=method, fold=fold, processed_index=int(i),
                                        group_id=rows[i]["group_id"], head_key=rows[i]["head_key"],
                                        error_fk_axes_mm=(delta*1000).tolist(), distance_mm=float(np.linalg.norm(delta)*1000)))
        except (ValueError, np.linalg.LinAlgError) as exc:
            split["fit_failed"] = str(exc)
        results.append(split)
    summaries = {}
    for method in dict.fromkeys(s[0] for s in splits):
        folds = [r for r in results if r["method"] == method]
        error = np.array([r["error_fk_axes_mm"] for r in predictions if r["method"] == method])/1000
        successful = [r for r in folds if "model" in r]
        parameters = np.array([r["model"]["pf_m"] for r in successful])*1000
        summaries[method] = dict(successful_folds=len(successful), failed_folds=len(folds)-len(successful),
                                 oof=residual_stats(error, np.zeros_like(error)) if len(error) else None,
                                 worst_fold_p95_mm=max((r["test"]["p95_mm"] for r in successful), default=None),
                                 pf_fold_min_mm=parameters.min(0).tolist() if len(parameters) else None,
                                 pf_fold_max_mm=parameters.max(0).tolist() if len(parameters) else None,
                                 pf_fold_std_mm=parameters.std(0).tolist() if len(parameters) else None)
    return dict(samples=len(rows), groups=len({r["group_id"] for r in rows}),
                head_counts=dict(Counter(heads)), full_fit=full,
                full_fit_training=residual_stats(full_error, np.zeros_like(full_error)),
                summaries=summaries, folds=results,
                validation_scope="new arm points at observed discrete heads; no unseen-head interpolation",
                parameter_stability_is_not_true_parameter_error=True,
                independent_future_acquisition_verified=False), predictions


def write_json(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-capture", nargs="+", default=FIXED_CAPTURES, type=Path)
    parser.add_argument("--multi-capture", type=Path, default=ROOT/"data/ball_point_head_20261018_v1.jsonl")
    parser.add_argument("--mdh", type=Path, default=ROOT/"configs/luna_new.ini")
    parser.add_argument("--stereo", type=Path, default=ROOT/"configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema="luna_pf_rt_grouped_cv_v1", target="relative FK–vision XYZ consistency only",
                  convention="X_camera = R_head @ (t_MF + R_MF @ pF) + t_head",
                  no_fk_or_camera_parameter_files_modified=True, datasets={})
    all_fixed = []
    datasets = []
    for n, path in enumerate(args.fixed_capture, 1):
        rows, provenance = load_capture(path, args.mdh, args.stereo)
        all_fixed.extend(rows)
        datasets.append((f"fixed_session_{n}", rows, [provenance], False))
    datasets.append(("fixed_leave_session_out", all_fixed, [], True))
    multi, provenance = load_capture(args.multi_capture, args.mdh, args.stereo, multi=True)
    datasets.append(("multi_head_snapshot", multi, [provenance], False))
    for name, rows, provenance, sessions in datasets:
        with (args.output_dir/f"{name}_processed.jsonl").open("x", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, allow_nan=False)+"\n")
        print(f"Fitting {name}: {len(rows)} observations", flush=True)
        result, predictions = evaluate_dataset(rows, args.mdh, sessions=sessions)
        result["provenance"] = provenance
        report["datasets"][name] = result
        with (args.output_dir/f"{name}_oof.jsonl").open("x", encoding="utf-8") as out:
            for row in predictions:
                out.write(json.dumps(row, allow_nan=False)+"\n")
        write_json(args.output_dir/f"{name}.json", result)
        print(json.dumps(result["summaries"], ensure_ascii=False), flush=True)
    write_json(args.output_dir/"report.json", report)


if __name__ == "__main__":
    main()
