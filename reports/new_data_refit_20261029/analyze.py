"""Refit the selected Luna V2 models on the four new-camera 360-point captures."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dynamic_handeye.calibration import fit_geometry, metrics, predict as predict_fixed
from dynamic_handeye.continuous_free_pf import ball
from dynamic_handeye.dataset import arrays, load_capture
from dynamic_handeye.geometry import Assets, apply, fixed_geometry_hashes, parameters, pose, torso_from_arm
from dynamic_handeye.optimizer import robust_optimize


CAPTURES = [
    ROOT / "datasets/raw/luna_v2_transferred_marvin360_fixed25_20261028.jsonl",
    ROOT / "datasets/raw/luna_v2_transferred_marvin360_fixed20_20261028.jsonl",
    ROOT / "datasets/raw/luna_v2_transferred_marvin360_yaw10_pitchm20_20261028.jsonl",
    ROOT / "datasets/raw/luna_v2_transferred_marvin360_yaw10_pitchm25_20261028.jsonl",
]
FIXED_PF_M = np.array([0., 0., .101])


def clamp_encoder_quantization(angles):
    angles = np.asarray(angles, float).copy()
    limits = np.array([[-173., 173.], [-25., 30.]])
    excess = np.maximum(limits[:, 0] - angles, angles - limits[:, 1])
    if np.max(excess) > .01:
        raise ValueError("Head feedback exceeds limits by more than encoder quantization tolerance")
    return np.clip(angles, limits[:, 0], limits[:, 1])


RBF_GRID = ((.02, .1), (.02, 1.), (.02, 10.),
            (.05, .1), (.05, 1.), (.05, 10.),
            (.10, .1), (.10, 1.), (.10, 10.))


def space_kernel(a, b, length=.05):
    return np.exp(-np.sum((a[:, None] - b[None]) ** 2, axis=2) / (2 * length ** 2))


def fit_selected(camera, angles, target_s, assets):
    fixed = fit_geometry(camera, apply(np.linalg.inv(torso_from_arm(assets)), target_s), angles, assets)
    root = np.asarray(fixed["torso_from_head_root_T"])
    mount = np.asarray(fixed["head_tip_from_rectified_camera_T"])
    initial = np.r_[parameters(root), parameters(mount)]
    head = assets.head(angles)

    def residual(x):
        return (apply(pose(x[:6]) @ head @ pose(x[6:]), camera) - target_s).ravel() * 1000

    fitted = robust_optimize(residual, initial, iterations=250)
    jac = np.column_stack([
        (residual(fitted + np.eye(12)[i] * 1e-5) - residual(fitted - np.eye(12)[i] * 1e-5)) / 2e-5
        for i in range(12)
    ])
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    learned = dict(kind="learned12", x=fitted.tolist(), jacobian_rank=rank,
                   jacobian_singular_values=singular.tolist())
    base = predict_selected(learned, assets, camera, angles)
    common = dict(approved=False, independently_verified=False, frame_id="S_MODEL",
                  camera_assets=assets.hashes, fixed_geometry_hashes=fixed_geometry_hashes(assets))
    fixed_model = dict(kind="fixed6", model=fixed, **common)
    learned.update(common)
    models = {"fixed6": fixed_model, "learned12": learned}
    for length, ridge in RBF_GRID:
        name = f"learned12_torso_rbf_l{int(length * 1000)}_r{ridge:g}"
        model = dict(kind="learned12_torso_rbf", baseline=learned,
                     centers_m=base.tolist(), length_m=length, ridge=ridge,
                     weights_m=np.linalg.solve(space_kernel(base, base, length) + ridge * np.eye(len(base)), target_s - base).tolist())
        model.update(common)
        models[name] = model
    return models


def predict_selected(model, assets, camera, angles):
    camera, angles = np.asarray(camera), np.asarray(angles)
    if model["kind"] == "fixed6":
        return predict_fixed(model["model"], assets, camera, angles)
    if model["kind"] == "learned12":
        x = np.asarray(model["x"])
        return apply(pose(x[:6]) @ assets.head(angles) @ pose(x[6:]), camera)
    if model["kind"] == "learned12_torso_rbf":
        base = predict_selected(model["baseline"], assets, camera, angles)
        return base + space_kernel(base, np.asarray(model["centers_m"]), model["length_m"]) @ np.asarray(model["weights_m"])
    raise ValueError("Unknown selected model")


def canonicalize_groups(dataset):
    expected = "luna_v2_transferred_marvin360_20261028.json"
    by_waypoint = {}
    for row in dataset["rows"]:
        if Path(row["taught_path"]).name != expected:
            raise ValueError("Capture does not use the common transferred trajectory")
        row["group_id"] = f'{row["arm_side"]}:{expected}:{row["waypoint_id"]}'
        by_waypoint.setdefault(row["waypoint_id"], []).append(row)
    for rows in by_waypoint.values():
        q = np.asarray([r["q7_deg"] for r in rows])
        if np.ptp(q, axis=0).max() > .002:
            raise ValueError("Cross-head waypoint feedback mismatch")
    return by_waypoint


def main():
    assets = Assets()
    dataset = load_capture(CAPTURES, assets)
    by_waypoint = canonicalize_groups(dataset)
    camera_archived, _, angles = arrays(dataset)
    angles = clamp_encoder_quantization(angles)
    camera = np.asarray([assets.triangulate(r["corrected_pixels"]) for r in dataset["rows"]])
    q7 = np.asarray([r["q7_deg"] for r in dataset["rows"]])
    poses = assets.flange(q7)
    heads = np.asarray([r["head_key"] for r in dataset["rows"]])
    waypoint = np.asarray([r["waypoint_id"] for r in dataset["rows"]])
    unique_waypoints = np.array(sorted(by_waypoint))

    waypoint_q = np.asarray([by_waypoint[w][0]["q7_deg"] for w in unique_waypoints])
    waypoint_position = assets.flange(waypoint_q)[:, :3, 3]
    spatial_axis = int(np.argmax(np.ptp(waypoint_position, axis=0)))
    configuration_axis = np.linalg.svd(waypoint_q - waypoint_q.mean(0), full_matrices=False)[2][0]
    orderings = {
        "trajectory": unique_waypoints,
        "space": unique_waypoints[np.argsort(waypoint_position[:, spatial_axis])],
        "configuration": unique_waypoints[np.argsort((waypoint_q - waypoint_q.mean(0)) @ configuration_axis)],
    }
    folds = [("all", np.ones(len(camera), bool), np.zeros(len(camera), bool))]
    block_waypoints = {}
    for family, ordered in orderings.items():
        for index, held in enumerate(np.array_split(ordered, 5)):
            test = np.isin(waypoint, held)
            name = f"block_{family}_{index}"
            folds.append((name, ~test, test))
            block_waypoints[name] = held.tolist()
    for head in sorted(set(heads)):
        test_head = heads == head
        folds.append((f"head_{head}", ~test_head, test_head))
        for index, held in enumerate(np.array_split(orderings["space"], 5)):
            joint_test = test_head & np.isin(waypoint, held)
            joint_train = ~test_head & ~np.isin(waypoint, held)
            folds.append((f"joint_{head}_space_{index}", joint_train, joint_test))

    result = dict(schema="luna_new_data_refit_v1", approved=False,
                  goal="Vision agrees with frozen current Luna arm FK in S_MODEL",
                  captures=dataset["sources"], assets=assets.hashes,
                  valid_observations=len(camera), rejected=dataset["rejected"],
                  flange_to_ball_center_m=FIXED_PF_M.tolist(),
                  flange_to_ball_source="user-confirmed precision measurement, 2026-10-29",
                  head_counts={h: int(np.sum(heads == h)) for h in sorted(set(heads))},
                  independent_waypoints=len(unique_waypoints),
                  pixel_vs_archived_camera=metrics(camera - camera_archived),
                  split_definition=dict(spatial_axis=spatial_axis,
                      configuration_principal_axis=configuration_axis.tolist(), block_waypoints=block_waypoints), folds={})
    full_models = None
    for name, train, test in folds:
        target_s = apply(torso_from_arm(assets), ball(poses, FIXED_PF_M))
        models = fit_selected(camera[train], angles[train], target_s[train], assets)
        scores = {}
        for model_name, model in models.items():
            train_error = predict_selected(model, assets, camera[train], angles[train]) - target_s[train]
            entry = {"train": metrics(train_error)}
            if test.any():
                test_error = predict_selected(model, assets, camera[test], angles[test]) - target_s[test]
                entry["test"] = metrics(test_error)
            scores[model_name] = entry
        result["folds"][name] = dict(training_samples=int(train.sum()), test_samples=int(test.sum()), models=scores)
        print(name, {k: round(v.get("test", v["train"])["p95_mm"], 3) for k, v in scores.items()}, flush=True)
        if name == "all":
            full_models = models
            for model in full_models.values():
                model.update(input="rectified [uL,vL,uR,vR] and measured [yaw,pitch] degrees",
                             reference="current nominal right-arm FK",
                             flange_to_ball_center_m=FIXED_PF_M.tolist(),
                             flange_to_ball_source="user-confirmed precision measurement, 2026-10-29",
                             source_capture_hashes=[s["sha256"] for s in dataset["sources"]])

    selection_validation = [v for k, v in result["folds"].items() if k.startswith("block_")]
    selected = min(full_models, key=lambda model: (
        max(f["models"][model]["test"]["p95_mm"] for f in selection_validation),
        np.mean([f["models"][model]["test"]["p95_mm"] for f in selection_validation]), model))
    result["selection"] = dict(model=selected,
        rule="minimum worst P95 across 5 trajectory, 5 spatial and 5 configuration blocks, then mean P95; head holdouts not used for selection",
        exploratory=True)
    validation = [v for k, v in result["folds"].items() if k != "all"]
    result["validation_summary"] = {
        model: dict(worst_p95_mm=max(f["models"][model]["test"]["p95_mm"] for f in validation),
                    median_p95_mm=float(np.median([f["models"][model]["test"]["p95_mm"] for f in validation])),
                    all_p95_below_1mm=all(f["models"][model]["test"]["p95_mm"] < 1 for f in validation))
        for model in full_models
    }
    result["limitations"] = [
        "Only a 2x2 head grid at yaw 0/10 and pitch -25/-20 was observed.",
        "All data are one right-arm trajectory and one acquisition day; no independent new session is held out.",
        "pF is fixed to the user-confirmed precision measurement; FK remains the selected nominal reference.",
        "Exposure timestamps are unavailable; software-paired frames passed the existing timing gates.",
    ]
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for name in ("fixed6", "learned12"):
        model = full_models[name]
        Path(__file__).with_name(name + "_model.json").write_text(
            json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    Path(__file__).with_name("selected_model.json").write_text(
        json.dumps(full_models[selected], ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
