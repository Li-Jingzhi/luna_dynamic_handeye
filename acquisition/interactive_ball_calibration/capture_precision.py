"""Instrumented repeatability/return/configuration experiments on a taught path."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from capt_head.drag_capture import EXPECTED_VERSION, load_sdk, mdh_provenance
from capt_head.measurement_contract import load_contract
from gq_calibration.capture import checked, head_angles, read_limits, wait_head
from .capture import (arm_limits, assert_fixed, connect_robot, head_reference_waypoints,
                      q7, read_rt, validate_path, wait_arm)
from .capture_point_head import capture_at_pose
from .geometry import arm_mdh, flange_pose
from .validate_pf_rt import write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = np.array([0., -25.])


def make_plan(path, reference_capture):
    taught = json.loads(Path(path).read_text())
    rows = [json.loads(x) for x in Path(reference_capture).read_text().splitlines() if x.strip()]
    rows = sorted([r for r in rows if r.get("capture_valid")], key=lambda r: r["waypoint_index"])
    X = np.array([r["sphere_center_left_camera_m"] for r in rows])
    q = np.array([r["q7_deg"] for r in rows])
    # Spatial farthest-point selection includes depth and lateral coverage.
    normalized = (X-X.mean(0))/np.maximum(np.ptp(X, axis=0), .05)
    selected = [int(np.argmin(X[:, 2])), int(np.argmax(X[:, 2]))]
    while len(selected) < 12:
        distances = np.min(np.sum((normalized[:, None]-normalized[selected])**2, axis=2), axis=1)
        selected.append(int(np.argmax(distances)))
    static = sorted({rows[i]["waypoint_index"] for i in selected}, reverse=True)
    distance = np.linalg.norm(X[:, None]-X[None, :], axis=2)*1000
    joint_distance = np.max(abs(q[:, None]-q[None, :]), axis=2)
    combinations = set()
    for i in range(len(rows)):
        for neighbors in itertools.combinations(np.where((distance[i] <= 20) & (joint_distance[i] >= 10))[0], 3):
            ids = tuple(sorted((i, *neighbors)))
            if all(joint_distance[a, b] >= 10 for a, b in itertools.combinations(ids, 2)):
                combinations.add(ids)
    candidates = sorted(combinations, key=lambda ids: max(distance[a, b] for a, b in itertools.combinations(ids, 2)))
    used, groups = set(), []
    for ids in candidates:
        if used.intersection(ids):
            continue
        groups.append(dict(group=len(groups), waypoints=[rows[i]["waypoint_index"] for i in ids],
                           prior_max_visual_distance_mm=float(max(distance[a, b] for a, b in itertools.combinations(ids, 2))),
                           prior_min_joint_difference_deg=float(min(joint_distance[a, b] for a, b in itertools.combinations(ids, 2)))))
        used.update(ids)
        if len(groups) == 8:
            break
    if len(static) != 12 or len(groups) != 8:
        raise ValueError("Taught data cannot supply requested spatial/configuration groups")
    return dict(schema="luna_precision_diagnostic_plan_v1", taught_path=str(path),
                taught_path_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                reference_capture=str(reference_capture), reference_head_deg=REFERENCE.tolist(),
                static_waypoints=static, static_windows_each=10, arm_returns_each=3,
                head_return_waypoints=static[::2], head_cycles_each=3,
                configuration_groups=groups,
                planned_windows=12*10+12*3+6*3*3+8*4,
                configuration_warning="Nearby positions, not identical physical XYZ; measure actual offsets and keep groups in validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(ROOT/"data/ball_taught_20261018_visible_v1.json"))
    parser.add_argument("--reference-capture", default=str(ROOT/"data/ball_auto_20261018_visible_v1.jsonl"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-baseline", action="store_true",
                        help="explicitly use original automatic AE/AWB warmup-and-lock baseline")
    parser.add_argument("--evidence-dir", help="Orin lossless ROI directory recorded in the session manifest")
    parser.add_argument("--sdk-root", default=str(ROOT.parent/"00040701"))
    parser.add_argument("--robot-ip", default="6.6.7.190")
    parser.add_argument("--mdh", default=str(ROOT/"configs/luna_new.ini"))
    parser.add_argument("--latest", type=Path, default=ROOT/"runtime/new_sh5_yellow/latest_pose.json")
    parser.add_argument("--stereo", type=Path, default=ROOT/"configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json")
    args = parser.parse_args()
    if args.plan_only:
        plan = make_plan(args.path, args.reference_capture)
        write_json(args.plan, plan)
        print(json.dumps(plan, indent=2))
        return
    if not args.run:
        parser.error("--run is required for automatic taught-path motion")
    if not args.evidence_dir:
        parser.error("--evidence-dir is required to locate lossless ROI evidence")
    plan = json.loads(args.plan.read_text())
    if plan["taught_path_sha256"] != hashlib.sha256(Path(args.path).read_bytes()).hexdigest():
        raise ValueError("Diagnostic plan taught-path hash mismatch")
    if args.output.exists():
        raise ValueError("Use a new output; partial diagnostic runs are not silently resumed")
    contract, _ = load_contract()
    camera_controls = contract.get("camera_controls")
    if not camera_controls and not args.allow_baseline:
        raise ValueError("A tested fixed profile or explicit --allow-baseline is required")
    manifest = json.loads(Path(args.path).read_text())
    waypoints = manifest["waypoints"]
    stereo = json.loads(args.stereo.read_text())
    frozen = json.loads((ROOT/"models/candidate_yellow_xyz_spatial_v2_final.json").read_text())
    frozen_pf = np.array(frozen["flange_to_ball_effective_m"], float)
    chain = arm_mdh(args.mdh)
    plan.update(camera_controls=camera_controls,
                evidence_remote_directory=args.evidence_dir,
                imaging_policy="fixed_profile" if camera_controls else "original_auto_warmup_then_lock",
                reboot_imaging_repeatability_verified=False,
                contract_sha256=contract["contract_sha256"],
                mdh_sha256=mdh_provenance(args.mdh), started_monotonic_ns=time.monotonic_ns(),
                frozen_label_reference="models/candidate_yellow_xyz_spatial_v2_final.json",
                frozen_pf_m=frozen_pf.tolist())
    write_json(args.output.with_suffix(".session.json"), plan)
    sdk = load_sdk(args.sdk_root)
    robot = connect_robot(sdk, args.robot_ip)
    arm, head = sdk.FXObjType.OBJ_ARM1, sdk.FXObjType.OBJ_HEAD
    last_stamp, saved, valid = 0, 0, 0
    try:
        if str(robot.get_sdk_version()) != EXPECTED_VERSION or str(robot.get_controller_version()) != EXPECTED_VERSION:
            raise ValueError("SDK/controller version mismatch")
        validate_path(waypoints, arm_limits(robot))
        head_limits = np.array(read_limits(robot))
        if any(sdk.state_map.get(robot.current_state(part)) != "IDLE" for part in (arm, head)):
            raise ValueError("Arm/head must be IDLE before this experiment")
        rt = read_rt(robot)
        body_ref = np.array(manifest["body_q_deg"])
        assert_fixed(rt, REFERENCE, body_ref)
        taught_q = np.array([w["q7_deg"] for w in waypoints])
        current = int(np.argmin(np.max(abs(taught_q-q7(rt)), axis=1)))
        if np.max(abs(taught_q[current]-q7(rt))) > .3:
            raise ValueError("Arm must start at a taught waypoint")
        third = float(head_angles(rt)[2])
        checked(robot.switch_to_position_mode(arm, 2000, 5, 5), "Diagnostic arm Position")
        checked(robot.switch_to_position_mode(head, 2000, 5, 5), "Diagnostic head Position")

        def move_arm(destination):
            nonlocal current
            if destination == current:
                return
            step = 1 if destination > current else -1
            for index in range(current+step, destination+step, step):
                start = q7(read_rt(robot))
                target = taught_q[index]
                checked(robot.runtime_set_joint_pos_cmd(arm, target.tolist()), f"Diagnostic path {index}")
                wait_arm(robot, sdk, start, target, REFERENCE, body_ref)
                current = index

        def move_head(target):
            for point in head_reference_waypoints(head_angles(read_rt(robot))[:2], target, head_limits):
                checked(robot.runtime_set_joint_pos_cmd(head, [*point.tolist(), third]), "Diagnostic head")
                wait_head(robot, point)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            def capture(stage, repeat, target_head=REFERENCE, **extra):
                nonlocal last_stamp, saved, valid
                target = taught_q[current]
                for attempt in range(2):
                    row = capture_at_pose(robot, args.latest, stereo, target, target_head, body_ref, last_stamp)
                    last_stamp = max(last_stamp, row["camera_window_last_seen_ns"])
                    if row["capture_valid"]:
                        break
                if row["capture_valid"] and not any(f.get("image_evidence") for f in row["camera_window_frames"]):
                    row.update(capture_valid=False, rejection_reason="Missing lossless ROI evidence")
                row.update(capture_kind="precision_diagnostic_window_v1", experiment=stage,
                           repeat_index=repeat, waypoint_index=current, control_mode="Position",
                           target_head_deg=list(map(float, target_head)),
                           mdh_sha256=plan["mdh_sha256"], contract_sha256=contract["contract_sha256"],
                           stereo_model_sha256=hashlib.sha256(args.stereo.read_bytes()).hexdigest(),
                           source_calibration_sha256=stereo["source_calibration_sha256"],
                           image_size=contract["image_size"], sensor_ids=contract["sensor_ids"],
                           pixel_domain=contract["pixel_domain"], camera_controls=camera_controls,
                           imaging_policy=plan["imaging_policy"],
                           **extra)
                flange = flange_pose(row["q7_deg"], chain)
                row["fk_ball_frozen_m"] = (flange[:3, 3]+flange[:3, :3]@frozen_pf).tolist()
                out.write(json.dumps(row, allow_nan=False)+"\n")
                out.flush()
                saved += 1
                valid += bool(row["capture_valid"])
                print(f"Saved {saved}/{plan['planned_windows']} point={current} {stage} repeat={repeat}: "
                      f"{'valid' if row['capture_valid'] else row['rejection_reason']}", flush=True)

            configuration = defaultdict(list)
            for group in plan["configuration_groups"]:
                for point in group["waypoints"]:
                    configuration[point].append(group["group"])
            destinations = sorted(set(plan["static_waypoints"])|set(configuration), reverse=True)
            for point in destinations:
                move_arm(point)
                if point in plan["static_waypoints"]:
                    for repeat in range(10):
                        capture("static_repeat", repeat)
                    for repeat in range(3):
                        offset = 2 if repeat % 2 == 0 else -2
                        away = max(0, min(len(waypoints)-1, point+offset))
                        if away == point:
                            away = max(0, point-2)
                        move_arm(away)
                        move_arm(point)
                        capture("arm_return", repeat, return_from_waypoint=away)
                if point in plan["head_return_waypoints"]:
                    for cycle, target in enumerate(([-10., -25.], [10., -25.], [0., -20.])):
                        capture("head_return", cycle, cycle_phase="A_before")
                        move_head(np.array(target))
                        capture("head_return", cycle, np.array(target), cycle_phase="B")
                        move_head(REFERENCE)
                        capture("head_return", cycle, cycle_phase="A_after")
                for group in configuration[point]:
                    capture("configuration", 0, configuration_group=group)
        write_json(args.output.with_suffix(".completed.json"), dict(complete=True, rows=saved, valid_rows=valid,
                   invalid_rows=saved-valid, last_waypoint=current,
                   sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()))
    finally:
        if robot._connected:
            if sdk.state_map.get(robot.current_state(arm)) == "Position":
                robot.switch_to_idle(arm, 2000)
            robot.switch_to_idle(head, 2000)
            robot.unlink()


if __name__ == "__main__":
    main()
