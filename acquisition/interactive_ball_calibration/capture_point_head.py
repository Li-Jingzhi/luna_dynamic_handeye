"""Capture several stationary head poses at each waypoint of a taught arm path."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from capt_head.drag_capture import (CAMERA_CONTRACT, CAMERA_IDS, CAMERA_SIZE,
                                    EXPECTED_VERSION, PIXEL_DOMAIN, load_sdk,
                                    mdh_provenance)
from gq_calibration.capture import checked, head_angles, read_limits, wait_head
from .capture import (MAX_STEP_DEG, arm_limits, assert_fixed, camera_observation,
                      connect_robot, fuse_camera_observations, head_reference_waypoints,
                      q7, read_rt, validate_path, wait_arm)
from .capture import capture_feedback_frame
from .geometry import arm_mdh, validate_stereo

ROOT = Path(__file__).resolve().parents[1]
HEAD_POSES = ((-10., -25.), (10., -25.), (0., -20.))
REFERENCE = np.array([0., -25.])


def capture_at_pose(robot, latest, stereo, target_q, target_head, body_ref, minimum_stamp_ns=0):
    observations = []
    seen = set()
    rejected = 0
    deadline = time.monotonic() + 10
    while len(observations) < CAMERA_CONTRACT["valid_frames_per_point"] and time.monotonic() < deadline:
        try:
            frame, pixels, stamp, error, sphere = camera_observation(latest, stereo, timeout=.2)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            rejected += 1
            continue
        identity = (stamp, str(frame.get("frame_number")))
        if stamp <= minimum_stamp_ns:
            time.sleep(.01)
            continue
        if identity not in seen:
            rt = read_rt(robot)
            assert_fixed(rt, target_head, body_ref)
            if np.max(np.abs(q7(rt)-target_q)) > .3:
                raise RuntimeError("Arm moved during stationary head-pose capture")
            observations.append((pixels, stamp, error, capture_feedback_frame(frame, rt), sphere))
            seen.add(identity)
        time.sleep(.02)
    before = time.monotonic_ns()
    rt = read_rt(robot)
    after = time.monotonic_ns()
    result = dict(q7_deg=q7(rt).tolist(), head_angles_deg=head_angles(rt)[:2].tolist(),
                  head_q_deg=head_angles(rt).tolist(), body_q_deg=rt["body"]["fb_pos"],
                  camera_rejected_frames=rejected,
                  robot_feedback_timestamp_monotonic_ns=(before+after)//2,
                  robot_feedback_read_span_ms=(after-before)/1e6,
                  camera_timestamp_basis="local_udp_dequeue_minus_processing_not_exposure",
                  exposure_time_valid=False,
                  camera_window_last_seen_ns=max((o[1] for o in observations), default=minimum_stamp_ns),
                  exposure_sync_status="unverified_no_sensor_exposure_timestamp")
    try:
        fused = fuse_camera_observations(observations)
        if "sphere_center_left_camera_m" not in fused:
            raise ValueError("Physical sphere centre unavailable")
        result.update(fused, capture_valid=True,
                      camera_timestamp_monotonic_ns=max(o[1] for o in observations),
                      camera_robot_offset_ms=(max(o[1] for o in observations)-(before+after)/2)/1e6,
                      pair_capture_skew_ms=float(max(o[3].get("pair_dequeue_skew_ms", 0) for o in observations)),
                      camera_pipeline_delay_ms=float(np.median([o[3].get("camera_pipeline_delay_ms", 0) for o in observations])),
                      stereo_reprojection_max_px=float(max(o[2] for o in observations)))
    except ValueError as exc:
        result.update(capture_valid=False, rejection_reason=str(exc), stereo_pixels=None,
                      sphere_center_left_camera_m=None, camera_timestamp_monotonic_ns=None)
    return result


def completed_rows(path, count):
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for n, row in enumerate(rows):
        if (row.get("waypoint_index"), row.get("head_pose_index")) != divmod(n, count):
            raise ValueError("Output order does not match the requested point/head sequence")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stereo", default=str(ROOT / "configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json"))
    parser.add_argument("--latest", default=str(ROOT / "runtime/new_sh5_yellow/latest_pose.json"))
    parser.add_argument("--robot-ip", default="6.6.7.190")
    parser.add_argument("--sdk-root", default=str(ROOT.parent / "00040701"))
    parser.add_argument("--mdh", default=str(ROOT / "configs/luna_new.ini"))
    parser.add_argument("--stop-after", type=int, default=323, help="last waypoint count, for a pilot run")
    args = parser.parse_args()
    path, output, latest = Path(args.path), Path(args.output), Path(args.latest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    waypoints = manifest["waypoints"]
    stereo = json.loads(Path(args.stereo).read_text(encoding="utf-8"))
    validate_stereo(stereo)
    stereo_hash = hashlib.sha256(Path(args.stereo).read_bytes()).hexdigest()
    mdh_hash = mdh_provenance(args.mdh)
    arm_mdh(args.mdh)
    if (manifest.get("schema") != "luna_sh5_taught_path_v1" or not manifest.get("complete") or
            manifest.get("mdh_sha256") != mdh_hash or
            manifest.get("image_size") != CAMERA_SIZE or manifest.get("sensor_ids") != CAMERA_IDS or
            manifest.get("pixel_domain") != PIXEL_DOMAIN):
        raise ValueError("Taught path belongs to a different robot/camera contract")
    if not 1 <= args.stop_after <= len(waypoints):
        parser.error("--stop-after outside taught path")
    done = completed_rows(output, len(HEAD_POSES))
    start_index = done // len(HEAD_POSES)
    start_pose = done % len(HEAD_POSES)
    if start_index >= args.stop_after:
        print(f"Already captured {start_index} points", flush=True)
        return
    sdk = load_sdk(args.sdk_root)
    robot = connect_robot(sdk, args.robot_ip)
    arm, head = sdk.FXObjType.OBJ_ARM1, sdk.FXObjType.OBJ_HEAD
    try:
        if (str(robot.get_sdk_version()) != EXPECTED_VERSION or
                str(robot.get_controller_version()) != EXPECTED_VERSION):
            raise RuntimeError("SDK/controller version mismatch")
        validate_path(waypoints, arm_limits(robot))
        limits = np.asarray(read_limits(robot), float)
        for target in (REFERENCE, *HEAD_POSES):
            if np.any(np.asarray(target) < limits[:, 0]) or np.any(np.asarray(target) > limits[:, 1]):
                raise ValueError(f"Head pose {target} outside live limits {limits.tolist()}")
        if sdk.state_map.get(robot.current_state(arm)) != "IDLE" or sdk.state_map.get(robot.current_state(head)) != "IDLE":
            raise RuntimeError("Arm and head must both start IDLE")
        rt = read_rt(robot)
        body_ref = np.asarray(manifest["body_q_deg"], float)
        if np.max(np.abs(np.asarray(rt["body"]["fb_pos"], float)-body_ref)) > .3:
            raise RuntimeError("Body moved since the taught trajectory")
        current_q = q7(rt)
        expected_q = np.asarray(waypoints[max(0, start_index-1)]["q7_deg"], float)
        next_q = np.asarray(waypoints[start_index]["q7_deg"], float)
        if min(np.max(np.abs(current_q-expected_q)), np.max(np.abs(current_q-next_q))) > .3:
            raise RuntimeError(f"Arm is not at either safe resume point near {start_index}")
        camera_observation(latest, stereo, timeout=3)
        output.parent.mkdir(parents=True, exist_ok=True)
        checked(robot.switch_to_position_mode(head, 2000, 5, 5), "Head Position")
        third = float(head_angles(rt)[2])
        for reference in head_reference_waypoints(head_angles(rt)[:2], REFERENCE, limits):
            checked(robot.runtime_set_joint_pos_cmd(head, [*reference.tolist(), third]), "Head resume reference")
            wait_head(robot, reference)
        assert_fixed(read_rt(robot), REFERENCE, body_ref)
        checked(robot.switch_to_position_mode(arm, 2000, 5, 5), "Right-arm Position")
        with output.open("a", encoding="utf-8") as out:
            for index in range(start_index, args.stop_after):
                target_q = np.asarray(waypoints[index]["q7_deg"], float)
                start_q = q7(read_rt(robot))
                if np.max(np.abs(target_q-start_q)) > MAX_STEP_DEG+.5:
                    raise RuntimeError("Arm left the dense taught path")
                checked(robot.runtime_set_joint_pos_cmd(arm, target_q.tolist()), f"Arm point {index}")
                wait_arm(robot, sdk, start_q, target_q, REFERENCE, body_ref)
                for pose_index in range(start_pose if index == start_index else 0, len(HEAD_POSES)):
                    target = HEAD_POSES[pose_index]
                    target = np.asarray(target, float)
                    for reference in head_reference_waypoints(head_angles(read_rt(robot))[:2], target, limits):
                        checked(robot.runtime_set_joint_pos_cmd(head, [*reference.tolist(), third]),
                                f"Head point {index} pose {pose_index}")
                        wait_head(robot, reference)
                    row = capture_at_pose(robot, latest, stereo, target_q, target, body_ref)
                    if not row["capture_valid"]:
                        time.sleep(.5)
                        row = capture_at_pose(robot, latest, stereo, target_q, target, body_ref)
                    row.update(schema_version=2, capture_kind="interactive_ball_point_head",
                               waypoint_index=index, head_pose_index=pose_index,
                               target_head_deg=target.tolist(), taught_path=str(path),
                               sdk_version=EXPECTED_VERSION, arm_side="right", control_mode="Position",
                               mdh_source="luna_new.ini:R.A1", mdh_sha256=mdh_hash,
                               stereo_model_sha256=stereo_hash,
                               taught_path_stereo_model_sha256=manifest.get("stereo_model_sha256"),
                               pixel_domain=PIXEL_DOMAIN, sensor_ids=CAMERA_IDS, image_size=CAMERA_SIZE,
                               source_calibration_sha256=stereo.get("source_calibration_sha256"))
                    out.write(json.dumps(row, allow_nan=False)+"\n")
                    out.flush()
                    print(f"Point {index+1}/{len(waypoints)} head {pose_index+1}/{len(HEAD_POSES)} "
                          f"{target.tolist()}: {'valid' if row['capture_valid'] else row['rejection_reason']}", flush=True)
                for reference in head_reference_waypoints(head_angles(read_rt(robot))[:2], REFERENCE, limits):
                    checked(robot.runtime_set_joint_pos_cmd(head, [*reference.tolist(), third]), "Head return reference")
                    wait_head(robot, reference)
    finally:
        if robot._connected:
            if sdk.state_map.get(robot.current_state(arm)) == "Position":
                robot.switch_to_idle(arm, 2000)
            robot.switch_to_idle(head, 2000)
            robot.unlink()


if __name__ == "__main__":
    main()
