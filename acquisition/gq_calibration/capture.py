"""Collect stationary arm/ball observations at measured Luna head poses."""
import argparse
from collections import deque
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np

from capt_head.drag_capture import (EXPECTED_VERSION, JOINT_D, JOINT_K, load_sdk,
                                    interpolate_q, make_record, mdh_provenance, wait_for_yellow,
                                    yellow_pixels)

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [(0, -25)] + [(yaw, pitch) for pitch in (-25, -10, 0)
                         for yaw in (-20, 20, -40, 40, -60, 60)]


def read_limits(robot):
    limits = []
    for i in range(2):
        values = []
        for key in ("LimitNeg", "LimitPos"):
            name = f"R.H.L{i}.BASIC.{key}"
            ret, value = robot.param_get_float(name)
            if ret != 0 or not np.isfinite(value):
                raise RuntimeError(f"Cannot read live controller head limit {name}: {ret}")
            values.append(float(value))
        if values[0] >= values[1]:
            raise RuntimeError(f"Invalid head limits for joint {i}: {values}")
        limits.append(values)
    return limits


def check_targets(limits):
    for yaw, pitch in TARGETS:
        for value, (lower, upper) in zip((yaw, pitch), limits):
            if not lower <= value <= upper:
                raise ValueError(f"Target {(yaw, pitch)} exceeds live head limits {limits}")


def checked(ret, action):
    if ret != 0:
        raise RuntimeError(f"{action} failed: SDK return {ret}")


def head_angles(rt):
    position = np.asarray(rt["head"]["fb_pos"], float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise RuntimeError("Invalid head feedback")
    return position


def wait_head(robot, target, timeout=40):
    deadline = time.monotonic() + timeout
    stable_since = None
    while time.monotonic() < deadline:
        position = head_angles(robot.get_rt_dict())
        if np.max(np.abs(position[:2] - target)) <= .3:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= .7:
                return
        else:
            stable_since = None
        time.sleep(.05)
    raise RuntimeError(f"Head failed to settle at {target} within {timeout}s")


def capture_pose(robot, sdk, latest, out, mdh_hash, index, target):
    arm = sdk.FXObjType.OBJ_ARM1
    print(f"Head {index + 1}/{len(TARGETS)} at {target}. Press Enter to enter right-arm drag; 'q' ends this pose.", flush=True)
    if input().strip():
        raise RuntimeError("Expected Enter; capture stopped")
    wait_for_yellow(latest, 15)
    state = sdk.state_map.get(robot.current_state(arm), "Unknown")
    if state == "Error":
        ret, code = robot.reset_error(arm, 2000)
        if ret != 0:
            raise RuntimeError(f"Right-arm clear error failed: {code}, {ret}")
        state = sdk.state_map.get(robot.current_state(arm), "Unknown")
    if state != "IDLE":
        checked(robot.switch_to_idle(arm, 2000), "Right-arm IDLE")
    checked(robot.switch_to_drag_joint(arm, 2000, JOINT_K, JOINT_D), "Right-arm DragJoint")
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        seen = None
        history = deque()
        saved = rejected = 0
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "q":
                    break
                if key == "\x03":
                    raise KeyboardInterrupt
            if sdk.state_map.get(robot.current_state(arm)) != "DragJoint":
                raise RuntimeError("Right arm left DragJoint")
            try:
                before = time.monotonic_ns()
                rt = robot.get_rt_dict()
                after = time.monotonic_ns()
                sample_ns = (before + after) // 2
                history.append((sample_ns, np.asarray(rt["arms"][1]["fb"]["fb_pos"], float)))
                while history and history[0][0] < sample_ns - 1_000_000_000:
                    history.popleft()
                frame = json.loads(latest.read_text(encoding="utf-8"))
                identity = (frame.get("timestamp_monotonic_ns"), str(frame.get("frame_number")))
                if identity == seen:
                    time.sleep(.002)
                    continue
                seen = identity
                position = head_angles(rt)
                if np.max(np.abs(position[:2] - target)) > .3:
                    raise RuntimeError("Head drifted more than 0.3 deg during capture")
                _, camera_ns, _ = yellow_pixels(frame, sample_ns)
                aligned_q, span_ms = interpolate_q(history, camera_ns)
                row = make_record(frame, rt, sample_ns, mdh_sha256=mdh_hash,
                                  q_deg=aligned_q, q_sample_ns=camera_ns,
                                  q_interpolation_span_ms=span_ms)
                row.update(capture_kind="gq_head_sweep", capture_policy="continuous",
                           target_head_deg=list(target),
                           head_angles_deg=position[:2].tolist(),
                           head_q_deg=position.tolist(),
                           head_angles_rad=np.radians(position[:2]).tolist(),
                           reference_head_angles_rad=np.radians([0, -25]).tolist(),
                           head_pose_index=index, calibration_sample_valid=True)
                out.write(json.dumps(row, allow_nan=False) + "\n")
                out.flush()
                saved += 1
                if saved % 20 == 0:
                    print(f"  {target}: saved {saved}, rejected {rejected}", flush=True)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                rejected += 1
                if rejected == 1 or rejected % 50 == 0:
                    print(f"  rejected {rejected}: {exc}", flush=True)
            time.sleep(.01)
        print(f"Pose {target}: saved {saved}, rejected {rejected}", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        checked(robot.switch_to_idle(arm, 2000), "Right-arm IDLE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", required=True)
    parser.add_argument("--robot-ip", default="6.6.7.190")
    parser.add_argument("--sdk-root", default=str(ROOT.parent / "00040701"))
    parser.add_argument("--mdh", default=str(ROOT / "configs/luna_new.ini"))
    parser.add_argument("--latest", default=str(ROOT / "runtime/new_sh5_yellow/latest_pose.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-pose", type=int, default=1,
                        help="1-based target index; use 2 when [0,-25] was already captured")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Interactive terminal required for Enter/q controls")
    latest = Path(args.latest)
    wait_for_yellow(latest, 15)
    mdh_hash = mdh_provenance(args.mdh)
    sdk = load_sdk(args.sdk_root)
    robot = sdk.GentoRobot()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as out:
        try:
            robot.link(*map(int, args.robot_ip.split(".")), log_level=sdk.FXLogMask.FX_LOG_INFO_FLAG)
            if not robot._connected:
                raise RuntimeError("Robot link failed")
            if str(robot.get_sdk_version()) != EXPECTED_VERSION or str(robot.get_controller_version()) != EXPECTED_VERSION:
                raise RuntimeError("SDK/controller version must both be 00040701")
            limits = read_limits(robot)
            check_targets(limits)
            if not 1 <= args.start_pose <= len(TARGETS):
                raise ValueError(f"start-pose must be 1..{len(TARGETS)}")
            print(f"Live head limits [yaw,pitch]: {limits}", flush=True)
            head = sdk.FXObjType.OBJ_HEAD
            arm = sdk.FXObjType.OBJ_ARM1
            for index, target in enumerate(TARGETS):
                if index + 1 < args.start_pose:
                    continue
                if sdk.state_map.get(robot.current_state(arm)) != "IDLE":
                    checked(robot.switch_to_idle(arm, 2000), "Right-arm IDLE before head move")
                if sdk.state_map.get(robot.current_state(head)) != "IDLE":
                    checked(robot.switch_to_idle(head, 2000), "Head IDLE before move")
                third = head_angles(robot.get_rt_dict())[2]
                checked(robot.switch_to_position_mode(head, 2000, 5, 5), "Head Position")
                checked(robot.runtime_set_joint_pos_cmd(head, [*target, float(third)]), "Head move")
                wait_head(robot, target)
                capture_pose(robot, sdk, latest, out, mdh_hash, index, target)
        finally:
            if robot._connected:
                arm = sdk.FXObjType.OBJ_ARM1
                head = sdk.FXObjType.OBJ_HEAD
                if sdk.state_map.get(robot.current_state(arm)) == "DragJoint":
                    robot.switch_to_idle(arm, 2000)
                robot.switch_to_idle(head, 2000)
                robot.unlink()
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
