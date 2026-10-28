"""Teach operator-reviewed arm waypoints, then replay and observe the flange ball."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import select
import signal
import sys
import time

import numpy as np

from capt_head.drag_capture import (CAMERA_CONTRACT, CAMERA_IDS, CAMERA_SIZE, EXPECTED_VERSION,
                                    JOINT_D, JOINT_K, PIXEL_DOMAIN, YELLOW_BALL_DIAMETER_M,
                                    load_sdk, mdh_provenance, yellow_pixels)
from gq_calibration.capture import checked, head_angles, read_limits, wait_head
from .geometry import (arm_mdh, flange_pose, stereo_sphere_center, triangulate,
                       validate_stereo)
from .fit_pose9d_consistency import matrix_to_rotation6d, rotation6d_to_matrix

ROOT = Path(__file__).resolve().parents[1]
HEAD_TARGET = np.array([0., -25.])
MAX_STEP_DEG = 5.
TRAJECTORY_STEP_DEG = 3.
MIN_VISUAL_POSE_FRAMES_PER_POINT = 8
MAX_VISUAL_ROTATION_JITTER_P95_DEG = 1.0
MAX_CAPTURE_WINDOW_RANGE_DEG = .02


def capture_feedback_frame(frame, rt):
    """Keep per-observation feedback; this timestamp is not an exposure time."""
    return dict(frame, capture_q7_deg=q7(rt).tolist(),
                capture_head_deg=head_angles(rt)[:2].tolist(),
                capture_feedback_monotonic_ns=time.monotonic_ns())


def connect_robot(sdk, robot_ip, attempts=10, delay_s=1.0):
    """Allow the controller time to release the preceding SDK process."""
    last_ret = None
    for attempt in range(1, attempts + 1):
        robot = sdk.GentoRobot()
        last_ret = robot.link(*map(int, robot_ip.split(".")),
                              log_level=sdk.FXLogMask.FX_LOG_INFO_FLAG)
        if robot._connected:
            return robot
        if attempt < attempts:
            print(f"Robot link attempt {attempt}/{attempts} failed; retrying in {delay_s:g}s...",
                  flush=True)
            time.sleep(delay_s)
    raise RuntimeError(f"Robot link failed after {attempts} attempts: "
                       f"{robot._get_operate_error_msg(last_ret)}")


def confirm_automatic_motion(stream=None):
    if stream is not None:
        return stream.readline().strip() == "RUN"
    try:
        with open("/dev/tty", "r", encoding="utf-8") as terminal:
            print("Type RUN to start automatic motion: ", end="", flush=True)
            return terminal.readline().strip() == "RUN"
    except OSError as exc:
        raise RuntimeError("Cannot read RUN confirmation from the controlling terminal") from exc


def read_rt(robot):
    rt = robot.get_rt_dict()
    if "error" in rt:
        raise RuntimeError(rt["error"])
    return rt


def q7(rt):
    q = np.asarray(rt["arms"][1]["fb"]["fb_pos"], float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise RuntimeError("Invalid right-arm feedback")
    return q


def arm_limits(robot):
    result = []
    for i in range(7):
        pair = []
        for key in ("LimitNeg", "LimitPos"):
            name = f"R.A1.L{i}.BASIC.{key}"
            ret, value = robot.param_get_float(name)
            if ret != 0 or not np.isfinite(value):
                raise RuntimeError(f"Cannot read live limit {name}: {ret}")
            pair.append(float(value))
        if pair[0] >= pair[1]:
            raise RuntimeError(f"Invalid arm limit {i}: {pair}")
        result.append(pair)
    return np.asarray(result)


def check_q(q, limits):
    if np.any(q < limits[:, 0] + 2) or np.any(q > limits[:, 1] - 2):
        raise ValueError(f"Right-arm waypoint too close to live joint limit: {q.tolist()}")


def validate_path(waypoints, limits):
    q = np.asarray([w["q7_deg"] for w in waypoints], float)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) < 18 or not np.isfinite(q).all():
        raise ValueError("Teach >=18 distinct finite q7 waypoints")
    for item in q:
        check_q(item, limits)
    if np.any(np.max(np.abs(np.diff(q, axis=0)), axis=1) > MAX_STEP_DEG):
        raise ValueError(f"Adjacent taught waypoints differ by more than {MAX_STEP_DEG:g} deg on one joint")
    if np.any(np.max(np.abs(np.diff(q, axis=0)), axis=1) < 1):
        raise ValueError("Adjacent taught waypoints too similar for an independent sample")


def replay_position_waypoints(current, waypoints):
    current = np.asarray(current, float)
    taught = np.asarray([waypoint["q7_deg"] for waypoint in waypoints], float)
    nearest = int(np.argmin(np.max(np.abs(taught - current), axis=1)))
    if np.max(np.abs(taught[nearest] - current)) <= .3:
        return waypoints[nearest + 1:]
    target = np.asarray(waypoints[-1]["q7_deg"], float)
    delta = target - current
    distance = float(np.max(np.abs(delta)))
    steps = int(np.ceil(distance / MAX_STEP_DEG))
    return [{"q7_deg": (current + delta * step / steps).tolist()}
            for step in range(1, steps + 1)]


def replay_return_waypoints(current, waypoints):
    last = np.asarray(waypoints[-1]["q7_deg"], float)
    if np.max(np.abs(np.asarray(current, float) - last)) > .3:
        raise RuntimeError("Arm did not reach the last taught waypoint")
    return list(reversed(waypoints))


def camera_observation(latest, stereo, timeout=2.):
    deadline = time.monotonic() + timeout
    while True:
        try:
            frame = json.loads(latest.read_text(encoding="utf-8"))
            if frame.get("source_calibration_sha256") != stereo.get("source_calibration_sha256"):
                raise ValueError("RTP rectification calibration does not match the stereo model")
            pixels, stamp, _ = yellow_pixels(frame, time.monotonic_ns())
            # Centre-pixel triangulation remains only a stereo-pair quality check.
            # For a physical sphere the silhouette ellipse centre is not generally
            # the perspective projection of the 3-D sphere centre.
            _, error = triangulate(pixels, stereo)
            sphere = stereo_sphere_center(frame["left"], frame["right"], stereo,
                                          YELLOW_BALL_DIAMETER_M)
            point = np.asarray(sphere["center_left_camera_m"], float)
            if not .15 <= point[2] <= 5:
                raise ValueError("Recovered sphere centre outside plausible 0.15..5 m range")
            return frame, pixels, stamp, error, sphere
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.03)


def fuse_camera_observations(observations, require_visual_pose=False):
    required = CAMERA_CONTRACT["valid_frames_per_point"]
    if len(observations) < required:
        raise ValueError(f"Only {len(observations)} fresh camera frames; need {required}")
    observations = observations[:required]
    identities = [(item[1], str(item[3].get('frame_number') if len(item)>3 else None)) for item in observations]
    if len(set(identities)) != required:
        raise ValueError("Repeated camera frames cannot form an independent fusion window")
    pixels = np.asarray([item[0] for item in observations[:required]], float)
    if pixels.shape != (required, 4) or not np.isfinite(pixels).all():
        raise ValueError("Invalid camera pixels in fusion window")
    median = np.median(pixels, axis=0)
    disparity = pixels[:, 0]-pixels[:, 2]
    disparity_jitter = float(np.percentile(
        np.abs(disparity-np.median(disparity)), 95))
    if disparity_jitter > CAMERA_CONTRACT["max_point_disparity_jitter_p95_px"]:
        raise ValueError(f"Disparity jitter {disparity_jitter:.3f} px exceeds contract")
    result = {
        "stereo_pixels": median.tolist(),
        "valid_frames": required,
        "pixel_std_px": pixels.std(axis=0).tolist(),
        "pixel_mad_px": np.median(np.abs(pixels-median), axis=0).tolist(),
        "pixel_p95_jitter_px": np.percentile(
            np.abs(pixels-median), 95, axis=0).tolist(),
        "disparity_jitter_p95_px": disparity_jitter,
        "stereo_pixels_framewise": pixels.tolist(),
        "camera_window_frames": [dict(
            frame_number=item[3].get("frame_number"),
            source_timestamp_ns=item[3].get("source_timestamp_ns"),
            source_clock_ns=item[3].get("source_clock_ns"),
            pair_source_clock_skew_ms=item[3].get("pair_source_clock_skew_ms"),
            pairing_policy=item[3].get("pairing_policy"),
            epipolar_check_domain=item[3].get("epipolar_check_domain"),
            capture_dequeue_timestamp_ns=item[3].get("capture_dequeue_timestamp_ns"),
            camera_timestamp_monotonic_ns=int(item[1]),
            pair_dequeue_skew_ms=item[3].get("pair_dequeue_skew_ms"),
            left=item[3].get("left"), right=item[3].get("right"),
            image_evidence=item[3].get("image_evidence"),
            camera_controls=item[3].get("camera_controls"),
        ) for item in observations if len(item) > 3],
    }
    if all(len(item) >= 5 and item[4] is not None for item in observations[:required]):
        sphere = np.asarray([item[4]["center_left_camera_m"] for item in observations[:required]], float)
        corrected_pixels = np.asarray([
            item[4]["corrected_stereo_pixels"] for item in observations[:required]], float)
        corrections = np.asarray([
            item[4]["projection_correction_px"] for item in observations[:required]], float)
        reprojection = np.asarray([
            item[4]["stereo_reprojection_error_px"] for item in observations[:required]], float)
        if sphere.shape != (required, 3) or not np.isfinite(sphere).all() or np.any(sphere[:,2] <= 0):
            raise ValueError("Invalid physical sphere observations")
        center = np.median(sphere, axis=0)
        corrected_median = np.median(corrected_pixels, axis=0)
        corrected_disparity = corrected_pixels[:, 0] - corrected_pixels[:, 2]
        radial_mm = 1000. * np.linalg.norm(sphere-center, axis=1)
        result.update(
            sphere_center_method="sphere_center_projection_stereo_v1",
            yellow_ball_diameter_m=YELLOW_BALL_DIAMETER_M,
            sphere_center_left_camera_m=center.tolist(),
            sphere_center_corrected_stereo_pixels=corrected_median.tolist(),
            sphere_center_corrected_pixels_framewise=corrected_pixels.tolist(),
            sphere_projection_correction_px=np.median(corrections, axis=0).tolist(),
            sphere_center_p95_jitter_mm=float(np.percentile(radial_mm, 95)),
            sphere_center_axis_p95_jitter_mm=(1000. * np.percentile(
                np.abs(sphere-center), 95, axis=0)).tolist(),
            sphere_corrected_disparity_jitter_p95_px=float(np.percentile(
                np.abs(corrected_disparity-np.median(corrected_disparity)), 95)),
            sphere_center_stereo_reprojection_p95_px=float(np.percentile(reprojection, 95)),
            sphere_center_framewise_m=sphere.tolist(),
            sphere_center_frame_covariance_mm2=np.cov(sphere*1000., rowvar=False).tolist(),
            sphere_center_half_window_shift_mm=float(1000*np.linalg.norm(
                np.median(sphere[:required//2],axis=0)-np.median(sphere[required//2:],axis=0))),
        )
    feedback = [item[3] for item in observations if len(item)>3 and 'capture_q7_deg' in item[3]]
    if feedback:
        q = np.asarray([f['capture_q7_deg'] for f in feedback], float)
        h = np.asarray([f['capture_head_deg'] for f in feedback], float)
        if (q.shape != (required,7) or h.shape != (required,2)
                or not np.isfinite(q).all() or not np.isfinite(h).all()):
            raise ValueError("Incomplete or invalid fusion-window robot feedback")
        q_span, h_span = np.ptp(q,axis=0), np.ptp(h,axis=0)
        if max(q_span.max(),h_span.max()) > MAX_CAPTURE_WINDOW_RANGE_DEG:
            raise ValueError("Robot feedback changed by more than 0.02 deg during fusion")
        result.update(robot_window_q7_deg=q.tolist(), robot_window_head_deg=h.tolist(),
                      robot_window_q7_range_deg=q_span.tolist(),
                      robot_window_head_range_deg=h_span.tolist(),
                      robot_window_feedback_monotonic_ns=[f['capture_feedback_monotonic_ns'] for f in feedback],
                      camera_window_monotonic_ns=[int(item[1]) for item in observations])
    pose_by_frame = {}
    for item in observations[:required]:
        frame = item[3] if len(item) > 3 else {}
        pose = frame.get("four_ball_pose") if isinstance(frame, dict) else None
        if (not isinstance(pose, dict) or pose.get("valid") is not True or
                pose.get("fresh") is not True or
                pose.get("visual_rotation6d_camera") is None):
            continue
        identity = tuple(pose.get("frame_number", ()))
        if identity:
            pose_by_frame[identity] = pose
    if pose_by_frame:
        values = list(pose_by_frame.values())
        rotations = np.asarray([
            rotation6d_to_matrix(item["visual_rotation6d_camera"]) for item in values])
        average = rotations.sum(axis=0)
        U, _, Vt = np.linalg.svd(average)
        mean_rotation = U @ Vt
        if np.linalg.det(mean_rotation) < 0:
            U[:, -1] *= -1
            mean_rotation = U @ Vt
        angular = []
        for rotation in rotations:
            cosine = np.clip((np.trace(mean_rotation.T @ rotation)-1.)/2., -1., 1.)
            angular.append(np.degrees(np.arccos(cosine)))
        rotation_p95 = float(np.percentile(angular, 95))
        if (require_visual_pose and
                (len(values) < MIN_VISUAL_POSE_FRAMES_PER_POINT or
                 rotation_p95 > MAX_VISUAL_ROTATION_JITTER_P95_DEG)):
            raise ValueError(
                f"Visual pose has {len(values)} unique frames and {rotation_p95:.3f} deg P95 jitter")
        if "sphere_center_left_camera_m" in result:
            result.update(
                visual_xyz_camera_m=result["sphere_center_left_camera_m"],
                visual_rotation6d_camera=matrix_to_rotation6d(mean_rotation).tolist(),
                visual_pose_method="stereo_colour_direction_pose_v1",
                visual_pose_unique_frames=len(values),
                visual_rotation_jitter_p95_deg=rotation_p95,
            )
    elif require_visual_pose:
        raise ValueError("No fresh four-ball visual orientation available")
    return result


def assert_fixed(rt, head_ref, body_ref):
    head = head_angles(rt)[:2]
    body = np.asarray(rt["body"]["fb_pos"], float)
    if head.shape != (2,) or body.shape != (6,) or not np.isfinite(head).all() or not np.isfinite(body).all():
        raise RuntimeError("Invalid head/body feedback")
    if np.max(np.abs(head - head_ref)) > .25 or np.max(np.abs(body - body_ref)) > .3:
        raise RuntimeError("Head or body moved; fixed-camera calibration cannot continue")


def head_reference_waypoints(current, target, limits, *, tolerance_deg=.05,
                             approach_deg=1.0):
    """Return a deterministic final-approach path for repeatable fixed-head capture.

    If feedback is already at the requested reference pose, return no motion at all.
    Otherwise first move to a point on a fixed side of the target (prefer + angle,
    fall back to - angle near a joint limit), then approach the target from that
    same side on every calibration session.
    """
    current = np.asarray(current, float)
    target = np.asarray(target, float)
    limits = np.asarray(limits, float)
    if current.shape != (2,) or target.shape != (2,) or limits.shape != (2, 2):
        raise ValueError("Invalid head reference inputs")
    if np.max(np.abs(current-target)) <= tolerance_deg:
        return []
    approach = target.copy()
    for index, (lo, hi) in enumerate(limits):
        positive = target[index] + approach_deg
        negative = target[index] - approach_deg
        if positive <= hi - .1:
            approach[index] = positive
        elif negative >= lo + .1:
            approach[index] = negative
        else:
            raise ValueError("Head target has no room for a repeatable approach")
    return [approach, target.copy()]


def wait_arm(robot, sdk, start, target, head_ref, body_ref, timeout=35):
    arm = sdk.FXObjType.OBJ_ARM1
    deadline = time.monotonic() + timeout
    stable_since = None
    while time.monotonic() < deadline:
        if sdk.state_map.get(robot.current_state(arm)) != "Position":
            raise RuntimeError("Right arm left Position mode")
        rt = read_rt(robot)
        assert_fixed(rt, head_ref, body_ref)
        q = q7(rt)
        vel = np.asarray(rt["arms"][1]["fb"]["fb_vel"], float)
        if (vel.shape != (7,) or not np.isfinite(vel).all() or np.max(np.abs(vel)) > 30 or
                np.any(q < np.minimum(start, target) - 2) or np.any(q > np.maximum(start, target) + 2)):
            raise RuntimeError("Arm feedback left the taught joint segment or exceeded 30 deg/s")
        if np.max(np.abs(q - target)) <= .25 and np.max(np.abs(vel)) <= .5:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= .7:
                return rt
        else:
            stable_since = None
        time.sleep(.05)
    raise RuntimeError(f"Arm failed to settle at taught waypoint within {timeout}s")


def teach(robot, sdk, latest, stereo, stereo_hash, output, mdh_hash, limits):
    arm = sdk.FXObjType.OBJ_ARM1
    state = sdk.state_map.get(robot.current_state(arm), "Unknown")
    if state == "Error":
        ret, code = robot.reset_error(arm, 2000)
        if ret:
            raise RuntimeError(f"Right-arm clear error failed: {ret}, {code}")
        state = sdk.state_map.get(robot.current_state(arm), "Unknown")
    if state != "IDLE":
        checked(robot.switch_to_idle(arm, 2000), "Right-arm IDLE")
    checked(robot.switch_to_drag_joint(arm, 2000, JOINT_K, JOINT_D), "Right-arm DragJoint")
    rt = read_rt(robot)
    head_ref = head_angles(rt)[:2]
    body_ref = np.asarray(rt["body"]["fb_pos"], float)
    waypoints = []
    partial = Path(str(output) + ".partial")
    last_limit_warning = 0.0
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("\nCtrl+C received; finishing and validating the taught trajectory...", flush=True)

    def manifest(complete):
        return {"schema": "luna_sh5_taught_path_v1", "complete": complete,
                "arm_side": "right", "sdk_version": EXPECTED_VERSION,
                "mdh_sha256": mdh_hash, "stereo_model_sha256": stereo_hash,
                "head_angles_deg": head_ref.tolist(), "body_q_deg": body_ref.tolist(),
                "image_size": CAMERA_SIZE, "sensor_ids": CAMERA_IDS,
                "pixel_domain": PIXEL_DOMAIN, "recording_mode": "continuous_drag_3deg",
                "waypoints": waypoints}

    print(f"Trajectory recording is active. Drag slowly; points are saved every {TRAJECTORY_STEP_DEG:g} deg. "
          "After at least 18 points, press Ctrl+C to finish and save.", flush=True)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
      while not stop_requested:
        if select.select([sys.stdin], [], [], 0)[0]:
            command = sys.stdin.readline().strip().lower()
            if command == "q":
                try:
                    validate_path(waypoints, limits)
                except ValueError as exc:
                    print(f"Cannot finish yet: {exc}. Continue dragging, then type q again.", flush=True)
                else:
                    break
            elif command:
                print("Type q then Enter to finish; trajectory points are saved automatically.", flush=True)
        if sdk.state_map.get(robot.current_state(arm)) != "DragJoint":
            raise RuntimeError("Right arm left DragJoint")
        rt = read_rt(robot)
        assert_fixed(rt, head_ref, body_ref)
        q = q7(rt)
        try:
            check_q(q, limits)
        except ValueError as exc:
            if time.monotonic() - last_limit_warning >= 2:
                print(f"Not recording near joint limit: {exc}. Move away from the limit.", flush=True)
                last_limit_warning = time.monotonic()
            time.sleep(.02)
            continue
        if waypoints:
            delta = np.max(np.abs(q - np.asarray(waypoints[-1]["q7_deg"])))
            if delta < TRAJECTORY_STEP_DEG:
                time.sleep(.02)
                continue
            if delta > MAX_STEP_DEG:
                raise RuntimeError(f"Trajectory sampling gap {delta:.1f} deg exceeds {MAX_STEP_DEG:g} deg; drag more slowly")
        waypoints.append({"q7_deg": q.tolist(), "head_angles_deg": head_ref.tolist(),
                          "body_q_deg": body_ref.tolist()})
        temp = Path(str(partial) + ".tmp")
        temp.write_text(json.dumps(manifest(False), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(partial)
        print(f"Recorded trajectory point {len(waypoints)}", flush=True)
        time.sleep(.02)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    validate_path(waypoints, limits)
    with output.open("x", encoding="utf-8") as out:
        json.dump(manifest(True), out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    partial.unlink(missing_ok=True)
    print(f"Saved {len(waypoints)} taught waypoints to {output}")


def replay(robot, sdk, latest, stereo, stereo_hash, path, output, mdh_hash, limits, head_ref, motion_confirmed=False, require_visual_pose=False):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    waypoints = manifest["waypoints"]
    validate_path(waypoints, limits)
    if (manifest.get("schema") != "luna_sh5_taught_path_v1" or manifest.get("complete") is not True or
            manifest.get("mdh_sha256") != mdh_hash or
            manifest.get("image_size") != CAMERA_SIZE or manifest.get("sensor_ids") != CAMERA_IDS or
            manifest.get("pixel_domain") != PIXEL_DOMAIN):
        raise ValueError("Taught path belongs to different robot MDH/camera contract")
    rt = read_rt(robot)
    body_ref = np.asarray(manifest["body_q_deg"], float)
    assert_fixed(rt, head_ref, body_ref)
    position_waypoints = replay_position_waypoints(q7(rt), waypoints)
    arm = sdk.FXObjType.OBJ_ARM1
    if sdk.state_map.get(robot.current_state(arm)) != "IDLE":
        raise RuntimeError("Right arm must be IDLE before automatic replay")
    print(f"Ready to position in <=5 deg joint steps to the reverse replay start without capture, then capture "
          f"{len(waypoints)} waypoints backward to the start at 5% SDK speed. "
          "Keep the terminal and emergency stop in reach.")
    if not motion_confirmed and not confirm_automatic_motion():
        raise RuntimeError("Automatic replay cancelled")
    checked(robot.switch_to_position_mode(arm, 2000, 5, 5), "Right-arm Position")
    for index, waypoint in enumerate(position_waypoints):
        target = np.asarray(waypoint["q7_deg"], float)
        start = q7(read_rt(robot))
        checked(robot.runtime_set_joint_pos_cmd(arm, target.tolist()),
                f"Position toward reverse replay start {index} motion")
        wait_arm(robot, sdk, start, target, head_ref, body_ref)
        print(f"Positioned toward reverse replay start {index + 1}/{len(position_waypoints)}; not captured",
              flush=True)
    return_waypoints = replay_return_waypoints(q7(read_rt(robot)), waypoints)
    with output.open("x", encoding="utf-8") as out:
        for reverse_index, waypoint in enumerate(return_waypoints):
            waypoint_index = len(waypoints) - 1 - reverse_index
            target = np.asarray(waypoint["q7_deg"], float)
            start = q7(read_rt(robot))
            if np.max(np.abs(start - target)) > MAX_STEP_DEG + .5:
                raise RuntimeError("Current arm pose has left the dense taught path")
            checked(robot.runtime_set_joint_pos_cmd(arm, target.tolist()),
                    f"Reverse waypoint {waypoint_index} motion")
            wait_arm(robot, sdk, start, target, head_ref, body_ref)
            observations = []
            required_frames = CAMERA_CONTRACT["valid_frames_per_point"]
            deadline = time.monotonic() + 8
            seen = set()
            rejected_frames = 0
            while len(observations) < required_frames and time.monotonic() < deadline:
                try:
                    frame, pixels, stamp, error, sphere = camera_observation(latest, stereo, timeout=.2)
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    rejected_frames += 1
                    continue
                identity = (stamp, str(frame.get("frame_number")))
                if identity not in seen:
                    rt = read_rt(robot)
                    assert_fixed(rt, head_ref, body_ref)
                    if np.max(np.abs(q7(rt) - target)) > .3:
                        raise RuntimeError("Arm moved during camera sampling")
                    observations.append((pixels, stamp, error, capture_feedback_frame(frame, rt), sphere))
                    seen.add(identity)
                time.sleep(.03)
            robot_before_ns = time.monotonic_ns()
            rt = read_rt(robot)
            robot_after_ns = time.monotonic_ns()
            row = {"schema_version": 2, "capture_kind": "interactive_ball_auto",
                   "waypoint_index": waypoint_index, "sdk_version": EXPECTED_VERSION, "arm_side": "right",
                   "control_mode": "Position", "mdh_source": "luna_new.ini:R.A1", "mdh_sha256": mdh_hash,
                   "stereo_model_sha256": stereo_hash,
                   "taught_path": str(path),
                   "taught_path_stereo_model_sha256": manifest.get("stereo_model_sha256"),
                   "q7_deg": q7(rt).tolist(),
                   "head_angles_deg": head_angles(rt)[:2].tolist(), "body_q_deg": rt["body"]["fb_pos"],
                   "head_q_deg": head_angles(rt).tolist(),
                   "pixel_domain": PIXEL_DOMAIN, "sensor_ids": CAMERA_IDS, "image_size": CAMERA_SIZE,
                   "source_calibration_sha256": stereo.get("source_calibration_sha256"),
                   "camera_rejected_frames": rejected_frames,
                   "robot_feedback_timestamp_monotonic_ns": (robot_before_ns + robot_after_ns) // 2,
                   "robot_feedback_read_span_ms": (robot_after_ns - robot_before_ns) / 1e6,
                   "camera_timestamp_basis": "local_udp_dequeue_minus_processing_not_exposure",
                   "exposure_time_valid": False,
                   "exposure_sync_status": "unverified_no_sensor_exposure_timestamp"}
            try:
                fused = fuse_camera_observations(
                    observations, require_visual_pose=require_visual_pose)
            except ValueError as exc:
                row.update(capture_valid=False, rejection_reason=str(exc), stereo_pixels=None,
                           sphere_center_left_camera_m=None,
                           camera_timestamp_monotonic_ns=None, stereo_reprojection_max_px=None)
                status = f"INVALID: {row['rejection_reason']}"
            else:
                try:
                    if "sphere_center_left_camera_m" not in fused:
                        raise ValueError("Sphere-aware capture did not produce a physical ball centre")
                    sphere_point = np.asarray(fused["sphere_center_left_camera_m"], float)
                    if (sphere_point.shape != (3,) or not np.isfinite(sphere_point).all() or
                            not .15 <= sphere_point[2] <= 5):
                        raise ValueError("Invalid fused physical sphere centre")
                except ValueError as exc:
                    row.update(capture_valid=False, rejection_reason=str(exc), stereo_pixels=None,
                               sphere_center_left_camera_m=None,
                               camera_timestamp_monotonic_ns=None, stereo_reprojection_max_px=None)
                    status = f"INVALID: {exc}"
                else:
                    row.update(fused, capture_valid=True,
                               camera_timestamp_monotonic_ns=max(o[1] for o in observations),
                               camera_robot_offset_ms=(max(o[1] for o in observations) -
                                                       (robot_before_ns + robot_after_ns) / 2) / 1e6,
                               pair_capture_skew_ms=float(max(
                                   o[3].get("pair_dequeue_skew_ms", 0.) for o in observations)),
                               camera_pipeline_delay_ms=float(np.median([
                                   o[3].get("camera_pipeline_delay_ms", 0.) for o in observations])),
                               stereo_reprojection_max_px=float(max(o[2] for o in observations)))
                    status = "valid"
            out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
            print(f"Collected reverse waypoint {reverse_index + 1}/{len(waypoints)} "
                  f"(taught index {waypoint_index}): {status}", flush=True)
    print(f"Saved {len(waypoints)} observations to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("teach", "replay"))
    parser.add_argument("--stereo", required=True, help="New SH5 camera_calibration.fit JSON")
    parser.add_argument("--latest", default=str(ROOT / "runtime/new_sh5_yellow/latest_pose.json"))
    parser.add_argument("--path", help="Taught path JSON (replay only)")
    parser.add_argument("--run", action="store_true", help="Explicitly authorize replay motion without typing RUN")
    parser.add_argument("--require-visual-pose9", action="store_true",
                        help="reject stopped points without stable four-ball Rotation6D")
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot-ip", default="6.6.7.190")
    parser.add_argument("--sdk-root", default=str(ROOT.parent / "00040701"))
    parser.add_argument("--mdh", default=str(ROOT / "configs/luna_new.ini"))
    parser.add_argument("--head-target", nargs=2, type=float, default=HEAD_TARGET.tolist(),
                        metavar=("YAW_DEG", "PITCH_DEG"))
    args = parser.parse_args()
    if not sys.stdin.isatty() or (args.mode == "replay" and not args.path):
        parser.error("Interactive terminal required; replay also requires --path")
    stereo = json.loads(Path(args.stereo).read_text(encoding="utf-8"))
    stereo_hash = hashlib.sha256(Path(args.stereo).read_bytes()).hexdigest()
    validate_stereo(stereo)
    latest, output = Path(args.latest), Path(args.output)
    if args.mode == "teach":
        camera_observation(latest, stereo)
    mdh_hash = mdh_provenance(args.mdh)
    arm_mdh(args.mdh)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    sdk = load_sdk(args.sdk_root)
    robot = connect_robot(sdk, args.robot_ip)
    try:
        if str(robot.get_sdk_version()) != EXPECTED_VERSION or str(robot.get_controller_version()) != EXPECTED_VERSION:
            raise RuntimeError("SDK/controller version must both be 00040701")
        limits = arm_limits(robot)
        head_limits = read_limits(robot)
        target = np.asarray(args.head_target, float)
        if any(not lo <= value <= hi for value, (lo, hi) in zip(target, head_limits)):
            raise RuntimeError(f"Head target {target.tolist()} outside live limits {head_limits}")
        arm = sdk.FXObjType.OBJ_ARM1
        arm_state = sdk.state_map.get(robot.current_state(arm), "Unknown")
        if arm_state == "Error" and args.mode == "teach":
            ret, code = robot.reset_error(arm, 2000)
            if ret:
                raise RuntimeError(f"Right-arm clear error failed: {ret}, {code}")
            arm_state = sdk.state_map.get(robot.current_state(arm), "Unknown")
        if arm_state != "IDLE":
            if args.mode == "replay":
                raise RuntimeError("Right arm must be IDLE before head or arm motion")
            checked(robot.switch_to_idle(arm, 2000), "Right-arm IDLE before head move")
        head = sdk.FXObjType.OBJ_HEAD
        if sdk.state_map.get(robot.current_state(head)) != "IDLE":
            checked(robot.switch_to_idle(head, 2000), "Head IDLE")
        current_head = head_angles(read_rt(robot))
        reference_waypoints = head_reference_waypoints(
            current_head[:2], target, np.asarray(head_limits, float))
        if reference_waypoints:
            third = float(current_head[2])
            checked(robot.switch_to_position_mode(head, 2000, 5, 5), "Head Position")
            for index, reference in enumerate(reference_waypoints):
                checked(robot.runtime_set_joint_pos_cmd(
                    head, [*reference.tolist(), third]),
                    f"Head repeatable reference move {index + 1}/{len(reference_waypoints)}")
                wait_head(robot, reference)
            print("Head reference reached with deterministic final approach.", flush=True)
        else:
            print("Head already within 0.05 deg of reference; no head motion commanded.", flush=True)
        if args.mode == "teach":
            teach(robot, sdk, latest, stereo, stereo_hash, output, mdh_hash, limits)
        else:
            replay(robot, sdk, latest, stereo, stereo_hash, Path(args.path), output,
                   mdh_hash, limits, target, args.run, args.require_visual_pose9)
    finally:
        if robot._connected:
            arm, head = sdk.FXObjType.OBJ_ARM1, sdk.FXObjType.OBJ_HEAD
            if sdk.state_map.get(robot.current_state(arm)) in ("DragJoint", "Position"):
                robot.switch_to_idle(arm, 2000)
            robot.switch_to_idle(head, 2000)
            robot.unlink()


if __name__ == "__main__":
    main()
