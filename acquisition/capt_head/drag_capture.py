"""Enter Luna right-arm joint drag mode and pair q7 feedback with yellow pixels."""
from __future__ import annotations

import argparse
from collections import deque
import configparser
import hashlib
import importlib
import json
import platform
from pathlib import Path
import signal
import sys
import time

import numpy as np
from .measurement_contract import load_contract, valid_measurement


EXPECTED_VERSION = "00040701"
RIGHT_ARM_INDEX = 1
CAMERA_CONTRACT, _ = load_contract()
CAMERA_SIZE = CAMERA_CONTRACT["image_size"]
CAMERA_IDS = CAMERA_CONTRACT["sensor_ids"]
PIXEL_DOMAIN = CAMERA_CONTRACT["pixel_domain"]
JOINT_K = [3, 3, 3, 2, 1, 1, 1]
JOINT_D = [.2] * 7
YELLOW_BALL_DIAMETER_M = .04


def stationary_q(history, window_ms, max_range_deg):
    if len(history) < 2 or (history[-1][0] - history[0][0]) / 1e6 < .8 * window_ms:
        return False
    return float(np.max(np.ptp(np.asarray([item[1] for item in history]), axis=0))) <= max_range_deg


def mdh_provenance(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"MDH file missing: {source}")
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(source, encoding="utf-8")
    required = ["R.A1.FLANGE"] + [f"R.A1.L{i}.DH" for i in range(7)]
    for section in required:
        if section not in cfg or any(key not in cfg[section] for key in ("Alpha", "A", "D", "Theta")):
            raise ValueError(f"Incomplete right-arm MDH section: {section}")
        if not np.isfinite([float(cfg[section][key]) for key in ("Alpha", "A", "D", "Theta")]).all():
            raise ValueError(f"Nonfinite right-arm MDH section: {section}")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def yellow_pixels(frame, now_ns, max_age_ms=200.0):
    """Validate the existing camera sidecar and return one yellow center per eye."""
    if frame.get("connected") is not True:
        raise ValueError("camera disconnected")
    if frame.get("detector_version") != 4:
        raise ValueError("expected Orin subpixel detector version 4")
    valid_measurement(frame, CAMERA_CONTRACT)
    stamp_value = frame.get("camera_capture_timestamp_monotonic_ns")
    stamp = int(frame["timestamp_monotonic_ns"] if stamp_value is None else stamp_value)
    if frame.get("timestamp_basis") in ("local_udp_dequeue_not_exposure",
                                         "local_receive_minus_orin_processing_not_exposure",
                                         "local_monotonic_estimated_capture_dequeue_not_exposure"):
        max_age_ms = max(max_age_ms, 500.0)
    age_ms = (now_ns - stamp) / 1e6
    if not 0 <= age_ms <= max_age_ms:
        raise ValueError("stale camera frame")
    if frame.get("pair_dequeue_skew_ms", 1e9) > 10:
        raise ValueError("stereo pair dequeue skew exceeds 10 ms")
    if frame.get("timestamp_basis") not in ("orin_host_dequeue_not_exposure",
                                             "local_dds_dequeue_not_exposure",
                                             "local_rtp_dequeue_not_exposure",
                                             "local_udp_dequeue_not_exposure",
                                             "local_receive_minus_orin_processing_not_exposure",
                                             "local_monotonic_estimated_capture_dequeue_not_exposure"):
        raise ValueError("unexpected camera timestamp basis")
    size = frame.get("image_size")
    if size != CAMERA_CONTRACT["image_size"]:
        raise ValueError("expected calibrated 1280x992 images from the SH5 ROS2/Apex pipeline")
    if frame.get("sensor_ids") != CAMERA_CONTRACT["sensor_ids"]:
        raise ValueError("expected head_left csi://0 then head_right csi://1")
    if frame.get("pixel_domain") != CAMERA_CONTRACT["pixel_domain"]:
        raise ValueError("expected pixels from native-resolution undistorted ROS2 images")
    if frame.get("measurement_backend") != CAMERA_CONTRACT["measurement_backend"]:
        raise ValueError("expected Orin subpixel yellow detector")
    if frame.get("source_calibration_sha256") != CAMERA_CONTRACT["calibration_sha256"]:
        raise ValueError("yellow pixels use a different checkerboard calibration")
    detection_fps = float(frame.get("detection_fps", 0))
    if not np.isfinite(detection_fps) or detection_fps < 0:
        raise ValueError("invalid yellow detection rate")
    observations = frame["image_observations"]
    left, right = observations["left"]["yellow"], observations["right"]["yellow"]
    if len(left) != 1 or len(right) != 1:
        raise ValueError("yellow ball missing or ambiguous")
    pixels = np.asarray(left[0]["pixel"] + right[0]["pixel"], float)
    if pixels.shape != (4,) or not np.isfinite(pixels).all():
        raise ValueError("invalid yellow pixels")
    w, h = size
    if (np.any(pixels[[0, 2]] < 0) or np.any(pixels[[0, 2]] >= w) or
            np.any(pixels[[1, 3]] < 0) or np.any(pixels[[1, 3]] >= h)):
        raise ValueError("yellow pixels outside image")
    return pixels.tolist(), stamp, age_ms


def wait_for_yellow(path, timeout_s):
    deadline = time.monotonic() + timeout_s
    last_error = "no camera frame"
    while time.monotonic() < deadline:
        try:
            frame = json.loads(path.read_text(encoding="utf-8"))
            yellow_pixels(frame, time.monotonic_ns())
            return frame
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
            time.sleep(.05)
    raise ValueError(f"camera did not produce a fresh stereo yellow frame within {timeout_s:g}s: {last_error}")


def interpolate_q(history, target_ns, max_span_ms=50.0):
    """Interpolate q7 from locally timestamped SDK feedback around an image time."""
    if len(history) < 2 or target_ns < history[0][0] or target_ns > history[-1][0]:
        raise ValueError("q7 history does not bracket delayed camera frame")
    samples = list(history)
    for (before_ns, before_q), (after_ns, after_q) in zip(samples, samples[1:]):
        if before_ns <= target_ns <= after_ns:
            span = after_ns - before_ns
            if span <= 0:
                raise ValueError("q7 history timestamps are not increasing")
            if span / 1e6 > max_span_ms:
                raise ValueError("q7 feedback gap exceeds synchronization limit")
            weight = (target_ns - before_ns) / span
            return before_q + weight * (after_q - before_q), span / 1e6
    raise ValueError("q7 history does not bracket delayed camera frame")


def make_record(frame, rt, sample_ns, *, max_pair_ms=100.0, mdh_sha256=None,
                q_deg=None, q_sample_ns=None, camera_delay_ms=0.0,
                q_interpolation_span_ms=None):
    pixels, camera_ns, age_ms = yellow_pixels(frame, sample_ns)
    q_deg = np.asarray(rt["arms"][RIGHT_ARM_INDEX]["fb"]["fb_pos"] if q_deg is None else q_deg, float)
    if q_deg.shape != (7,) or not np.isfinite(q_deg).all():
        raise ValueError("invalid right-arm q7 feedback")
    q_sample_ns = sample_ns if q_sample_ns is None else int(q_sample_ns)
    target_ns = camera_ns - int(camera_delay_ms * 1e6)
    pair_ms = abs(q_sample_ns - target_ns) / 1e6
    if pair_ms > max_pair_ms:
        raise ValueError("camera/q7 pairing exceeds limit")
    record = {
        "schema": "luna_yellow_measurement_v1",
        "schema_version": 1,
        "sdk_version": EXPECTED_VERSION,
        "arm_side": "right",
        "sdk_object": "ARM1",
        "control_mode": "DragJoint",
        "mdh_source": "luna_new.ini:R.A1",
        "mdh_sha256": mdh_sha256,
        "yellow_ball_diameter_m": YELLOW_BALL_DIAMETER_M,
        "yellow_center_flange_m": None,
        "q7_deg": q_deg.tolist(),
        "q7_rad": np.radians(q_deg).tolist(),
        "body_q_deg": rt["body"]["fb_pos"],
        "stereo_pixels": pixels,
        "pixel_order": ["left_u", "left_v", "right_u", "right_v"],
        "pixel_domain": frame.get("pixel_domain"),
        "image_size": frame["image_size"],
        "sensor_ids": frame.get("sensor_ids"),
        "measurement_backend": frame.get("measurement_backend"),
        "source_calibration_sha256": frame.get("source_calibration_sha256"),
        "contract_sha256": frame.get("contract_sha256"),
        "quality": frame.get("quality"),
        "left": frame.get("left"), "right": frame.get("right"),
        "disparity_px": frame.get("disparity_px"),
        "epipolar_dv_px": frame.get("epipolar_dv_px"),
        "exposure_locked": frame.get("exposure_locked"),
        "awb_locked": frame.get("awb_locked"),
        "brightness_stable": frame.get("brightness_stable"),
        "antibanding_hz": frame.get("antibanding_hz"),
        "detection_fps": float(frame["detection_fps"]),
        "measurement_fps": float(frame["measurement_fps"]),
        "frame_number": frame.get("frame_number"),
        "sdk_frame_serial": rt.get("frame_serial"),
        "camera_timestamp_monotonic_ns": camera_ns,
        "camera_capture_timestamp_monotonic_ns": camera_ns,
        "metadata_received_timestamp_monotonic_ns": frame.get(
            "metadata_received_timestamp_monotonic_ns"),
        "q7_sample_timestamp_monotonic_ns": q_sample_ns,
        "camera_q7_pair_ms": pair_ms,
        "camera_pipeline_delay_ms": float(frame.get("camera_pipeline_delay_ms", camera_delay_ms)),
        "q7_interpolation_span_ms": q_interpolation_span_ms,
        "camera_age_ms": age_ms,
        "pair_dequeue_skew_ms": frame.get("pair_dequeue_skew_ms"),
        "pair_capture_skew_ms": frame.get("pair_capture_skew_ms", frame.get("pair_dequeue_skew_ms")),
        "timestamp_basis": frame.get("timestamp_basis"),
        "network_latency_uncompensated": frame.get("network_latency_uncompensated", True),
        "timestamp_alignment": "q7_interpolated_at_estimated_camera_capture_dequeue",
        "exposure_time_valid": bool(frame.get("exposure_time_valid", False)),
        "training_eligible": False,
    }
    head = np.asarray(rt.get("head", {}).get("fb_pos", []), float)
    if head.shape == (3,) and np.isfinite(head).all():
        record.update(head_q_deg=head.tolist(), head_angles_deg=head[:2].tolist(),
                      head_angles_rad=np.radians(head[:2]).tolist())
    return record


def make_pivot_record(rt, sample_ns, mdh_sha256):
    q_deg = np.asarray(rt["arms"][RIGHT_ARM_INDEX]["fb"]["fb_pos"], float)
    if q_deg.shape != (7,) or not np.isfinite(q_deg).all():
        raise ValueError("invalid right-arm q7 feedback")
    return {
        "schema_version": 1,
        "capture_kind": "pivot_q7",
        "sdk_version": EXPECTED_VERSION,
        "arm_side": "right",
        "sdk_object": "ARM1",
        "control_mode": "DragJoint",
        "mdh_source": "luna_new.ini:R.A1",
        "mdh_sha256": mdh_sha256,
        "yellow_ball_diameter_m": YELLOW_BALL_DIAMETER_M,
        "q7_deg": q_deg.tolist(),
        "q7_rad": np.radians(q_deg).tolist(),
        "body_q_deg": rt["body"]["fb_pos"],
        "q7_sample_timestamp_monotonic_ns": sample_ns,
    }


def load_sdk(sdk_root):
    root = Path(sdk_root).resolve()
    wrapper = root / "PYTHON_SDK" / "GentoRobot.py"
    if not wrapper.is_file():
        raise FileNotFoundError(f"00040701 Python SDK missing: {wrapper}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("PYTHON_SDK.GentoRobot")
    if platform.machine().lower() in ("aarch64", "arm64"):
        native = root / "dist_aarch64" / "libGentoSDKPY.so"
        if not native.is_file():
            raise FileNotFoundError(f"00040701 aarch64 library missing: {native}")
        module.current_path = str(native.parent)
    return module


def run(args):
    mdh_sha256 = mdh_provenance(args.mdh)
    latest = Path(args.latest) if args.latest else None
    if not args.pivot:
        # Do not change robot state unless the new camera contract and ball detection are live.
        print(f"Waiting up to {args.camera_wait_seconds:g}s for a fresh stereo yellow frame...", flush=True)
        wait_for_yellow(latest, args.camera_wait_seconds)
    sdk = load_sdk(args.sdk_root)
    robot = sdk.GentoRobot()
    arm = sdk.FXObjType.OBJ_ARM1
    stopped = False
    entered_drag = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path before changing robot state; calibration logs are never overwritten.
    with output.open("x", encoding="utf-8") as out:
        try:
            ret = robot.link(*map(int, args.robot_ip.split(".")), log_level=sdk.FXLogMask.FX_LOG_INFO_FLAG)
            if not robot._connected:
                raise RuntimeError(f"robot link failed: {robot._get_operate_error_msg(ret)}")
            sdk_version = str(robot.get_sdk_version())
            controller_version = str(robot.get_controller_version())
            if sdk_version != EXPECTED_VERSION or controller_version != EXPECTED_VERSION:
                raise RuntimeError(f"version mismatch: SDK={sdk_version}, controller={controller_version}")

            state = robot.current_state(arm)
            state_name = sdk.state_map.get(state, str(state))
            if state_name == "Error":
                ret, code = robot.reset_error(arm, 2000)
                if ret != 0:
                    raise RuntimeError(f"right-arm clear error failed: system_error={code}, ret={ret}")
                state_name = sdk.state_map.get(robot.current_state(arm), "Unknown")
            if state_name != "IDLE":
                ret = robot.switch_to_idle(arm, 2000)
                if ret != 0:
                    raise RuntimeError(f"right-arm switch to IDLE failed: {robot._get_operate_error_msg(ret)}")
            ret = robot.switch_to_drag_joint(arm, 2000, JOINT_K, JOINT_D)
            if ret != 0:
                raise RuntimeError(f"right-arm DragJoint failed: {robot._get_operate_error_msg(ret)}")
            entered_drag = True
            if sdk.state_map.get(robot.current_state(arm)) != "DragJoint":
                raise RuntimeError("right arm did not enter DragJoint")

            print("Right arm is in DragJoint. Hold the arm terminal drag button; Ctrl-C stops and returns to IDLE.", flush=True)
            seen = None
            saved = skipped = moving = 0
            last_report = 0.0
            q_history = deque()
            deadline = time.monotonic() + args.duration if args.duration else None
            while not stopped and (deadline is None or time.monotonic() < deadline):
                if sdk.state_map.get(robot.current_state(arm)) != "DragJoint":
                    raise RuntimeError("right arm left DragJoint unexpectedly")
                try:
                    frame = None if args.pivot else json.loads(latest.read_text(encoding="utf-8"))
                    before = time.monotonic_ns()
                    rt = robot.get_rt_dict()
                    after = time.monotonic_ns()
                    if "error" in rt:
                        raise ValueError(rt["error"])
                    sample_ns = (before + after) // 2
                    current_q = np.asarray(rt["arms"][RIGHT_ARM_INDEX]["fb"]["fb_pos"], float)
                    q_history.append((sample_ns, current_q))
                    history_ms = 1000. if args.capture_motion else args.stationary_window_ms
                    cutoff = sample_ns - int(history_ms * 1e6)
                    while q_history and q_history[0][0] < cutoff:
                        q_history.popleft()
                    if frame is not None:
                        identity = (frame.get("timestamp_monotonic_ns"), frame.get("frame_number"))
                        if identity == seen:
                            time.sleep(.005)
                            continue
                        seen = identity
                    paired_q = None
                    paired_ns = None
                    interpolation_span_ms = None
                    if not args.pivot and args.capture_motion:
                        _, camera_ns, _ = yellow_pixels(frame, sample_ns)
                        paired_ns = int(camera_ns - args.camera_delay_ms * 1e6)
                        paired_q, interpolation_span_ms = interpolate_q(q_history, paired_ns)
                    elif not args.pivot:
                        if not stationary_q(q_history, args.stationary_window_ms, args.stationary_range_deg):
                            moving += 1
                            time.sleep(.005)
                            continue
                    record = (make_pivot_record(rt, sample_ns, mdh_sha256) if args.pivot else
                              make_record(frame, rt, sample_ns,
                                          max_pair_ms=args.max_pair_ms,
                                          mdh_sha256=mdh_sha256,
                                          q_deg=paired_q,
                                          q_sample_ns=paired_ns,
                                          camera_delay_ms=args.camera_delay_ms if args.capture_motion else 0.,
                                          q_interpolation_span_ms=interpolation_span_ms))
                    if not args.pivot:
                        record["capture_policy"] = "continuous" if args.capture_motion else "stationary"
                    out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    out.flush()
                    saved += 1
                    now = time.monotonic()
                    if now - last_report >= .25:
                        print(f"\rCaptured {saved} samples; skipped {skipped}", end="", flush=True)
                        last_report = now
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    skipped += 1
                    if skipped == 1 or skipped % 30 == 0:
                        print(f"Skipped {skipped}: {exc}", flush=True)
                time.sleep(.005)
            if saved:
                print()
            print(f"Saved {saved} samples to {output}; skipped {skipped}, moving {moving} frames.", flush=True)
        finally:
            if entered_drag and robot._connected:
                ret = robot.switch_to_idle(arm, 2000)
                if ret != 0:
                    print(f"WARNING: failed to return right arm to IDLE: {ret}", file=sys.stderr)
            if robot._connected:
                robot.unlink()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Required: clear faults and enter right-arm DragJoint")
    parser.add_argument("--robot-ip", default="6.6.7.190")
    parser.add_argument("--sdk-root", default=str(root / "00040701"))
    parser.add_argument("--mdh", default=str(root / "head_2dof_capt" / "configs" / "luna_new.ini"))
    parser.add_argument("--latest",
                        help="PASS sidecar produced by Orin 1280x992 formal measurement")
    parser.add_argument("--pivot", action="store_true",
                        help="Record q7 for fixed-ball pivot calibration; camera is not required")
    parser.add_argument("--output", required=True, help="New JSONL path; existing files are never overwritten")
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 runs until Ctrl-C")
    parser.add_argument("--max-pair-ms", type=float, default=100.0)
    parser.add_argument("--camera-wait-seconds", type=float, default=45.0)
    parser.add_argument("--capture-motion", action="store_true",
                        help="Save every new stereo frame while dragging instead of only stationary poses")
    parser.add_argument("--camera-delay-ms", type=float, default=0.0,
                        help="Image pipeline delay compensated by interpolating earlier q7 feedback")
    parser.add_argument("--stationary-window-ms", type=float,
                        default=CAMERA_CONTRACT["stationary_window_ms"])
    parser.add_argument("--stationary-range-deg", type=float,
                        default=CAMERA_CONTRACT["max_stationary_q_range_deg"])
    args = parser.parse_args()
    if not args.start:
        parser.error("--start is required because this command clears faults and changes right-arm mode")
    if not args.pivot and not args.latest:
        parser.error("--latest is required unless --pivot is used")
    if (args.duration < 0 or not 0 < args.max_pair_ms <= 200 or args.camera_wait_seconds <= 0 or
            not 0 <= args.camera_delay_ms <= 500):
        parser.error("invalid duration, pairing limit, camera wait, or camera delay")
    if args.camera_delay_ms and not args.capture_motion:
        parser.error("camera-delay-ms requires --capture-motion")
    if args.stationary_window_ms < 200 or not 0 < args.stationary_range_deg <= 2:
        parser.error("stationary-window-ms must be >=200 and stationary-range-deg must be in (0, 2]")
    parts = args.robot_ip.split(".")
    if len(parts) != 4 or any(not p.isdigit() or not 0 <= int(p) <= 255 for p in parts):
        parser.error("robot-ip must be an IPv4 address")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
