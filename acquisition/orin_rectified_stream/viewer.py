#!/usr/bin/env python3
"""Receive Orin measurements and draw them over independent 640x496 previews."""
import argparse
from collections import deque
import json
from pathlib import Path
import socket
import sys
import threading
import time

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capt_head.measurement_contract import DEFAULT_CONTRACT, load_contract, valid_measurement


def measurement_record(observation, config, receive_ns, drop_count=0):
    """Convert Orin processing latency into this host's estimated capture-dequeue time."""
    valid_measurement(observation, config)
    processing_ms = float(observation["processing_latency_ms"])
    if not np.isfinite(processing_ms) or not 0 <= processing_ms <= 1000:
        raise ValueError("invalid Orin processing latency")
    capture_ns = int(receive_ns - processing_ms * 1e6)
    return {
        "schema": "luna_yellow_measurement_v1",
        "connected": True, "detector_version": 4,
        "timestamp_monotonic_ns": capture_ns,
        "camera_capture_timestamp_monotonic_ns": capture_ns,
        "metadata_received_timestamp_monotonic_ns": int(receive_ns),
        "timestamp_basis": "local_receive_minus_orin_processing_not_exposure",
        "network_latency_uncompensated": True,
        "source_timestamp_ns": observation.get("source_timestamp_ns"),
        "source_clock_ns": observation["source_clock_ns"],
        "pair_source_clock_skew_ms": observation["pair_source_clock_skew_ms"],
        "pairing_policy": observation["pairing_policy"],
        "pairing_stats": observation["pairing_stats"],
        "epipolar_check_domain": observation["epipolar_check_domain"],
        "corrected_stereo_pixels": observation["corrected_stereo_pixels"],
        "raw_epipolar_dv_px": observation["raw_epipolar_dv_px"],
        "raw_disparity_px": observation["raw_disparity_px"],
        "capture_dequeue_timestamp_ns": observation.get("capture_dequeue_timestamp_ns"),
        "pair_capture_skew_ms": observation["pair_capture_skew_ms"],
        "pair_dequeue_skew_ms": observation["pair_capture_skew_ms"],
        "exposure_time_valid": False, "image_size": config["image_size"],
        "preview_size": observation.get("preview_size"),
        "sensor_ids": observation["sensor_ids"],
        "pixel_domain": observation["pixel_domain"],
        "measurement_backend": observation.get("measurement_backend"),
        "source_calibration_sha256": observation.get("source_calibration_sha256"),
        "contract_sha256": observation["contract_sha256"],
        "quality": observation["quality"],
        "measurement_fps": observation["measurement_fps"],
        "detection_fps": observation["measurement_fps"],
        "exposure_locked": observation["exposure_locked"],
        "awb_locked": observation["awb_locked"],
        "brightness_stable": observation["brightness_stable"],
        "antibanding_hz": observation["antibanding_hz"],
        "camera_controls": observation.get("camera_controls"),
        "camera_controls_readback": observation.get("camera_controls_readback"),
        "mean_luma": observation.get("mean_luma"),
        "image_evidence": observation.get("image_evidence"),
        "left": observation["left"], "right": observation["right"],
        "disparity_px": observation["disparity_px"],
        "epipolar_dv_px": observation["epipolar_dv_px"],
        "frame_number": observation["frame_number"],
        "measurement_drop_count": drop_count,
        "camera_pipeline_delay_ms": processing_ms,
        "four_ball_pose": observation.get("four_ball_pose"),
        "image_observations": {
            "left": {"yellow": [{"pixel": observation["left"]["center"]}]},
            "right": {"yellow": [{"pixel": observation["right"]["center"]}]},
        },
    }


def pipeline(port):
    return (
        f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,"
        "payload=96,clock-rate=90000 ! rtpjitterbuffer latency=20 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! openh264dec ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--metadata-port", type=int, default=5004)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--manual-preview", action="store_true", help="do not publish calibration samples or show FK comparison")
    parser.add_argument("--latest-only", action="store_true",
                        help="draw detections from --latest without binding the metadata UDP port")
    parser.add_argument("--snapshot-dir",
                        help="optionally save the latest decoded left/right preview frames")
    args = parser.parse_args()
    config, _ = load_contract(args.contract)
    expected_size = tuple(config["image_size"])
    latest = Path(args.latest) if args.latest and not args.manual_preview else None
    overlay = Path("runtime/live_base_comparison.json")
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_at = 0.
    if latest:
        latest.parent.mkdir(parents=True, exist_ok=True)
    running = True
    latest_measurement = [None]
    latest_status = [None]
    measurement_count = [0]
    measurement_drop_count = [0]
    previous_frame_number = [None]
    metadata = None
    if not args.latest_only:
        metadata = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        metadata.bind(("0.0.0.0", args.metadata_port))
        metadata.settimeout(.5)

    def receive_metadata():
        while running:
            try:
                packet, _ = metadata.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                observation = json.loads(packet)
            except json.JSONDecodeError:
                continue
            if not isinstance(observation, dict):
                continue
            latest_status[0] = observation
            if latest:
                status_path = latest.with_name("latest_status.json")
                status_temp = status_path.with_suffix(".json.tmp")
                status_temp.write_text(json.dumps(observation), encoding="utf-8")
                status_temp.replace(status_path)
            try:
                valid_measurement(observation, config)
            except ValueError:
                continue
            try:
                frame_number = observation["frame_number"]
                record = measurement_record(observation, config, time.monotonic_ns(),
                                            measurement_drop_count[0])
            except (KeyError, TypeError, ValueError):
                continue
            if previous_frame_number[0] is not None:
                measurement_drop_count[0] += max(
                    max(current - previous - 1, 0)
                    for current, previous in zip(frame_number, previous_frame_number[0]))
                record["measurement_drop_count"] = measurement_drop_count[0]
            previous_frame_number[0] = frame_number
            if latest:
                temporary = latest.with_suffix(latest.suffix + ".tmp")
                temporary.write_text(json.dumps(record), encoding="utf-8")
                temporary.replace(latest)
            latest_measurement[0] = record
            measurement_count[0] += 1

    if metadata:
        metadata_thread = threading.Thread(target=receive_metadata, daemon=True)
        metadata_thread.start()
    if args.metadata_only:
        previous_count = 0
        try:
            while True:
                time.sleep(5)
                record = latest_measurement[0]
                age_ms = ((time.monotonic_ns() - record["timestamp_monotonic_ns"]) / 1e6
                          if record else float("inf"))
                print(f"measurement={(measurement_count[0]-previous_count)/5:.1f} FPS; "
                      f"latest age={age_ms:.0f} ms; "
                      f"dropped={measurement_drop_count[0]}; "
                      f"yellow={'1/1' if record else '0/0'}", flush=True)
                previous_count = measurement_count[0]
        except KeyboardInterrupt:
            pass
        finally:
            running = False
            metadata.close()
        return
    captures = [cv2.VideoCapture(pipeline(port), cv2.CAP_GSTREAMER) for port in (5000, 5002)]
    if not all(capture.isOpened() for capture in captures):
        running = False
        metadata.close()
        raise RuntimeError("Cannot open both rectified RTP streams")
    queues = [deque(maxlen=4), deque(maxlen=4)]
    serials = [0, 0]
    lock = threading.Lock()

    def receive(index):
        nonlocal running
        while running:
            ok, frame = captures[index].read()
            if not ok:
                time.sleep(.01)
                continue
            with lock:
                serials[index] += 1
                queues[index].append((time.monotonic_ns(), serials[index], frame))

    threads = [threading.Thread(target=receive, args=(i,), daemon=True) for i in range(2)]
    for thread in threads:
        thread.start()
    report_at = time.monotonic()
    last_pair_at = report_at
    previous_measurement_count = 0
    previous_video_count = 0
    video_count = 0
    cv2.namedWindow("Luna Orin measurement preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Luna Orin measurement preview", 1280, 496)
    try:
        while True:
            with lock:
                if not queues[0] or not queues[1]:
                    pair = None
                else:
                    left, right = min(((left, right) for left in queues[0] for right in queues[1]),
                                      key=lambda items: abs(items[0][0] - items[1][0]))
                    pair = [left[2], right[2]]
                    pair_stamps = [left[0], right[0]]
                    pair_serials = [left[1], right[1]]
                    for queue, selected in zip(queues, (left, right)):
                        while queue and queue[0][1] <= selected[1]:
                            queue.popleft()
            if pair is None:
                if args.manual_preview and time.monotonic()-last_pair_at > 1.:
                    waiting = np.zeros((config['preview_size'][1], 2*config['preview_size'][0], 3), dtype=np.uint8)
                    cv2.putText(waiting, "WAITING FOR BOTH CAMERAS - restart may take 40-60s", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 1)
                    cv2.imshow("Luna Orin measurement preview", waiting)
                if cv2.waitKey(1) & 255 == ord("q"):
                    break
                time.sleep(.002)
                continue
            last_pair_at = time.monotonic()
            actual_size = (pair[0].shape[1], pair[0].shape[0])
            preview_size = tuple(config["preview_size"])
            if actual_size != preview_size or pair[1].shape[:2] != pair[0].shape[:2]:
                raise ValueError(f"Expected preview {preview_size}, received {actual_size}")
            observation = latest_measurement[0]
            if args.latest_only and latest:
                try:
                    observation = json.loads(latest.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    observation = None
            centers = [None, None]
            if observation and time.monotonic_ns()-observation["metadata_received_timestamp_monotonic_ns"] >= 500_000_000:
                observation = None
            if observation:
                full = [observation["image_observations"][eye]["yellow"][0]["pixel"]
                        for eye in ("left", "right")]
                centers = [np.asarray(point) * np.asarray(preview_size) / np.asarray(expected_size)
                           for point in full]
                for eye, frame, center, name in zip(("L", "R"), pair, centers, ("left", "right")):
                    x, y = np.rint(center).astype(int)
                    ellipse = observation[name]["ellipse"]
                    axes = (int(ellipse["major_px"]*preview_size[0]/expected_size[0]/2),
                            int(ellipse["minor_px"]*preview_size[1]/expected_size[1]/2))
                    cv2.ellipse(frame, (x, y), axes, ellipse["angle_deg"], 0, 360, (0, 255, 0), 1)
                    cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
                    cv2.putText(frame, f"{eye} ({full[0 if eye == 'L' else 1][0]:.3f}, {full[0 if eye == 'L' else 1][1]:.3f})",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
            skew_ms = abs(pair_stamps[0] - pair_stamps[1]) / 1e6
            video_count += 1
            now = time.monotonic()
            if snapshot_dir and now-snapshot_at >= 1.:
                cv2.imwrite(str(snapshot_dir / "left.jpg"), pair[0])
                cv2.imwrite(str(snapshot_dir / "right.jpg"), pair[1])
                snapshot_at = now
            if now - report_at >= 5:
                elapsed = now - report_at
                print(f"preview={((video_count-previous_video_count)/elapsed):.1f} FPS; "
                      f"metadata_packets={((measurement_count[0]-previous_measurement_count)/elapsed):.1f}/s; "
                      f"Orin_measurement={observation['measurement_fps']:.1f} FPS "
                      f"RTP pair skew={skew_ms:.1f} ms" if observation else
                      f"preview={((video_count-previous_video_count)/elapsed):.1f} FPS; "
                      f"metadata_packets={((measurement_count[0]-previous_measurement_count)/elapsed):.1f}/s; "
                      f"RTP pair skew={skew_ms:.1f} ms", flush=True)
                previous_video_count = video_count
                previous_measurement_count = measurement_count[0]
                report_at = now
            display = np.hstack(pair)
            status = latest_status[0]
            title = (f"MEASURE {expected_size[0]}x{expected_size[1]}  PREVIEW {preview_size[0]}x{preview_size[1]}  "
                     f"quality={status.get('quality', 'WAIT') if status else 'WAIT'}"
                     if not observation else
                     f"d={observation['disparity_px']:.3f}px dv={observation['epipolar_dv_px']:.3f}px "
                     f"FPS={observation['measurement_fps']:.1f} {observation['quality']} AE/AWB=LOCK 50Hz")
            if observation and isinstance(observation.get("four_ball_pose"), dict):
                pose = observation["four_ball_pose"]
                if pose.get("valid") and pose.get("fresh", True):
                    pose_text = (f"POSE9 VALID score={pose.get('score', float('nan')):.3f} "
                                 f"age={pose.get('age_ms', float('nan')):.0f}ms")
                    pose_color = (0, 255, 0)
                else:
                    counts = pose.get("candidate_counts", {})
                    color_counts = " ".join(
                        f"{color}:{'/'.join(str(v) for v in counts.get(color, ['?', '?']))}"
                        for color in ("red", "blue", "green"))
                    pose_text = (f"POSE9 INVALID {pose.get('reason', 'stale_or_missing')} "
                                 f"{color_counts}")
                    pose_color = (0, 165, 255)
                cv2.putText(display, pose_text, (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, pose_color, 1)

            cv2.putText(display, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 1)
            try:
                if args.manual_preview:
                    raise OSError("manual preview has no synchronized FK comparison")
                value = json.loads(overlay.read_text(encoding="utf-8"))
                if value.get("valid"):
                    lines = (f"Vision XYZ m: {np.round(value['vision_xyz_m'], 4)}",
                             f"FK XYZ m: {np.round(value['fk_xyz_m'], 4)}",
                             f"Error mm: {np.round(value['error_xyz_mm'], 2)}  norm={value['error_norm_mm']:.2f}")
                else:
                    lines = (f"COMPARISON INVALID: {value.get('reason', 'waiting')}",)
                for row, line in enumerate(lines, 1):
                    cv2.putText(display, line, (20, 38 + 34 * row),
                                cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
            cv2.imshow("Luna Orin measurement preview", display)
            if cv2.waitKey(1) & 255 == ord("q"):
                break
    finally:
        running = False
        for capture in captures:
            capture.release()
        if metadata:
            metadata.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
