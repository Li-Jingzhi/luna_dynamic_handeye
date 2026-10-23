#!/usr/bin/env python3
"""Automatically save synchronized raw NV12 stereo chessboard pairs on a ROS2 host."""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


def gray(message):
    if message.encoding.lower() != "nv12" or message.step < message.width:
        raise ValueError("expected NV12 with a valid stride")
    data = np.frombuffer(message.data, np.uint8)
    return data[:message.height * message.step].reshape(message.height, message.step)[:, :message.width].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--corners", nargs=2, type=int, default=(11, 8), metavar=("COLS", "ROWS"))
    p.add_argument("--square-mm", type=float, default=35)
    p.add_argument("--left-topic", default="/camera/left_eye/image_nv12")
    p.add_argument("--right-topic", default="/camera/right_eye/image_nv12")
    p.add_argument("--seconds", type=float, default=1800)
    p.add_argument("--max-skew-ms", type=float, default=20)
    p.add_argument("--detection-scale", type=float, default=.25)
    p.add_argument("--stable-seconds", type=float, default=.15)
    p.add_argument("--min-pose-change-px", type=float, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()
    pattern = tuple(args.corners)
    if min(pattern) < 3 or args.square_mm <= 0 or not 0 < args.detection_scale <= 1:
        p.error("invalid board or detection scale")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    output = Path(args.output); output.mkdir(parents=True, exist_ok=args.resume)
    manifest = {"pixel_domain": "raw", "distortion_model": "opencv_fisheye",
                "sensor_ids": ["left", "right"], "source_topics": [args.left_topic, args.right_topic],
                "board_inner_corners": list(pattern), "square_size_m": args.square_mm / 1000}
    manifest_path = output / "capture_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing manifest uses different camera/board parameters")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rclpy.init(); node = rclpy.create_node("portable_stereo_chessboard_capture")
    latest = [None, None]
    def receive(message, eye): latest[eye] = message
    subscriptions = [node.create_subscription(Image, topic, lambda m, i=i: receive(m, i), qos_profile_sensor_data)
                     for i, topic in enumerate((args.left_topic, args.right_topic))]
    count = len(list(output.glob("*_left.png"))); previous = None; stable_since = None; saved = []
    last_stamps = None; last_save = 0.; started = time.monotonic()
    if args.preview:
        cv2.namedWindow("RAW stereo chessboard - q finishes", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RAW stereo chessboard - q finishes", 1280, 500)
    try:
        while rclpy.ok() and time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=.03)
            if any(x is None for x in latest): continue
            stamps = [x.header.stamp.sec * 10**9 + x.header.stamp.nanosec for x in latest]
            if stamps == last_stamps or min(stamps) <= 0 or abs(stamps[0]-stamps[1])/1e6 > args.max_skew_ms:
                continue
            images = [gray(x) for x in latest]
            if images[0].shape != images[1].shape: raise ValueError("left/right image sizes differ")
            found = [cv2.findChessboardCornersSB(cv2.resize(x, None, fx=args.detection_scale,
                                                            fy=args.detection_scale), pattern) for x in images]
            found = [(ok, pts / args.detection_scale if ok else None) for ok, pts in found]
            ready = all(ok for ok, _ in found); status = "SHOW COMPLETE BOARD IN BOTH EYES"
            if ready:
                points = np.concatenate([pts.reshape(-1, 2) for _, pts in found])
                drift = np.inf if previous is None else np.linalg.norm(points-previous, axis=1).mean()
                previous = points.copy()
                stable_since = (stable_since or time.monotonic()) if drift < 20 else None
                distinct = not saved or np.linalg.norm(points-saved[-1], axis=1).mean() > args.min_pose_change_px
                ready = distinct and stable_since is not None and time.monotonic()-stable_since >= args.stable_seconds
                ready &= time.monotonic()-last_save >= .5
                status = "HOLD STILL" if distinct else "SAVED - MOVE BOARD"
            else:
                previous = None; stable_since = None
            if args.preview:
                views = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in images]
                for view, (ok, points) in zip(views, found):
                    if ok: cv2.drawChessboardCorners(view, pattern, points, ok)
                    cv2.putText(view, f"SAVED {count} | {pattern[0]}x{pattern[1]} | {args.square_mm:g}mm",
                                (25, 50), cv2.FONT_HERSHEY_SIMPLEX, .9, (0,255,255), 2)
                    cv2.putText(view, status, (25, 95), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
                cv2.imshow("RAW stereo chessboard - q finishes",
                           np.hstack([cv2.resize(x, (640,496)) for x in views]))
                if cv2.waitKey(1) & 255 == ord("q"): break
            if not ready: continue
            saved.append(points.copy()); count += 1
            for eye, image in zip(("left", "right"), images):
                if not cv2.imwrite(str(output / f"{count:04d}_{eye}.png"), image):
                    raise OSError("failed to save image")
            (output / f"{count:04d}.json").write_text(json.dumps({
                "stamp_ns": stamps, "pair_skew_ms": abs(stamps[0]-stamps[1])/1e6,
                "image_size": [images[0].shape[1], images[0].shape[0]],
                "board_inner_corners": list(pattern), "square_size_m": args.square_mm/1000}, indent=2)+"\n")
            last_stamps = stamps; last_save = time.monotonic()
            print(f"Saved pair {count}; skew={abs(stamps[0]-stamps[1])/1e6:.2f}ms", flush=True)
    finally:
        cv2.destroyAllWindows(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__": main()
