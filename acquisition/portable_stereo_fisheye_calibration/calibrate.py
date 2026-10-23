#!/usr/bin/env python3
"""Calibrate fisheye intrinsics, distortion, stereo extrinsics and rectification."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect(path, pattern):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    ok, points = cv2.findChessboardCornersSB(image, pattern)
    return image.shape[::-1], points.astype(np.float64).reshape(1, -1, 2) if ok else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory containing NNNN_left/right.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--corners", nargs=2, type=int, default=(11, 8), metavar=("COLS", "ROWS"))
    parser.add_argument("--square-mm", type=float, default=35)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--balance", type=float, default=0.0, help="rectification FOV balance, 0..1")
    args = parser.parse_args()
    pattern = tuple(args.corners)
    if min(pattern) < 3 or args.square_mm <= 0 or args.min_pairs < 10 or not 0 <= args.balance <= 1:
        parser.error("invalid board/min-pairs/balance setting")
    directory = Path(args.images)
    manifest = json.loads((directory / "capture_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pixel_domain") != "raw" or manifest.get("board_inner_corners") != list(pattern):
        raise ValueError("manifest pixel domain or board geometry does not match CLI arguments")
    if abs(float(manifest["square_size_m"]) - args.square_mm / 1000) > 1e-9:
        raise ValueError("manifest square size does not match --square-mm")
    grid = np.zeros((1, pattern[0] * pattern[1], 3), np.float64)
    grid[0, :, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2) * args.square_mm / 1000
    objects, left, right, used, rejected = [], [], [], [], []
    image_size = None
    for left_path in sorted(directory.glob("*_left.png")):
        right_path = left_path.with_name(left_path.name.replace("_left.png", "_right.png"))
        if not right_path.is_file():
            rejected.append(f"{left_path.name}: missing right image")
            continue
        size_l, points_l = detect(left_path, pattern)
        size_r, points_r = detect(right_path, pattern)
        if size_l != size_r or (image_size and size_l != image_size):
            raise ValueError(f"image size mismatch at {left_path.name}")
        image_size = size_l
        if points_l is None or points_r is None:
            rejected.append(f"{left_path.name}: corners missing")
            continue
        objects.append(grid.copy()); left.append(points_l); right.append(points_r); used.append(left_path.name)
    if len(used) < args.min_pairs:
        raise ValueError(f"only {len(used)} valid pairs; need {args.min_pairs}")
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-8)
    rms_l, K_l, D_l, _, _ = cv2.fisheye.calibrate(objects, left, image_size, None, None, flags=flags, criteria=criteria)
    rms_r, K_r, D_r, _, _ = cv2.fisheye.calibrate(objects, right, image_size, None, None, flags=flags, criteria=criteria)
    rms_s, K_l, D_l, K_r, D_r, R, T = cv2.fisheye.stereoCalibrate(
        objects, left, right, K_l, D_l, K_r, D_r, image_size,
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC, criteria=criteria)
    baseline = float(np.linalg.norm(T))
    if not .005 < baseline < 1:
        raise ValueError(f"implausible stereo baseline: {baseline:.4f} m")
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=args.balance)
    vertical = []
    for l, r in zip(left, right):
        ul = cv2.fisheye.undistortPoints(l, K_l, D_l, R=R1, P=P1).reshape(-1, 2)
        ur = cv2.fisheye.undistortPoints(r, K_r, D_r, R=R2, P=P2).reshape(-1, 2)
        vertical.extend(np.abs(ul[:, 1] - ur[:, 1]))
    result = {
        "schema": "portable_raw_fisheye_stereo_v1", "pixel_domain": "raw",
        "distortion_model": "opencv_fisheye", "image_size": list(image_size),
        "sensor_ids": manifest.get("sensor_ids", ["left", "right"]),
        "board_inner_corners": list(pattern), "square_size_m": args.square_mm / 1000,
        "valid_pairs": len(used), "rejected_pairs": rejected, "pair_files": used,
        "left_K": K_l.tolist(), "left_D": D_l.ravel().tolist(),
        "right_K": K_r.tolist(), "right_D": D_r.ravel().tolist(),
        "right_from_left_R": R.tolist(), "right_from_left_T_m": T.ravel().tolist(),
        "rectify_R1": R1.tolist(), "rectify_R2": R2.tolist(),
        "rectify_P1": P1.tolist(), "rectify_P2": P2.tolist(), "rectify_Q": Q.tolist(),
        "left_rms_px": float(rms_l), "right_rms_px": float(rms_r),
        "stereo_rms_px": float(rms_s),
        "rectified_vertical_p95_px": float(np.percentile(vertical, 95)), "baseline_m": baseline,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(f"Saved {output}; valid={len(used)}, stereo RMS={rms_s:.3f}px, "
          f"vertical P95={result['rectified_vertical_p95_px']:.3f}px, baseline={baseline*1000:.2f}mm")


if __name__ == "__main__":
    main()
