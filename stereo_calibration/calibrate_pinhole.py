"""Calibrate the new SH5 stereo pair in its undistorted ROS2 pixel domain."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def corners(path, pattern):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, points = cv2.findChessboardCornersSB(image, pattern)
    else:
        ok, points = cv2.findChessboardCorners(image, pattern)
        if ok:
            points = cv2.cornerSubPix(image, points, (7, 7), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, .001))
    return image.shape[::-1], points.astype(np.float32) if ok else None


def fit(directory, pattern=(9, 6), square_mm=20):
    if min(pattern) < 3 or square_mm <= 0:
        raise ValueError("Need at least 3x3 inner corners and a positive square size")
    manifest_path = Path(directory) / "capture_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Missing undistorted stereo capture manifest; raw NV12 images cannot be used")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("pixel_domain") != "ideal_pinhole" or
            manifest.get("image_size") != [2560, 1984] or
            manifest.get("sensor_ids") != ["head_left:csi://0", "head_right:csi://1"] or
            manifest.get("source_topics") != ["/camera/left_eye/image_h264_undistorted",
                                              "/camera/right_eye/image_h264_undistorted"]):
        raise ValueError("Chessboard images are not from the new SH5 undistorted stereo topics")
    if manifest.get("board_inner_corners") != list(pattern) or abs(float(manifest.get("square_size_m", -1)) - square_mm / 1000) > 1e-9:
        raise ValueError("Manifest board dimensions do not match CLI")
    paths = sorted(Path(directory).glob("*_left.png"))
    if len(paths) < 20:
        raise ValueError("Capture at least 20 stereo board pairs at varied poses")
    grid = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    grid[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2) * (square_mm / 1000)
    objects, left, right, used = [], [], [], []
    image_size = None
    for path in paths:
        opposite = path.with_name(path.name.replace("_left.png", "_right.png"))
        if not opposite.is_file():
            raise ValueError(f"Missing right image for {path}")
        size_l, points_l = corners(path, pattern)
        size_r, points_r = corners(opposite, pattern)
        if size_l != size_r or (image_size is not None and size_l != image_size):
            raise ValueError(f"Image size mismatch at {path}")
        image_size = size_l
        if points_l is None or points_r is None:
            print(f"Rejected pair (corners missing): {path.name}")
            continue
        objects.append(grid.copy())
        left.append(points_l)
        right.append(points_r)
        used.append(path.name)
    if len(used) < 20:
        raise ValueError(f"Only {len(used)} valid pairs; need >=20")
    if image_size != (2560, 1984):
        raise ValueError(f"Expected native undistorted 2560x1984 pixels, got {image_size}")
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
             cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(objects, left, image_size, None, None, flags=flags)
    rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(objects, right, image_size, None, None, flags=flags)
    rms_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objects, left, right, K_l, D_l, K_r, D_r, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC)
    if np.linalg.norm(T) < .005 or np.linalg.norm(T) > 1:
        raise ValueError(f"Implausible stereo baseline {np.linalg.norm(T):.4f} m")
    epipolar = []
    for l, r in zip(left, right):
        lines = cv2.computeCorrespondEpilines(l, 1, F).reshape(-1, 3)
        points = r.reshape(-1, 2)
        epipolar.extend(np.abs(lines[:, 0] * points[:, 0] + lines[:, 1] * points[:, 1] + lines[:, 2]) /
                        np.linalg.norm(lines[:, :2], axis=1))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, image_size, R, T)
    return {
        "schema_version": 1, "pixel_domain": "ideal_pinhole", "image_size": list(image_size),
        "sensor_ids": ["head_left:csi://0", "head_right:csi://1"],
        "board_inner_corners": list(pattern), "square_size_m": square_mm / 1000,
        "valid_pairs": len(used), "pair_files": used,
        "left_K": K_l.tolist(), "right_K": K_r.tolist(),
        "left_D": D_l.ravel().tolist(), "right_D": D_r.ravel().tolist(),
        "right_from_left_R": R.tolist(), "right_from_left_T_m": T.ravel().tolist(),
        "essential_E": E.tolist(), "fundamental_F": F.tolist(),
        "rectify_R1": R1.tolist(), "rectify_R2": R2.tolist(),
        "rectify_P1": P1.tolist(), "rectify_P2": P2.tolist(), "rectify_Q": Q.tolist(),
        "left_rms_px": float(rms_l), "right_rms_px": float(rms_r), "stereo_rms_px": float(rms_s),
        "epipolar_p95_px": float(np.percentile(epipolar, 95)),
        "baseline_m": float(np.linalg.norm(T)),
        "timing_note": "Pairs captured while board stationary; DDS arrival times are not exposure synchronization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corners", nargs=2, type=int, default=(9, 6), metavar=("X", "Y"))
    parser.add_argument("--square-mm", type=float, default=20)
    args = parser.parse_args()
    result = fit(args.images, tuple(args.corners), args.square_mm)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Saved {args.output}; stereo RMS {result['stereo_rms_px']:.2f} px, "
          f"epipolar p95 {result['epipolar_p95_px']:.2f} px")


if __name__ == "__main__":
    main()
