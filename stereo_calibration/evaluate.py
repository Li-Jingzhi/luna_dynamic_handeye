#!/usr/bin/env python3
"""Evaluate a frozen calibration on separately captured raw board pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from calibrate import detect


def board_metrics(calibration, left, right, pattern, square_m):
    c = calibration
    rectified = []
    for eye, index, points in (("left", 1, left), ("right", 2, right)):
        rectified.append(cv2.fisheye.undistortPoints(
            points, np.array(c[f"{eye}_K"]), np.array(c[f"{eye}_D"]),
            R=np.array(c[f"rectify_R{index}"]), P=np.array(c[f"rectify_P{index}"])).reshape(-1, 2))
    l, r = rectified
    homogeneous = cv2.triangulatePoints(np.array(c["rectify_P1"]), np.array(c["rectify_P2"]), l.T, r.T)
    if np.any(np.abs(homogeneous[3]) < 1e-12):
        raise ValueError("degenerate triangulation")
    xyz = (homogeneous[:3] / homogeneous[3]).T
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError("nonfinite or negative-depth board")
    board = xyz.reshape(pattern[1], pattern[0], 3)
    lengths = np.r_[np.linalg.norm(np.diff(board, axis=0), axis=2).ravel(),
                    np.linalg.norm(np.diff(board, axis=1), axis=2).ravel()]
    grid = np.zeros_like(xyz)
    grid[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2) * square_m
    x, y = grid - grid.mean(0), xyz - xyz.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3); correction[-1, -1] = np.linalg.det(u @ vt)
    residual = x @ (u @ correction @ vt) - y  # rigid alignment; no fitted scale
    return {"vertical_px": np.abs(l[:, 1] - r[:, 1]),
            "edge_error_mm": np.abs(lengths - square_m) * 1000,
            "rigid_board_error_mm": np.linalg.norm(residual, axis=1) * 1000}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", required=True)
    p.add_argument("--images", required=True, help="independent raw board capture directory")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = json.loads(Path(a.calibration).read_text())
    root = Path(a.images)
    manifest = json.loads((root / "capture_manifest.json").read_text())
    if c.get("distortion_model") != "opencv_fisheye" or manifest.get("pixel_domain") != "raw":
        raise ValueError("this evaluator requires raw fisheye images/calibration")
    for key in ("sensor_ids", "board_inner_corners", "square_size_m"):
        if manifest.get(key) != c[key]:
            raise ValueError(f"manifest mismatch: {key}")
    hashes = set(c.get("training_image_sha256", {}).values())
    if not hashes:
        raise ValueError("calibration lacks training hashes; rerun local calibrate.py")
    values = {}; rejected = []; used = []
    for left in sorted(root.glob("*_left.png")):
        right = left.with_name(left.name.replace("_left.png", "_right.png"))
        if not right.is_file():
            rejected.append({"pair": left.name, "reason": "missing right image"}); continue
        for path in (left, right):
            if hashlib.sha256(path.read_bytes()).hexdigest() in hashes:
                raise ValueError(f"training image reused in evaluation: {path}")
        sl, l = detect(left, tuple(c["board_inner_corners"]))
        sr, r = detect(right, tuple(c["board_inner_corners"]))
        if list(sl) != c["image_size"] or sl != sr:
            raise ValueError("image size mismatch")
        if l is None or r is None:
            rejected.append({"pair": left.name, "reason": "corners missing"}); continue
        metrics = board_metrics(c, l, r, c["board_inner_corners"], c["square_size_m"])
        for key, value in metrics.items(): values.setdefault(key, []).extend(value.tolist())
        used.append(left.name)
    if not used: raise ValueError("no valid evaluation pairs")
    result = {"approved": False, "valid_pairs": len(used), "pair_files": used,
              "rejected_pairs": rejected,
              "calibration_sha256": hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),
              "note": "Board shape/scale diagnostics only; not torso/FK accuracy certification.",
              "metrics": {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                              "max": float(np.max(v))} for k, v in values.items()}}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__": main()
