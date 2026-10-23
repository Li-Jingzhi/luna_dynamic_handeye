"""Fit native SH5 fisheye intrinsics, distortion and stereo extrinsics from raw pairs."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .fit import corners


def fit(directory, pattern=(11, 8), square_mm=35):
    directory = Path(directory)
    manifest = json.loads((directory / 'capture_manifest.json').read_text())
    if manifest.get('pixel_domain') != 'raw' or manifest.get('board_inner_corners') != list(pattern):
        raise ValueError('Expected raw images and matching board geometry')
    if square_mm <= 0 or min(pattern) < 3:
        raise ValueError('Invalid board geometry')
    grid = np.zeros((1, pattern[0]*pattern[1], 3), np.float64)
    grid[0, :, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2)*square_mm/1000
    objects, left, right, used = [], [], [], []
    size = None
    for path in sorted(directory.glob('*_left.png')):
        sl, pl = corners(path, pattern)
        sr, pr = corners(path.with_name(path.name.replace('_left.png', '_right.png')), pattern)
        if sl != sr or (size is not None and size != sl):
            raise ValueError('Image sizes differ')
        size = sl
        if pl is None or pr is None:
            continue
        objects.append(grid.copy())
        left.append(pl.reshape(1, -1, 2).astype(np.float64))
        right.append(pr.reshape(1, -1, 2).astype(np.float64))
        used.append(path.name)
    if len(used) < 20:
        raise ValueError(f'Only {len(used)} corner-complete pairs; need at least 20 varied poses')
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-8)
    rl, kl, dl, _, _ = cv2.fisheye.calibrate(objects, left, size, None, None, flags=flags, criteria=crit)
    rr, kr, dr, _, _ = cv2.fisheye.calibrate(objects, right, size, None, None, flags=flags, criteria=crit)
    rms, kl, dl, kr, dr, R, T = cv2.fisheye.stereoCalibrate(
        objects, left, right, kl, dl, kr, dr, size,
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC, criteria=crit)
    if not .005 < np.linalg.norm(T) < 1:
        raise ValueError('Implausible stereo baseline')
    r1, r2, p1, p2, q = cv2.fisheye.stereoRectify(kl, dl, kr, dr, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0)
    vertical = []
    for l, r in zip(left, right):
        a = cv2.fisheye.undistortPoints(l, kl, dl, R=r1, P=p1).reshape(-1, 2)
        b = cv2.fisheye.undistortPoints(r, kr, dr, R=r2, P=p2).reshape(-1, 2)
        vertical.extend(abs(a[:, 1]-b[:, 1]))
    return dict(schema='luna_raw_fisheye_stereo_v1', pixel_domain='raw',
        distortion_model='opencv_fisheye', image_size=list(size),
        sensor_ids=manifest['sensor_ids'], board_inner_corners=list(pattern),
        square_size_m=square_mm/1000, valid_pairs=len(used), pair_files=used,
        left_K=kl.tolist(), right_K=kr.tolist(), left_D=dl.ravel().tolist(), right_D=dr.ravel().tolist(),
        right_from_left_R=R.tolist(), right_from_left_T_m=T.ravel().tolist(),
        rectify_R1=r1.tolist(), rectify_R2=r2.tolist(), rectify_P1=p1.tolist(),
        rectify_P2=p2.tolist(), rectify_Q=q.tolist(), left_rms_px=float(rl),
        right_rms_px=float(rr), stereo_rms_px=float(rms),
        rectified_vertical_p95_px=float(np.percentile(vertical, 95)), baseline_m=float(np.linalg.norm(T)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = fit(a.images)
    with Path(a.output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"Saved {a.output}; stereo RMS={result['stereo_rms_px']:.3f}px; vertical P95={result['rectified_vertical_p95_px']:.3f}px")
