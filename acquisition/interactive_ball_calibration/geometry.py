"""New Luna right-arm MDH FK and SH5 stereo sphere-center geometry."""
from __future__ import annotations

import configparser
from pathlib import Path

import cv2
import numpy as np

from capt_head.measurement_contract import load_contract


_CAMERA_CONTRACT, _ = load_contract()
CAMERA_IDS = _CAMERA_CONTRACT["sensor_ids"]
CAMERA_SIZE = _CAMERA_CONTRACT["image_size"]
PIXEL_DOMAIN = _CAMERA_CONTRACT["pixel_domain"]


def arm_mdh(path):
    cfg = configparser.ConfigParser()
    cfg.read(Path(path), encoding="utf-8")
    rows = []
    for section in [*(f"R.A1.L{i}.DH" for i in range(7)), "R.A1.FLANGE"]:
        if section not in cfg:
            raise ValueError(f"Missing {section}")
        rows.append([float(cfg[section][key]) for key in ("A", "Alpha", "D", "Theta")])
    values = np.asarray(rows, float)
    if values.shape != (8, 4) or not np.isfinite(values).all():
        raise ValueError("Invalid new Luna right-arm MDH")
    return values


def flange_pose(q_deg, rows):
    """T_MF in the R.A1 MDH parent frame; dimensions are metres."""
    q = np.asarray(q_deg, float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Expected finite right-arm q7 degrees")
    T = np.eye(4)
    for i, (a_mm, alpha_deg, d_mm, theta_deg) in enumerate(rows):
        alpha = np.radians(alpha_deg)
        theta = np.radians(theta_deg + (q[i] if i < 7 else 0))
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        step = np.array([[ct, -st, 0, a_mm / 1000],
                         [ca * st, ca * ct, -sa, -d_mm * sa / 1000],
                         [sa * st, sa * ct, ca, d_mm * ca / 1000],
                         [0, 0, 0, 1]], float)
        T = T @ step
    return T


def validate_stereo(model):
    if (model.get("pixel_domain") != PIXEL_DOMAIN or model.get("image_size") != CAMERA_SIZE or
            model.get("sensor_ids") != CAMERA_IDS or model.get("valid_pairs", 0) < 20):
        raise ValueError("Stereo calibration does not match new SH5 undistorted native images")
    try:
        stereo_rms = float(model["stereo_rms_px"])
        epipolar_p95 = float(model["epipolar_p95_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("SH5 stereo calibration lacks quality metrics") from exc
    if (not np.isfinite(stereo_rms) or not np.isfinite(epipolar_p95) or
            stereo_rms > 3 or epipolar_p95 > 3):
        raise ValueError("SH5 stereo calibration needs <=3 px stereo RMS and epipolar p95")
    K_l = np.asarray(model["left_K"], float)
    K_r = np.asarray(model["right_K"], float)
    D_l = np.asarray(model["left_D"], float)
    D_r = np.asarray(model["right_D"], float)
    R = np.asarray(model["right_from_left_R"], float)
    t = np.asarray(model["right_from_left_T_m"], float)
    if (K_l.shape != (3, 3) or K_r.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,) or
            not all(np.isfinite(x).all() for x in (K_l, K_r, D_l, D_r, R, t)) or
            np.max(np.abs(D_l)) > 1e-8 or np.max(np.abs(D_r)) > 1e-8 or
            min(K_l[0, 0], K_l[1, 1], K_r[0, 0], K_r[1, 1]) <= 0 or
            abs(np.linalg.det(R) - 1) > .01 or not .005 <= np.linalg.norm(t) <= 1):
        raise ValueError("Invalid SH5 stereo geometry")
    return K_l, K_r, R, t


def triangulate(pixels, model):
    K_l, K_r, R, t = validate_stereo(model)
    uv = np.asarray(pixels, float)
    if uv.shape != (4,) or not np.isfinite(uv).all():
        raise ValueError("Expected four finite native stereo pixels")
    if not (0 <= uv[0] < CAMERA_SIZE[0] and 0 <= uv[2] < CAMERA_SIZE[0] and
            0 <= uv[1] < CAMERA_SIZE[1] and 0 <= uv[3] < CAMERA_SIZE[1]):
        raise ValueError("Stereo pixels outside native images")
    P_l = K_l @ np.c_[np.eye(3), np.zeros(3)]
    P_r = K_r @ np.c_[R, t]
    h = cv2.triangulatePoints(P_l, P_r, uv[:2].reshape(2, 1), uv[2:].reshape(2, 1)).ravel()
    if abs(h[3]) < 1e-12:
        raise ValueError("Stereo point at infinity")
    point = h[:3] / h[3]
    if point[2] <= 0 or (R @ point + t)[2] <= 0:
        raise ValueError("Ball is behind a camera")
    def project(P):
        v = P @ np.r_[point, 1.]
        return v[:2] / v[2]
    error = max(np.linalg.norm(project(P_l) - uv[:2]), np.linalg.norm(project(P_r) - uv[2:]))
    if error > 3:
        raise ValueError(f"Stereo reprojection error {error:.2f} px >3 px")
    return point, float(error)


def _ellipse_conic(ellipse):
    """Return the pixel-domain homogeneous conic for a fitted ellipse."""
    try:
        center = np.asarray([ellipse["cx"], ellipse["cy"]], float)
        major = float(ellipse["major_px"])
        minor = float(ellipse["minor_px"])
        angle = np.radians(float(ellipse["angle_deg"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Incomplete fitted ellipse") from exc
    if (center.shape != (2,) or not np.isfinite(center).all() or
            not np.isfinite([major, minor, angle]).all() or major <= 0 or minor <= 0):
        raise ValueError("Invalid fitted ellipse")
    semi_major, semi_minor = major / 2., minor / 2.
    axis = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-np.sin(angle), np.cos(angle)])
    quadratic = (np.outer(axis, axis) / semi_major**2 +
                 np.outer(normal, normal) / semi_minor**2)
    linear = -quadratic @ center
    conic = np.zeros((3, 3), float)
    conic[:2, :2] = quadratic
    conic[:2, 2] = linear
    conic[2, :2] = linear
    conic[2, 2] = center @ quadratic @ center - 1.
    return conic


def sphere_center_from_ellipse(ellipse, K, radius_m):
    """Recover a calibrated pinhole-camera sphere center from its silhouette ellipse.

    The centre of the projected silhouette ellipse is generally not the perspective
    projection of the physical sphere centre. In normalized camera coordinates the
    tangent cone is proportional to C C^T - (||C||^2-r^2) I; the distinct
    eigenvector gives the centre ray and the eigenvalue ratio gives its distance.
    """
    K = np.asarray(K, float)
    radius_m = float(radius_m)
    if K.shape != (3, 3) or not np.isfinite(K).all() or radius_m <= 0:
        raise ValueError("Invalid sphere intrinsics or radius")
    normalized_conic = K.T @ _ellipse_conic(ellipse) @ K
    values, vectors = np.linalg.eigh(normalized_conic)
    candidates = []
    for distinct in range(3):
        pair = [values[index] for index in range(3) if index != distinct]
        perpendicular = .5 * (pair[0] + pair[1])
        parallel = values[distinct]
        if parallel * perpendicular >= 0 or abs(parallel) < 1e-15:
            continue
        distance_sq = radius_m**2 * (1. - perpendicular / parallel)
        if distance_sq <= radius_m**2:
            continue
        mismatch = abs(pair[0] - pair[1]) / max(abs(pair[0]) + abs(pair[1]), 1e-15)
        candidates.append((mismatch, distinct, distance_sq))
    if not candidates:
        raise ValueError("Ellipse conic is not a valid calibrated sphere silhouette")
    mismatch, distinct, distance_sq = min(candidates, key=lambda item: item[0])
    if mismatch > .12:
        raise ValueError("Ellipse conic is inconsistent with a sphere silhouette")
    direction = vectors[:, distinct]
    if direction[2] < 0:
        direction = -direction
    direction = direction / np.linalg.norm(direction)
    center = direction * np.sqrt(distance_sq)
    if not np.isfinite(center).all() or center[2] <= radius_m:
        raise ValueError("Recovered sphere center is invalid")
    return center, float(mismatch)


def stereo_sphere_center(left_detection, right_detection, model, diameter_m=.04,
                         iterations=6):
    """Triangulate the physical sphere centre from fitted silhouette centres.

    A sphere silhouette's ellipse centre is slightly farther from the principal point
    than the perspective projection of the 3-D sphere centre.  Use the known radius
    and the current stereo depth to iteratively move each ellipse centre to the true
    centre projection, while retaining stereo disparity as the depth source.
    """
    K_l, K_r, R_rl, t_rl = validate_stereo(model)
    radius = float(diameter_m) / 2.
    if radius <= 0:
        raise ValueError("Sphere diameter must be positive")
    try:
        original = np.asarray(left_detection["center"] + right_detection["center"], float)
    except (KeyError, TypeError) as exc:
        raise ValueError("Detector metadata lacks fitted sphere centres") from exc
    if original.shape != (4,) or not np.isfinite(original).all():
        raise ValueError("Invalid fitted sphere centres")
    corrected = original.copy()
    center, reprojection = triangulate(corrected, model)
    inv_l, inv_r = np.linalg.inv(K_l), np.linalg.inv(K_r)
    for _ in range(int(iterations)):
        right_center = R_rl @ center + t_rl
        updated = []
        for pixel, K, inv_K, depth in (
                (original[:2], K_l, inv_l, center[2]),
                (original[2:], K_r, inv_r, right_center[2])):
            if depth <= radius:
                raise ValueError("Sphere is too close to camera for centre correction")
            ellipse_ray = inv_K @ np.r_[pixel, 1.]
            factor = 1. - (radius / depth)**2
            centre_ray = np.array([ellipse_ray[0] * factor,
                                   ellipse_ray[1] * factor, 1.])
            homogeneous = K @ centre_ray
            updated.extend((homogeneous[:2] / homogeneous[2]).tolist())
        updated = np.asarray(updated, float)
        new_center, reprojection = triangulate(updated, model)
        corrected = updated
        if np.linalg.norm(new_center - center) < 1e-12:
            center = new_center
            break
        center = new_center
    return {
        "center_left_camera_m": center.tolist(),
        "corrected_stereo_pixels": corrected.tolist(),
        "projection_correction_px": (corrected-original).tolist(),
        "stereo_reprojection_error_px": float(reprojection),
        "method": "sphere_center_projection_stereo_v1",
        "diameter_m": float(diameter_m),
    }


def rigid_fit(source, target):
    a, b = np.asarray(source, float), np.asarray(target, float)
    ac, bc = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ac).T @ (b - bc))
    sign = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, sign]) @ U.T
    return R, bc - R @ ac


def solve_joint(poses, observed, *, iterations=200):
    """Alternate exact rigid alignment and linear flange-offset estimation."""
    poses = np.asarray(poses, float)
    observed = np.asarray(observed, float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or observed.shape != (len(poses), 3) or len(poses) < 12:
        raise ValueError("Need >=12 paired flange poses and stereo points")
    rotations, origins = poses[:, :3, :3], poses[:, :3, 3]
    diversity = np.linalg.svd((rotations - rotations.mean(0)).reshape(-1, 3), compute_uv=False)
    if diversity[-1] < .12:
        raise ValueError("Insufficient flange orientation diversity to identify p_F")
    best = None
    for seed in ([0, 0, 0], [0, 0, .02], [0, 0, -.02]):
        offset = np.asarray(seed, float)
        for _ in range(iterations):
            points = origins + np.einsum("nij,j->ni", rotations, offset)
            R, t = rigid_fit(points, observed)
            # With R and t fixed, the offset is an ordinary linear least-squares problem.
            A = np.einsum("ij,njk->nik", R, rotations).reshape(-1, 3)
            rhs = (observed - (R @ origins.T).T - t).reshape(-1)
            updated = np.linalg.lstsq(A, rhs, rcond=None)[0]
            if np.linalg.norm(updated - offset) < 1e-10:
                offset = updated
                break
            offset = updated
        predicted = (R @ (origins + np.einsum("nij,j->ni", rotations, offset)).T).T + t
        residual = np.linalg.norm(predicted - observed, axis=1)
        candidate = (float(np.median(residual)), offset, R, t, residual)
        if best is None or candidate[0] < best[0]:
            best = candidate
    _, offset, R, t, residual = best
    return offset, R, t, residual
