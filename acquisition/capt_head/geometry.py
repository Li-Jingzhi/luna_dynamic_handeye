"""Camera pixel geometry and two-revolute head kinematics.

Frame B is the immovable robot base (metres). P_L0/P_R0 project base coordinates
to pixels at q0. G(q) maps reference-camera rigid coordinates into the current
camera's rigid pose in B. Dynamic P = P0 @ inv(G). No robot interface.
"""
from __future__ import annotations
import numpy as np
import cv2


def project(P, points):
    P = np.asarray(P, float)
    X = np.atleast_2d(np.asarray(points, float))
    if P.shape != (3, 4) or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("Expected projection 3x4 and Nx3 points")
    y = np.column_stack((X, np.ones(len(X)))) @ P.T
    if np.any(np.abs(y[:, 2]) < 1.e-12):
        raise ValueError("Projection at infinity")
    return y[:, :2] / y[:, 2, None]


def _normalize(points, target):
    c = points.mean(axis=0)
    d = np.linalg.norm(points - c, axis=1).mean()
    if d < 1.e-9:
        raise ValueError("Insufficient point spread")
    s = target / d
    H = np.eye(points.shape[1] + 1)
    H[:-1, :-1] *= s
    H[:-1, -1] = -s*c
    return (points - c)*s, H


def dlt(points, pixels):
    X, uv = np.asarray(points, float), np.asarray(pixels, float)
    if X.ndim != 2 or X.shape[1] != 3 or uv.shape != (len(X), 2) or len(X) < 12:
        raise ValueError("At least 12 XYZ/pixel pairs required")
    if not np.isfinite(X).all() or not np.isfinite(uv).all():
        raise ValueError("Nonfinite DLT samples")
    singular = np.linalg.svd(X-X.mean(0), compute_uv=False)
    if singular[-1] / max(singular[0], 1.e-12) < .03:
        raise ValueError("DLT 3D reference points are nearly coplanar")
    a, U = _normalize(X, np.sqrt(3.))
    b, V = _normalize(uv, np.sqrt(2.))
    homogeneous = np.column_stack((a, np.ones(len(a))))
    M = np.zeros((len(X)*2, 12), float)
    M[0::2, :4] = homogeneous
    M[0::2, 8:12] = -b[:, 0, None] * homogeneous
    M[1::2, 4:8] = homogeneous
    M[1::2, 8:12] = -b[:, 1, None] * homogeneous
    _, _, VT = np.linalg.svd(M, full_matrices=False)
    P = np.linalg.inv(V) @ VT[-1].reshape(3, 4) @ U
    P /= np.linalg.norm(P[2, :3])
    if np.median(np.column_stack((X, np.ones(len(X)))) @ P[2]) < 0:
        P = -P
    return P


def triangulate(Pl, Pr, pixels):
    pair = np.asarray(pixels, float)
    if pair.shape != (4,) or not np.isfinite(pair).all():
        raise ValueError("Expected four finite stereo pixel coordinates")
    Pl, Pr = np.asarray(Pl, float), np.asarray(Pr, float)
    A = np.stack((pair[0]*Pl[2]-Pl[0], pair[1]*Pl[2]-Pl[1],
                  pair[2]*Pr[2]-Pr[0], pair[3]*Pr[2]-Pr[1]))
    _, _, VT = np.linalg.svd(A)
    h = VT[-1]
    if abs(h[3]) < 1.e-12:
        raise ValueError("Degenerate 3D triangulation")
    X = h[:3]/h[3]
    if (Pl @ np.r_[X, 1.])[2] <= 0 or (Pr @ np.r_[X, 1.])[2] <= 0:
        raise ValueError("Point behind one of the cameras")
    return X


def revolute(axis, pivot, angle):
    a = np.asarray(axis, float).copy()
    n = float(np.linalg.norm(a))
    if n < 1.e-9 or not np.isfinite(n):
        raise ValueError("Invalid head axis")
    a /= n
    c = np.asarray(pivot, float)
    if c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError("Invalid pivot")
    R = cv2.Rodrigues(a*float(angle))[0]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (np.eye(3)-R) @ c
    return T


def head_motion(parameters, head_rad, reference_rad):
    params = np.asarray(parameters, float).reshape(2, 6)
    delta = np.asarray(head_rad, float)-np.asarray(reference_rad, float)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("Measured [yaw,pitch] in radians required")
    return (revolute(params[0, :3], params[0, 3:], delta[0]) @
            revolute(params[1, :3], params[1, 3:], delta[1]))


def inverse_rigid(T):
    A = np.asarray(T, float)
    if A.shape != (4, 4):
        raise ValueError("Rigid 4x4 required")
    o = np.eye(4)
    o[:3, :3] = A[:3, :3].T
    o[:3, 3] = -o[:3, :3] @ A[:3, 3]
    return o


def dynamic_projections(Pl0, Pr0, params, angles, reference):
    D = inverse_rigid(head_motion(params, angles, reference))
    return np.asarray(Pl0) @ D, np.asarray(Pr0) @ D


def ideal_pixels(pairs, lens):
    """Raw pixels to *ideal* pixels with identical output K; never rectify twice."""
    data = np.asarray(pairs, float).reshape(-1, 4)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite pixels")
    if lens is None:
        return data.copy()
    if lens.get("model") not in ("fisheye", "pinhole"):
        raise ValueError("Specify fisheye or pinhole lens")
    output = []
    for i, name in enumerate(("left", "right")):
        K = np.asarray(lens["K_"+name], float)
        D = np.asarray(lens["D_"+name], float)
        if K.shape != (3, 3) or np.min(np.diag(K)[:2]) <= 0 or not np.isfinite(K).all() or not np.isfinite(D).all():
            raise ValueError("Invalid lens K/D")
        fn = cv2.fisheye.undistortPoints if lens["model"] == "fisheye" else cv2.undistortPoints
        x = data[:, (2*i):(2*i+2)].reshape(-1, 1, 2)
        y = fn(np.ascontiguousarray(x), K, D, P=K).reshape(-1, 2)
        if not np.isfinite(y).all():
            raise ValueError("Lens undistortion failed")
        output.append(y)
    return np.hstack(output)


def robust_optimize(fun, initial, iterations=70):
    """Small dependency-free Huber damped least-squares for offline fit only."""
    x = np.asarray(initial, float).copy()
    damping = .001
    def cost(v):
        a = np.abs(v)
        return float(np.sum(np.where(a <= 3, .5*v*v, 3*(a-1.5))))
    residual = fun(x)
    best = cost(residual)
    for _ in range(iterations):
        J = np.column_stack([(fun(x+np.eye(len(x))[i]*1.e-5)-residual)/1.e-5
                             for i in range(len(x))])
        w = np.minimum(1., 3/np.maximum(np.abs(residual), 1.e-12))
        JW = J*w[:, None]
        H = J.T@JW
        direction = J.T@(w*residual)
        try:
            step = np.linalg.solve(H + damping*np.diag(np.maximum(np.diag(H), 1)), direction)
        except np.linalg.LinAlgError:
            raise ValueError("Head calibration ill-conditioned")
        trial = x-step
        try:
            r2 = fun(trial)
            updated = cost(r2)
        except (ValueError, FloatingPointError):
            updated = float("inf")
        if updated < best:
            x, residual, best = trial, r2, updated
            damping = max(damping/3, 1.e-8)
            if np.linalg.norm(step) < 1.e-7:
                break
        else:
            damping = min(damping*10, 1.e12)
    return x


def inverse_head_points(params, angles, reference, xyz):
    """Vectorized reference-coordinates of world points for an array of heads."""
    prm = np.asarray(params, float).reshape(2, 6)
    ang = np.asarray(angles, float)
    xyz = np.asarray(xyz, float)
    if ang.shape != (len(xyz), 2) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected Nx2 head angles and Nx3 base points")
    v = xyz.copy()
    for joint in (0, 1):  # Inverse yaw first, then inverse pitch
        axis = prm[joint, :3]
        pivot = prm[joint, 3:]
        norm = np.linalg.norm(axis)
        if norm < 1.e-9:
            raise ValueError("Degenerate head axis")
        axis = axis/norm
        theta = -(ang[:, joint] - reference[joint])
        d = v - pivot
        co, si = np.cos(theta)[:, None], np.sin(theta)[:, None]
        v = pivot + d*co + np.cross(axis, d)*si + (
            (1-co)*np.sum(d*axis, axis=1)[:, None]*axis)
    return v
