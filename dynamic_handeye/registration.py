"""Rigid camera registration and flange sphere-offset fitting, without device defaults."""
import numpy as np


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
