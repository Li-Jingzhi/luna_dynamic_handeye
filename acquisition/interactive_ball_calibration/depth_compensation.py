"""Low-order disparity correction with fixed FK labels and rigid registration."""
from __future__ import annotations

import numpy as np

from .geometry import rigid_fit


def basis(camera, order):
    camera = np.asarray(camera, float)
    if (camera.ndim != 2 or camera.shape[1] != 3 or
            not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0)):
        raise ValueError("Expected finite positive-depth camera XYZ")
    x, y, z = camera.T
    u, v, d = x/z, y/z, .4/z-1.
    terms = [np.ones(len(x))]
    if order >= 1:
        terms += [u, v, d]
    if order == 2:
        terms += [u*u, v*v, d*d, u*v, u*d, v*d]
    if order not in (0, 1, 2):
        raise ValueError("Disparity polynomial order must be 0, 1 or 2")
    return np.column_stack(terms)


def correct(model, camera):
    camera = np.asarray(camera, float)
    b = basis(camera, model['order'])
    delta = b @ np.asarray(model['coefficients_px'], float)
    disparity = model['focal_baseline_px_m']/camera[:, 2]
    if (not np.isfinite(delta).all() or
            np.any(abs(delta) > model['max_correction_px']) or
            np.any(disparity+delta <= 0)):
        raise ValueError("Disparity correction outside calibrated bounds")
    return camera * (disparity/(disparity+delta))[:, None]


def fit(camera, target, focal_baseline, order=0, ridge=.1):
    """Variable projection eliminates SE3; finite-difference damped least squares."""
    camera, target = np.asarray(camera, float), np.asarray(target, float)
    if target.shape != camera.shape or not np.isfinite(target).all() or len(camera) < 12:
        raise ValueError("Need at least 12 finite paired points")
    if not np.isfinite([focal_baseline, ridge]).all() or focal_baseline <= 0 or ridge < 0:
        raise ValueError("Expected positive focal-baseline and nonnegative ridge")
    b = basis(camera, order)
    model = dict(order=order, coefficients_px=np.zeros(b.shape[1]).tolist(),
                 focal_baseline_px_m=float(focal_baseline), max_correction_px=1.)
    def evaluate(coef):
        candidate = dict(model, coefficients_px=coef.tolist())
        corrected = correct(candidate, camera)
        R, t = rigid_fit(target, corrected)
        error = ((corrected-t)@R-target)*1000.
        return np.r_[error.ravel(), np.sqrt(ridge)*coef], R, t
    coef = np.zeros(b.shape[1]); damping = 1e-3
    for _ in range(60):
        error, R, t = evaluate(coef)
        step = np.eye(len(coef))*1e-5
        try:
            J = np.column_stack([(evaluate(coef+s)[0]-evaluate(coef-s)[0])/2e-5 for s in step])
        except ValueError:
            break
        delta = np.linalg.solve(J.T@J+damping*np.eye(len(coef)), -J.T@error)
        if np.linalg.norm(delta) < 1e-8:
            break
        try:
            proposal = evaluate(coef+delta)[0]
        except ValueError:
            damping *= 10
            continue
        if proposal@proposal < error@error:
            coef += delta
            damping = max(damping/3, 1e-9)
        else:
            damping *= 10
        if damping > 1e12:
            break
    _, R, t = evaluate(coef)
    model['coefficients_px'] = coef.tolist()
    return model, R, t
