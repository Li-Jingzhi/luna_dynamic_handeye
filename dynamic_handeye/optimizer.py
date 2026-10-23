"""Copied NumPy Huber optimizer from head_2dof_capt/capt_head/geometry.py."""
import numpy as np

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
