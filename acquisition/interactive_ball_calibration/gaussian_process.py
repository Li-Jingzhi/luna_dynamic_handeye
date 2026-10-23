"""Small NumPy ARD Gaussian process for three-axis residuals in millimetres."""
from __future__ import annotations

import numpy as np


def kernel(x, y, lengths):
    x, y = np.asarray(x, float)/lengths, np.asarray(y, float)/lengths
    d2 = np.maximum(0., np.sum(x*x, 1)[:, None]+np.sum(y*y, 1)[None, :]-2*x@y.T)
    return np.exp(-.5*d2)


def fit(camera, error_m, *, quality=None):
    x, y = np.asarray(camera, float), np.asarray(error_m, float)*1000.
    if (x.ndim != 2 or x.shape[1] != 3 or y.shape != x.shape or len(x) < 12
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Need at least 12 finite paired camera/residual points')
    weights = np.ones(len(x))
    if quality is not None:
        quality = np.asarray(quality, float)
        if quality.shape != (len(x),) or not np.isfinite(quality).all() or np.any(quality <= 0):
            raise ValueError('Quality proxies must be finite and positive')
        # Point jitter ranks reliability; it is NOT a fused-mean standard error.
        weights = np.clip(quality/np.median(quality), .5, 3.)
    mean, scale = y.mean(0), np.maximum(y.std(0), .1)
    z = (y-mean)/scale
    bounds = np.log(np.array([[.005,.5]]*3+[[.05,2.]]))

    def objective(theta, keep=False):
        lengths, noise = np.exp(theta[:3]), np.exp(theta[3])
        K = kernel(x, x, lengths)
        K[np.diag_indices(len(x))] += (noise*weights)**2+1e-8
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
        nll = .5*np.sum(z*alpha)+3*np.log(np.diag(L)).sum()
        return (float(nll), lengths, noise, L, alpha) if keep else float(nll)

    best = None
    for initial in (.03, .10):
        theta = np.log([initial]*3+[.4]); cost = objective(theta)
        for step in (1., .5, .2):
            for _ in range(3):
                improved = False
                for axis in range(4):
                    for direction in (-1., 1.):
                        proposal = theta.copy()
                        proposal[axis] = np.clip(proposal[axis]+direction*step, *bounds[axis])
                        value = objective(proposal)
                        if value < cost:
                            theta, cost, improved = proposal, value, True
                if not improved:
                    break
        result = objective(theta, keep=True)
        if best is None or result[0] < best[0]:
            best = result
    cost, lengths, noise, L, alpha = best
    return dict(kind='ard_gp', centers_m=x.tolist(), lengths_m=lengths.tolist(),
                noise_normalized=float(noise), weights=weights.tolist(),
                output_mean_mm=mean.tolist(), output_scale_mm=scale.tolist(),
                coefficients_normalized=alpha.tolist(), cholesky=L.tolist(),
                negative_log_likelihood=cost,
                uncertainty_note='Model-dependent predictive uncertainty, not a calibrated accuracy guarantee')


def predict(model, camera, *, with_std=False):
    x = np.asarray(camera, float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite camera XYZ array')
    K = kernel(x, np.asarray(model['centers_m']), np.asarray(model['lengths_m']))
    scale = np.asarray(model['output_scale_mm'])
    mean = (K@np.asarray(model['coefficients_normalized']))*scale+model['output_mean_mm']
    if not with_std:
        return mean/1000.
    projected = np.linalg.solve(np.asarray(model['cholesky']), K.T)
    latent_var = np.maximum(0., 1.-np.sum(projected*projected, axis=0))
    # Unit query weight: typical observation noise, not per-frame input covariance.
    std = np.sqrt(latent_var+model['noise_normalized']**2)[:, None]*scale
    return mean/1000., std
