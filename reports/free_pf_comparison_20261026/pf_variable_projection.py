"""NumPy pF variable projection, copied unchanged from head_2dof_capt validate_pf_rt.fit_model."""
import numpy as np
from dynamic_handeye.registration import rigid_fit

def fit_model(poses, xyz, heads):
    """Optimize three pF coordinates, eliminating each head's R/t by Kabsch."""
    poses, xyz, heads = np.asarray(poses), np.asarray(xyz), np.asarray(heads)
    rotations, origins = poses[:, :3, :3], poses[:, :3, 3]
    keys = sorted(set(heads))
    if any(np.sum(heads == h) < 12 for h in keys):
        raise ValueError("Each training head needs at least 12 arm points")
    diversity = np.linalg.svd((rotations-rotations.mean(0)).reshape(-1, 3), compute_uv=False)
    if diversity[-1] < .12:
        raise ValueError("Insufficient flange orientation diversity")

    def evaluate(pf):
        fk = origins + np.einsum("nij,j->ni", rotations, pf)
        error, transforms = np.empty_like(xyz), {}
        for h in keys:
            ids = heads == h
            R, t = rigid_fit(fk[ids], xyz[ids])
            error[ids] = fk[ids] @ R.T + t-xyz[ids]
            transforms[h] = dict(R=R.tolist(), t_m=t.tolist())
        return error.ravel(), transforms

    best = None
    for seed in ([0., 0., 0.], [0., 0., .1]):
        pf = np.array(seed)
        converged = False
        for iteration in range(100):
            error, transforms = evaluate(pf)
            step = np.eye(3)*1e-5
            J = np.column_stack([(evaluate(pf+d)[0]-evaluate(pf-d)[0])/2e-5 for d in step])
            delta = np.linalg.lstsq(J, -error, rcond=None)[0]
            if np.linalg.norm(delta) < 1e-9:
                converged = True
                break
            cost = error@error
            for scale in 2.**-np.arange(20):
                proposal = pf+scale*delta
                candidate_error = evaluate(proposal)[0]
                if candidate_error@candidate_error < cost:
                    pf = proposal
                    break
            else:
                converged = np.linalg.norm(J.T@error) < 1e-8
                break
        error, transforms = evaluate(pf)
        candidate = (float(error@error), pf, transforms, converged, iteration+1, J)
        if best is None or candidate[0] < best[0]:
            best = candidate
    cost, pf, transforms, converged, iterations, J = best
    singular = np.linalg.svd(J, compute_uv=False)
    if np.linalg.norm(pf) > .2 or singular[-1] < 1e-8 or not converged:
        raise ValueError("pF optimization unconverged, unobservable or outside 200 mm")
    return dict(pf_m=pf.tolist(), camera_from_mdh_by_head=transforms,
                optimizer="variable_projection_pF_exact_Kabsch_per_head",
                iterations=iterations, converged=converged,
                pf_projected_jacobian_singular_values=singular.tolist(),
                pf_projected_jacobian_condition=float(singular[0]/singular[-1]))
