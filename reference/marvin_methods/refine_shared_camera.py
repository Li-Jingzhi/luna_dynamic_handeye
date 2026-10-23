"""Shared-pF joint fits with one regularized camera correction for BOTH arms.

Experimental observation correction, not a new intrinsic/stereo calibration.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .compare_dual_pf import SIDES, dataset, fit_pair, subset
from .compare_pro_calibration import splits
from .pro_capture import ROOT, digest
from .pro_fit import metrics

# Predeclared before evaluation: strength is selected within outer-training data only.
VARIANTS = {
    'shared_rigid': ('none', 0.),
    'disparity_plane_01px': ('disparity_plane', .1),
    'disparity_plane_03px': ('disparity_plane', .3),
    'disparity_quad_01px': ('disparity_quad', .1),
    'disparity_quad_03px': ('disparity_quad', .3),
    'radial_disparity_01px': ('radial_disparity', .1),
    'radial_disparity_03px': ('radial_disparity', .3),
    'xyz_quad_05mm': ('xyz_quad', .0005),
    'xyz_quad_1mm': ('xyz_quad', .001),
    'xyz_quad_2mm': ('xyz_quad', .002),
    'xyz_cubic_05mm': ('xyz_cubic', .0005),
    'xyz_cubic_1mm': ('xyz_cubic', .001),
    'affine_shared': ('none', 0.),
    'affine_xyz_quad_05mm': ('xyz_quad', .0005),
    'affine_xyz_quad_1mm': ('xyz_quad', .001),
    'affine_xyz_quad_2mm': ('xyz_quad', .002),
    'affine_xyz_cubic_05mm': ('xyz_cubic', .0005),
    'affine_xyz_cubic_1mm': ('xyz_cubic', .001),
}
CENTER = np.array([0., 0., .4])
SCALE = .2


def polynomial(points, cubic=False):
    x, y, z = ((points-CENTER)/SCALE).T
    columns = [x*x, x*y, x*z, y*y, y*z, z*z]
    if cubic:
        columns += [x**3, x*x*y, x*x*z, x*y*y, x*y*z, x*z*z,
                    y**3, y*y*z, y*z*z, z**3]
    return np.column_stack(columns)


def ray_features(points, quadratic=False):
    u, v = (points[:, :2]/points[:, 2:]/.5).T
    columns = [np.ones(len(points)), u, v]
    if quadratic:
        columns += [u*u, u*v, v*v]
    return np.column_stack(columns)


def correction_priors(kind, sigma):
    if kind == 'none': return np.empty(0)
    if kind.startswith('xyz_'): return np.full((16 if kind=='xyz_cubic' else 6)*3, sigma)
    n = 3 if kind=='disparity_plane' else 6
    return np.r_[np.full(n,sigma), [.005,.005] if kind=='radial_disparity' else []]


def correct_camera(points, model):
    """Apply a common camera-space correction; no arm/joints/pF are inputs."""
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or np.any(points[:,2]<=0):
        raise ValueError('Expected finite Nx3 positive-depth camera observations')
    kind=model['correction_kind'];c=np.asarray(model['correction_coefficients'],float)
    if kind=='none':return points.copy()
    if kind.startswith('xyz_'):
        return points+polynomial(points,kind=='xyz_cubic')@c.reshape(-1,3)
    F=ray_features(points,kind!='disparity_plane')
    bias=F@c[:F.shape[1]]
    # Signed baseline: native disparity=-P_right[0,3]/Z for this calibrated rig.
    denominator=1-bias*points[:,2]/model['signed_projection_baseline_px_m']
    if np.any(denominator<=.1):raise ValueError('Disparity correction becomes singular')
    corrected=points/denominator[:,None]
    if kind=='radial_disparity':
        radius2=np.sum((points[:,:2]/points[:,2:])**2,axis=1)
        corrected[:,:2]*=(1+c[-2]*radius2+c[-1]*radius2**2)[:,None]
    return corrected


def camera_to_arm(points, model, side):
    if side not in SIDES:raise ValueError('Unknown arm')
    corrected=correct_camera(points,model)
    maps=model['observation_from_mdh_T'] if 'observation_from_mdh_T' in model else model['camera_from_mdh_T']
    T=np.asarray(maps[side])
    return np.linalg.solve(T[:3,:3],(corrected-T[:3,3]).T).T


def flange_errors(data, model, side):
    pose=data['poses']
    return camera_to_arm(data['points'],model,side)-pose[:,:3,3]-np.einsum('nij,j->ni',pose[:,:3,:3],model['shared_pf_m'])


def fit(data, variant, seed=None):
    if variant not in VARIANTS:raise ValueError('Unknown correction variant')
    kind,sigma=VARIANTS[variant]
    affine=variant.startswith('affine_');block=12 if affine else 6;pf_start=2*block;correction_start=pf_start+3
    if seed is None:seed=fit_pair(data,'affine' if affine else 'rigid_l2','shared')
    coefficients=correction_priors(kind,sigma)
    x0=np.r_[*(np.asarray(seed['models'][s]['parameters'])[:block] for s in SIDES),
              seed['models']['right']['flange_to_ball_center_m'],np.zeros(len(coefficients))]
    lower,upper=np.full(len(x0),-np.inf),np.full(len(x0),np.inf)
    lower[pf_start:correction_start],upper[pf_start:correction_start]=-.2,.2
    if affine:
        for i in range(2):
            lower[i*block:i*block+9]=x0[i*block:i*block+9]-.03
            upper[i*block:i*block+9]=x0[i*block:i*block+9]+.03
    if len(coefficients):
        limits=np.full(len(coefficients),.01 if kind.startswith('xyz_') else 1.)
        if kind=='radial_disparity':limits[-2:]=.03
        lower[correction_start:],upper[correction_start:]=-limits,limits
    projection=float(data['right']['P'][1,0,3])
    if abs(projection)<1e-6 or not np.allclose(data['right']['P'],data['left']['P'],atol=1e-12,rtol=0):
        raise ValueError('Both arms must use the same nondegenerate stereo projection')
    def unpack(x):
        maps={}
        for i,s in enumerate(SIDES):
            T=np.eye(4)
            if affine:
                T[:3,:3]=x[i*block:i*block+9].reshape(3,3);T[:3,3]=x[i*block+9:i*block+12]
            else:
                T[:3,:3]=Rotation.from_rotvec(x[i*block:i*block+3]).as_matrix();T[:3,3]=x[i*block+3:i*block+6]
            maps[s]=T.tolist()
        return dict(variant=variant,correction_kind=kind,correction_coefficients=x[correction_start:].tolist(),
                    map_kind='affine' if affine else 'rigid',observation_from_mdh_T=maps,shared_pf_m=x[pf_start:correction_start].tolist(),
                    signed_projection_baseline_px_m=projection)
    def residual(x):
        model=unpack(x);delta=[]
        for s in SIDES:
            corrected=correct_camera(data[s]['points'],model)
            pose=data[s]['poses'];arm=pose[:,:3,3]+np.einsum('nij,j->ni',pose[:,:3,:3],x[pf_start:correction_start])
            T=np.asarray(model['observation_from_mdh_T'][s])
            delta.extend(((arm@T[:3,:3].T+T[:3,3]-corrected)/.001).ravel())
        if len(coefficients):delta.extend(x[correction_start:]/coefficients)
        return np.asarray(delta)
    optimized=least_squares(residual,x0,bounds=(lower,upper),x_scale='jac',max_nfev=500,
                             ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise RuntimeError(f'{variant}: {optimized.message}')
    if np.linalg.norm(optimized.x[pf_start:correction_start])>.2:raise ValueError('pF exceeds 200mm')
    model=unpack(optimized.x)
    for T in model['observation_from_mdh_T'].values():
        singular=np.linalg.svd(np.asarray(T)[:3,:3],compute_uv=False)
        if singular.min()<.9 or singular.max()>1.1:raise ValueError('Unphysical affine scale')
    model.update(schema='shared_pf_camera_correction_experiment_v1',units='m',
                 coefficient_prior_sigma=coefficients.tolist(),data_residual_scale_m=.001,
                 parameter_count=len(x0),optimizer_evaluations=int(optimized.nfev),
                 bound_active=bool(np.any(optimized.active_mask)),
                 polynomial_center_m=CENTER.tolist(),polynomial_scale_m=SCALE,
                 training_camera_bounds_m={s:dict(min=data[s]['points'].min(0).tolist(),max=data[s]['points'].max(0).tolist()) for s in SIDES},
                 independently_verified=False,right_anchor_fixed=False,
                 equation='camera_corrected=f(observed_camera_xyz); MDH_xyz=inverse(M)@(camera_corrected-t), M=R for rigid models; pF only for calibration flange comparison.',
                 note='Common empirical camera correction, not independently identified intrinsics; runtime needs no robot joint angles.')
    return model


def fit_all(data):
    seeds={False:fit_pair(data,'rigid_l2','shared'),True:fit_pair(data,'affine','shared')}
    return {v:fit(data,v,seeds[v.startswith('affine_')]) for v in VARIANTS}


def choose(scores):
    return min(scores,key=lambda name:(max(scores[name][s]['p95_mm'] for s in SIDES),
                                       max(scores[name][s]['rms_mm'] for s in SIDES),name))


def compare(data):
    indices={s:np.arange(len(data[s]['points'])) for s in SIDES}
    outer={s:list(splits(indices[s])) for s in SIDES}
    cv={v:{s:np.empty_like(data[s]['points']) for s in SIDES} for v in VARIANTS}
    nested={s:np.empty_like(data[s]['points']) for s in SIDES};folds=[]
    for f in range(5):
        train={s:outer[s][f][0] for s in SIDES};held={s:outer[s][f][1] for s in SIDES}
        inner={s:list(splits(train[s],folds=4)) for s in SIDES}
        errors={v:{s:[] for s in SIDES} for v in VARIANTS}
        for j in range(4):
            fitted=fit_all({s:subset(data[s],inner[s][j][0]) for s in SIDES})
            for v,model in fitted.items():
                for s in SIDES:errors[v][s].extend(flange_errors(subset(data[s],inner[s][j][1]),model,s))
            print(f'Outer {f+1}/5 inner {j+1}/4 done',flush=True)
        inner_scores={v:{s:metrics(np.asarray(errors[v][s])) for s in SIDES} for v in VARIANTS}
        chosen=choose(inner_scores)
        models=fit_all({s:subset(data[s],train[s]) for s in SIDES})
        for v,model in models.items():
            for s in SIDES:cv[v][s][held[s]]=flange_errors(subset(data[s],held[s]),model,s)
        for s in SIDES:nested[s][held[s]]=cv[chosen][s][held[s]]
        folds.append(dict(fold=f,selected=chosen,inner_scores=inner_scores,models=models,
                          splits={s:dict(train=train[s].tolist(),held=held[s].tolist()) for s in SIDES}))
        print(f'Outer {f+1}/5 selected {chosen}',flush=True)
    models=fit_all(data)
    records={v:dict(model=m,fit={s:metrics(flange_errors(data[s],m,s)) for s in SIDES},
                    cv={s:metrics(cv[v][s]) for s in SIDES}) for v,m in models.items()}
    selected=choose({v:r['cv'] for v,r in records.items()})
    return dict(candidates=records,selected=selected,nested={s:metrics(nested[s]) for s in SIDES},folds=folds),cv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output directory')
    data={s:dataset(ROOT/('datasets/drag_replay_20261020_091535_901340' if s=='right' else 'datasets/left_drag_replay_20261020_110645_673779'),s) for s in SIDES}
    args.output.mkdir(parents=True)
    result,cv=compare(data)
    result.update(schema='shared_camera_refinement_comparison_v1',source_script_sha256=digest(Path(__file__)),
        data={s:dict(path=data[s]['path'],manifest=data[s]['manifest']) for s in SIDES},
        warnings=['Same latest two sessions reused for further model research; nested selection reduces within-run selection bias, not adaptive research bias.',
                  'No independent holdout session or absolute position truth; no proof that empirical correction is camera-only error.',
                  'No residual-based exclusion; both arms held out with purge 2; identical shared pF in every model.',
                  'Deployment remains unchanged. Candidate right extrinsics are re-estimated, not the prior fixed anchor.'])
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for v,r in result['candidates'].items():(args.output/f'{v}.json').write_text(json.dumps(r['model'],indent=2,allow_nan=False)+'\n')
    with (args.output/'heldout_errors.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['variant','arm','waypoint_index','dx_mm','dy_mm','dz_mm','error_mm'],lineterminator='\n');writer.writeheader()
        for v in cv:
            for s in SIDES:
                for row,d in zip(data[s]['rows'],cv[v][s]*1000):writer.writerow(dict(variant=v,arm=s,waypoint_index=row['waypoint_index'],dx_mm=d[0],dy_mm=d[1],dz_mm=d[2],error_mm=np.linalg.norm(d)))
    print(json.dumps(dict(selected=result['selected'],nested=result['nested']),indent=2))


if __name__=='__main__':main()
