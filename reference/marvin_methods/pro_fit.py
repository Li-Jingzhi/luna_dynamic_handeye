"""Fit fixed-camera registration and flange ball offset from Pro no-head captures."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .pro_capture import ROOT, digest, flange_pose, fuse, mdh_rows, validate_camera
from .registration import solve_joint


def read_capture(path):
    data=Path(path).read_bytes()
    return gzip.decompress(data) if str(path).endswith('.gz') else data


def load_capture(path,stereo,mdh,raw,*,arm='right'):
    if arm not in ('left','right'):raise ValueError('arm must be left or right')
    arm_index=0 if arm=='left' else 1
    model=json.loads(Path(stereo).read_text());raw_hash=digest(raw);stereo_hash=digest(stereo);mdh_hash=digest(mdh)
    if model.get('schema')!='head_eye_pro_rectified_stereo_v1' or model.get('source_calibration_sha256')!=raw_hash:
        raise ValueError('Wrong Pro stereo/raw contract')
    records=[json.loads(line) for line in read_capture(path).splitlines() if line.strip()]
    rows=[r for r in records if r.get('capture_valid') is True]
    if len(rows)<30:raise ValueError('Need at least30 valid positions for blocked validation')
    if len({r['waypoint_index'] for r in rows})!=len(rows):raise ValueError('Repeated waypoint')
    expected=dict(mdh_sha256=mdh_hash,stereo_model_sha256=stereo_hash,source_calibration_sha256=raw_hash,
        sdk_version='00040701',arm_index=arm_index,arm_side=arm,robot_ip='192.168.12.190',
        image_size=model['image_size'],sensor_ids=model['sensor_ids'],pixel_domain=model['pixel_domain'],
        capture_kind='interactive_ball_auto',valid_frames=50,yellow_ball_diameter_m=.04,
        sphere_center_method='sphere_center_projection_stereo_v1')
    chain=mdh_rows(mdh,arm_index);seen=[set(),set()]
    for row in rows:
        if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('Capture contract mismatch')
        frames=row['camera_window_frames']
        for frame in frames:
            identity=validate_camera(frame,raw_hash,stereo_hash,frame['capture_feedback_monotonic_ns'])
            for eye in (0,1):
                if identity[eye] in seen[eye]:raise ValueError('Source frame reused across positions')
                seen[eye].add(identity[eye])
        fused=fuse(frames)
        for field in ('q7_deg','sphere_center_left_camera_m','sphere_center_corrected_stereo_pixels'):
            if not np.allclose(row[field],fused[field],atol=1e-12,rtol=0):raise ValueError('Stored fusion differs from raw frames')
        pose=flange_pose(row['q7_deg'],chain)
        if not (np.allclose(pose[:3,3],row['flange_position_m'],atol=1e-12,rtol=0) and
                np.allclose(pose[:3,:3],row['flange_rotation'],atol=1e-12,rtol=0)):
            raise ValueError('Stored flange pose differs from MDH FK')
    return sorted(rows,key=lambda r:r['waypoint_index']),len(records)-len(rows)


def predict(poses,offset,R,t):
    return (R@(poses[:,:3,3]+np.einsum('nij,j->ni',poses[:,:3,:3],offset)).T).T+t


def metrics(delta_m):
    error=np.linalg.norm(delta_m,axis=1)*1000
    return dict(samples=len(error),mean_mm=float(np.mean(error)),rms_mm=float(np.sqrt(np.mean(error**2))),
        median_mm=float(np.median(error)),p95_mm=float(np.percentile(error,95)),max_mm=float(np.max(error)),
        axis_bias_mm=(np.mean(delta_m,axis=0)*1000).tolist(),
        axis_rms_mm=(np.sqrt(np.mean(delta_m**2,axis=0))*1000).tolist(),fraction_within_1mm=float(np.mean(error<=1)))


def blocked_splits(count):
    # shortcut: one session only, acquire independent sessions for absolute accuracy validation.
    indices=np.arange(count)
    for held in np.array_split(indices,5):
        train=indices[(indices<held[0]-2)|(indices>held[-1]+2)]
        if len(train)<12:raise ValueError('Insufficient training poses after purging neighbours')
        yield train,held


def solve_fixed_pf_z(poses, points, fixed_pf_z_m):
    """Optimize eight free parameters; pF Z is an exact constraint, not a penalty."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    if not np.isfinite(fixed_pf_z_m) or abs(fixed_pf_z_m) >= .2:
        raise ValueError('Fixed pF Z must be finite and within 200mm')
    offset, R, t, _ = solve_joint(poses, points, iterations=2000)
    seed = np.r_[Rotation.from_matrix(R).as_rotvec(), t, offset[:2]]
    def unpack(x):
        return np.r_[x[6:8], fixed_pf_z_m], Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]
    def residual(x):
        return (predict(poses, *unpack(x)) - points).ravel()
    result = least_squares(residual, seed, x_scale='jac', max_nfev=1000,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Constrained pF optimization failed: ' + result.message)
    offset, R, t = unpack(result.x)
    if np.linalg.norm(offset) > .2:
        raise ValueError('Ball offset exceeds200mm')
    if np.linalg.matrix_rank(result.jac) < 8:
        raise ValueError('Insufficient pose excitation for constrained calibration')
    return offset, R, t, result.nfev


def fit(path,stereo,mdh,raw,*,arm='right',fixed_pf_z_m=None):
    rows,rejected=load_capture(path,stereo,mdh,raw,arm=arm);chain=mdh_rows(mdh,0 if arm=='left' else 1)
    poses=np.array([flange_pose(r['q7_deg'],chain) for r in rows]);points=np.array([r['sphere_center_left_camera_m'] for r in rows])
    solver = (lambda poses, points: solve_joint(poses, points, iterations=2000)) if fixed_pf_z_m is None else (lambda poses, points: solve_fixed_pf_z(poses, points, fixed_pf_z_m))
    cv=np.empty_like(points);fold_ids=np.empty(len(rows),int);folds=[]
    for index,(train,held) in enumerate(blocked_splits(len(rows))):
        offset,R,t,_=solver(poses[train],points[train])
        cv[held]=predict(poses[held],offset,R,t);fold_ids[held]=index
        folds.append(dict(fold=index,train_waypoint_indices=[rows[i]['waypoint_index'] for i in train],
            heldout_waypoint_indices=[rows[i]['waypoint_index'] for i in held],metrics=metrics(cv[held]-points[held]),
            flange_to_ball_center_m=offset.tolist()))
    offset,R,t,_=solver(poses,points)
    if np.linalg.norm(offset)>.2:raise ValueError('Ball offset exceeds200mm')
    fitted=predict(poses,offset,R,t);fit_stats=metrics(fitted-points);cv_stats=metrics(cv-points)
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;inverse=np.eye(4);inverse[:3,:3]=R.T;inverse[:3,3]=-R.T@t
    result=dict(schema='head_eye_pro_fixed_camera_registration_v1',units='m',
        frame_definition=f'left_camera_from_{arm}_arm_MDH_parent',arm_side=arm,arm_index=0 if arm=='left' else 1,
        flange_to_ball_center_m=offset.tolist(),
        left_camera_from_arm_mdh_R=R.tolist(),left_camera_from_arm_mdh_t_m=t.tolist(),
        left_camera_from_arm_mdh_T=T.tolist(),arm_mdh_from_left_camera_T=inverse.tolist(),
        mdh_sha256=digest(mdh),stereo_model_sha256=digest(stereo),source_calibration_sha256=digest(raw),
        capture_uncompressed_sha256=hashlib.sha256(read_capture(path)).hexdigest(),
        captured_waypoints=len(rows)+rejected,valid_waypoints=len(rows),rejected_waypoints=rejected,
        observations_per_position=50,stereo_observations=len(rows)*50,sphere_diameter_assumed_m=.04,
        fit_metrics=fit_stats,blocked_cv_metrics=cv_stats,blocked_cv_folds=folds,
        validation_method='5 contiguous trajectory blocks; purge2 neighbouring positions each side; no residual-based exclusion',
        internal_consistency_1mm_p95_passed=cv_stats['p95_mm']<=1.,independently_verified=False,
        camera_xyz_span_m=np.ptp(points,axis=0).tolist(),joint_span_deg=np.ptp([r['q7_deg'] for r in rows],axis=0).tolist(),
        camera_xyz_bounds_m=dict(min=points.min(0).tolist(),max=points.max(0).tolist()),
        warnings=['Single-session internal consistency, not external absolute accuracy.',
                  '40mm ball diameter assumed; hardware exposure synchronization unverified.',
                  'MDH parent not independently verified as whole-robot base.'])
    if fixed_pf_z_m is not None:
        result['parameter_constraints'] = dict(flange_to_ball_center_z_m=fixed_pf_z_m, free_parameter_count=8, source='User-specified exact constraint; not independently measured by this fit')
    errors=[dict(waypoint_index=r['waypoint_index'],fold=int(fold_ids[i]),
        fit_error_mm=float(np.linalg.norm(fitted[i]-points[i])*1000),
        cv_error_mm=float(np.linalg.norm(cv[i]-points[i])*1000)) for i,r in enumerate(rows)]
    return result,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',required=True)
    parser.add_argument('--arm',choices=('left','right'),default='right')
    parser.add_argument('--stereo',default=str(ROOT/'configs/orin_pro_rectified_stereo_20261019.json'))
    parser.add_argument('--raw',default=str(ROOT/'configs/orin_pro_raw_fisheye_stereo_20261019.json'))
    parser.add_argument('--mdh',default=str(ROOT/'configs/1004.ini'));parser.add_argument('--output',required=True)
    parser.add_argument('--fixed-pf-z-mm',type=float,help='Fix native flange pF Z exactly, in mm')
    args=parser.parse_args();result,errors=fit(args.capture,args.stereo,args.mdh,args.raw,arm=args.arm,
        fixed_pf_z_m=None if args.fixed_pf_z_mm is None else args.fixed_pf_z_mm/1000)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with output.with_suffix('.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(errors[0]),lineterminator='\n');writer.writeheader();writer.writerows(errors)
    print(json.dumps({k:result[k] for k in ('flange_to_ball_center_m','fit_metrics','blocked_cv_metrics','internal_consistency_1mm_p95_passed')},indent=2))


if __name__=='__main__':main()
