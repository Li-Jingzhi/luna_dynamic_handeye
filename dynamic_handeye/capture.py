"""Plan-driven adapter to the preserved Luna SDK capture implementation."""
import json
import functools
import time
from pathlib import Path
import sys
import numpy as np
from .geometry import ROOT, digest
from .dataset import write_json


def load_plan(path, assets):
    plan = json.loads(Path(path).read_text())
    mode=plan.get('mode')
    if (plan.get('robot') != 'gento_luna' or mode not in ('stationary_crossed','stationary_yaw_sweep') or
            plan.get('robot_version','luna_v2_0') != 'luna_v2_0'):
        raise ValueError('Expected old Luna stationary crossed or yaw-sweep capture plan')
    if plan.get('head_feedback_order',['controller_yaw','controller_pitch']) != ['controller_yaw','controller_pitch']:
        raise ValueError('Head feedback order must be yaw then pitch')
    heads = np.asarray(plan['head_angles_deg'], float)
    minimum=6 if mode=='stationary_crossed' else 3
    if heads.ndim != 2 or heads.shape[1] != 2 or len(heads) < minimum or not np.isfinite(heads).all():
        raise ValueError('Insufficient finite two-angle head settings')
    if len(np.unique(heads,axis=0)) != len(heads):raise ValueError('Duplicate head settings')
    if mode=='stationary_crossed' and np.linalg.matrix_rank(heads-heads.mean(0))<2:
        raise ValueError('Collinear head settings cannot define a crossed plan')
    reference=np.asarray(plan.get('reference_head_deg',[0.,-25.]),float)
    if reference.shape!=(2,) or not np.isfinite(reference).all():raise ValueError('Invalid reference head')
    if mode=='stationary_yaw_sweep' and (np.any(heads[:,1]!=0) or np.any(reference!=[0,0]) or
            not (heads[:,0].min()<0<heads[:,0].max()) or not np.any(heads[:,0]==0)):
        raise ValueError('Yaw sweep requires pitch0, reference[0,0], left/right and center yaw')
    if plan.get('observations_per_position') != assets.contract['valid_frames_per_point']:
        raise ValueError('Plan frame count differs from camera fusion contract')
    for i in range(2):
        limits = [float(assets.ini[f'R.H.L{i}.BASIC'][k]) for k in ('LimitNeg', 'LimitPos')]
        if min(heads[:,i].min(),reference[i])<limits[0] or max(heads[:,i].max(),reference[i])>limits[1]:
            raise ValueError('Plan exceeds recorded Luna head limits; live limits checked before motion')
    return plan


def fuse_pro_window(original_fuse,observations):
    """Require exactly 50 fresh source frames per eye, then reuse Luna sphere fusion."""
    if len(observations)!=50:raise ValueError('Exactly 50 unique source frames required')
    source=np.asarray([o[3].get('source_timestamp_ns') for o in observations],dtype=float)
    if source.shape!=(50,2) or not np.isfinite(source).all() or np.any(source<0):
        raise ValueError('Missing per-eye source timestamps')
    # Keep integer identities: nanosecond timestamps need not be exactly representable as floats.
    for eye in (0,1):
        if len({o[3]['source_timestamp_ns'][eye] for o in observations})!=50:
            raise ValueError('Repeated source frames in one eye')
    result=original_fuse(observations)
    jitter=result.get('sphere_corrected_disparity_jitter_p95_px',float('nan'))
    if not np.isfinite(jitter) or not 0<=jitter<=.1:
        raise ValueError('Corrected disparity jitter exceeds 0.1px')
    q=np.asarray(result['robot_window_q7_deg'],float);h=np.asarray(result['robot_window_head_deg'],float)
    if q.shape!=(50,7) or h.shape!=(50,2) or not np.isfinite(q).all() or not np.isfinite(h).all():
        raise ValueError('Incomplete fusion-window robot feedback')
    if max(np.ptp(q,axis=0).max(),np.ptp(h,axis=0).max())>.02:
        raise ValueError('Robot varied by more than 0.02deg')
    result.update(q7_deg=np.median(q,axis=0).tolist(),head_angles_deg=np.median(h,axis=0).tolist(),
                  robot_version='luna_v2_0',fusion_policy='50_unique_source_frames_per_eye')
    return result


def capture_fresh(original_capture,*args,minimum_stamp_ns=0,**kwargs):
    return original_capture(*args,minimum_stamp_ns=max(minimum_stamp_ns,time.monotonic_ns()),**kwargs)


def yaw_reference_waypoints(original_reference,current,target,limits,**kwargs):
    if np.asarray(target).shape!=(2,) or target[1]!=0:raise ValueError('Yaw-sweep pitch must remain zero')
    path=original_reference(current,target,limits,**kwargs)
    return [np.array([point[0],0.]) for point in path]


def run_capture(args, assets):
    plan = load_plan(args.plan, assets)
    trajectory = json.loads(Path(args.path).read_text())
    if (trajectory.get('schema') != 'luna_sh5_taught_path_v1' or trajectory.get('complete') is not True or
            trajectory.get('mdh_sha256') != assets.hashes['mdh'] or
            trajectory.get('stereo_model_sha256') != assets.hashes['stereo']):
        raise ValueError('Trajectory uses different Luna robot/camera assets')
    if not 1 <= args.stop_after <= len(trajectory['waypoints']):
        raise ValueError('stop-after outside recorded trajectory')
    if not args.run:
        return dict(status='plan_checked_no_hardware_access', head_settings=len(plan['head_angles_deg']))
    output = Path(args.output).resolve()
    plan_record = output.with_suffix(output.suffix+'.plan.json')
    binding = dict(plan_sha256=digest(args.plan), trajectory_sha256=digest(args.path), assets=assets.hashes)
    if plan_record.exists():
        if json.loads(plan_record.read_text()) != binding:
            raise ValueError('Resume plan/trajectory/assets differ from original capture')
    elif output.exists():
        raise ValueError('Existing capture lacks plan binding; select a new output')
    else:
        write_json(plan_record, binding)
    sys.path.insert(0, str(ROOT/'acquisition'))
    from interactive_ball_calibration import capture_point_head as original
    from interactive_ball_calibration.capture import confirm_automatic_motion
    if not confirm_automatic_motion():
        raise RuntimeError('Capture cancelled')
    saved=(original.HEAD_POSES,original.REFERENCE,original.fuse_camera_observations,
           original.capture_at_pose,original.head_reference_waypoints,sys.argv)
    try:
        original.HEAD_POSES=tuple(tuple(h) for h in plan['head_angles_deg'])
        original.REFERENCE=np.asarray(plan.get('reference_head_deg',[0.,-25.]),float)
        if plan['mode']=='stationary_yaw_sweep':
            original.fuse_camera_observations=functools.partial(fuse_pro_window,saved[2])
            original.capture_at_pose=functools.partial(capture_fresh,saved[3])
            original.head_reference_waypoints=functools.partial(yaw_reference_waypoints,saved[4])
        sys.argv = ['capture_point_head', '--path', str(Path(args.path).resolve()), '--output', str(output),
                    '--stereo', str(ROOT/'configs/camera/luna_rectified_stereo.json'),
                    '--mdh', str(ROOT/'configs/robot/luna_new.ini'), '--sdk-root', str(Path(args.sdk_root).resolve()),
                    '--latest', str(Path(args.latest).resolve()), '--robot-ip', args.robot_ip,
                    '--stop-after', str(args.stop_after)]
        original.main()
    finally:
        (original.HEAD_POSES,original.REFERENCE,original.fuse_camera_observations,
         original.capture_at_pose,original.head_reference_waypoints,sys.argv)=saved
    return dict(status='capture_finished', output=str(output))
