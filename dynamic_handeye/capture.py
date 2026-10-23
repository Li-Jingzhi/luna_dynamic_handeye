"""Plan-driven adapter to the preserved Luna SDK capture implementation."""
import json
from pathlib import Path
import sys
import numpy as np
from .geometry import ROOT, digest
from .dataset import write_json


def load_plan(path, assets):
    plan = json.loads(Path(path).read_text())
    if plan.get('robot') != 'gento_luna' or plan.get('mode') != 'stationary_crossed':
        raise ValueError('Expected Luna stationary crossed capture plan')
    heads = np.asarray(plan['head_angles_deg'], float)
    if heads.ndim != 2 or heads.shape[1] != 2 or len(heads) < 6 or not np.isfinite(heads).all():
        raise ValueError('Need at least six finite two-axis head settings')
    if len(np.unique(heads, axis=0)) != len(heads) or np.linalg.matrix_rank(heads-heads.mean(0)) < 2:
        raise ValueError('Duplicate or collinear head settings')
    if plan.get('observations_per_position') != assets.contract['valid_frames_per_point']:
        raise ValueError('Plan frame count differs from camera fusion contract')
    for i in range(2):
        limits = [float(assets.ini[f'R.H.L{i}.BASIC'][k]) for k in ('LimitNeg', 'LimitPos')]
        if np.any(heads[:, i] < limits[0]) or np.any(heads[:, i] > limits[1]):
            raise ValueError('Plan exceeds recorded Luna head limits; live limits checked before motion')
    return plan


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
    original.HEAD_POSES = tuple(tuple(h) for h in plan['head_angles_deg'])
    sys.argv = ['capture_point_head', '--path', str(Path(args.path).resolve()), '--output', str(output),
                '--stereo', str(ROOT/'configs/camera/luna_rectified_stereo.json'),
                '--mdh', str(ROOT/'configs/robot/luna_new.ini'), '--sdk-root', str(Path(args.sdk_root).resolve()),
                '--latest', str(Path(args.latest).resolve()), '--robot-ip', args.robot_ip,
                '--stop-after', str(args.stop_after)]
    original.main()
    return dict(status='capture_finished', output=str(output))
