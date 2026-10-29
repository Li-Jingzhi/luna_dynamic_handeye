"""Luna V2: operator drag teaching followed by automatic fixed-head replay capture."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'acquisition'))
from dynamic_handeye.geometry import Assets
from dynamic_handeye.dataset import load_capture
from capt_head.drag_capture import load_sdk, yellow_pixels
from gq_calibration.capture import checked, head_angles, read_limits, wait_head
from interactive_ball_calibration.capture import (connect_robot, read_rt, arm_limits, validate_path,
                                                  head_reference_waypoints)
from orin_rectified_stream.capture_progress import capture_indices

STEREO = ROOT / 'configs/camera/luna_rectified_stereo.json'
MDH = ROOT / 'configs/robot/luna_new.ini'
CONTRACT = ROOT / 'acquisition/configs/yellow_measurement_v1.json'
LATEST = ROOT / 'acquisition/runtime/new_sh5_yellow/latest_pose.json'
VIEWER = ROOT / 'acquisition/orin_rectified_stream/viewer.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    capture_indices(1,plan.get('capture_stride',1))
    if type(plan.get('alternate_direction', False)) is not bool:
        raise ValueError('alternate_direction must be boolean')
    q = np.asarray(plan['head_angles_deg'], float)
    if (q.ndim != 2 or q.shape[1] != 2 or not len(q) or not np.isfinite(q).all()
            or len({tuple(x) for x in q}) != len(q) or np.any(abs(q) > [90,25])
            or plan.get('observations_per_position') != 50):
        raise ValueError('Expected unique finite yaw/pitch targets in model limits and exactly 50 frames')
    return plan, q


def validate_trajectory(path, assets):
    value = json.loads(Path(path).read_text())
    if (value.get('schema') != 'luna_sh5_taught_path_v1' or value.get('complete') is not True
            or value.get('arm_side') != 'right' or value.get('sdk_version') != '00040701'
            or value.get('pixel_domain') != 'ideal_pinhole'
            or value.get('mdh_sha256') != assets.hashes['mdh']
            or value.get('stereo_model_sha256') != assets.hashes['stereo']
            or value.get('image_size') != [1280,992]
            or value.get('sensor_ids') != assets.stereo['sensor_ids']):
        raise ValueError('Trajectory must be complete and bound to current Luna MDH/camera')
    validate_path(value['waypoints'], np.tile([-360.,360.], (7,1)))
    return value


def summarize(path, count, head, code, assets, stride=1):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    indices = [r['waypoint_index'] for r in rows]
    selected = capture_indices(count,stride)
    if len(set(indices)) != len(indices) or any(i not in selected for i in indices):
        raise ValueError('Invalid/repeated waypoint in capture')
    for row in rows:
        if row.get('capture_stride',1) != stride:
            raise ValueError('Capture stride mismatch')
        if (row.get('target_head_deg') != list(head) or row.get('mdh_sha256') != assets.hashes['mdh']
                or row.get('stereo_model_sha256') != assets.hashes['stereo']):
            raise ValueError('Capture target/asset mismatch')
        if row.get('capture_valid') is True and row.get('valid_frames') != 50:
            raise ValueError('Valid capture must contain fifty frames')
    valid = sum(r.get('capture_valid') is True for r in rows)
    return dict(complete=code == 0 and len(rows) == len(selected), observations=len(rows), valid=valid,
                planned_observations=len(selected),trajectory_waypoints=count,capture_stride=stride,
                invalid=len(rows)-valid, head_target_deg=list(head),
                sha256=digest(path) if path.exists() else None, process_exit_code=code)


def completed_prefix(directory, plan, trajectory_path, assets):
    """Resume only a contiguous, hash-verified completed prefix; never append to partial rounds."""
    manifest=json.loads((directory/'session.json').read_text())
    if manifest['plan']!=plan or Path(manifest['trajectory']).resolve()!=trajectory_path.resolve():
        raise ValueError('Resume plan/trajectory differs from original session')
    for path,sha in manifest['asset_hashes'].items():
        if digest(Path(path))!=sha:raise ValueError('Resume asset changed: '+path)
    trajectory=validate_trajectory(trajectory_path,assets)
    count=len(trajectory['waypoints']);done=[];gap=False
    for i,head in enumerate(plan['head_angles_deg']):
        output=directory/f'head_{i:02}.jsonl'
        if not output.exists():
            gap=True
            if output.with_suffix('.completed.json').exists():raise ValueError('Resume data file missing')
            continue
        if gap:raise ValueError('Resume rounds are not a contiguous prefix')
        saved=json.loads(output.with_suffix('.completed.json').read_text())
        actual=summarize(output,count,head,saved['process_exit_code'],assets,plan.get('capture_stride',1))
        if not actual['complete'] or actual!=saved:raise ValueError('Resume incomplete/tampered round: '+str(output))
        metadata=json.loads(output.with_suffix('.session.json').read_text())
        if metadata['head_target_deg']!=head:raise ValueError('Resume head mismatch')
        if metadata.get('replay_direction','reverse')!=('forward' if plan.get('alternate_direction',False) and i%2 else 'reverse'):
            raise ValueError('Resume replay direction mismatch')
        for path,sha in metadata['asset_hashes'].items():
            if digest(Path(path))!=sha:raise ValueError('Resume trajectory/asset changed: '+path)
        if actual['valid']:load_capture([output],assets)
        done.append(actual)
    return done


@contextmanager
def connected(sdk_root, robot_ip):
    sdk = load_sdk(sdk_root)
    robot = connect_robot(sdk, robot_ip)
    try:
        if str(robot.get_sdk_version()) != '00040701' or str(robot.get_controller_version()) != '00040701':
            raise RuntimeError('SDK/controller must both be 00040701')
        for obj in (sdk.FXObjType.OBJ_ARM1, sdk.FXObjType.OBJ_HEAD):
            if sdk.state_map.get(robot.current_state(obj)) != 'IDLE':
                raise RuntimeError('Right arm and head must be IDLE')
        yield robot, sdk
    finally:
        if robot._connected:
            robot.unlink()


def stop_receivers(latest):
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = (entry/'cmdline').read_bytes().split(b'\0')
            if any(a.endswith(b'orin_rectified_stream/viewer.py') for a in args):
                for i, arg in enumerate(args[:-1]):
                    if arg == b'--latest' and Path(os.fsdecode(args[i+1])).resolve() == latest.resolve():
                        os.kill(int(entry.name), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def start_receiver(latest, output, log_path):
    stop_receivers(latest)
    time.sleep(1)
    started = time.monotonic_ns()
    with log_path.open('a') as log:
        child = subprocess.Popen(['/usr/bin/python3', '-u', str(VIEWER), '--contract', str(CONTRACT),
            '--latest', str(latest), '--capture-log', str(output)], stdout=log, stderr=log, start_new_session=True)
    seen = set()
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError('Receiver exited; inspect viewer log')
        try:
            frame = json.loads(latest.read_text())
            _, stamp, _ = yellow_pixels(frame, time.monotonic_ns())
            if stamp > started:
                seen.add((stamp, str(frame.get('frame_number'))))
            if len(seen) >= 50:
                return child
        except (OSError, ValueError, KeyError, TypeError):
            pass
        time.sleep(.03)
    raise RuntimeError('No fifty fresh PASS frames; check Orin stream and ball visibility')


def run_child(command, log_path=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT/'acquisition'))
    # The vendor SDK may release stdin; acquire the controlling terminal explicitly.
    with open('/dev/tty', 'r') as terminal:
        if not terminal.isatty():
            raise RuntimeError('Interactive terminal required')
        log = log_path.open('a') if log_path else None
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=terminal,
                                     stdout=log, stderr=log)
            try:
                return child.wait()
            except BaseException:
                child.send_signal(signal.SIGINT)
                child.wait(timeout=20)
                raise
        finally:
            if log:
                log.close()


def capture_command(mode, trajectory, output, head, args):
    cmd = [sys.executable, '-u', '-m', 'interactive_ball_calibration.capture', mode,
        '--stereo', str(STEREO), '--mdh', str(MDH), '--sdk-root', str(args.sdk_root),
        '--robot-ip', args.robot_ip, '--latest', str(args.latest),
        '--head-target', *map(str, head), '--output', str(output)]
    if mode == 'replay':
        cmd += ['--path', str(trajectory), '--direct-approach', '--run',
                '--direction', getattr(args, 'direction', 'reverse'),
                '--capture-stride',str(getattr(args,'capture_stride',1))]
    return cmd


def model_check(trajectory, rt, head, destination, direction='reverse'):
    import mujoco
    import xml.etree.ElementTree as ET
    path = ROOT/'reference/LunaUdpTool/model/gento_luna/gento_luna_contact_filtered.xml'
    tree = ET.parse(path)
    tree.getroot().find('compiler').set('meshdir', str(path.parent/'meshes'))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))
    data = mujoco.MjData(model)
    def set_joints(names, values):
        for name, v in zip(names, values):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError('Missing model joint: '+name)
            data.qpos[model.jnt_qposadr[j]] = np.deg2rad(v)
    set_joints([f'Gento Luna_Leg_J{i}_Joint' for i in range(1,7)], rt['body']['fb_pos'])
    set_joints([f'Arm_L{i}_Joint' for i in range(1,8)], rt['arms'][0]['fb']['fb_pos'])
    set_joints(['Head_J1_joint','Head_J2_Joint'], head)
    taught = np.array([w['q7_deg'] for w in trajectory['waypoints']])
    if direction not in ('forward','reverse'):
        raise ValueError('Unknown replay direction')
    points = np.vstack([rt['arms'][1]['fb']['fb_pos'], taught[::-1] if direction == 'reverse' else taught])
    samples = 0
    for a,b in zip(points[:-1], points[1:]):
        n = max(1,int(np.ceil(np.max(abs(b-a)))))
        for fraction in np.linspace(0,1,n+1):
            set_joints([f'Arm_R{i}_Joint' for i in range(1,8)], a+(b-a)*fraction)
            mujoco.mj_forward(model,data)
            samples += 1
            for contact in data.contact[:data.ncon]:
                names = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or 'world'
                         for g in (contact.geom1,contact.geom2)]
                if contact.dist < 0 and any(name.startswith('Arm_R') for name in names):
                    raise RuntimeError('Nominal model right-arm collision: '+str(names))
    write_json(destination, dict(samples=samples, head_target_deg=list(head), replay_direction=direction, model_sha256=digest(path),
               feedback=rt, note='Nominal CAD only; table, ball rod and people are not modeled.'))


def prepare(args, trajectory, body_ref, left_ref, expected_right, head, report):
    with connected(args.sdk_root, args.robot_ip) as (robot, sdk):
        rt = read_rt(robot)
        for value, reference in ((rt['body']['fb_pos'], body_ref),
                                 (rt['arms'][0]['fb']['fb_pos'], left_ref),
                                 (rt['arms'][1]['fb']['fb_pos'], expected_right)):
            if np.max(abs(np.array(value)-reference)) > .3:
                raise RuntimeError('Robot changed from expected session pose')
        validate_path(trajectory['waypoints'], arm_limits(robot))
        limits = np.array(read_limits(robot))
        if np.any(head < limits[:,0]) or np.any(head > limits[:,1]):
            raise ValueError('Head target exceeds live limits')
        model_check(trajectory, rt, head, report, getattr(args,'direction','reverse'))
        obj = sdk.FXObjType.OBJ_HEAD
        try:
            current = head_angles(rt)
            for point in head_reference_waypoints(current[:2], head, limits):
                if sdk.state_map.get(robot.current_state(obj)) == 'IDLE':
                    checked(robot.switch_to_position_mode(obj,2000,5,5), 'Head Position')
                checked(robot.runtime_set_joint_pos_cmd(obj,[*point.tolist(),float(current[2])]), 'Head target')
                wait_head(robot,point)
        finally:
            checked(robot.switch_to_idle(obj,2000), 'Head IDLE')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, help='New session name; never overwritten')
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('plans')/'training.json')
    parser.add_argument('--trajectory', type=Path, help='Complete existing trajectory; otherwise teach first')
    parser.add_argument('--sdk-root', type=Path, default=ROOT.parent/'00040701')
    parser.add_argument('--robot-ip', default='6.6.7.190')
    parser.add_argument('--latest', type=Path, default=LATEST)
    parser.add_argument('--run', action='store_true', help='Authorize teaching and all plan replay rounds')
    parser.add_argument('--resume',action='store_true',help='Continue after verified completed rounds; refuse partial rounds')
    parser.add_argument('--continue-low-valid',action='store_true',help='Keep rejected records and continue after a complete round below80%% valid')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.session):
        parser.error('Session must contain only letters, digits, underscore or dash')
    args.sdk_root = args.sdk_root.resolve()
    args.latest = args.latest.resolve()
    args.plan = args.plan.resolve()
    assets = Assets()
    if not np.array_equal(assets.pf, [0,0,.101]):
        raise ValueError('This workflow requires the precision-measured pF=[0,0,101]mm')
    plan, heads = read_plan(args.plan)
    args.capture_stride = plan.get('capture_stride',1)
    directory = ROOT/'datasets/raw'/args.session
    if directory.exists() and not args.resume:
        raise FileExistsError('Session exists; choose a new session name')
    if args.resume and (not directory.is_dir() or not args.trajectory):
        parser.error('--resume requires an existing session and explicit --trajectory')
    if args.trajectory:
        args.trajectory = args.trajectory.resolve()
        existing = validate_trajectory(args.trajectory, assets)
    done=completed_prefix(directory,plan,args.trajectory,assets) if args.resume else []
    if not args.continue_low_valid and any(s['valid']/s['planned_observations']<.8 for s in done):
        raise RuntimeError('Completed round below80%; explicit --continue-low-valid required')
    if not args.run:
        print(json.dumps(dict(session=str(directory), heads=heads.tolist(), frames_per_point=50,
                              positions_per_head=len(existing['waypoints']) if args.trajectory else None,
                              capture_stride=args.capture_stride,
                              measurement_stops_per_head=len(capture_indices(len(existing['waypoints']),args.capture_stride)) if args.trajectory else None,
                              alternate_direction=plan.get('alternate_direction',False),
                              completed_rounds=len(done),continue_low_valid=args.continue_low_valid,
                              teach_first=args.trajectory is None, motion=False), indent=2))
        return
    runtime = ROOT/'acquisition/runtime'
    runtime.mkdir(exist_ok=True)
    with (runtime/'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume:directory.mkdir()
        status = directory/'status.json'
        trajectory_path = args.trajectory or directory/'trajectory.json'
        hashes = {str(p):digest(p) for p in (args.plan,STEREO,MDH,CONTRACT)}
        if args.resume:
            # Revalidate under the collection lock before any motion.
            done=completed_prefix(directory,plan,trajectory_path,assets)
            saved_manifest=json.loads((directory/'session.json').read_text())
            if saved_manifest['robot_ip']!=args.robot_ip:raise ValueError('Resume robot IP mismatch')
            write_json(directory/'resume.json',dict(completed_rounds=len(done),continue_low_valid=args.continue_low_valid))
        else:
            write_json(directory/'session.json',dict(plan=plan, asset_hashes=hashes, measured_pf_m=[0,0,.101],
                       sdk_speed_percent=5, trajectory=str(trajectory_path), robot_ip=args.robot_ip))
        try:
            with connected(args.sdk_root,args.robot_ip) as (robot,sdk):
                initial = read_rt(robot)
                live_limits = np.array(read_limits(robot))
                if np.any(heads < live_limits[:,0]) or np.any(heads > live_limits[:,1]):
                    raise ValueError('Plan exceeds live head limits')
            if args.resume:
                original=json.loads((directory/'initial_feedback.json').read_text())
                if done:
                    last_index=0 if not plan.get('alternate_direction',False) or (len(done)-1)%2==0 else -1
                    expected=existing['waypoints'][last_index]['q7_deg']
                else:expected=original['arms'][1]['fb']['fb_pos']
                for value,reference in ((initial['body']['fb_pos'],original['body']['fb_pos']),
                                        (initial['arms'][0]['fb']['fb_pos'],original['arms'][0]['fb']['fb_pos']),
                                        (initial['arms'][1]['fb']['fb_pos'],expected)):
                    if np.max(abs(np.array(value)-reference))>.3:raise RuntimeError('Resume robot pose changed; refusing automatic motion')
            else:write_json(directory/'initial_feedback.json',initial)
            # Import before any teaching/motion so a missing model dependency stops early.
            import mujoco
            start_receiver(args.latest,directory/'head_00.jsonl',directory/'viewer.log')
            if args.trajectory is None:
                write_json(status,dict(phase='teaching',trajectory=str(trajectory_path)))
                print('Drag slowly; q + Enter finishes teaching. Replay begins automatically after validation.',flush=True)
                code = run_child(capture_command('teach',None,trajectory_path,[0,-25],args))
                if code != 0:
                    raise RuntimeError('Teaching ended unsuccessfully; partial trajectory will not be replayed')
            trajectory = validate_trajectory(trajectory_path,assets)
            count = len(trajectory['waypoints'])
            selected = capture_indices(count,args.capture_stride)
            with connected(args.sdk_root,args.robot_ip) as (robot,sdk):
                rt = read_rt(robot)
            if np.max(abs(np.array(rt['body']['fb_pos'])-trajectory['body_q_deg'])) > .3:
                raise RuntimeError('Body differs from teaching session')
            body_ref = rt['body']['fb_pos']
            left_ref = rt['arms'][0]['fb']['fb_pos']
            expected_right = rt['arms'][1]['fb']['fb_pos']
            hashes[str(trajectory_path)] = digest(trajectory_path)
            for i,head in enumerate(heads):
                if i<len(done):continue
                args.direction = 'forward' if plan.get('alternate_direction',False) and i % 2 else 'reverse'
                if any(digest(Path(p)) != h for p,h in hashes.items()):
                    raise RuntimeError('Session assets changed')
                output = directory/f'head_{i:02}.jsonl'
                write_json(status,dict(phase='preparing',round=i+1,total_rounds=len(heads),head=head.tolist(),direction=args.direction))
                prepare(args,trajectory,body_ref,left_ref,expected_right,head,directory/f'model_check_{i:02}.json')
                start_receiver(args.latest,output,directory/'viewer.log')
                write_json(output.with_suffix('.session.json'),dict(head_target_deg=head.tolist(),
                           purpose=plan.get('purpose','development'),trajectory=str(trajectory_path),
                           replay_direction=args.direction,trajectory_id=trajectory.get('trajectory_id'),
                           capture_stride=args.capture_stride,capture_waypoint_indices=selected,planned_observations=len(selected),
                           asset_hashes=hashes,total_waypoints=count,frames_per_point=50,measured_pf_m=[0,0,.101]))
                write_json(status,dict(phase='replay',round=i+1,total_rounds=len(heads),head=head.tolist(),
                           direction=args.direction,
                           planned_observations=len(selected),
                           output=str(output),total_waypoints=count))
                print(f'Round {i+1}/{len(heads)} head={head.tolist()}, {len(selected)} stops over {count} waypoints, {args.direction}',flush=True)
                code = -1
                try:
                    code = run_child(capture_command('replay',trajectory_path,output,head,args),directory/f'capture_{i:02}.log')
                finally:
                    summary = summarize(output,count,head,code,assets,args.capture_stride)
                    write_json(output.with_suffix('.completed.json'),summary)
                if not summary['complete']:
                    raise RuntimeError('Replay incomplete; later rounds cancelled')
                if summary['valid']:
                    load_capture([output], assets)
                print(json.dumps(summary),flush=True)
                if summary['valid'] / len(selected) < .8:
                    if not args.continue_low_valid:
                        raise RuntimeError('Less than 80% valid; fix visibility/quality before further rounds')
                    write_json(output.with_suffix('.warning.json'),dict(reason='Below80% valid; explicitly continuing',valid=summary['valid'],invalid=summary['invalid']))
                expected_right = trajectory['waypoints'][0 if args.direction == 'reverse' else -1]['q7_deg']
            write_json(status,dict(phase='complete',rounds=len(heads),trajectory=str(trajectory_path)))
        except BaseException as exc:
            write_json(status,dict(phase='stopped',reason=str(exc)))
            raise


if __name__ == '__main__':
    main()
