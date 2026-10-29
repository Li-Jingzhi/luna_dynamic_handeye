"""Collect the operator-taught independent validation trajectory at four heads."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/validationB_capture_20261029'
sys.path.insert(0, str(ROOT / 'acquisition'))
spec = importlib.util.spec_from_file_location('previous_batch', ROOT / 'reports/two_head_replay_20261028/run_batch.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
TRAJECTORY = ROOT / 'datasets/raw/luna_v2_validationB_taught_20261029.json'
HEADS = ([0, -25], [0, -20], [10, -25], [10, -20])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop_project_receivers():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            if (any(a.endswith(b'orin_rectified_stream/viewer.py') for a in args)
                    and str(previous.LATEST).encode() in args):
                os.kill(int(entry.name), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def feedback():
    sdk = previous.load_sdk(ROOT.parent / '00040701')
    robot = previous.connect_robot(sdk, '6.6.7.190')
    try:
        for obj in (sdk.FXObjType.OBJ_ARM1, sdk.FXObjType.OBJ_HEAD):
            if sdk.state_map.get(robot.current_state(obj)) != 'IDLE':
                raise RuntimeError('Arm/head must be IDLE between rounds')
        return previous.read_rt(robot)
    finally:
        robot.unlink()


def main():
    os.chdir(ROOT)
    path = json.loads(TRAJECTORY.read_text())
    count = len(path['waypoints'])
    assert path['complete'] is True and count >= 18
    assert path['purpose'] == 'independent_validation_only'
    assert path['measured_flange_to_ball_center_m'] == [0, 0, .101]
    outputs = [ROOT / f'datasets/raw/luna_v2_validationB_yaw{h[0]}_pitchm{abs(h[1])}_20261029.jsonl' for h in HEADS]
    assert len(set(outputs)) == len(HEADS)
    assert all(not p.exists() for p in outputs), 'Refuse to overwrite capture'
    if '--check' in sys.argv:
        print(json.dumps(dict(waypoints=count, heads=HEADS, outputs=list(map(str, outputs)), motion=False)))
        return
    assert sys.stdin.isatty(), 'Interactive terminal required'
    initial = feedback()
    hashes = {str(p): digest(p) for p in (TRAJECTORY, previous.CONTRACT,
              ROOT / 'configs/camera/luna_rectified_stereo.json', ROOT / 'configs/robot/luna_new.ini')}
    previous.TRAJECTORY = TRAJECTORY
    viewer_pid = 30290
    status = OUT / 'batch_status.json'
    try:
        for index, (head, output) in enumerate(zip(HEADS, outputs)):
            assert all(digest(Path(p)) == h for p, h in hashes.items()), 'Capture assets changed'
            rt = feedback()
            for a, b in ((rt['body']['fb_pos'], path['body_q_deg']),
                         (rt['arms'][0]['fb']['fb_pos'], initial['arms'][0]['fb']['fb_pos'])):
                assert np.max(abs(np.asarray(a) - b)) <= .3, 'Body/left arm changed'
            expected = initial['arms'][1]['fb']['fb_pos'] if index == 0 else path['waypoints'][0]['q7_deg']
            assert np.max(abs(np.asarray(rt['arms'][1]['fb']['fb_pos']) - expected)) <= .3, 'Arm start changed'
            previous.SNAPSHOT = dict(feedback=rt)
            previous.write_json(status, dict(phase='preparing', round=index + 1, head=head, output=str(output)))
            previous.prepare(head)
            stop_project_receivers()
            time.sleep(1)
            receiver_start = time.monotonic_ns()
            with (OUT / f'viewer_{index}.log').open('a') as log:
                viewer = subprocess.Popen(['/usr/bin/python3', '-u', str(ROOT / 'acquisition/orin_rectified_stream/viewer.py'),
                    '--contract', str(previous.CONTRACT), '--latest', str(previous.LATEST),
                    '--capture-log', str(output)], stdout=log, stderr=log, start_new_session=True)
            viewer_pid = viewer.pid
            seen = set()
            deadline = time.monotonic() + 45
            while len(seen) < 50 and time.monotonic() < deadline:
                if viewer.poll() is not None:
                    raise RuntimeError('Preview/measurement receiver exited')
                try:
                    frame = json.loads(previous.LATEST.read_text())
                    _, stamp, _ = previous.yellow_pixels(frame, time.monotonic_ns())
                    if stamp > receiver_start:
                        seen.add((stamp, str(frame.get('frame_number'))))
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                time.sleep(.03)
            if len(seen) < 50:
                raise RuntimeError('Receiver did not deliver 50 fresh PASS frames before replay')
            previous.write_json(output.with_suffix('.session.json'), dict(purpose='independent_validation_only',
                head_target_deg=head, trajectory=str(TRAJECTORY), total_waypoints=count,
                frames_per_point=50, sdk_speed_percent=5, measured_flange_to_ball_center_m=[0, 0, .101],
                asset_hashes=hashes, approach_mode='direct_joint_interpolation_max5deg'))
            cmd = ['/usr/bin/python3', '-u', '-m', 'interactive_ball_calibration.capture', 'replay',
                   '--path', str(TRAJECTORY), '--stereo', str(ROOT / 'configs/camera/luna_rectified_stereo.json'),
                   '--mdh', str(ROOT / 'configs/robot/luna_new.ini'), '--sdk-root', str(ROOT.parent / '00040701'),
                   '--latest', str(previous.LATEST), '--head-target', *map(str, head), '--output', str(output),
                   '--direct-approach', '--run']
            env = dict(os.environ, PYTHONPATH=str(ROOT / 'acquisition'))
            with (OUT / f'capture_{index}.log').open('a') as log, open('/dev/tty', 'r') as terminal:
                assert terminal.isatty(), 'Cannot acquire replay terminal'
                child = subprocess.Popen(cmd, stdin=terminal, stdout=log, stderr=log, env=env)
                previous.write_json(status, dict(phase='replay', round=index + 1, head=head,
                    output=str(output), collector_pid=child.pid, viewer_pid=viewer_pid, total_waypoints=count))
                print(f'ROUND {index + 1}/4 head={head}, {count} points, collector={child.pid}', flush=True)
                try:
                    code = child.wait()
                except BaseException:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=20)
                    raise
            rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
            complete = code == 0 and len(rows) == count
            assert all(r.get('target_head_deg') == head for r in rows)
            assert len({r['waypoint_index'] for r in rows}) == len(rows)
            summary = dict(complete=complete, observations=len(rows),
                           valid=sum(r.get('capture_valid') is True for r in rows),
                           invalid=sum(r.get('capture_valid') is not True for r in rows),
                           head_target_deg=head, purpose='independent_validation_only',
                           sha256=digest(output) if output.exists() else None)
            previous.write_json(output.with_suffix('.completed.json'), summary)
            print('ROUND FINISHED ' + json.dumps(summary), flush=True)
            if not complete:
                raise RuntimeError('Round incomplete; remaining rounds cancelled')
        previous.write_json(status, dict(phase='complete', outputs=list(map(str, outputs))))
    except BaseException as exc:
        previous.write_json(status, dict(phase='stopped', reason=str(exc)))
        raise


if __name__ == '__main__':
    main()
