"""Execute the two head poses explicitly requested on 2026-10-28."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acquisition'))
from capt_head.drag_capture import load_sdk,yellow_pixels
from interactive_ball_calibration.capture import connect_robot,read_rt,arm_limits,validate_path,head_reference_waypoints
from gq_calibration.capture import checked,read_limits,head_angles,wait_head

TRAJECTORY=ROOT/'datasets/raw/luna_v2_transferred_marvin360_20261028.json'
LATEST=ROOT/'acquisition/runtime/new_sh5_yellow/latest_pose.json'
CONTRACT=ROOT/'acquisition/configs/yellow_measurement_v1.json'
HEADS=([10,-20],[10,-25])
SNAPSHOT=json.loads((OUT/'controller_snapshot.json').read_text())
STATUS=OUT/'batch_status.json'


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def prepare(head):
    sdk=load_sdk(ROOT.parent/'00040701');robot=connect_robot(sdk,'6.6.7.190')
    obj=sdk.FXObjType.OBJ_HEAD
    try:
        assert str(robot.get_sdk_version())=='00040701' and str(robot.get_controller_version())=='00040701'
        assert sdk.state_map.get(robot.current_state(sdk.FXObjType.OBJ_ARM1))=='IDLE','Arm not IDLE'
        assert sdk.state_map.get(robot.current_state(obj))=='IDLE','Head not IDLE'
        rt=read_rt(robot);previous=SNAPSHOT['feedback']
        for a,b in [(rt['body']['fb_pos'],previous['body']['fb_pos']),
                    (rt['arms'][0]['fb']['fb_pos'],previous['arms'][0]['fb']['fb_pos']),
                    (rt['arms'][1]['fb']['fb_pos'],previous['arms'][1]['fb']['fb_pos'])]:
            assert np.max(np.abs(np.asarray(a)-b))<=.3,'Robot changed since approach model check'
        path=json.loads(TRAJECTORY.read_text());validate_path(path['waypoints'],arm_limits(robot))
        limits=np.asarray(read_limits(robot));target=np.asarray(head,float)
        assert np.all(target>=limits[:,0]) and np.all(target<=limits[:,1])
        current=head_angles(rt)
        for point in head_reference_waypoints(current[:2],target,limits):
            if sdk.state_map.get(robot.current_state(obj))=='IDLE':
                checked(robot.switch_to_position_mode(obj,2000,5,5),'Head Position')
            checked(robot.runtime_set_joint_pos_cmd(obj,[*point.tolist(),float(current[2])]),'Requested batch head')
            wait_head(robot,point)
        actual=head_angles(read_rt(robot))[:2];assert np.max(abs(actual-target))<=.3
    finally:
        if robot._connected:
            robot.switch_to_idle(obj,2000);robot.unlink()
    minimum=time.monotonic_ns();deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        try:
            frame=json.loads(LATEST.read_text());yellow_pixels(frame,time.monotonic_ns())
            if frame['camera_capture_timestamp_monotonic_ns']>minimum:return
        except (OSError,ValueError,KeyError,TypeError):pass
        time.sleep(.05)
    raise RuntimeError(f'No fresh valid ball observation at head {head}')


def stop_viewer(pid):
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'orin_rectified_stream/viewer.py' in command:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
    except FileNotFoundError:pass


def main():
    os.chdir(ROOT)
    paths=[ROOT/f'datasets/raw/luna_v2_transferred_marvin360_yaw10_pitchm{abs(head[1])}_20261028.jsonl' for head in HEADS]
    assert all(not path.exists() for path in paths),'Refuse to overwrite capture'
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [TRAJECTORY,CONTRACT,ROOT/'configs/camera/luna_rectified_stereo.json',ROOT/'configs/robot/luna_new.ini']}
    viewer_pid=1353501
    try:
        for index,(head,path) in enumerate(zip(HEADS,paths)):
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'Capture assets changed during batch'
            write_json(STATUS,dict(phase='preparing',round=index+1,head_target_deg=head,output=str(path)))
            prepare(head)
            stop_viewer(viewer_pid);time.sleep(1)
            with (OUT/f'viewer_pitchm{abs(head[1])}.log').open('w') as log:
                viewer=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'acquisition/orin_rectified_stream/viewer.py'),'--contract',str(CONTRACT),'--latest',str(LATEST),'--capture-log',str(path)],stdout=log,stderr=log,start_new_session=True)
            viewer_pid=viewer.pid
            write_json(path.with_suffix('.session.json'),dict(head_target_deg=head,trajectory=str(TRAJECTORY),total_waypoints=360,frames_per_point=50,sdk_speed_percent=5,approach_mode='direct_joint_interpolation_max5deg',asset_hashes=hashes))
            cmd=['/usr/bin/python3','-u','-m','interactive_ball_calibration.capture','replay','--path',str(TRAJECTORY),'--stereo',str(ROOT/'configs/camera/luna_rectified_stereo.json'),'--mdh',str(ROOT/'configs/robot/luna_new.ini'),'--sdk-root',str(ROOT.parent/'00040701'),'--latest',str(LATEST),'--head-target',*map(str,head),'--output',str(path),'--direct-approach','--run']
            assert cmd[cmd.index('--head-target')+1:cmd.index('--head-target')+3]==list(map(str,head))
            env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'acquisition')
            write_json(STATUS,dict(phase='replay',round=index+1,head_target_deg=head,output=str(path),viewer_pid=viewer_pid))
            with (OUT/f'capture_pitchm{abs(head[1])}.log').open('x') as log:
                child=subprocess.Popen(cmd,stdin=sys.stdin,stdout=log,stderr=log,env=env)
                try:code=child.wait()
                except BaseException:
                    child.send_signal(signal.SIGINT);child.wait(timeout=20);raise
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
            complete=code==0 and len(rows)==360
            assert all(r.get('target_head_deg')==head for r in rows),'Wrong head metadata'
            summary=dict(complete=complete,observations=len(rows),valid=sum(r.get('capture_valid') is True for r in rows),invalid=sum(r.get('capture_valid') is not True for r in rows),head_target_deg=head,sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
            write_json(path.with_suffix('.completed.json'),summary)
            print('ROUND FINISHED',json.dumps(summary),flush=True)
            if not complete:raise RuntimeError(f'Round at {head} did not complete; next round cancelled')
            time.sleep(2)
        write_json(STATUS,dict(phase='complete',outputs=list(map(str,paths))))
    except BaseException as error:
        write_json(STATUS,dict(phase='stopped',reason=str(error)))
        raise


if __name__=='__main__':main()
