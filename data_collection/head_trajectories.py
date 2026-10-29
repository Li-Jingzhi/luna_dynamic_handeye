"""Teach a different visible arm trajectory at each head setting; replay right-arm paths by head."""
import argparse
import configparser
import fcntl
import json
from pathlib import Path
import re
import select
import sys
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'acquisition'))
from data_collection import workflow as w
from capt_head.drag_capture import load_sdk, JOINT_K, JOINT_D, yellow_pixels
from interactive_ball_calibration.capture import connect_robot, read_rt, validate_path, check_q, head_reference_waypoints
from gq_calibration.capture import checked, head_angles, read_limits, wait_head
from dynamic_handeye.geometry import Assets

GRID = [(yaw,pitch) for pitch in (-25,-20,-15) for yaw in (-10,0,10)]


def token(value):
    value=float(value)
    if not np.isfinite(value):
        raise ValueError('Nonfinite angle')
    return ('p' if value>0 else 'm' if value<0 else 'z')+f'{abs(value):g}'.replace('.','p')


def filename(arm, head):
    if arm not in ('left','right'):
        raise ValueError('Unknown arm')
    return f'luna_v2_{arm}_yaw_{token(head[0])}_pitch_{token(head[1])}.json'


def arm_index(arm):
    return 1 if arm=='right' else 0


def limits(robot, index):
    result=[]
    for i in range(7):
        pair=[]
        for key in ('LimitNeg','LimitPos'):
            ret,value=robot.param_get_float(f'R.A{index}.L{i}.BASIC.{key}')
            if ret or not np.isfinite(value):
                raise RuntimeError('Cannot read online arm limit')
            pair.append(float(value))
        result.append(pair)
    result=np.array(result)
    if np.any(result[:,0]>=result[:,1]):
        raise ValueError('Invalid online limits')
    return result


def robot_identity(robot):
    ret,sub=robot.param_get_string('R.BASIC.SubName')
    if ret:
        raise RuntimeError('Cannot read robot submodel')
    return dict(robot_type=robot.get_robot_type(),submodel=str(sub),sdk_version=str(robot.get_sdk_version()))


def verify_live_mdh(robot,index):
    ini=configparser.ConfigParser();ini.read(w.MDH)
    sections=[*(f'R.A{index}.L{i}.DH' for i in range(7)),f'R.A{index}.FLANGE']
    for section in sections:
        for key in ('A','Alpha','D','Theta'):
            ret,value=robot.param_get_float(f'{section}.{key}')
            if ret or not np.isfinite(value) or abs(float(value)-float(ini[section][key]))>1e-3:
                raise RuntimeError(f'Live MDH differs or is unreadable: {section}.{key}')


def validate_record(record,arm,head,assets,identity=None):
    if (record.get('schema')!='luna_sh5_taught_path_v1' or record.get('complete') is not True
            or record.get('arm_side')!=arm or record.get('mdh_sha256')!=assets.hashes['mdh']
            or record.get('robot_version')!='luna_v2_0'
            or record.get('target_head_deg')!=list(head) or not record.get('trajectory_id')):
        raise ValueError('Incompatible/incomplete head-specific trajectory')
    if identity is not None and record.get('robot_identity')!=identity:
        raise ValueError('Robot model/SDK differs; offline trajectory migration is required')
    validate_path(record['waypoints'],np.tile([-360.,360.],(7,1)))


def visible(latest):
    try:
        r=json.loads(latest.read_text());yellow_pixels(r,time.monotonic_ns())
        return dict(pass_detected=True,frame_number=r['frame_number'],pixels=r['corrected_stereo_pixels'])
    except (OSError,ValueError,KeyError,TypeError):
        return dict(pass_detected=False)


def assert_stationary_others(rt,index,head,body,other):
    if (np.max(abs(np.asarray(rt['head']['fb_pos'][:2])-head))>.25
            or np.max(abs(np.asarray(rt['body']['fb_pos'])-body))>.3
            or np.max(abs(np.asarray(rt['arms'][1-index]['fb']['fb_pos'])-other))>.3):
        raise RuntimeError('Head/body/opposite arm moved during teaching')


def teach_one(robot,sdk,index,head,output,assets,identity,terminal,latest,arm_limits):
    arm=sdk.FXObjType.OBJ_ARM1 if index==1 else sdk.FXObjType.OBJ_ARM0
    rt=read_rt(robot);body=np.array(rt['body']['fb_pos']);other=np.array(rt['arms'][1-index]['fb']['fb_pos'])
    waypoints=[];tid=uuid.uuid4().hex;side='right' if index==1 else 'left'
    partial=Path(str(output)+'.partial')
    def record(complete):
        return dict(schema='luna_sh5_taught_path_v1',complete=complete,robot_version='luna_v2_0',
            robot_identity=identity,trajectory_id=tid,arm_side=side,sdk_version='00040701',
            mdh_sha256=assets.hashes['mdh'],stereo_model_sha256=assets.hashes['stereo'],
            source_calibration_sha256=assets.hashes['raw'],head_angles_deg=head_angles(rt)[:2].tolist(),
            target_head_deg=list(head),body_q_deg=body.tolist(),opposite_arm_q_deg=other.tolist(),
            image_size=[1280,992],sensor_ids=assets.stereo['sensor_ids'],pixel_domain='ideal_pinhole',
            recording_mode='head_specific_continuous_drag_3deg',waypoints=waypoints,
            visible_waypoints=sum(p['visibility']['pass_detected'] for p in waypoints),
            physical_group_rule='trajectory_id + waypoint_index; never match indices across independent trajectories')
    checked(robot.switch_to_drag_joint(arm,2000,JOINT_K,JOINT_D),'Selected arm DragJoint')
    print(f'开始{side}臂示教，头姿{list(head)}。保持球可见、缓慢拖动。q回车保存；Ctrl+C暂停整个九姿任务。',flush=True)
    try:
        while True:
            if select.select([terminal],[],[],0)[0]:
                line=terminal.readline()
                if not line:
                    raise KeyboardInterrupt('Teaching terminal closed')
                command=line.strip().lower()
                if command=='q':
                    try:
                        validate_path(waypoints,arm_limits)
                    except ValueError as exc:
                        print('暂不能保存：'+str(exc),flush=True)
                    else:
                        break
            if sdk.state_map.get(robot.current_state(arm))!='DragJoint':
                raise RuntimeError('Selected arm left DragJoint')
            now=read_rt(robot);assert_stationary_others(now,index,np.array(head),body,other)
            q=np.array(now['arms'][index]['fb']['fb_pos'],float)
            check_q(q,arm_limits)
            delta=np.max(abs(q-np.array(waypoints[-1]['q7_deg']))) if waypoints else 3.
            if delta>5:
                raise RuntimeError('Sampling gap exceeds5deg; incomplete path preserved, drag more slowly')
            if delta>=3:
                sight=visible(latest)
                waypoints.append(dict(q7_deg=q.tolist(),head_angles_deg=head_angles(now)[:2].tolist(),
                    body_q_deg=body.tolist(),visibility=sight))
                w.write_json(partial,record(False))
                print(f'[{side} yaw={head[0]} pitch={head[1]}] 点数={len(waypoints)} 球检测={"PASS" if sight["pass_detected"] else "不可见/不合格"}',flush=True)
            time.sleep(.02)
        if output.exists():
            raise FileExistsError('Refuse to overwrite trajectory')
        w.write_json(output,record(True))
        partial.unlink(missing_ok=True)
        print(f'已保存{len(waypoints)}点：{output}',flush=True)
    finally:
        checked(robot.switch_to_idle(arm,2000),'Selected arm IDLE')


def set_head(robot,sdk,head,rt):
    limits=np.array(read_limits(robot));q=np.array(head,float)
    if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):
        raise ValueError('Head target exceeds online limits')
    obj=sdk.FXObjType.OBJ_HEAD
    try:
        current=head_angles(rt)
        for point in head_reference_waypoints(current[:2],q,limits):
            if sdk.state_map.get(robot.current_state(obj))=='IDLE':
                checked(robot.switch_to_position_mode(obj,2000,5,5),'Head Position')
            checked(robot.runtime_set_joint_pos_cmd(obj,[*point.tolist(),float(current[2])]),'Head target')
            wait_head(robot,point)
    finally:
        checked(robot.switch_to_idle(obj,2000),'Head IDLE')


def replay_grid(args,directory,assets):
    if args.arm!='right':
        raise ValueError('Current fifty-frame replay adapter is right-arm only; left paths are joint-teaching only')
    records=[]
    for head in args.heads:
        p=directory/filename(args.arm,head);r=json.loads(p.read_text());validate_record(r,args.arm,head,assets)
        records.append((head,p,r))
    output_dir=ROOT/'datasets/raw'/f'{args.session}_replay_{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True,exist_ok=False)
    state=output_dir/'status.json'
    try:
        for i,(head,path,r) in enumerate(records):
            with w.connected(args.sdk_root,args.robot_ip) as (robot,sdk):
                validate_record(r,args.arm,head,assets,robot_identity(robot));verify_live_mdh(robot,1)
                rt=read_rt(robot)
                # Reusing a joint path on a compatible unit binds this execution to its current fixed body.
                execution=dict(r,body_q_deg=rt['body']['fb_pos'],stereo_model_sha256=assets.hashes['stereo'],
                    source_trajectory=str(path),source_trajectory_sha256=w.digest(path),
                    execution_binding='current robot/camera; original q7 unchanged')
            bound=output_dir/f'execution_{i:02}.json';w.write_json(bound,execution)
            output=output_dir/filename(args.arm,head).replace('.json','.jsonl')
            w.write_json(state,dict(phase='preparing',round=i+1,head=head,output=str(output)))
            w.prepare(args,execution,rt['body']['fb_pos'],rt['arms'][0]['fb']['fb_pos'],
                      rt['arms'][1]['fb']['fb_pos'],np.array(head),output_dir/f'model_check_{i:02}.json')
            w.start_receiver(args.latest,output,output_dir/'viewer.log')
            code=-1
            try:
                code=w.run_child(w.capture_command('replay',bound,output,head,args),output_dir/f'capture_{i:02}.log')
            finally:
                summary=w.summarize(output,len(r['waypoints']),head,code,assets)
                summary.update(trajectory_id=r['trajectory_id'],source_trajectory=str(path))
                w.write_json(output.with_suffix('.completed.json'),summary)
            if not summary['complete'] or summary['valid']/len(r['waypoints'])<.8:
                raise RuntimeError('Incomplete/low-visibility round; following heads cancelled')
            w.load_capture([output],assets)
        w.write_json(state,dict(phase='complete',rounds=len(records)))
    except BaseException as exc:
        w.write_json(state,dict(phase='stopped',reason=str(exc)));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('teach','replay'))
    parser.add_argument('--arm',choices=('right','left'),default='right')
    parser.add_argument('--session',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--head',nargs=2,type=float,metavar=('YAW','PITCH'),help='Teach/replay only this head pose')
    parser.add_argument('--robot-ip',default='6.6.7.190')
    parser.add_argument('--sdk-root',type=Path,default=ROOT.parent/'00040701')
    parser.add_argument('--latest',type=Path,default=w.LATEST)
    args=parser.parse_args(argv)
    args.heads=[tuple(args.head)] if args.head is not None else GRID
    if not np.isfinite(args.heads).all() or np.any(abs(np.asarray(args.heads))>[90,25]):
        parser.error('Head angles exceed supported model limits')
    if not re.fullmatch('[A-Za-z0-9_-]+',args.session):
        parser.error('Invalid session name')
    args.sdk_root=args.sdk_root.resolve();args.latest=args.latest.resolve()
    directory=ROOT/'trajectories/by_head_pose'/args.session
    assets=Assets()
    if not args.run:
        print(json.dumps(dict(mode=args.mode,arm=args.arm,heads=args.heads,directory=str(directory),
            files=[filename(args.arm,h) for h in args.heads],motion=False),indent=2));return
    runtime=ROOT/'acquisition/runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'collection.lock').open('a') as lock,open('/dev/tty','r') as terminal:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='replay':
            replay_grid(args,directory,assets);return
        if directory.exists() and not args.resume:
            raise FileExistsError('Session exists; use --resume to skip completed head trajectories')
        directory.mkdir(parents=True,exist_ok=True)
        status=directory/'status.json'
        # Receiver is independent of dragging; do not require the ball to be visible before entering the view.
        w.stop_receivers(args.latest);time.sleep(1)
        with (directory/'viewer.log').open('a') as log:
            import subprocess
            subprocess.Popen(['/usr/bin/python3','-u',str(w.VIEWER),'--contract',str(w.CONTRACT),
                '--latest',str(args.latest)],stdout=log,stderr=log,start_new_session=True)
        sdk=load_sdk(args.sdk_root);robot=connect_robot(sdk,args.robot_ip)
        selected=sdk.FXObjType.OBJ_ARM1 if args.arm=='right' else sdk.FXObjType.OBJ_ARM0
        try:
            if str(robot.get_sdk_version())!='00040701' or str(robot.get_controller_version())!='00040701':
                raise RuntimeError('Expected SDK/controller00040701')
            for obj in (sdk.FXObjType.OBJ_ARM0,sdk.FXObjType.OBJ_ARM1,sdk.FXObjType.OBJ_HEAD):
                if sdk.state_map.get(robot.current_state(obj))!='IDLE':
                    raise RuntimeError('Both arms and head must be IDLE')
            identity=robot_identity(robot)
            if identity['robot_type']!='GentoLuna':
                raise ValueError('Offline migration needed for another robot model')
            index=arm_index(args.arm);live=limits(robot,index);verify_live_mdh(robot,index)
            body=np.array(read_rt(robot)['body']['fb_pos'])
            for i,head in enumerate(args.heads):
                output=directory/filename(args.arm,head)
                if output.exists():
                    validate_record(json.loads(output.read_text()),args.arm,head,assets,identity)
                    print('跳过已完成：'+output.name,flush=True);continue
                if Path(str(output)+'.partial').exists():
                    raise RuntimeError('Incomplete path exists; preserve/review it before teaching this head again')
                w.write_json(status,dict(phase='waiting_operator',round=i+1,total=len(args.heads),head=head,output=str(output)))
                print(f'\n第{i+1}/{len(args.heads)}头姿{head}，机械臂停稳、保持头部转动区域通畅。',flush=True)
                if i:
                    print('按Enter移动到下一头姿并开始示教；输入stop退出。',flush=True)
                    line=terminal.readline()
                    if not line or line.strip().lower()=='stop':
                        raise KeyboardInterrupt('Operator stopped before next head')
                rt=read_rt(robot)
                if np.max(abs(np.array(rt['body']['fb_pos'])-body))>.3:
                    raise RuntimeError('Body changed during head-grid teaching')
                set_head(robot,sdk,head,rt)
                w.write_json(status,dict(phase='teaching',round=i+1,total=len(args.heads),head=head,output=str(output)))
                teach_one(robot,sdk,index,head,output,assets,identity,terminal,args.latest,live)
            w.write_json(status,dict(phase='complete',trajectories=len(args.heads),arm=args.arm))
        except BaseException as exc:
            w.write_json(status,dict(phase='stopped',reason=str(exc)));raise
        finally:
            if robot._connected:
                robot.switch_to_idle(selected,2000);robot.switch_to_idle(sdk.FXObjType.OBJ_HEAD,2000);robot.unlink()


if __name__=='__main__':
    main()
