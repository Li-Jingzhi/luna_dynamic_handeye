"""Supplement an existing successful suite with static repeats and arm-fixed head repeats."""
import argparse
import fcntl
import json
import shutil
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'acquisition'))
from data_collection.diagnostics.cli import MODEL,preflight
from data_collection.diagnostics.core import default_plan,digest
from data_collection.diagnostics.camera import Camera
from data_collection.diagnostics.collect import Collector,write_json,append,skippable_window_error
from data_collection.diagnostics.robot import Robot

HEADS=((0,-25),(10,-15),(-10,-20),(0,-15),(10,-25),(-10,-15),(0,-20),(10,-20),(-10,-25))


def head_tasks(site):
    result=[]
    for repeat in range(3):
        order=HEADS[repeat*3:]+HEADS[:repeat*3]
        for index,head in enumerate(((0,-25),*order,(0,-25))):
            result.append(dict(id=f'head_site{site}_round{repeat}_visit{index}',site=site,repeat=repeat,
                head_deg=list(head),reference_revisit=index in (0,10),kind='arm_fixed_head'))
    return result


def move_head(robot,target):
    from interactive_ball_calibration.capture import head_reference_waypoints
    from gq_calibration.capture import read_limits,wait_head,checked
    target=np.asarray(target,float);obj=robot.sdk.FXObjType.OBJ_HEAD
    checked(robot.robot.switch_to_idle(robot.arm,2000),'Right arm IDLE before head-only experiment')
    robot.position=False;rt=robot.w.read_rt(robot.robot);anchor=np.array(rt['arms'][1]['fb']['fb_pos'])
    limits=np.array(read_limits(robot.robot))
    try:
        for point in head_reference_waypoints(rt['head']['fb_pos'][:2],target,limits):
            start=np.array(robot.w.read_rt(robot.robot)['head']['fb_pos'][:2])
            for head in np.linspace(start,point,max(1,int(np.ceil(np.max(abs(point-start)))))+1)[1:]:
                feedback=robot.w.read_rt(robot.robot)
                if np.max(abs(np.array(feedback['arms'][1]['fb']['fb_pos'])-anchor))>.02:
                    raise RuntimeError('Arm moved during head-only experiment')
                robot.model_check(feedback,anchor,head)
                if robot.sdk.state_map.get(robot.robot.current_state(obj))=='IDLE':
                    checked(robot.robot.switch_to_position_mode(obj,2000,5,5),'Head Position')
                checked(robot.robot.runtime_set_joint_pos_cmd(obj,[*head.tolist(),float(rt['head']['fb_pos'][2])]),'Head target')
                wait_head(robot.robot,head)
        robot.plan['head_deg']=target.tolist()
    finally:checked(robot.robot.switch_to_idle(obj,2000),'Head IDLE')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--after-suite',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    plan=default_plan();plan['image_gamma']=1.6;plan['continue_measurement_rejects']=True
    assets,model,summary=preflight(plan,MODEL,'all')
    if not args.run:
        print(json.dumps(dict(static_windows=30,head_windows=99,head_grid_unique=81,reference_extra_windows=18,
            head_order=HEADS,frames_per_window=50,independent_absolute_scale=False,motion=False)));return
    if args.wait_pid:
        while Path(f'/proc/{args.wait_pid}/cmdline').exists():
            command=Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
            if b'data_collection.diagnostics.cli' not in command:break
            time.sleep(2)
    finished=json.loads((args.after_suite/'completed.json').read_text())
    if finished.get('complete') is not True or finished.get('simulated') is not False:
        raise RuntimeError('Preceding real suite did not complete; no supplementary motion')
    if not Path('/proc/mounts').read_text().count(str((ROOT/'datasets/orin_diagnostic_storage').resolve())):
        raise RuntimeError('Verified Orin image-storage mount missing')
    if args.output.exists():raise FileExistsError('Use a new supplementary session')
    with (ROOT/'acquisition/runtime/collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        write_json(args.output/'manifest.json',dict(schema='luna_ball_extra_v1',plan=plan,model_sha256=digest(MODEL),
            assets=assets.hashes,after_suite=str(args.after_suite),static_windows=30,head_windows=99,
            note='Fixed q within site; camera head changes only in head experiment. No independent size target.'))
        (args.output/'frozen_model.json').write_bytes(MODEL.read_bytes())
        asset_source=next((p for p in args.after_suite.glob('*/assets') if p.is_dir()),None)
        if asset_source is None:raise ValueError('Preceding suite has no asset snapshot')
        shutil.copytree(asset_source,args.output/'assets')
        robot=camera=None;total=0
        try:
            robot=Robot(ROOT.parent/'00040701','6.6.7.190',plan,args.output)
            camera=Camera('6.6.7.100','nvidia',8768,image_gamma=1.6,evidence_directory=args.output)
            collector=Collector(assets,model,plan,robot,camera,args.output)
            collector.origin_camera=collector.control()
            write_json(args.output/'initial.json',dict(feedback=robot.sample(),camera_m=collector.origin_camera.tolist()))
            origin=collector.origin_camera.copy()
            for kind in ('static','head'):
                for site,z in enumerate((-.02,0.,.02)):
                    if kind=='head':move_head(robot,[0,-25])
                    collector=Collector(assets,model,plan,robot,camera,args.output)
                    collector.origin_camera=origin.copy();collector.servo(origin+[0,0,z],robot.origin)
                    anchor=np.array(robot.sample()['q7_deg'])
                    tasks=[dict(id=f'static_site{site}_repeat{r}',kind='static',site=site,repeat=r,head_deg=[0,-25]) for r in range(10)] if kind=='static' else head_tasks(site)
                    for task in tasks:
                        if kind=='head':
                            move_head(robot,task['head_deg'])
                            collector=Collector(assets,model,plan,robot,camera,args.output)
                            collector.control()  # No reused camera-frame identities: persisted source ledger is shared.
                        feedback=robot.sample()
                        if np.max(abs(np.array(feedback['q7_deg'])-anchor))>.02:
                            raise RuntimeError('Same-q site constraint violated')
                        print(task['id'],f'{total}/129',flush=True)
                        try:sample=collector.window(task)
                        except (ValueError,RuntimeError) as error:
                            if not skippable_window_error(error):raise
                            append(args.output/'observations.jsonl',dict(task=task,capture_valid=False,simulated=False,
                                rejection_reason=str(error),fixed_site_q7_deg=anchor.tolist(),feedback=robot.sample()))
                            total+=1;write_json(args.output/'status.json',dict(phase='collecting',attempted=total,planned=129,last_task=task))
                            continue
                        if np.max(abs(np.array(sample['q7_deg'])-anchor))>.02:
                            raise RuntimeError('Same-q capture constraint violated')
                        append(args.output/'observations.jsonl',dict(task=task,capture_valid=True,simulated=False,
                            fixed_site_q7_deg=anchor.tolist(),**sample))
                        total+=1;write_json(args.output/'status.json',dict(phase='collecting',saved=total,planned=129,last_task=task))
                    if kind=='head':move_head(robot,[0,-25])
            recorded=[json.loads(line) for line in (args.output/'observations.jsonl').read_text().splitlines() if line.strip()]
            valid=sum(row.get('capture_valid') is True for row in recorded)
            write_json(args.output/'completed.json',dict(complete=True,simulated=False,observations=total,
                valid_windows=valid,rejected_windows=total-valid,sha256=digest(args.output/'observations.jsonl')))
            write_json(args.output/'status.json',dict(phase='complete',saved=total,planned=129))
        except BaseException as error:
            write_json(args.output/'status.json',dict(phase='stopped',saved=total,planned=129,reason=str(error)));raise
        finally:
            if robot is not None:
                try:
                    ret=robot.robot.switch_to_idle(robot.arm,2000)
                    if ret!=0:raise RuntimeError('Extra capture arm IDLE failed')
                finally:robot.close()
            if camera is not None:camera.close()


if __name__=='__main__':main()
