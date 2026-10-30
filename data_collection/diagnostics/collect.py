"""Automatic local visual servo, sampling and immutable experiment sessions."""
import json
import hashlib
import shutil
from pathlib import Path
import time
import uuid
import numpy as np
from .core import ROOT,digest,tasks,servo_step,guard_q,window_summary,validate_metadata,planned_step_limit
from dynamic_handeye.geometry import apply,torso_from_arm


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def skippable_window_error(error):
    return (isinstance(error,ValueError) and str(error)=='Corrected disparity jitter exceeds 0.1px') or (
        isinstance(error,RuntimeError) and str(error).startswith('Only ') and str(error).endswith((' of 20 unique qualified frames; window preserved',' of 50 unique qualified frames; window preserved')))


def development_joints(model):
    values=[]
    for source in model['source_captures']:
        path=ROOT/source['path']
        if digest(path)!=source['sha256']:raise ValueError('Development source changed: '+str(path))
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('capture_valid'):values.append(row['q7_deg'])
    return np.array(values)


class Collector:
    def __init__(self,assets,model,plan,robot,camera,directory):
        self.a=assets;self.model=model;self.p=plan;self.robot=robot;self.camera=camera;self.dir=Path(directory)
        self.seen=[set(),set()];self.process=None;self.origin_camera=None;self.control_id=0
        self.previous_pixels=None;self.previous_center=None
        self.expected=dict(source_calibration_sha256=assets.hashes['raw'],stereo_model_sha256=assets.hashes['stereo'],
            controls_sha256=digest(self.dir/'assets/configs/camera/luna_v2_fixed_controls.json'))
        self.seen_file=self.dir/'source_identities.jsonl'
        if self.seen_file.exists():
            for line in self.seen_file.read_text().splitlines():
                value=json.loads(line)
                for eye in (0,1):self.seen[eye].add(tuple(value[eye]))

    def measure(self,barrier,save=True):
        before=self.robot.sample()
        metadata,identity,blobs,images=(self.camera.capture(barrier) if getattr(self.camera,'simulated',False) else self.camera.capture(barrier,save=save))
        after=self.robot.sample()
        validate_metadata(metadata,self.expected,barrier['after_ns'])
        if self.process is None:self.process=metadata['process_id']
        if metadata['process_id']!=self.process:raise RuntimeError('Camera process restarted; stop and create a new camera session')
        identities=[(metadata['process_id'],int(s)) for s in metadata['source_clock_ns']]
        if any(identities[i] in self.seen[i] for i in (0,1)):raise ValueError('A source eye frame was reused')
        for eye in (0,1):self.seen[eye].add(identities[eye])
        append(self.seen_file,identities)
        uid=metadata.get('evidence_id',uuid.uuid4().hex);files={}
        if save:
            if metadata.get('stored_evidence'):
                files=metadata['stored_evidence']
                required={f'{eye}_{kind}.png' for eye in ('left','right') for kind in ('input','rectified','rectified_unenhanced')}
                if set(files)!=required:raise ValueError('Incomplete remote image evidence')
            else:
                folder=self.dir/'images'/uid;folder.mkdir(parents=True,exist_ok=False)
                for name,blob in blobs.items():
                    path=folder/name
                    with path.open('xb') as f:f.write(blob)
                    files[name]=dict(path=str(path.relative_to(self.dir)),sha256=digest(path))
            append(self.dir/'camera_evidence.jsonl',dict(id=uid,metadata=metadata,feedback=after,image_files=files))
        if np.max(abs(np.array(before['q7_deg'])-after['q7_deg']))>self.p['max_window_joint_span_deg']:
            raise RuntimeError('Arm moved while transferring camera evidence')
        if getattr(self.camera,'simulated',False):measurement=metadata['synthetic_measurement']
        else:
            from acquisition.capt_head.subpixel_ball import stereo_measure
            config=self.a.contract
            measurement=stereo_measure(images,config,previous=self.previous_pixels,stereo=self.a.stereo)
        if measurement is None or measurement.get('quality')!='PASS':raise ValueError('Ball detection/epipolar quality failed')
        c=np.array(measurement['center_left_camera_m'],float)
        if not np.isfinite(c).all() or not .15<c[2]<.8:raise RuntimeError('Ball depth outside diagnostic local range')
        if self.previous_center is not None and np.linalg.norm(c-self.previous_center)>.01:
            raise ValueError('Ball detection jumped more than10mm between stationary feedback samples')
        for p in self.a.P:
            image=p@np.r_[c,1];uv=image[:2]/image[2]
            if not (30<uv[0]<1250 and 30<uv[1]<962):raise RuntimeError('Ball approached stereo field boundary')
        if self.origin_camera is not None and np.linalg.norm(c-self.origin_camera)>self.p['max_camera_excursion_m']:
            raise RuntimeError('Measured ball left local camera workspace')
        self.previous_center=c.copy()
        if not getattr(self.camera,'simulated',False):
            self.previous_pixels=[np.array(measurement[eye]['center']) for eye in ('left','right')]
        return dict(id=uid,simulated=self.robot.simulated,metadata=metadata,measurement=measurement,
            feedback=after,feedback_before=before,image_files=files,request_feedback_note='Static before/after SDK reads; clocks are not hardware exposure-synchronized')

    def control(self):
        self.robot.settle();barrier=self.camera.barrier();values=[];deadline=time.monotonic()+30
        while len(values)<self.p['control_samples'] and time.monotonic()<deadline:
            try:
                frame=self.measure(barrier,save=False);values.append(frame['measurement']['center_left_camera_m'])
            except (ValueError,OSError) as error:
                append(self.dir/'events.jsonl',dict(kind='control_reject',reason=str(error)));time.sleep(.02)
        if len(values)!=self.p['control_samples']:raise RuntimeError('No fresh valid ball feedback for visual servo')
        return np.median(values,axis=0)

    def jacobian(self,q):
        offsets=np.array(self.model['effective_joint_offsets_deg']);eps=.01
        def fk(x):return apply(torso_from_arm(self.a)@self.a.flange(x+offsets),self.a.pf)[0]
        j=np.column_stack([(fk(q+np.eye(7)[i]*eps)-fk(q-np.eye(7)[i]*eps))/(2*eps) for i in range(7)])
        return np.array(self.model['camera_to_torso']['R']).T@j/self.model['camera_scale']

    def servo(self,target,posture=None):
        target=np.array(target,float)
        if np.linalg.norm(target-self.origin_camera)>self.p['max_camera_excursion_m']:raise ValueError('Requested target exceeds workspace')
        previous=None;diverging=0
        for iteration in range(self.p['servo_max_iterations']):
            point=self.control();q=np.array(self.robot.sample()['q7_deg']);error=target-point;j=self.jacobian(q)
            dq,null=servo_step(j,error,q,posture,planned_step_limit(self.p))
            if np.linalg.norm(error)<=self.p['servo_tolerance_m'] and (posture is None or np.max(abs(null@(posture-q)))<.1):
                return dict(iterations=iteration,actual_camera_m=point.tolist(),desired_camera_m=target.tolist(),error_mm=float(np.linalg.norm(error)*1000),q7_deg=q.tolist())
            if previous is not None and np.linalg.norm(error)>previous+.001:diverging+=1
            else:diverging=0
            if diverging>=3:raise RuntimeError('Visual servo diverged; no blind continuation')
            previous=float(np.linalg.norm(error));goal=q+dq
            guard_q(goal,self.robot.origin,self.robot.limits,self.p);self.robot.move(goal)
        raise RuntimeError('Visual servo did not converge within iteration budget')

    def joint_path(self,target):
        current=np.array(self.robot.sample()['q7_deg']);target=np.array(target,float);history=[current.tolist()]
        guard_q(target,self.robot.origin,self.robot.limits,self.p)
        steps=max(1,int(np.ceil(np.max(abs(target-current))/planned_step_limit(self.p))))
        for goal in np.linspace(current,target,steps+1)[1:]:
            self.robot.move(goal);self.control();history.append(self.robot.sample()['q7_deg'])
        if np.max(abs(np.array(history[-1])-target))>self.p['joint_target_tolerance_deg']:raise RuntimeError('Return feedback tolerance failed')
        return history

    def window(self,task):
        self.robot.settle();barrier=self.camera.barrier();frames=[];deadline=time.monotonic()+self.p['window_timeout_s']
        while len(frames)<self.p['samples_per_window'] and time.monotonic()<deadline:
            try:
                frame=self.measure(barrier);frames.append(frame)
                append(self.dir/'frames.jsonl',dict(task_id=task['id'],accepted=True,**frame))
            except (ValueError,OSError) as error:
                append(self.dir/'events.jsonl',dict(kind='window_reject',task_id=task['id'],reason=str(error)))
                time.sleep(.02)
        if len(frames)!=self.p['samples_per_window']:raise RuntimeError(f"Only {len(frames)} of {self.p['samples_per_window']} unique qualified frames; window preserved")
        return window_summary(frames,self.p)


def run_session(a,model_path,plan,experiment,directory,robot_factory,camera_factory,simulated=False,resume=False,limit=None,reference=None):
    directory=Path(directory);model_path=Path(model_path);model_bytes=model_path.read_bytes();model=json.loads(model_bytes)
    expected_tasks=tasks(plan,experiment)
    if limit is not None:expected_tasks=expected_tasks[:limit]
    if directory.exists() and not resume:raise FileExistsError('Use a new session or explicit --resume')
    if not directory.exists():directory.mkdir(parents=True)
    manifest_path=directory/'manifest.json';model_hash=hashlib.sha256(model_bytes).hexdigest()
    if resume:
        manifest=json.loads(manifest_path.read_text())
        if manifest['plan']!=plan or manifest['experiment']!=experiment or manifest['model_sha256']!=model_hash or manifest['assets']!=a.hashes or manifest['simulated']!=simulated:
            raise ValueError('Resume configuration/model/assets mismatch')
        if manifest.get('collection_code_hashes')!={str(path.relative_to(ROOT)):digest(path) for path in Path(__file__).parent.glob('*.py')}:
            raise ValueError('Collection code changed since session creation')
        if digest(directory/'frozen_model.json')!=model_hash:raise ValueError('Frozen model changed')
        for name,value in manifest['asset_files'].items():
            if digest(directory/'assets'/name)!=value:raise ValueError('Session asset changed')
    else:
        manifest=dict(schema='luna_auto_diagnostic_session_v1',experiment=experiment,plan=plan,assets=a.hashes,
            model_sha256=model_hash,simulated=simulated,tasks=expected_tasks,
            full_protocol=limit is None or limit>=len(tasks(plan,experiment)),
            reference='Original and separately calibrated FK are both retained; model never refitted by collection',
            independent_known_distance_reference=False,physical_fixed_center_fixture=False,
            collection_code_hashes={str(path.relative_to(ROOT)):digest(path) for path in Path(__file__).parent.glob('*.py')})
        (directory/'frozen_model.json').write_bytes(model_bytes)
        snapshot=directory/'assets'
        paths=[*ROOT.glob('configs/camera/*.json'),*ROOT.glob('configs/robot/*.json'),ROOT/'configs/robot/luna_new.ini',
               ROOT/'sources.json',ROOT/'reference/LunaUdpTool/model/gento_luna/gento_luna_contact_filtered.xml']
        manifest['asset_files']={}
        for path in paths:
            relative=path.relative_to(ROOT);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest);manifest['asset_files'][str(relative)]=digest(dest)
        write_json(manifest_path,manifest)
    if manifest['tasks']!=expected_tasks:raise ValueError('Resume task list differs')
    output=directory/'observations.jsonl';existing=[json.loads(x) for x in output.read_text().splitlines()] if output.exists() else []
    done=[x['task']['id'] for x in existing]
    if len(done)!=len(set(done)):raise ValueError('Duplicate completed task')
    if not set(done)<=set(t['id'] for t in expected_tasks):raise ValueError('Observation belongs to another task list')
    if len(done)==len(expected_tasks) and (directory/'completed.json').exists():
        completed=json.loads((directory/'completed.json').read_text())
        if completed['sha256']!=digest(output):raise ValueError('Completed observations modified')
        return dict(complete=True,simulated=simulated,observations=len(done),directory=str(directory),already_complete=True)
    robot=camera=None;complete=False;active=None
    try:
        robot=robot_factory();camera=camera_factory(robot);collector=Collector(a,model,plan,robot,camera,directory)
        if resume and 'checkpoint_q7_deg' in manifest:
            if np.max(abs(np.array(robot.sample()['q7_deg'])-manifest['checkpoint_q7_deg']))>.3:
                raise ValueError('Robot moved from saved checkpoint; use a new session, not blind resume')
            fb=robot.sample();initial=manifest['initial_feedback']
            if any(np.max(abs(np.array(fb[k])-initial[k]))>.3 for k in ('body_deg','opposite_arm_deg')):
                raise ValueError('Body/opposite arm changed since session creation')
            collector.origin_camera=np.array(manifest['origin_camera_m']);robot.origin=np.array(manifest['guard_origin_q7_deg'])
        elif reference is not None:
            robot.origin=np.array(reference['q7_deg']);collector.origin_camera=np.array(reference['camera_m'])
            fb=robot.sample()
            if np.max(abs(np.array(fb['body_deg'])-reference['body_deg']))>.3 or np.max(abs(np.array(fb['opposite_arm_deg'])-reference['opposite_arm_deg']))>.3:
                raise ValueError('Suite body/opposite arm reference changed')
            collector.control();manifest.update(initial_feedback=fb,origin_camera_m=collector.origin_camera.tolist(),
                guard_origin_q7_deg=robot.origin.tolist(),robot_identity=robot.identity)
            write_json(manifest_path,manifest)
        else:
            collector.origin_camera=collector.control();manifest.update(initial_feedback=robot.sample(),origin_camera_m=collector.origin_camera.tolist(),robot_identity=robot.identity)
            manifest['guard_origin_q7_deg']=robot.origin.tolist()
            write_json(manifest_path,manifest)
        dev=development_joints(model) if experiment=='acceptance' else None
        anchors=manifest.get('anchors',{})
        for task in expected_tasks:
            if task['id'] in done:continue
            active=task;print(f"{experiment}: {task['id']} ({len(done)}/{len(expected_tasks)})",flush=True)
            goal=collector.origin_camera+np.array(task['offset_m']);audit={}
            if experiment in ('hold','return'):
                site=str(task['site'])
                if site not in anchors:
                    audit['anchor_servo']=collector.servo(goal,robot.origin);anchors[site]=robot.sample()['q7_deg'];manifest['anchors']=anchors;write_json(manifest_path,manifest)
                anchor=np.array(anchors[site]);collector.joint_path(anchor)
                if experiment=='hold':
                    # Normalize the reachable null-space direction: a raw axis
                    # request can project to an almost unchanged configuration.
                    _,null=servo_step(collector.jacobian(anchor),np.zeros(3),anchor,None,plan['max_joint_step_deg'])
                    direction=null[:,task['axis']]*task['sign']
                    magnitude=float(np.max(abs(direction)))
                    if magnitude<.05:raise ValueError('Requested axis has insufficient position null-space freedom')
                    desired=anchor+direction/magnitude*task['amplitude_deg']
                    audit['projected_posture_target_q7_deg']=desired.tolist()
                    audit['servo']=collector.servo(goal,desired)
                    if np.max(abs(np.array(robot.sample()['q7_deg'])-anchor))<1:raise ValueError('Null-space posture did not produce an independent configuration')
                    audit['constraint']='vision-held center, not mechanical fixture; no absolute scale truth'
                else:
                    pre=anchor.copy();pre[task['axis']]+=task['sign']*plan['approach_deg']
                    audit['outbound']=collector.joint_path(pre);audit['approach']=collector.joint_path(anchor)
                    axis=task['axis'];actual=np.array(audit['approach'])[:,axis]
                    if task['sign']*(actual[0]-actual[-1])<.8*plan['approach_deg'] or np.max(task['sign']*np.diff(actual))>.03:
                        raise RuntimeError('Recorded feedback does not prove requested return direction')
                    audit['anchor_q7_deg']=anchor.tolist()
            else:
                posture=robot.origin.copy()
                if experiment=='acceptance':
                    index=expected_tasks.index(task);posture=robot.origin.copy();axis=plan['posture_axes_0based'][index%len(plan['posture_axes_0based'])]
                    posture[axis]+=(-1 if index%2 else 1)*plan['posture_delta_deg']
                audit['servo']=collector.servo(goal,posture)
            try:sample=collector.window(task)
            except (ValueError,RuntimeError) as error:
                if not plan.get('continue_measurement_rejects',False) or not skippable_window_error(error):raise
                append(output,dict(task=task,capture_valid=False,simulated=simulated,model_sha256=model_hash,
                    rejection_reason=str(error),audit=audit,feedback=robot.sample(),image_evidence='frames.jsonl and camera_evidence.jsonl retain failed window'))
                done.append(task['id']);manifest['checkpoint_q7_deg']=robot.sample()['q7_deg'];write_json(manifest_path,manifest)
                append(directory/'events.jsonl',dict(kind='window_rejected_continue',task=task,reason=str(error)))
                write_json(directory/'status.json',dict(complete=False,attempted=len(done),planned=len(expected_tasks),last_task=task['id']))
                continue
            if experiment=='hold' and np.linalg.norm(np.array(sample['camera_xyz_m'])-goal)>plan['hold_tolerance_m']:
                raise RuntimeError('Visual center did not remain at the declared site')
            if dev is not None:
                novelty=float(np.min(np.max(abs(dev-np.array(sample['q7_deg'])),axis=1)))
                audit['nearest_development_joint_max_diff_deg']=novelty
                if novelty<plan['acceptance_min_joint_novelty_deg']:raise RuntimeError('Acceptance posture too close to a development sample; data retained but not counted')
            row=dict(task=task,capture_valid=True,simulated=simulated,model_sha256=model_hash,audit=audit,**sample)
            append(output,row);done.append(task['id']);manifest['checkpoint_q7_deg']=robot.sample()['q7_deg'];write_json(manifest_path,manifest)
            write_json(directory/'status.json',dict(complete=False,saved=len(done),planned=len(expected_tasks),last_task=task['id']))
        complete=len(done)==len(expected_tasks)
    except BaseException as error:
        append(directory/'events.jsonl',dict(kind='session_abort',task=active,reason=str(error)))
        if robot is not None:
            try:manifest['checkpoint_q7_deg']=robot.sample()['q7_deg'];write_json(manifest_path,manifest)
            except Exception as feedback_error:append(directory/'events.jsonl',dict(kind='cleanup_feedback_error',reason=str(feedback_error)))
        raise
    finally:
        cleanup_errors=[]
        for name,resource in (('robot',robot),('camera',camera)):
            if resource is not None:
                try:resource.close()
                except Exception as error:cleanup_errors.append(name+': '+str(error))
        if cleanup_errors:complete=False;append(directory/'events.jsonl',dict(kind='cleanup_failure',errors=cleanup_errors))
        write_json(directory/'status.json',dict(complete=complete,simulated=simulated,saved=len(done),planned=len(expected_tasks)))
        if complete:
            recorded=[json.loads(line) for line in output.read_text().splitlines() if line.strip()]
            valid=sum(row.get('capture_valid') is True for row in recorded)
            write_json(directory/'completed.json',dict(complete=True,simulated=simulated,observations=len(done),
                valid_windows=valid,rejected_windows=len(done)-valid,all_windows_valid=valid==len(done),
                sha256=digest(output),model_sha256=model_hash))
        if cleanup_errors:raise RuntimeError('Cleanup failed: '+'; '.join(cleanup_errors))
    return dict(complete=complete,simulated=simulated,observations=len(done),directory=str(directory))
