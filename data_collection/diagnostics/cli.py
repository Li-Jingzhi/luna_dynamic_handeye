"""Automatic yellow-ball error diagnostics. No hardware is accessed without --run."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

from .core import ROOT,default_plan,validate_plan,tasks,digest
from dynamic_handeye.geometry import Assets,fixed_geometry_hashes

MODEL=ROOT/'reports/fixed_view_zero_diagnosis_20261030/fixed_head_best_candidate.json'


def execute(a,model,plan,args,simulated=False):
    from .collect import run_session,write_json
    if simulated:
        from .simulation import SimRobot,SimCamera
        def factory(folder):return lambda:SimRobot(a,model,plan)
        camera_factory=lambda robot:SimCamera(robot,a)
    else:
        from .robot import Robot
        from .camera import Camera
        def factory(folder):return lambda:Robot(args.sdk_root,args.robot_ip,plan,folder)
        camera_factory=lambda robot:Camera(args.host,args.user,args.port,image_gamma=plan.get('image_gamma',1.),evidence_directory=robot.directory)
    if args.experiment!='all':
        return run_session(a,args.model,plan,args.experiment,args.output,factory(args.output),camera_factory,
            simulated=simulated,resume=getattr(args,'resume',False),limit=getattr(args,'limit',None))
    directory=args.output;resume=getattr(args,'resume',False)
    if directory.exists() and not resume:raise FileExistsError('Suite directory exists')
    directory.mkdir(parents=True,exist_ok=resume)
    spec=dict(plan=plan,model_sha256=digest(args.model),simulated=simulated,limit=getattr(args,'limit',None))
    if getattr(args,'reference_session',None):spec['reference_manifest_sha256']=digest(args.reference_session/'manifest.json')
    spec_path=directory/'suite.json'
    if resume and json.loads(spec_path.read_text())!=spec:raise ValueError('Suite configuration changed')
    if not resume:write_json(spec_path,spec)
    reference=None;results={}
    if getattr(args,'reference_session',None):
        manifest=json.loads((args.reference_session/'manifest.json').read_text())
        if manifest['assets']!=a.hashes or manifest['model_sha256']!=digest(args.model):raise ValueError('Reference source assets/model mismatch')
        reference=dict(q7_deg=manifest['guard_origin_q7_deg'],camera_m=manifest['origin_camera_m'],
            body_deg=manifest['initial_feedback']['body_deg'],opposite_arm_deg=manifest['initial_feedback']['opposite_arm_deg'])
        write_json(directory/'reference_source.json',dict(path=str(args.reference_session),manifest_sha256=digest(args.reference_session/'manifest.json')))
    for name in ('optical','hold','return','acceptance'):
        if not tasks(plan,name):
            results[name]=dict(complete=True,observations=0,note='Previously captured tasks excluded by plan');continue
        folder=directory/name
        results[name]=run_session(a,args.model,plan,name,folder,factory(folder),camera_factory,
            simulated=simulated,resume=resume and folder.exists(),limit=getattr(args,'limit',None),reference=reference)
        m=json.loads((folder/'manifest.json').read_text())
        if reference is None:reference=dict(q7_deg=m['guard_origin_q7_deg'],camera_m=m['origin_camera_m'],
            body_deg=m['initial_feedback']['body_deg'],opposite_arm_deg=m['initial_feedback']['opposite_arm_deg'])
    write_json(directory/'completed.json',dict(complete=True,simulated=simulated,experiments=results))
    return results


def preflight(plan,model_path,experiment):
    a=Assets();model=json.loads(Path(model_path).read_text())
    if model.get('schema')!='luna_fixed_view_calibrated_fk_v1' or model['assets']!=a.hashes or model['fixed_geometry_hashes']!=fixed_geometry_hashes(a):
        raise ValueError('Fixed-view model/assets mismatch')
    if not all(abs(x-y)<=.001 for x,y in zip(model['fixed_head_deg'],plan['head_deg'])):raise ValueError('Wrong fixed head')
    validate_plan(plan)
    names=('optical','hold','return','acceptance') if experiment=='all' else (experiment,)
    return a,model,dict(hardware_accessed=False,model_sha256=digest(model_path),assets=a.hashes,
        tasks={name:len(tasks(plan,name)) for name in names},frames_per_task=50,
        max_joint_step_deg=plan['max_joint_step_deg'],max_joint_excursion_deg=plan['max_total_joint_excursion_deg'],
        local_camera_workspace_radius_m=plan['max_camera_excursion_m'],
        optical_reference='Existing nominal40mm ball only; no independent known-distance ground truth',
        hold_reference='Automatic visual servo, not a mechanical pivot fixture')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    generate=sub.add_parser('plan');generate.add_argument('--output',required=True,type=Path)
    deploy=sub.add_parser('deploy-camera');deploy.add_argument('--host',default='6.6.7.100');deploy.add_argument('--user',default='nvidia');deploy.add_argument('--port',type=int,default=8768);deploy.add_argument('--run',action='store_true')
    analyze_parser=sub.add_parser('analyze');analyze_parser.add_argument('--session',required=True,type=Path)
    for name in ('check','collect','simulate'):
        command=sub.add_parser(name);command.add_argument('--plan',type=Path);command.add_argument('--model',type=Path,default=MODEL)
        command.add_argument('--experiment',choices=['optical','hold','return','acceptance','all'],default='all')
        if name!='check':command.add_argument('--output',required=True,type=Path)
        if name=='collect':
            command.add_argument('--reference-session',type=Path,help='Use a previous fixed-view origin, body and opposite-arm reference without changing source data')
            command.add_argument('--run',action='store_true');command.add_argument('--resume',action='store_true')
            command.add_argument('--sdk-root',type=Path,default=ROOT.parent/'00040701');command.add_argument('--robot-ip',default='6.6.7.190')
            command.add_argument('--host',default='6.6.7.100');command.add_argument('--user',default='nvidia');command.add_argument('--port',type=int,default=8768)
            command.add_argument('--no-deploy-camera',action='store_true',help='use an already prepared matching camera service')
        if name=='simulate':command.add_argument('--limit',type=int,default=2)
    args=p.parse_args(argv)
    if args.command=='plan':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(default_plan(),f,indent=2)
        print(args.output);return
    if args.command=='deploy-camera':
        from .camera import deployment
        print(json.dumps(deployment(args.host,args.user,args.port,args.run),indent=2));return
    if args.command=='analyze':
        from .analyze import analyze
        print(json.dumps(analyze(args.session),indent=2));return
    plan=json.loads(args.plan.read_text()) if args.plan else default_plan()
    a,model,summary=preflight(plan,args.model,args.experiment)
    if args.command=='check' or (args.command=='collect' and not args.run):
        print(json.dumps(summary,indent=2));return
    if args.command=='simulate':
        if args.limit<1:raise ValueError('Positive simulation task limit required')
        print(json.dumps(execute(a,model,plan,args,simulated=True),indent=2));return
    runtime=ROOT/'acquisition/runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(summary,indent=2),flush=True)
        print('RUN: automatic local joint motion and visual servo; keep the local workspace clear.',flush=True)
        if not args.no_deploy_camera:
            from .camera import deployment
            deployment(args.host,args.user,args.port,run=True)
        result=execute(a,model,plan,args)
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
