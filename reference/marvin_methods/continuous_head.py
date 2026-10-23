"""Nominal two-joint head MDH, calibrated mounts and exposure-time pose checks.

Static arbitrary angles and measurements DURING movement require different
validation. Historical stopped captures cannot certify moving-head accuracy.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from capt_head.drag_capture import interpolate_q
from capt_head.geometry import robust_optimize
from .fit_yellow_xyz_consistency import metrics
from .fit_yellow_xyz_v2 import write_new
from .geometry import arm_mdh, flange_pose, rigid_fit
from .validate_pf_rt import load_capture

SCHEMA = 'luna_continuous_head_xyz_v1'


def head_mdh(path):
    cfg=configparser.ConfigParser();cfg.read(path,encoding='utf-8')
    rows=np.array([[float(cfg[f'R.H.L{i}.DH'][k]) for k in ('A','Alpha','D','Theta')] for i in range(2)])
    if rows.shape != (2,4) or not np.isfinite(rows).all():
        raise ValueError('Invalid head MDH')
    return rows


def head_poses(rows, angles):
    angles=np.asarray(angles,float)
    rows=np.asarray(rows,float)
    if rows.shape!=(2,4) or not np.isfinite(rows).all():
        raise ValueError('Expected finite two-joint MDH rows')
    if angles.ndim != 2 or angles.shape[1] != 2 or not np.isfinite(angles).all():
        raise ValueError('Expected finite Nx2 yaw/pitch in degrees')
    out=np.broadcast_to(np.eye(4),(len(angles),4,4)).copy()
    for i,(a,alpha,d,theta) in enumerate(np.asarray(rows,float)):
        ca,sa=np.cos(np.radians(alpha)),np.sin(np.radians(alpha))
        ct,st=np.cos(np.radians(angles[:,i]+theta)),np.sin(np.radians(angles[:,i]+theta))
        step=np.zeros_like(out)
        step[:,0,0]=ct;step[:,0,1]=-st;step[:,0,3]=a/1000.
        step[:,1,0]=ca*st;step[:,1,1]=ca*ct;step[:,1,2]=-sa;step[:,1,3]=-d*sa/1000.
        step[:,2,0]=sa*st;step[:,2,1]=sa*ct;step[:,2,2]=ca;step[:,2,3]=d*ca/1000.
        step[:,3,3]=1.;out=out@step
    return out


def coverage(angles):
    angles=np.asarray(angles,float)
    if angles.ndim!=2 or angles.shape[1]!=2 or len(angles)==0 or not np.isfinite(angles).all():
        raise ValueError('Expected nonempty finite Nx2 angles')
    unique=np.unique(np.round(angles,2),axis=0)
    yaw,pitch=len(np.unique(unique[:,0])),len(np.unique(unique[:,1]))
    hull=cv2.convexHull(unique.astype(np.float32))
    area=float(cv2.contourArea(hull))
    return dict(head_count=len(unique),yaw_settings=yaw,pitch_settings=pitch,
                hull_deg=hull[:,0,:].tolist(),hull_area_deg2=area,
                coverage_qualified=bool(len(unique)>=6 and yaw>=3 and pitch>=3 and area>=4.))


def map_points(model, camera, angles):
    """Offline geometry only. Public validated runtime entry is query()."""
    camera=np.asarray(camera,float);poses=head_poses(model['head_mdh'],angles)
    if camera.shape != (len(poses),3) or not np.isfinite(camera).all():
        raise ValueError('Expected finite Nx3 camera XYZ')
    mount=np.asarray(model['mount_parameters']);R=cv2.Rodrigues(mount[:3])[0]
    head_points=camera@R.T+mount[3:]
    parent=np.einsum('nij,nj->ni',poses[:,:3,:3],head_points)+poses[:,:3,3]
    return parent@np.asarray(model['mdh_from_head_parent_R']).T+model['mdh_from_head_parent_t_m']


def fit_geometry(camera, target, angles, rows):
    camera,target,angles=np.asarray(camera,float),np.asarray(target,float),np.asarray(angles,float)
    if (camera.shape != target.shape or camera.shape != (len(angles),3) or len(camera)<36
            or not np.isfinite(target).all() or not np.isfinite(camera).all()):
        raise ValueError('Need >=36 finite paired head/camera/FK samples')
    unique=np.unique(angles,axis=0)
    if len(unique)<3 or np.linalg.matrix_rank(unique-unique.mean(0))<2:
        raise ValueError('Two-dimensional head excitation required')
    poses=head_poses(rows,angles)
    def eliminate(p):
        R=cv2.Rodrigues(p[:3])[0]
        local=camera@R.T+p[3:]
        parent=np.einsum('nij,nj->ni',poses[:,:3,:3],local)+poses[:,:3,3]
        world_R,world_t=rigid_fit(parent,target)
        return parent@world_R.T+world_t-target,world_R,world_t
    best=None
    for rotation in ([0,0,0],[np.pi/2,0,0],[-np.pi/2,0,0],[0,np.pi/2,0],[0,-np.pi/2,0],[0,0,np.pi]):
        initial=np.r_[rotation,np.zeros(3)]
        p=robust_optimize(lambda v:eliminate(v)[0].ravel()*1000.,initial,iterations=65)
        error,R,t=eliminate(p)
        cost=float(np.mean(np.minimum(np.linalg.norm(error,axis=1)*1000,10.)**2))
        if np.isfinite(cost) and np.linalg.norm(p[3:])<1.:
            if best is None or cost<best[0]:best=(cost,p,R,t)
    if best is None:
        raise ValueError('Head mount optimization failed')
    _,p,R,t=best
    return dict(schema=SCHEMA,head_mdh=np.asarray(rows).tolist(),mount_parameters=p.tolist(),
                mdh_from_head_parent_R=R.tolist(),mdh_from_head_parent_t_m=t.tolist())


def pose_at_exposure(history, timestamp_ns, clock_id):
    if not clock_id or len(history)<2 or any(r.get('clock_id')!=clock_id for r in history):
        raise ValueError('Exposure and head feedback must share a verified clock')
    stamps=np.array([r['timestamp_ns'] for r in history],dtype=np.int64)
    q=np.array([r['head_angles_deg'] for r in history],float)
    if q.shape!=(len(stamps),2) or not np.isfinite(q).all() or np.any(np.diff(stamps)<=0):
        raise ValueError('Invalid or unordered head history')
    angle,span=interpolate_q(list(zip(stamps,q)),timestamp_ns,max_span_ms=20.)
    j=min(max(int(np.searchsorted(stamps,timestamp_ns))-1,0),len(stamps)-2)
    speed=np.abs(q[j+1]-q[j])/((stamps[j+1]-stamps[j])/1e9)
    return angle,speed,span


def query(model, observation, *, history=None):
    if model.get('schema')!=SCHEMA or model.get('approved') is not True or model.get('do_not_deploy',True):
        raise ValueError('Continuous head model is not independently approved')
    validation=model.get('validation',{})
    if not validation.get('unseen_head_static_passed') or not model['coverage']['coverage_qualified']:
        raise ValueError('Unseen-head static validation and angle coverage required')
    for key in ('mdh_sha256','head_mdh_sha256','stereo_model_sha256','source_calibration_sha256',
                'pixel_domain','image_size','sensor_ids','sphere_center_method'):
        if observation.get(key)!=model[key]:raise ValueError('Observation contract mismatch: '+key)
    if observation.get('capture_valid') is not True:
        raise ValueError('Qualified visual measurement required')
    camera=np.asarray(observation['sphere_center_left_camera_m'],float)
    if camera.shape!=(3,) or not np.isfinite(camera).all() or camera[2]<=0:
        raise ValueError('Invalid camera XYZ')
    mode=observation.get('measurement_mode')
    if mode=='moving_single_frame':
        if not validation.get('dynamic_passed'):
            raise ValueError('Moving-head accuracy is not validated')
        if observation.get('valid_frames')!=1 or observation.get('exposure_time_valid') is not True:
            raise ValueError('Moving head requires a single synchronized exposure, not fused frames')
        left=observation['left_exposure_timestamp_ns'];right=observation['right_exposure_timestamp_ns']
        uncertainty=observation.get('timestamp_uncertainty_ns')
        if (not isinstance(left,int) or not isinstance(right,int) or not isinstance(uncertainty,int)
                or uncertainty<0 or history is None):raise ValueError('Verified exposure timestamps required')
        angle,speed,_=pose_at_exposure(history,left,observation.get('clock_id'))
        right_angle,right_speed,_=pose_at_exposure(history,right,observation.get('clock_id'))
        speed_limit=np.asarray(validation.get('dynamic_speed_limit_deg_s'),float)
        if (speed_limit.shape!=(2,) or not np.isfinite(speed_limit).all() or np.any(speed_limit<=0)
                or np.any(np.maximum(speed,right_speed)>speed_limit)):
            raise ValueError('Outside validated head speed range')
        radius=float(model['timing_radius_m'])
        budget=float(model['max_timing_error_mm'])
        if not np.isfinite([radius,budget]).all() or radius<=0 or budget<=0:
            raise ValueError('Invalid timing error budget')
        timing_error=radius*np.sum(np.radians(abs(angle-right_angle)+np.maximum(speed,right_speed)*uncertainty/1e9))*1000
        if not np.isfinite(timing_error) or timing_error>budget:
            raise ValueError('Head motion and timestamp uncertainty exceed timing budget')
    elif mode=='stationary_fused':
        angle=np.asarray(observation['head_angles_deg'],float)
        span=np.asarray(observation.get('robot_window_head_range_deg'),float)
        speed=np.asarray(observation.get('head_velocity_deg_s'),float)
        if (observation.get('valid_frames',0)<50 or span.shape!=(2,) or speed.shape!=(2,)
                or not np.isfinite(span).all() or not np.isfinite(speed).all()
                or np.any(span<0) or np.max(span)>.02 or np.max(abs(speed))>.02):
            raise ValueError('Stopped head and a stable fusion window required')
    else:
        raise ValueError('Explicit stationary or moving measurement mode required')
    if angle.shape!=(2,) or not np.isfinite(angle).all():raise ValueError('Invalid head angles')
    hull=np.array(model['coverage']['hull_deg'],np.float32).reshape(-1,1,2)
    if cv2.pointPolygonTest(hull,tuple(float(a) for a in angle),True)<-1e-5:
        raise ValueError('Outside calibrated yaw/pitch hull')
    point=map_points(model,camera[None,:],angle[None,:])[0]
    if np.min(np.linalg.norm(np.array(model['support_mdh_m'])-point,axis=1))*1000>model['support_radius_mm']:
        raise ValueError('Outside calibrated workspace')
    return point


def evaluate(capture, mdh, stereo, reference):
    rows,provenance=load_capture(capture,mdh,stereo,multi=True)
    manifest_path=Path(capture).with_suffix('.completed.json')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('complete') is not True or manifest['sha256']!=hashlib.sha256(Path(capture).read_bytes()).hexdigest():
            raise ValueError('Completed capture manifest mismatch')
        provenance.update(completed_manifest_verified=True,snapshot_is_final=True)
    ref=json.loads(Path(reference).read_text());raw=[json.loads(s) for s in Path(capture).read_text().splitlines()]
    for key in ('mdh_sha256','stereo_model_sha256'):
        if provenance[key]!=ref[key]:raise ValueError('Frozen reference mismatch: '+key)
    camera=np.array([r['sphere_center_left_camera_m'] for r in rows]);angles=np.array([r['head_angles_deg'] for r in rows])
    chain=arm_mdh(mdh);poses=np.array([flange_pose(r['q7_deg'],chain) for r in rows]);pf=np.asarray(ref['flange_to_ball_effective_m'])
    target=poses[:,:3,3]+np.einsum('nij,j->ni',poses[:,:3,:3],pf)
    groups=list(dict.fromkeys(r['group_id'] for r in rows));cut=int(.8*len(groups));g=set(groups[:cut])
    train=np.array([i for i,r in enumerate(rows) if r['group_id'] in g]);held=np.setdiff1d(np.arange(len(rows)),train)
    head_rows=head_mdh(mdh);selection=fit_geometry(camera[train],target[train],angles[train],head_rows)
    error=map_points(selection,camera,angles)-target
    model=fit_geometry(camera,target,angles,head_rows)
    c=coverage(angles)
    unseen=[]
    for head in np.unique(angles,axis=0):
        test=np.where(np.all(angles==head,axis=1))[0];tr=np.setdiff1d(np.arange(len(rows)),test)
        if not coverage(angles[tr])['coverage_qualified']:
            unseen.append(dict(head_deg=head.tolist(),status='insufficient_training_head_coverage'))
            continue
        fold=fit_geometry(camera[tr],target[tr],angles[tr],head_rows)
        unseen.append(dict(head_deg=head.tolist(),status='diagnostic',error=metrics(map_points(fold,camera[test],angles[test])-target[test])))
    metadata=raw[rows[0]['source_line']-1]
    model.update(approved=False,do_not_deploy=True,coverage=c,validation=dict(unseen_head_static_passed=False,dynamic_passed=False),
                 flange_to_ball_effective_m=pf.tolist(),support_mdh_m=target.tolist(),support_radius_mm=20.,
                 timing_radius_m=float(np.max(np.linalg.norm(target,axis=1))+np.linalg.norm(model['mdh_from_head_parent_t_m'])+1.),
                 max_timing_error_mm=.2,mdh_sha256=provenance['mdh_sha256'],
                 head_mdh_sha256=hashlib.sha256(head_rows.tobytes()).hexdigest(),
                 frame_id='right_arm_mdh_parent',units='m')
    for key in ('stereo_model_sha256','source_calibration_sha256','pixel_domain','image_size','sensor_ids','sphere_center_method'):
        model[key]=metadata[key]
    report=dict(provenance=provenance,coverage=c,model_approved=False,
                reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
                frozen_pf_mm=(pf*1000).tolist(),train_groups=cut,heldout_groups=len(groups)-cut,
                observed_head_train=metrics(error[train]),observed_head_heldout=metrics(error[held]),
                all_fit_training=metrics(map_points(model,camera,angles)-target),
                leave_one_head_out=unseen,dynamic_validation_available=False,
                note='Nominal head MDH and estimated mounts; stopped observations do not certify motion or arbitrary angles')
    return model,report


def main():
    root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',default=str(root/'data/ball_point_head_20261018_v1.jsonl'))
    parser.add_argument('--mdh',default=str(root/'configs/luna_new.ini'))
    parser.add_argument('--stereo',default=str(root/'configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json'))
    parser.add_argument('--reference',default=str(root/'models/candidate_yellow_xyz_spatial_v2_final.json'))
    parser.add_argument('--candidate',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();model,report=evaluate(args.capture,args.mdh,args.stereo,args.reference)
    write_new(args.candidate,model);write_new(args.report,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
