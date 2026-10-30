"""Explicitly synthetic robot/camera fixtures; never reported as hardware observations."""
import json
import time
import cv2
import numpy as np
from .core import ROOT,guard_q
from .camera import expected_assets
from dynamic_handeye.geometry import apply,torso_from_arm


class SimRobot:
    simulated=True
    def __init__(self,assets,model,plan):
        self.assets=assets;self.model=model;self.plan=plan
        source=model['source_captures'][0]['path']
        with (ROOT/source).open() as stream:
            row=next(json.loads(line) for line in stream if json.loads(line).get('capture_valid'))
        self.q=np.array(row['q7_deg'],float);self.origin=self.q.copy();self.limits=np.tile([-175.,175.],(7,1))
        self.body=row['body_q_deg'];self.identity=dict(simulated=True,submodel='SYNTHETIC_LUNA')
    def sample(self):return dict(q7_deg=self.q.tolist(),head_deg=self.plan['head_deg'],body_deg=self.body,
        opposite_arm_deg=[0.]*7,sample_monotonic_ns=time.monotonic_ns(),read_span_ms=0.,velocity_deg_s=[0.]*7)
    def move(self,target):
        target=np.array(target);guard_q(target,self.origin,self.limits,self.plan)
        if np.max(abs(target-self.q))>self.plan['max_joint_step_deg']+1e-8:raise ValueError('Synthetic step guard')
        self.q=target.copy()
    def settle(self):pass
    def close(self):pass
    def camera_position(self):
        p=apply(torso_from_arm(self.assets)@self.assets.flange(self.q+np.array(self.model['effective_joint_offsets_deg'])),self.assets.pf)[0]
        pose=self.model['camera_to_torso']
        return np.array(pose['R']).T@(p-pose['t_m'])/self.model['camera_scale']


class SimCamera:
    simulated=True
    def __init__(self,robot,assets):
        self.robot=robot;self.assets=assets;self.rng=np.random.default_rng(30);self.expected=expected_assets()
        self.identity='SIMULATED_CAMERA';self.count=0
        image=np.full((992,1280,3),80,np.uint8)
        self.blob=cv2.imencode('.png',image)[1].tobytes()
        self.images=[image,image]
    def close(self):pass
    def barrier(self):return dict(process_id=self.identity,after_ns=time.monotonic_ns())
    def capture(self,barrier):
        self.count+=1;stamp=max(time.monotonic_ns(),barrier['after_ns']+1)
        c=self.robot.camera_position()+self.rng.normal(0,2e-6,3);pixels=[]
        for p in self.assets.P:
            image=p@np.r_[c,1];pixels.extend(image[:2]/image[2])
        detection={}
        for i,eye in enumerate(('left','right')):
            center=c if i==0 else np.array(self.assets.stereo['right_from_left_R'])@c+self.assets.stereo['right_from_left_T_m']
            invk=np.linalg.inv(self.assets.K[i]);cone=(center@center-.02**2)*np.eye(3)-np.outer(center,center)
            conic=invk.T@cone@invk;A=conic[:2,:2];b=conic[:2,2];pixel_center=-np.linalg.solve(A,b)
            factor=-(conic[2,2]+b@pixel_center);values,vectors=np.linalg.eigh(A/factor)
            sizes=2/np.sqrt(values);angle=np.rad2deg(np.arctan2(vectors[1,0],vectors[0,0]))
            detection[eye]=dict(center=pixel_center.tolist(),ellipse=dict(major_px=float(sizes[0]),minor_px=float(sizes[1]),angle_deg=float(angle)),
                               ellipse_rms_px=.1,edge_coverage_deg=360.)
        m=dict(service='luna_diagnostic_camera_v1',process_id=self.identity,image_size=[1280,992],controls_verified=True,
            exposure_time_valid=False,source_clock_ns=[stamp,stamp+1],arrival_monotonic_ns=[stamp,stamp+1],
            frame_number=[self.count,self.count],simulated=True,**self.expected)
        m['synthetic_measurement']=dict(quality='PASS',center_left_camera_m=c.tolist(),corrected_stereo_pixels=pixels,**detection)
        blobs={f'{eye}_{kind}.png':self.blob for eye in ('left','right') for kind in ('input','rectified')}
        return m,(self.identity,stamp,stamp+1),blobs,self.images
