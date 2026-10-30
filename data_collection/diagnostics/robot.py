"""Opt-in SDK adapter; local model checks and small joint commands only."""
import time
import uuid
import numpy as np
from .core import guard_q


class Robot:
    simulated=False
    def __init__(self,sdk_root,ip,plan,directory):
        from data_collection import workflow as w
        from data_collection.head_trajectories import verify_live_mdh,robot_identity
        from interactive_ball_calibration.capture import read_rt,arm_limits,head_reference_waypoints
        from gq_calibration.capture import checked,read_limits,wait_head
        self.w=w;self.plan=plan;self.directory=directory;self.ctx=w.connected(sdk_root,ip)
        self.robot,self.sdk=self.ctx.__enter__();self.arm=self.sdk.FXObjType.OBJ_ARM1
        self.position=False
        try:
            verify_live_mdh(self.robot,1);self.identity=robot_identity(self.robot)
            initial=read_rt(self.robot);self.body=np.array(initial['body']['fb_pos']);self.left=np.array(initial['arms'][0]['fb']['fb_pos'])
            self.limits=arm_limits(self.robot);self.origin=np.array(initial['arms'][1]['fb']['fb_pos'])
            guard_q(self.origin,self.origin,self.limits,plan)
            head=np.array(initial['head']['fb_pos'][:2]);target=np.array(plan['head_deg']);limits=np.array(read_limits(self.robot))
            if np.any(target<limits[:,0]) or np.any(target>limits[:,1]):raise ValueError('Requested head exceeds live limits')
            obj=self.sdk.FXObjType.OBJ_HEAD
            try:
                for point in head_reference_waypoints(head,target,limits):
                    start=np.array(read_rt(self.robot)['head']['fb_pos'][:2])
                    steps=max(1,int(np.ceil(np.max(abs(point-start)))))
                    if self.sdk.state_map.get(self.robot.current_state(obj))=='IDLE':checked(self.robot.switch_to_position_mode(obj,2000,5,5),'Head Position')
                    for goal in np.linspace(start,point,steps+1)[1:]:
                        rt=read_rt(self.robot);self.model_check(rt,np.array(rt['arms'][1]['fb']['fb_pos']),goal)
                        checked(self.robot.runtime_set_joint_pos_cmd(obj,[*goal.tolist(),float(rt['head']['fb_pos'][2])]),'Head target')
                        wait_head(self.robot,goal)
            finally:
                checked(self.robot.switch_to_idle(obj,2000),'Head IDLE')
            self.sample()
        except BaseException:
            self.close();raise

    def model_check(self,rt,target,head):
        folder=self.directory/'model_checks';folder.mkdir(exist_ok=True)
        path=folder/(uuid.uuid4().hex+'.json')
        self.w.model_check(dict(waypoints=[dict(q7_deg=np.array(target).tolist())]),rt,np.array(head),path,'forward')

    def sample(self):
        before=time.monotonic_ns();rt=self.w.read_rt(self.robot);after=time.monotonic_ns()
        q=np.array(rt['arms'][1]['fb']['fb_pos'],float);head=np.array(rt['head']['fb_pos'][:2],float)
        if np.max(abs(head-self.plan['head_deg']))>self.plan['head_tolerance_deg']:raise RuntimeError('Head left fixed view')
        if np.max(abs(np.array(rt['body']['fb_pos'])-self.body))>.3 or np.max(abs(np.array(rt['arms'][0]['fb']['fb_pos'])-self.left))>.3:
            raise RuntimeError('Body/opposite arm moved')
        guard_q(q,self.origin,self.limits,self.plan)
        return dict(q7_deg=q.tolist(),head_deg=head.tolist(),body_deg=rt['body']['fb_pos'],
            opposite_arm_deg=rt['arms'][0]['fb']['fb_pos'],sample_monotonic_ns=(before+after)//2,
            read_span_ms=(after-before)/1e6,velocity_deg_s=rt['arms'][1]['fb']['fb_vel'])

    def move(self,target):
        from gq_calibration.capture import checked
        from interactive_ball_calibration.capture import wait_arm
        q=np.array(self.sample()['q7_deg']);target=np.array(target,float);guard_q(target,self.origin,self.limits,self.plan)
        if np.max(abs(target-q))>self.plan['max_joint_step_deg']+1e-9:raise ValueError('Joint step guard exceeded')
        rt=self.w.read_rt(self.robot);self.model_check(rt,target,self.plan['head_deg'])
        if not self.position:
            checked(self.robot.switch_to_position_mode(self.arm,2000,5,5),'Right arm Position');self.position=True
        checked(self.robot.runtime_set_joint_pos_cmd(self.arm,target.tolist()),'Diagnostic joint step')
        wait_arm(self.robot,self.sdk,q,target,np.array(self.plan['head_deg']),self.body)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            fb=self.sample()
            if np.max(abs(np.array(fb['q7_deg'])-target))<=self.plan['joint_target_tolerance_deg']:return
            time.sleep(.05)
        raise RuntimeError('Joint return did not reach precise feedback tolerance')

    def settle(self):
        start=time.monotonic();anchor=np.array(self.sample()['q7_deg'])
        while time.monotonic()-start<self.plan['settle_s']:
            sample=self.sample()
            if np.max(abs(np.array(sample['q7_deg'])-anchor))>self.plan['max_window_joint_span_deg']:
                raise RuntimeError('Arm moved during settling')
            time.sleep(.05)

    def close(self):
        try:
            if self.robot._connected:
                from gq_calibration.capture import checked
                checked(self.robot.switch_to_idle(self.arm,2000),'Right arm IDLE cleanup')
        finally:self.ctx.__exit__(None,None,None)
