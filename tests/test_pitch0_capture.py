"""Pitch-zero plan, fifty fresh per-eye frames and adapter wiring without a robot."""
import copy
import json
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dynamic_handeye.geometry import Assets,ROOT
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.capture import load_plan,fuse_pro_window,capture_fresh,yaw_reference_waypoints,run_capture


class PitchZeroCaptureTests(unittest.TestCase):
    def setUp(self):self.assets=Assets()

    def test_explicit_yaw_mode_preserves_crossed_checks(self):
        p=ROOT/'configs/capture_pitch0_yaw_plan.json';plan=load_plan(p,self.assets)
        self.assertEqual(plan['reference_head_deg'],[0,0])
        self.assertEqual(plan['head_angles_deg'],[[-10,0],[0,0],[10,0]])
        for change in [dict(mode='stationary_crossed'),dict(reference_head_deg=[0,-25]),
                       dict(head_angles_deg=[[-10,0],[0,1],[10,0]]),dict(observations_per_position=49),
                       dict(head_angles_deg=[[-100,0],[0,0],[10,0]]),dict(robot_version='luna_v3'),
                       dict(head_feedback_order=['controller_pitch','controller_yaw'])]:
            with tempfile.TemporaryDirectory() as t:
                bad=Path(t)/'plan.json';bad.write_text(json.dumps(dict(plan,**change)))
                with self.assertRaises(ValueError):load_plan(bad,self.assets)
        self.assertEqual(len(load_plan(ROOT/'configs/capture_plan.json',self.assets)['head_angles_deg']),9)

    @staticmethod
    def observations():
        return [(None,None,None,dict(source_timestamp_ns=[10**18+i,10**18+100+i])) for i in range(50)]

    @staticmethod
    def fused(_):
        return dict(sphere_corrected_disparity_jitter_p95_px=.05,robot_window_q7_deg=[[i*.0001]*7 for i in range(50)],
                    robot_window_head_deg=[[0,0]]*50,valid_frames=50)

    def test_fifty_independent_eyes_and_stable_median_feedback(self):
        obs=self.observations();result=fuse_pro_window(self.fused,obs)
        self.assertEqual(result['valid_frames'],50)
        self.assertEqual(result['head_angles_deg'],[0,0])
        np.testing.assert_allclose(result['q7_deg'],[.00245]*7)
        self.assertEqual(result['robot_version'],'luna_v2_0')
        with self.assertRaises(ValueError):fuse_pro_window(self.fused,obs[:49])
        reused=copy.deepcopy(obs);reused[-1][3]['source_timestamp_ns'][1]=reused[0][3]['source_timestamp_ns'][1]
        with self.assertRaisesRegex(ValueError,'Repeated source'):fuse_pro_window(self.fused,reused)
        missing=copy.deepcopy(obs);missing[0][3]['source_timestamp_ns']=None
        with self.assertRaises(ValueError):fuse_pro_window(self.fused,missing)
        def noisy(o):return dict(self.fused(o),sphere_corrected_disparity_jitter_p95_px=.11)
        with self.assertRaisesRegex(ValueError,'disparity'):fuse_pro_window(noisy,obs)
        def moving(o):
            out=self.fused(o);out['robot_window_head_deg'][-1]=[.03,0];return out
        with self.assertRaisesRegex(ValueError,'Robot varied'):fuse_pro_window(moving,obs)

    def test_freshness_and_yaw_approach(self):
        with patch('dynamic_handeye.capture.time.monotonic_ns',return_value=500):
            f=lambda *args,**kw:kw['minimum_stamp_ns']
            self.assertEqual(capture_fresh(f,'robot'),500)
            self.assertEqual(capture_fresh(f,'robot',minimum_stamp_ns=600),600)
        fn=lambda *args,**kw:[np.array([11.,1.]),np.array([10.,0.])]
        path=yaw_reference_waypoints(fn,[0,0],[10,0],[[-90,90],[-25,25]])
        np.testing.assert_allclose(path,[[11,0],[10,0]])
        with self.assertRaises(ValueError):yaw_reference_waypoints(fn,[0,0],[10,-25],[[-90,90],[-25,25]])

    def test_zero_pitch_fixed_replay_requires_explicit_import_target(self):
        row=json.loads((ROOT/'datasets/raw/ball_auto_20261018_visible_v1.jsonl').read_text().splitlines()[0])
        row['head_angles_deg']=[0.,0.];row['head_q_deg']=[0.,0.,0.];row.pop('target_head_deg',None)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'pitch0.jsonl';p.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Head did not reach'):load_capture([p],self.assets)
            result=load_capture([p],self.assets,fixed_head_target=[0,0])
            self.assertEqual(result['rows'][0]['head_key'],'0,0')

    def test_no_sdk_adapter_sets_zero_reference_and_restores_state(self):
        original=ModuleType('interactive_ball_calibration.capture_point_head')
        original.HEAD_POSES=((-10,-25),(10,-25),(0,-20));original.REFERENCE=np.array([0,-25])
        original.fuse_camera_observations=self.fused
        original.capture_at_pose=lambda *args,**kw:kw
        original.head_reference_waypoints=lambda *args,**kw:[np.array([1.,1.]),np.array([0.,0.])]
        saved=(original.HEAD_POSES,original.REFERENCE,original.fuse_camera_observations,original.capture_at_pose,original.head_reference_waypoints,sys.argv)
        def main():
            self.assertEqual(original.HEAD_POSES,((-10,0),(0,0),(10,0)))
            np.testing.assert_array_equal(original.REFERENCE,[0,0])
            self.assertEqual(original.fuse_camera_observations(self.observations())['valid_frames'],50)
            self.assertGreater(original.capture_at_pose()['minimum_stamp_ns'],0)
            np.testing.assert_allclose(original.head_reference_waypoints([0,0],[0,0],[]),[[1,0],[0,0]])
        original.main=main
        package=ModuleType('interactive_ball_calibration');package.capture_point_head=original
        cap=ModuleType('interactive_ball_calibration.capture');cap.confirm_automatic_motion=lambda:True
        modules={'interactive_ball_calibration':package,'interactive_ball_calibration.capture_point_head':original,'interactive_ball_calibration.capture':cap}
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,modules):
            args=SimpleNamespace(plan=ROOT/'configs/capture_pitch0_yaw_plan.json',path=ROOT/'datasets/raw/ball_taught_20261018_visible_v1.json',
                output=Path(t)/'capture.jsonl',stop_after=12,run=False,sdk_root=Path(t)/'NO_SDK',latest=Path(t)/'NO_CAMERA',robot_ip='invalid')
            self.assertEqual(run_capture(args,self.assets)['status'],'plan_checked_no_hardware_access')
            self.assertFalse(list(Path(t).iterdir()))
            args.run=True
            self.assertEqual(run_capture(args,self.assets)['status'],'capture_finished')
            def fail():raise RuntimeError('simulated stop')
            original.main=fail
            with self.assertRaisesRegex(RuntimeError,'simulated stop'):run_capture(args,self.assets)
        self.assertIs(original.REFERENCE,saved[1]);self.assertEqual(original.HEAD_POSES,saved[0])
        self.assertIs(original.fuse_camera_observations,saved[2]);self.assertIs(original.capture_at_pose,saved[3])
        self.assertIs(original.head_reference_waypoints,saved[4]);self.assertIs(sys.argv,saved[5])
