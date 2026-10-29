import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from data_collection import head_trajectories as h
from dynamic_handeye.geometry import Assets


class HeadTrajectoryTests(unittest.TestCase):
    def test_nine_unique_names_include_side_and_sign(self):
        self.assertEqual(len(h.GRID),9)
        self.assertEqual(len({h.filename('right',q) for q in h.GRID}),9)
        self.assertNotEqual(h.filename('left',[-10,-25]),h.filename('right',[-10,-25]))
        self.assertIn('yaw_m10_pitch_m25',h.filename('right',[-10,-25]))

    def test_dry_run_does_not_access_hardware_or_create_directory(self):
        name='offline_head_grid_check_20261029'
        directory=h.ROOT/'trajectories/by_head_pose'/name
        with patch.object(h,'load_sdk',side_effect=AssertionError('Hardware access')), contextlib.redirect_stdout(io.StringIO()) as out:
            h.main(['teach','--session',name])
        self.assertFalse(directory.exists())
        self.assertFalse(json.loads(out.getvalue())['motion'])

    def test_recorded_pose_cannot_be_reused_for_another_head_or_robot(self):
        a=Assets()
        record=dict(schema='luna_sh5_taught_path_v1',complete=True,arm_side='right',robot_version='luna_v2_0',
            trajectory_id='unique1',target_head_deg=[-10,-25],mdh_sha256=a.hashes['mdh'],robot_identity={'robot_type':'GentoLuna'},
            waypoints=[dict(q7_deg=[i*3]*7) for i in range(18)])
        h.validate_record(record,'right',[-10,-25],a,{'robot_type':'GentoLuna'})
        with self.assertRaises(ValueError):h.validate_record(record,'right',[0,-25],a)
        with self.assertRaises(ValueError):h.validate_record(record,'right',[-10,-25],a,{'robot_type':'MarvinPro'})

    def test_single_head_mode_plans_one_shared_trajectory(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            h.main(['teach','--session','single_head_dry_check','--head','0','0'])
        value=json.loads(out.getvalue())
        self.assertEqual(value['heads'],[[0,0]])
        self.assertEqual(len(value['files']),1)

    def test_both_arms_teach_correct_side_and_save_visibility_without_deleting_connections(self):
        a=Assets()
        sdk=SimpleNamespace(FXObjType=SimpleNamespace(OBJ_ARM0=0,OBJ_ARM1=1),state_map={5:'DragJoint'})
        for index in (0,1):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as temp:
                def rt(q):
                    arms=[dict(fb=dict(fb_pos=[0]*7)),dict(fb=dict(fb_pos=[0]*7))]
                    arms[index]['fb']['fb_pos']=[q]*7
                    return dict(head=dict(fb_pos=[-10,-25,0]),body=dict(fb_pos=[0]*6),arms=arms)
                robot=Mock();robot.current_state.return_value=5;robot.switch_to_drag_joint.return_value=0;robot.switch_to_idle.return_value=0
                readings=[rt(0)]+[rt(i*3) for i in range(18)]
                selection=[([],[],[])]*18+[([1],[],[])]
                terminal=io.StringIO('q\n');out=Path(temp)/'trajectory.json'
                with patch.object(h,'read_rt',side_effect=readings),patch.object(h.select,'select',side_effect=selection), \
                     patch.object(h,'visible',return_value={'pass_detected':False}),patch.object(h.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
                    h.teach_one(robot,sdk,index,[-10,-25],out,a,{'robot_type':'GentoLuna'},terminal,Path('unused'),np.tile([-100,100],(7,1)))
                value=json.loads(out.read_text())
                self.assertTrue(value['complete']);self.assertEqual(len(value['waypoints']),18)
                self.assertEqual(value['visible_waypoints'],0)
                self.assertEqual(value['arm_side'],'right' if index else 'left')
                robot.switch_to_drag_joint.assert_called_once_with(index,2000,h.JOINT_K,h.JOINT_D)
                robot.switch_to_idle.assert_called_once_with(index,2000)

    def test_head_body_and_opposite_arm_guard(self):
        rt=dict(head=dict(fb_pos=[-10,-25,0]),body=dict(fb_pos=[0]*6),arms=[dict(fb=dict(fb_pos=[1]*7)),dict(fb=dict(fb_pos=[0]*7))])
        with self.assertRaises(RuntimeError):h.assert_stationary_others(rt,1,[-10,-25],[0]*6,[0]*7)


if __name__=='__main__':unittest.main()
