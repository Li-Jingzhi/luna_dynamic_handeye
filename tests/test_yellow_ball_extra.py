import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from data_collection.yellow_ball_extra import head_tasks,move_head,HEADS

class ExtraCaptureTests(unittest.TestCase):
    def test_head_grid_repeated_three_times_with_reference_revisits(self):
        for site in range(3):
            tasks=head_tasks(site);self.assertEqual(len(tasks),33)
            self.assertEqual(len({t['id'] for t in tasks}),33)
            for repeat in range(3):
                group=[t for t in tasks if t['repeat']==repeat]
                self.assertEqual({tuple(t['head_deg']) for t in group[1:-1]},set(HEADS))
                self.assertEqual(sum(t['reference_revisit'] for t in group),2)
                self.assertEqual(group[0]['head_deg'],group[-1]['head_deg'])

    def test_head_motion_never_sends_an_arm_position_command(self):
        raw=Mock();raw.switch_to_idle.return_value=0;raw.switch_to_position_mode.return_value=0
        raw.runtime_set_joint_pos_cmd.return_value=0;raw.current_state.return_value=0
        sdk=SimpleNamespace(FXObjType=SimpleNamespace(OBJ_HEAD=2),state_map={0:'IDLE'})
        rt=dict(arms=[{},dict(fb=dict(fb_pos=[0]*7))],head=dict(fb_pos=[0,-25,0]))
        robot=SimpleNamespace(robot=raw,sdk=sdk,arm=1,position=True,w=SimpleNamespace(read_rt=lambda _:rt),
                              plan=dict(head_deg=[0,-25]),model_check=Mock())
        with patch('interactive_ball_calibration.capture.head_reference_waypoints',return_value=[np.array([1.,-25.])]), \
             patch('gq_calibration.capture.read_limits',return_value=[[-90,90],[-25,25]]), \
             patch('gq_calibration.capture.wait_head'):
            move_head(robot,[1,-25])
        self.assertTrue(raw.runtime_set_joint_pos_cmd.called)
        self.assertTrue(all(c.args[0]==2 for c in raw.runtime_set_joint_pos_cmd.call_args_list))
        self.assertTrue(robot.position)
        self.assertTrue(all(c.args[0]==2 for c in raw.switch_to_idle.call_args_list))

    def test_twenty_frame_window_and_summary(self):
        import contextlib,io,json,tempfile
        from pathlib import Path
        from dynamic_handeye.geometry import Assets
        from data_collection.diagnostics.cli import MODEL
        from data_collection.diagnostics.core import default_plan,validate_plan
        from data_collection.diagnostics.collect import run_session
        from data_collection.diagnostics.simulation import SimRobot,SimCamera
        assets=Assets();model=json.loads(MODEL.read_text());plan=default_plan();plan['samples_per_window']=20
        validate_plan(plan)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(temp)/'capture'
            result=run_session(assets,MODEL,plan,'optical',folder,lambda:SimRobot(assets,model,plan),
                lambda robot:SimCamera(robot,assets),simulated=True,limit=1)
            row=json.loads((folder/'observations.jsonl').read_text())
            self.assertTrue(result['complete']);self.assertEqual(len(row['frames']),20)
            self.assertTrue(row['capture_valid'])
