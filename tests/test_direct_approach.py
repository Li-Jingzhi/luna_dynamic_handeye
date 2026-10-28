import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acquisition'))
from interactive_ball_calibration.capture import replay_position_waypoints


class DirectApproachTests(unittest.TestCase):
    def test_direct_skips_detour_but_bounds_joint_steps(self):
        path=[dict(q7_deg=[x]*7) for x in [0,3,6,9,12,9,6,3]]
        self.assertEqual(len(replay_position_waypoints(np.zeros(7),path)),7)
        direct=replay_position_waypoints(np.zeros(7),path,direct=True)
        self.assertEqual(len(direct),1)
        np.testing.assert_allclose(direct[-1]['q7_deg'],[3]*7)
        path[-1]['q7_deg']=[17]*7
        points=np.array([[0]*7]+[w['q7_deg'] for w in replay_position_waypoints(np.zeros(7),path,direct=True)])
        self.assertLessEqual(np.max(np.abs(np.diff(points,axis=0))),5)
        self.assertEqual(replay_position_waypoints(np.array([17]*7),path,direct=True),[])
