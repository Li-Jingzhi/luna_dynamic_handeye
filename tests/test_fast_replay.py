"""Exercise complete motion traversal and sparse fifty-frame stops without hardware."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from data_collection import workflow as w
from interactive_ball_calibration import capture as c
from orin_rectified_stream.capture_progress import capture_indices, read_progress


class FastReplayTests(unittest.TestCase):
    def test_partial_prefix_preserves_original_forward_and_reverse_indices(self):
        path=[dict(q7_deg=[i]*7) for i in range(18)]
        self.assertEqual(c.remaining_capture_path(path,[dict(waypoint_index=i) for i in [0,5]],'forward',5),list(range(6,18)))
        self.assertEqual(c.remaining_capture_path(path,[dict(waypoint_index=i) for i in [17,15]],'reverse',5),list(range(14,-1,-1)))
        for indices in ([],[5],[0,10],[0,5,10,15,17]):
            with self.assertRaises(ValueError):c.remaining_capture_path(path,[dict(waypoint_index=i) for i in indices],'forward',5)

    def test_stop_selection_and_invalid_strides(self):
        selected=capture_indices(1003,5)
        self.assertEqual(len(selected),202)
        self.assertEqual((selected[0],selected[-1]),(0,1002))
        self.assertEqual(capture_indices(18),list(range(18)))
        for stride in (0,-1,1.5,True,'5'):
            with self.assertRaises(ValueError):capture_indices(1003,stride)

    def test_both_directions_traverse_every_point_and_capture_fifty_at_selected_stops(self):
        for direction in ('reverse','forward'):
            with self.subTest(direction=direction),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path=root/'trajectory.json';output=root/'capture.jsonl'
                manifest=dict(schema='luna_sh5_taught_path_v1',complete=True,mdh_sha256='mdh',
                    image_size=c.CAMERA_SIZE,sensor_ids=c.CAMERA_IDS,pixel_domain=c.PIXEL_DOMAIN,
                    body_q_deg=[0]*6,trajectory_id='same-path',
                    waypoints=[dict(q7_deg=[3*i]*7) for i in range(18)])
                path.write_text(json.dumps(manifest))
                current=[51. if direction=='reverse' else 0.]*7
                robot=Mock();sdk=SimpleNamespace(FXObjType=SimpleNamespace(OBJ_ARM1=1),state_map={0:'IDLE'})
                robot.current_state.return_value=0
                commands=[]
                def move(arm,target):
                    current[:]=target;commands.append(target[:]);return 0
                robot.runtime_set_joint_pos_cmd.side_effect=move
                def feedback(_):
                    return dict(arms=[{},dict(fb=dict(fb_pos=current[:]))],body=dict(fb_pos=[0]*6))
                observed=[]
                def observation(*args,**kwargs):
                    observed.append(current[0]/3);n=len(observed)
                    return dict(frame_number=n),[0]*4,n,0.,{}
                fused=dict(valid_frames=50,sphere_center_left_camera_m=[0,0,1])
                with patch.object(c,'read_rt',side_effect=feedback),patch.object(c,'assert_fixed'), \
                     patch.object(c,'checked'),patch.object(c,'wait_arm') as settled, \
                     patch.object(c,'head_angles',return_value=np.array([0,-25,0])), \
                     patch.object(c,'camera_observation',side_effect=observation), \
                     patch.object(c,'capture_feedback_frame',return_value={}), \
                     patch.object(c,'fuse_camera_observations',return_value=fused) as fuse, \
                     patch.object(c.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
                    c.replay(robot,sdk,root/'unused',{},'stereo',path,output,'mdh',
                        np.tile([-360.,360.],(7,1)),[0,-25],motion_confirmed=True,
                        direct_approach=True,direction=direction,capture_stride=5)
                order=list(range(18)) if direction=='forward' else list(range(17,-1,-1))
                self.assertEqual([round(q[0]/3) for q in commands],order)
                self.assertEqual(settled.call_count,18)
                expected=[i for i in order if i in capture_indices(18,5)]
                rows=[json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual([r['waypoint_index'] for r in rows],expected)
                self.assertEqual(observed,[i for i in expected for _ in range(50)])
                self.assertTrue(all(len(call.args[0])==50 for call in fuse.call_args_list))
                self.assertTrue(all(r['capture_stride']==5 and r['trajectory_id']=='same-path' for r in rows))
                self.assertIn('5/5 (100.0%)',read_progress(output))
                assets=SimpleNamespace(hashes=dict(mdh='mdh',stereo='stereo'))
                self.assertTrue(w.summarize(output,18,[0,-25],0,assets,5)['complete'])
                with self.assertRaises(ValueError):w.summarize(output,18,[0,-25],0,assets,1)


if __name__=='__main__':unittest.main()
