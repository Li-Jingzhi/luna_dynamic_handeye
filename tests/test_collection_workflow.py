import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch, Mock, mock_open

from data_collection import workflow as w
from dynamic_handeye.geometry import Assets


class WorkflowTests(unittest.TestCase):
    def test_invalid_plans_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'plan.json'
            for angles, frames in [([[0,-25],[0,-25]],50), ([[0,-26]],50), ([[float('nan'),0]],50), ([[0,0]],49)]:
                p.write_text(json.dumps(dict(head_angles_deg=angles,observations_per_position=frames)))
                with self.assertRaises(ValueError):
                    w.read_plan(p)

    def test_partial_trajectory_not_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'trajectory.json'
            p.write_text(json.dumps(dict(complete=False)))
            with self.assertRaises(ValueError):
                w.validate_trajectory(p,Assets())

    def test_summary_preserves_rejected_points_and_incomplete_state(self):
        a = Assets()
        rows = [dict(waypoint_index=i,target_head_deg=[0,-25],mdh_sha256=a.hashes['mdh'],
                     stereo_model_sha256=a.hashes['stereo'],capture_valid=(i==0),valid_frames=50) for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'data.jsonl'
            p.write_text('\n'.join(map(json.dumps, rows))+'\n')
            summary=w.summarize(p,3,[0,-25],-1,a)
            self.assertFalse(summary['complete'])
            self.assertEqual((summary['valid'],summary['invalid']),(1,1))
            rows[1]['waypoint_index']=0
            p.write_text('\n'.join(map(json.dumps, rows))+'\n')
            with self.assertRaises(ValueError):
                w.summarize(p,3,[0,-25],0,a)

    def test_dry_run_does_not_connect_or_create_session(self):
        session='offline_test_no_hardware_20261029'
        directory=w.ROOT/'datasets/raw'/session
        self.assertFalse(directory.exists())
        with patch.object(w,'connected',side_effect=AssertionError('hardware accessed')), contextlib.redirect_stdout(io.StringIO()) as output:
            w.main(['--session',session,'--plan',str(w.ROOT/'data_collection/plans/validation.json')])
        self.assertFalse(directory.exists())
        self.assertFalse(json.loads(output.getvalue())['motion'])

    def test_child_interrupt_is_forwarded_and_waited(self):
        child=Mock()
        child.wait.side_effect=[KeyboardInterrupt(),0]
        with patch('builtins.open',mock_open()), patch.object(w.subprocess,'Popen',return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                w.run_child(['unused'])
        child.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertEqual(child.wait.call_count,2)

    def test_full_replay_orchestration_and_quality_stop_without_hardware(self):
        assets=Assets()
        for reject_count,stride,alternate in ((0,1,False),(12,1,False),(0,5,True)):
            with self.subTest(reject_count=reject_count,stride=stride), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                (root/'datasets/raw').mkdir(parents=True)
                (root/'acquisition').mkdir()
                plan=root/'plan.json'
                plan.write_text(json.dumps(dict(head_angles_deg=[[0,-25],[0,-20]],observations_per_position=50,capture_stride=stride,alternate_direction=alternate)))
                trajectory=root/'trajectory.json'
                trajectory.write_text(json.dumps(dict(schema='luna_sh5_taught_path_v1',complete=True,
                    arm_side='right',sdk_version='00040701',pixel_domain='ideal_pinhole',
                    mdh_sha256=assets.hashes['mdh'],stereo_model_sha256=assets.hashes['stereo'],
                    image_size=[1280,992],sensor_ids=assets.stereo['sensor_ids'],body_q_deg=[0]*6,
                    waypoints=[dict(q7_deg=[i*3]*7) for i in range(18)])))
                rt=dict(body=dict(fb_pos=[0]*6),arms=[dict(fb=dict(fb_pos=[0]*7)),dict(fb=dict(fb_pos=[51]*7))])
                @contextlib.contextmanager
                def connection(*args):
                    yield Mock(),Mock()
                def child(command,log_path=None):
                    output=Path(command[command.index('--output')+1])
                    i=command.index('--head-target')
                    head=list(map(float,command[i+1:i+3]))
                    rows=[dict(waypoint_index=n,target_head_deg=head,mdh_sha256=assets.hashes['mdh'],
                        stereo_model_sha256=assets.hashes['stereo'],capture_valid=n>=reject_count,valid_frames=50,capture_stride=stride) for n in w.capture_indices(18,stride)]
                    output.write_text('\n'.join(map(json.dumps,rows))+'\n')
                    return 0
                with patch.object(w,'ROOT',root), patch.object(w,'connected',connection), \
                     patch.object(w,'read_rt',return_value=rt), patch.object(w,'read_limits',return_value=[[-90,90],[-25,25]]), \
                     patch.object(w,'prepare'), patch.object(w,'start_receiver'), patch.object(w,'load_capture') as imported, \
                     patch.object(w,'run_child',side_effect=child) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    args=['--session','mock','--plan',str(plan),'--trajectory',str(trajectory),'--run']
                    if reject_count:
                        with self.assertRaisesRegex(RuntimeError,'80%'):
                            w.main(args)
                        rt['arms'][1]['fb']['fb_pos']=[0]*7
                        first_hash=w.digest(root/'datasets/raw/mock/head_00.jsonl')
                        with self.assertRaisesRegex(RuntimeError,'continue-low-valid'):
                            w.main(args+['--resume'])
                        rt['arms'][1]['fb']['fb_pos']=[10]*7
                        with self.assertRaisesRegex(RuntimeError,'Resume robot pose changed'):
                            w.main(args+['--resume','--continue-low-valid'])
                        self.assertEqual(run.call_count,1)
                        rt['arms'][1]['fb']['fb_pos']=[0]*7
                        w.main(args+['--resume','--continue-low-valid'])
                        self.assertEqual(first_hash,w.digest(root/'datasets/raw/mock/head_00.jsonl'))
                        self.assertEqual(run.call_count,2)
                        status=json.loads((root/'datasets/raw/mock/status.json').read_text())
                        self.assertEqual(status['phase'],'complete')
                        first=root/'datasets/raw/mock/head_00.jsonl'
                        first.write_text(first.read_text()+'\n')
                        with self.assertRaisesRegex(ValueError,'tampered'):
                            w.main(args+['--resume','--continue-low-valid'])
                    else:
                        w.main(args)
                status=json.loads((root/'datasets/raw/mock/status.json').read_text())
                self.assertEqual(status['phase'],'complete')
                self.assertEqual(run.call_count,2)
                self.assertGreaterEqual(imported.call_count,run.call_count)
                directions=[call.args[0][call.args[0].index('--direction')+1] for call in run.call_args_list]
                self.assertEqual(directions,['reverse','forward' if alternate else 'reverse'])
                for call in run.call_args_list:
                    command=call.args[0]
                    self.assertEqual(command[command.index('--capture-stride')+1],str(stride))


if __name__=='__main__':
    unittest.main()
