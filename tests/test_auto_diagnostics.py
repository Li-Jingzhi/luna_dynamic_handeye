"""Offline contract, closed-loop orchestration, evidence and failure-path verification."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from dynamic_handeye.geometry import Assets
from data_collection.diagnostics.core import default_plan,validate_plan,tasks,servo_step,guard_q,validate_metadata,window_summary,ROOT
from data_collection.diagnostics.cli import main,MODEL
from data_collection.diagnostics.collect import run_session,Collector
from data_collection.diagnostics.simulation import SimRobot,SimCamera
from data_collection.diagnostics.analyze import analyze


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.a=Assets();cls.model=json.loads(MODEL.read_text())

    def test_dry_commands_never_open_network_or_sdk(self):
        with patch('data_collection.diagnostics.camera.connect',side_effect=AssertionError('network forbidden')),patch('data_collection.diagnostics.robot.Robot',side_effect=AssertionError('SDK forbidden')):
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                target=Path(tmp)/'not_created'
                main(['check']);main(['deploy-camera']);main(['collect','--output',str(target)])
                self.assertFalse(target.exists())

    def test_plan_and_joint_guards(self):
        p=default_plan();validate_plan(p)
        self.assertEqual([len(tasks(p,x)) for x in ('optical','hold','return','acceptance')],[15,48,120,30])
        for key,value in [('max_joint_step_deg',3),('max_camera_excursion_m',.3),('samples_per_window',49),('ball_diameter_m',.05)]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):validate_plan(bad)
        with self.assertRaises(ValueError):guard_q(np.ones(7)*13,np.zeros(7),np.tile([-170,170],(7,1)),p)

    def test_servo_small_steps_and_nullspace(self):
        rng=np.random.default_rng(3);j=rng.normal(size=(3,7))*.005
        q=np.zeros(7);desired=np.ones(7)*4
        step,null=servo_step(j,np.zeros(3),q,desired,.3)
        self.assertLessEqual(np.max(abs(step)),.30000001)
        self.assertLess(np.linalg.norm(j@step),.00001)
        err=np.array([.01,-.005,.002]);step,_=servo_step(j,err,q,None,.3)
        self.assertLess(np.linalg.norm(err-j@step),np.linalg.norm(err))

    def test_pairing_and_capture_barrier(self):
        expected=dict(source_calibration_sha256='a',stereo_model_sha256='b',controls_sha256='c')
        m=dict(service='luna_diagnostic_camera_v1',process_id='one',image_size=[1280,992],controls_verified=True,
            exposure_time_valid=False,source_clock_ns=[100_000_000,101_000_000],arrival_monotonic_ns=[200_000_000,201_000_000],**expected)
        validate_metadata(m,expected,199_000_000)
        with self.assertRaises(ValueError):validate_metadata(m,expected,200_000_000)
        m['source_clock_ns'][1]+=20_000_000
        with self.assertRaises(ValueError):validate_metadata(m,expected)

    def test_four_automatic_experiments_and_frozen_analysis(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()),patch('data_collection.diagnostics.camera.connect',side_effect=AssertionError('no network')):
            path=Path(tmp)/'suite'
            main(['simulate','--experiment','all','--limit','2','--output',str(path)])
            report=analyze(path)
            self.assertEqual(set(report['experiments']),{'optical','hold','return','acceptance'})
            self.assertTrue(report['simulated'])
            self.assertFalse(report['experiments']['acceptance']['calibrated_fk_consistency_passed'])
            for kind in report['experiments']:
                self.assertEqual(report['experiments'][kind]['valid'],2)
                self.assertFalse(report['experiments'][kind]['model_refitted'])
                self.assertFalse(report['experiments'][kind]['independent_absolute_scale_verified'])
            comparison=report['experiments']['return']['return_comparisons'][0]
            self.assertEqual(comparison['repeats_from_below'],1);self.assertEqual(comparison['repeats_from_above'],1)
            self.assertLess(report['experiments']['optical']['optical_consistency'][0]['nominal_diameter_mono_vs_stereo_p95_mm'],1e-5)
            observation=json.loads((path/'hold/observations.jsonl').read_text().splitlines()[0])
            evidence=next(iter(observation['frames'][0]['image_files'].values()))
            image=path/'hold'/evidence['path'];image.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'evidence'):analyze(path/'hold')

    def test_completed_resume_is_read_only_and_model_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            folder=Path(tmp)/'run';p=default_plan()
            factory=lambda:SimRobot(self.a,self.model,p)
            run_session(self.a,MODEL,p,'optical',folder,factory,lambda r:SimCamera(r,self.a),simulated=True,limit=1)
            result=run_session(self.a,MODEL,p,'optical',folder,lambda:(_ for _ in ()).throw(AssertionError('no reconnect')),None,simulated=True,resume=True,limit=1)
            self.assertTrue(result['already_complete'])
            with (folder/'frozen_model.json').open('a') as f:f.write(' ')
            with self.assertRaisesRegex(ValueError,'Frozen model'):analyze(folder)

    def test_resources_close_when_camera_initialization_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=default_plan();robot=SimRobot(self.a,self.model,p);closed=[]
            robot.close=lambda:closed.append(True)
            folder=Path(tmp)/'failure'
            with self.assertRaisesRegex(RuntimeError,'offline'):
                run_session(self.a,MODEL,p,'hold',folder,lambda:robot,
                    lambda r:(_ for _ in ()).throw(RuntimeError('offline')),simulated=True,limit=1)
            self.assertEqual(closed,[True]);self.assertFalse(json.loads((folder/'status.json').read_text())['complete'])
            self.assertFalse((folder/'completed.json').exists())

    def test_duplicate_source_and_rejected_image_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);dest=folder/'assets/configs/camera';dest.mkdir(parents=True)
            shutil.copyfile(ROOT/'configs/camera/luna_v2_fixed_controls.json',dest/'luna_v2_fixed_controls.json')
            p=default_plan();robot=SimRobot(self.a,self.model,p);camera=SimCamera(robot,self.a)
            collector=Collector(self.a,self.model,p,robot,camera,folder);barrier=camera.barrier();original=camera.capture(barrier)
            with patch.object(camera,'capture',return_value=original):
                collector.measure(barrier,save=False)
                with self.assertRaisesRegex(ValueError,'reused'):collector.measure(barrier,save=False)
            rejected=camera.capture(barrier);rejected[0]['synthetic_measurement']['quality']='REJECT'
            with patch.object(camera,'capture',return_value=rejected),self.assertRaises(ValueError):collector.measure(barrier)
            self.assertTrue((folder/'camera_evidence.jsonl').exists())
            self.assertEqual(len(list((folder/'images').glob('*/*.png'))),4)


if __name__=='__main__':unittest.main()
