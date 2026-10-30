import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import time
from unittest.mock import Mock
import cv2
import numpy as np
from data_collection.diagnostics.camera import Camera
from data_collection.diagnostics.camera_server import evidence_folder
from data_collection.workflow import collision_model
from data_collection.diagnostics.robot import Robot
from data_collection.diagnostics.core import planned_step_limit,default_plan

class TransportTests(unittest.TestCase):
    def test_planning_margin_does_not_relax_hard_command_guard(self):
        plan=default_plan();self.assertAlmostEqual(planned_step_limit(plan),.28)
        robot=Robot.__new__(Robot);robot.plan=plan;robot.origin=np.zeros(7);robot.limits=np.tile([-175.,175.],(7,1))
        robot.sample=Mock(return_value=dict(q7_deg=[0]*7))
        with self.assertRaisesRegex(ValueError,'Joint step guard exceeded'):robot.move([.301]*7)
        plan['joint_target_tolerance_deg']=.2
        with self.assertRaises(ValueError):planned_step_limit(plan)

    def test_settle_reuses_only_recent_unchanged_low_velocity_proof(self):
        robot=Robot.__new__(Robot);robot.plan=dict(settle_s=.7,max_window_joint_span_deg=.02)
        robot.sample=Mock(return_value=dict(q7_deg=[0]*7,velocity_deg_s=[0]*7))
        robot.last_settled=(time.monotonic(),np.zeros(7))
        with patch('data_collection.diagnostics.robot.time.sleep',side_effect=AssertionError('extra wait')):
            robot.settle()
            robot.last_settled=(time.monotonic()-10,np.zeros(7))
            with self.assertRaisesRegex(AssertionError,'extra wait'):robot.settle()
            robot.last_settled=(time.monotonic(),np.zeros(7));robot.sample.return_value['velocity_deg_s']=[1]*7
            with self.assertRaisesRegex(AssertionError,'extra wait'):robot.settle()

    def test_remote_path_cannot_escape_evidence_root(self):
        for path in ('','../escape','/tmp/escape'):
            with self.assertRaises(ValueError):evidence_folder(path)
        self.assertTrue(str(evidence_folder('session/hold')).endswith('/session/hold'))

    def test_remote_saved_detection_input_is_not_enhanced_twice(self):
        image=np.full((992,1280,3),120,np.uint8);blob=cv2.imencode('.png',image)[1].tobytes()
        metadata=dict(process_id='same',image_processing=dict(gamma=1.6),evidence_session='session/hold',
            stored_evidence={f'{eye}_{kind}.png':dict(path=f'images/id/{eye}_{kind}.png',sha256=hashlib.sha256(blob).hexdigest())
                for eye in ('left','right') for kind in ('input','rectified','rectified_unenhanced')})
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('metadata.json',json.dumps(metadata))
            for eye in ('left','right'):z.writestr(f'{eye}_rectified.png',blob)
        camera=Camera.__new__(Camera);camera.client=None;camera.port=8768;camera.expected={};camera.gamma=1.6;camera.store='session/hold'
        with patch('data_collection.diagnostics.camera.request',return_value=archive.getvalue()), \
             patch('data_collection.diagnostics.camera.validate_metadata',return_value='identity'), \
             patch('data_collection.diagnostics.camera.enhance_images',side_effect=AssertionError('double enhancement')):
            m,identity,blobs,images=camera.capture(dict(after_ns=1,process_id='same'))
        self.assertEqual(blobs,{})
        np.testing.assert_array_equal(images[0],image)
        self.assertEqual(len(m['stored_evidence']),6)

    def test_collision_model_cache_keeps_source_hash_in_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.xml';path.write_text('<mujoco><compiler/><worldbody/></mujoco>')
            first=collision_model(path,'hash1');self.assertIs(first,collision_model(path,'hash1'))
            self.assertIsNot(first,collision_model(path,'hash2'))
