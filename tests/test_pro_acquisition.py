import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acquisition'))
from capt_head.subpixel_ball import stereo_measure
from capt_head.measurement_contract import load_contract,valid_measurement
from orin_rectified_stream.stereo_pairing import StereoPairer

class ProAcquisitionTests(unittest.TestCase):
    def test_corrected_gate_and_raw_evidence(self):
        config,_=load_contract()
        left={'center':[650.,480.],'ellipse_rms_px':.1}
        right={'center':[600.,480.6],'ellipse_rms_px':.1}
        sphere={'corrected_stereo_pixels':[650.,480.,600.,480.4],'center_left_camera_m':[0.,0.,.5]}
        with patch('capt_head.subpixel_ball.candidates',side_effect=[[left],[right]]),patch('interactive_ball_calibration.geometry.stereo_sphere_center',return_value=sphere):
            out=stereo_measure([None,None],config,stereo={})
        self.assertEqual(out['quality'],'PASS')
        self.assertAlmostEqual(out['raw_epipolar_dv_px'],-.6)
        self.assertAlmostEqual(out['epipolar_dv_px'],-.4)
        sphere['corrected_stereo_pixels'][3]=480.7
        with patch('capt_head.subpixel_ball.candidates',side_effect=[[left],[right]]),patch('interactive_ball_calibration.geometry.stereo_sphere_center',return_value=sphere):
            self.assertEqual(stereo_measure([None,None],config,stereo={})['quality'],'REJECT')

    def test_receiver_requires_source_gate_and_corrected_geometry(self):
        c,_=load_contract()
        def detection(u,v):
            return dict(center=[u,v],ellipse=dict(cx=u,cy=v,major_px=30,minor_px=29),edge_count=100,ellipse_rms_px=.1,edge_coverage_deg=300)
        x=dict(schema='luna_yellow_measurement_v1',quality='PASS',image_size=c['image_size'],pixel_domain=c['pixel_domain'],sensor_ids=c['sensor_ids'],measurement_backend=c['measurement_backend'],source_calibration_sha256=c['calibration_sha256'],contract_sha256=c['contract_sha256'],exposure_locked=True,awb_locked=True,brightness_stable=True,antibanding_hz=50,measurement_fps=28,pair_capture_skew_ms=2,pair_source_clock_skew_ms=3,pairing_policy='shared_pipeline_source_and_arrival_10ms',epipolar_check_domain='sphere_corrected_projection',left=detection(650,480),right=detection(600,480.6),corrected_stereo_pixels=[650,480,600,480.4],disparity_px=50,epipolar_dv_px=-.4)
        valid_measurement(x,c)
        x['pair_source_clock_skew_ms']=11
        with self.assertRaises(ValueError):valid_measurement(x,c)

    def test_each_pair_consumed_once_and_both_time_gates(self):
        p=StereoPairer()
        p.push(0,(None,1,0,0));p.push(1,(None,1,1_000_000,1_000_000))
        self.assertIsNotNone(p.take_latest());self.assertIsNone(p.take_latest())
        p.push(0,(None,2,30_000_000,30_000_000));p.push(1,(None,2,50_000_000,31_000_000))
        self.assertIsNone(p.take_latest())
