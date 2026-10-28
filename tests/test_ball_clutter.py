import unittest
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acquisition'))
from capt_head.subpixel_ball import candidates
from capt_head.measurement_contract import load_contract


class BallClutterTests(unittest.TestCase):
    def test_oversized_yellow_regions_do_not_hide_ball(self):
        config,_=load_contract()
        image=np.full((992,1280,3),35,np.uint8)
        for row in range(2):
            for col in range(4):
                cv2.rectangle(image,(col*300+10,row*200+10),(col*300+210,row*200+110),(0,180,230),-1)
        cv2.circle(image,(700,700),25,(0,150,220),-1,lineType=cv2.LINE_AA)
        balls=candidates(image,config)
        self.assertTrue(balls)
        np.testing.assert_allclose(balls[0]['center'],[700,700],atol=.3)


if __name__=='__main__':unittest.main()

class FixedControlTests(unittest.TestCase):
    def test_readback_must_match_both_eyes(self):
        from capt_head.camera_controls import controls,verify_readback
        eye=dict(exposure_ns=5000000,analog_gain=1.,digital_gain=1.,wbmode=5,
                 tnr_mode=1,ee_mode=1,tnr_strength=.5,ee_strength=.5,saturation=1.)
        profile=dict(mode='fixed',eyes=[eye.copy(),eye.copy()])
        readback=[controls(eye),controls(eye)]
        class ArgusEnum(int):
            def __str__(self):return '<enum preset>'
        readback[0]['wbmode']=ArgusEnum(5)
        verify_readback(profile,readback)
        readback[1]['exposuretimerange']='40000 10000000'
        with self.assertRaisesRegex(ValueError,'exposuretimerange'):verify_readback(profile,readback)

class ShadedEdgeTests(unittest.TestCase):
    def test_valid_weaker_side_is_not_dropped_by_percentile(self):
        from capt_head.subpixel_ball import _edge_points
        yy,xx=np.mgrid[:100,:100];angle=np.arctan2(yy-50,xx-50)
        image=np.where((xx-50)**2+(yy-50)**2<=25**2,40+30*np.cos(angle),0).astype(np.uint8)
        theta=np.linspace(0,2*np.pi,360,endpoint=False)
        contour=np.stack([50+25*np.cos(theta),50+25*np.sin(theta)],axis=1).astype(np.float32).reshape(-1,1,2)
        points=_edge_points(image,contour,[50,50])
        self.assertGreater(len(points),.95*len(contour))
