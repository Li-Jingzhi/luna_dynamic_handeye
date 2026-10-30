import unittest
import cv2
import numpy as np
from data_collection.diagnostics.camera import enhance_images
from data_collection.diagnostics.core import default_plan,validate_plan

class BrightnessTests(unittest.TestCase):
    def test_detection_png_matches_enhanced_pixels_and_original_is_preserved(self):
        original=np.arange(256,dtype=np.uint8).reshape(16,16)
        image=np.repeat(original[:,:,None],3,axis=2)
        ok,png=cv2.imencode('.png',image);self.assertTrue(ok)
        blobs={f'{eye}_{kind}.png':png.tobytes() for eye in ('left','right') for kind in ('input','rectified')}
        result,saved=enhance_images([image,image],blobs,1.6)
        lut=np.rint(255*(np.arange(256)/255)**(1/1.6)).astype(np.uint8)
        np.testing.assert_array_equal(result[0],lut[image])
        self.assertEqual(saved['left_rectified_unenhanced.png'],blobs['left_rectified.png'])
        self.assertEqual(saved['left_input.png'],blobs['left_input.png'])
        decoded=cv2.imdecode(np.frombuffer(saved['left_rectified.png'],np.uint8),cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded,result[0])
        np.testing.assert_array_equal(image[:,:,0],original)

    def test_plan_gamma_bounds(self):
        for gamma in (0,3,float('nan'),True):
            plan=default_plan();plan['image_gamma']=gamma
            with self.assertRaises(ValueError):validate_plan(plan)
