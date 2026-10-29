import unittest
import numpy as np
from reports.pro_static_comparison_20261029.compare import predict


class StaticComparisonTests(unittest.TestCase):
    def test_all_models_recover_same_rigid_mapping_on_unseen_points(self):
        rng=np.random.default_rng(29)
        target=rng.uniform([-.1,-.1,.2],[.1,.1,.5],(70,3))
        held=rng.uniform([-.09,-.09,.21],[.09,.09,.49],(20,3))
        angle=.3;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        camera=target@r.T+[.02,-.03,.1];test=held@r.T+[.02,-.03,.1]
        for method in ('rigid','affine','joint_camera_gaussian_l60_a3'):
            with self.subTest(method=method):
                result,info=predict(camera,target,test,method)
                np.testing.assert_allclose(result,held,atol=1e-10)
                if info:
                    self.assertTrue(info['within_Pro_scale_bounds'])
                    self.assertTrue(info['within_Pro_element_seed_bounds'])
