import unittest
import numpy as np
from reports.algorithm_search_20261029.compare import fit_correction, correct, predict_pixels
from dynamic_handeye.geometry import Assets, fixed_geometry_hashes, apply


class JointTrendTests(unittest.TestCase):
    def test_global_affine_extrapolation_and_fixed_input(self):
        rng=np.random.default_rng(29)
        train=rng.uniform([.1,-.2,.2],[.4,.2,.6],(80,3))
        held=rng.uniform([.12,-.18,.22],[.42,.18,.62],(20,3))
        matrix=np.array([[.01,0,0],[0,-.005,.002],[.001,0,.003]])
        offset=np.array([.003,-.002,.001])
        target=train+train@matrix+offset
        model=fit_correction(train,target,('joint_affine_gaussian',1,.06,1.))
        error=correct(model,held)-(held+held@matrix+offset)
        self.assertLess(np.max(np.linalg.norm(error,axis=1)),.0001)

    def test_pixel_interface_needs_no_arm_input_and_checks_assets(self):
        a=Assets()
        point=np.array([.01,.02,.4])
        pixels=[]
        for p in a.P:
            x=p@np.r_[point,1]
            pixels.extend(x[:2]/x[2])
        model=dict(assets=a.hashes,fixed_geometry_hashes=fixed_geometry_hashes(a),
                   baseline=dict(x=[0]*12),correction=dict(kind='geometry'))
        with self.assertRaises(ValueError):
            predict_pixels(model,a,pixels,[0,-20])
        got=predict_pixels(model,a,pixels,[0,-20],diagnostic=True)
        np.testing.assert_allclose(got,apply(a.head(np.array([[0,-20]]))[0],point),atol=1e-10)
        model['assets']={}
        with self.assertRaises(ValueError):
            predict_pixels(model,a,pixels,[0,-20],diagnostic=True)


if __name__=='__main__':
    unittest.main()
