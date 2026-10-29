import unittest
import numpy as np
from dynamic_handeye.geometry import Assets
from reports.submillimeter_study_20261029.experiment import flexible_head,fit_predict,distances


class StudyTests(unittest.TestCase):
    def test_zero_axis_perturbation_is_original_head_and_offset_is_rigid(self):
        q=np.array([[-10,-25],[0,-20],[10,-15],[0,0]])
        np.testing.assert_allclose(flexible_head(q,np.zeros(6)),Assets().head(q),atol=1e-12)
        h=flexible_head(q,[.02,.01,.1,-.1,.01,-.01])
        np.testing.assert_allclose(np.einsum('nji,njk->nik',h[:,:3,:3],h[:,:3,:3]),np.tile(np.eye(3),(4,1,1)),atol=1e-12)
        np.testing.assert_allclose(h[-1],np.eye(4),atol=1e-12)

    def test_new_methods_reproduce_affine_mapping_on_unseen_inputs(self):
        rng=np.random.default_rng(29);train=rng.uniform(-.1,.1,(90,3));test=rng.uniform(-.07,.07,(15,3))
        q=rng.uniform([-10,-25],[10,-15],(90,2));qt=rng.uniform([-10,-25],[10,-15],(15,2))
        def target(p):return p+p@np.diag([.01,-.01,.02])+[.001,-.002,.001]
        for kind in ('matern32','head_additive','local_affine'):
            with self.subTest(kind=kind):
                prediction=fit_predict(kind,train,q,target(train),test,qt,ridge=.01)
                self.assertLess(np.max(np.linalg.norm(prediction-target(test),axis=1)),.0002)
        np.testing.assert_allclose(np.diag(distances(train,train)),0,atol=1e-14)
