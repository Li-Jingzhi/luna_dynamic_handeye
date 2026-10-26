"""Continuous arbitrary-angle math, free-pF recovery, smooth kernels and approval gate."""
import copy
import unittest
import numpy as np
from dynamic_handeye.geometry import Assets,pose,apply,fixed_head_root,torso_from_arm
from dynamic_handeye.continuous_free_pf import fit,predict,kernel


class ContinuousFreePFTests(unittest.TestCase):
    def test_synthetic_unseen_angles_free_pf_and_gates(self):
        a=Assets();rng=np.random.default_rng(2610)
        q=np.tile(np.array([(y,p) for y in (-20,0,20) for p in (-25,-15,-5)],float),(10,1))
        X=np.eye(4);X[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];X[:3,3]=[.08,.02,.06]
        pf=np.array([-.003,.005,.101])
        target_s=rng.uniform([.30,-.06,.43],[.46,.06,.60],(len(q),3))
        target_m=apply(np.linalg.inv(torso_from_arm(a)),target_s)
        poses=np.array([pose(np.r_[rng.normal(0,.6,3),[0,0,0]]) for _ in q])
        poses[:,:3,3]=target_m-np.einsum('nij,j->ni',poses[:,:3,:3],pf)
        camera=apply(np.linalg.inv(fixed_head_root(a)@a.head(q)@X),target_s)
        model=fit(camera,q,poses,a)
        np.testing.assert_allclose(model['pf_m'],pf,atol=1e-6)
        unseen=np.array([[-12,-21],[7,-12],[14,-8]],float)
        points=np.array([[0,0,.35],[.02,.01,.40],[-.01,.02,.38]])
        expected=apply(fixed_head_root(a)@a.head(unseen)@X,points)
        actual=predict(model,a,points,unseen,diagnostic=True)
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        with self.assertRaisesRegex(ValueError,'acceptance'):predict(model,a,points,unseen)
        with self.assertRaises(ValueError):predict(model,a,points,np.array([[0,-26]]*3),diagnostic=True)
        # Position prediction itself must not read pF or arm q7.
        stripped=copy.deepcopy(model);stripped.pop('pf_m')
        np.testing.assert_allclose(predict(stripped,a,points,unseen,diagnostic=True),actual)
        for kind in ('spatial','conditional','additive'):
            spec=dict(kind=kind,length_m=.05,angle_length_deg=15.)
            k=kernel(points,unseen,points,unseen,spec)
            np.testing.assert_allclose(k,k.T,atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-10)
            perturbed=kernel(points,unseen+1e-6,points,unseen,spec)
            self.assertLess(np.max(abs(perturbed-k)),1e-6)


if __name__=='__main__':unittest.main()
