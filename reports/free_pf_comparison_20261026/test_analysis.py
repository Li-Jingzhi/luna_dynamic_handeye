"""Self-contained checks for diagnostic variable projection and kernel prediction."""
import copy
import unittest
import numpy as np
from analyze import fit_model, baseline, rbf, predict_selected, splits
from dynamic_handeye.geometry import pose, apply


class DiagnosticTests(unittest.TestCase):
    def test_recovers_shared_pf_and_independent_head_maps(self):
        rng=np.random.default_rng(26)
        n=50
        poses=np.array([pose(np.r_[rng.normal(0,.5,3),rng.uniform(-.2,.2,3)]) for _ in range(n)])
        poses=np.repeat(poses,3,axis=0)
        pf=np.array([-.003,.005,.101])
        points=poses[:,:3,3]+np.einsum('nij,j->ni',poses[:,:3,:3],pf)
        heads=np.tile(['a','b','c'],n)
        camera=np.empty_like(points)
        for h,rotation in zip(('a','b','c'),(-.2,0,.2)):
            camera[heads==h]=apply(pose([rotation,.1,-.1,0,0,.8]),points[heads==h])
        model=fit_model(poses,camera,heads)
        np.testing.assert_allclose(model['pf_m'],pf,atol=1e-7)
        np.testing.assert_allclose(baseline(model,camera,heads),points,atol=1e-7)
        centers=camera[:12];b=points[:12]
        diagnostic=dict(rigid=copy.deepcopy(model),spec=['additive',.02,1.],
            centers_camera_m=centers.tolist(),centers_base_m=b.tolist(),heads=heads[:12].tolist(),
            weights_m=np.zeros_like(centers).tolist())
        diagnostic['rigid'].pop('pf_m')
        np.testing.assert_allclose(predict_selected(camera,heads,diagnostic),points,atol=1e-7)
        K=rbf(centers,centers,.02)
        np.testing.assert_allclose(np.diag(K),1)
        np.testing.assert_allclose(K,K.T)
        groups=np.repeat(np.arange(n).astype(str),3)
        q=rng.normal(size=(len(poses),7))
        for _,_,tr,te in splits(groups,poses,q,np.arange(len(poses))):
            self.assertFalse(set(groups[tr]) & set(groups[te]))


if __name__=='__main__':unittest.main()
