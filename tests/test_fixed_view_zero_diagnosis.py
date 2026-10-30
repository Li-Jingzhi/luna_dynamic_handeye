"""Identifiable synthetic joint/scale hypotheses and physically grouped holdouts."""
import unittest
import numpy as np
from dynamic_handeye.geometry import Assets,fixed_geometry_hashes
from reports.fixed_view_zero_diagnosis_20261030.analyze import target_fk,fit,predict,block_splits
from reports.fixed_view_zero_diagnosis_20261030.predict import predict_pixels


class FixedZeroTests(unittest.TestCase):
    def test_camera_only_fixed_view_interface_and_rejection(self):
        a=Assets();point=np.array([.01,-.01,.4]);pixels=[]
        for p in a.P:
            x=p@np.r_[point,1];pixels.extend(x[:2]/x[2])
        model=dict(schema='luna_fixed_view_calibrated_fk_v1',assets=a.hashes,fixed_geometry_hashes=fixed_geometry_hashes(a),fixed_head_deg=[0,-25],
            fixed_head_tolerance_deg=.02,camera_scale=1.,effective_joint_offsets_deg=[0]*7,
            camera_to_torso=dict(R=np.eye(3).tolist(),t_m=[0,0,0]),correction=dict(kind='geometry'))
        got=predict_pixels(model,a,pixels,[0,-25.001],diagnostic=True)
        np.testing.assert_allclose(got['xyz_m'],point,atol=1e-10)
        with self.assertRaisesRegex(ValueError,'acceptance'):predict_pixels(model,a,pixels,[0,-25])
        with self.assertRaisesRegex(ValueError,'fixed view'):predict_pixels(model,a,pixels,[0,-20],diagnostic=True)
        model['assets']={}
        with self.assertRaisesRegex(ValueError,'assets changed'):predict_pixels(model,a,pixels,[0,-25],diagnostic=True)

    def test_known_offsets_recovered_without_hardware_changes(self):
        a=Assets();rng=np.random.default_rng(30)
        q=rng.uniform([-70,-65,-55,-110,-60,-35,-60],[50,60,55,10,60,35,60],(500,7))
        known=np.array([0,.15,.65,-1.3,.05,-.7,-.9])
        target=target_fk(a,q,known)
        angle=.4;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        camera=target@r.T+[.03,-.1,.05];groups=np.array(['view']*len(q))
        result=fit(a,camera,q,groups,'zeros')
        self.assertLess(np.max(np.abs(np.array(result['effective_offsets_deg'])-known)),.03)
        self.assertLess(result['training_corrected_fk']['p95_mm'],.02)
        self.assertGreater(result['training_original_fk']['p95_mm'],1)
        self.assertEqual(result['data_only_jacobian']['rank'],6)
        np.testing.assert_allclose(predict(result,camera,groups),target,atol=.00003)

    def test_camera_scale_hypothesis_preserves_original_fk(self):
        a=Assets();rng=np.random.default_rng(3);q=rng.uniform(-40,40,(100,7))
        target=target_fk(a,q,np.zeros(7));camera=target/1.006+[.02,-.03,.1]
        model=fit(a,camera,q,np.array(['fixed']*len(q)),'scale')
        self.assertAlmostEqual(model['camera_scale'],1.006,places=4)
        self.assertLess(model['training_original_fk']['p95_mm'],.02)
        np.testing.assert_array_equal(model['effective_offsets_deg'],np.zeros(7))

    def test_all_head_observations_stay_with_physical_point(self):
        groups=[];batch=[]
        for b in ('A','B','C'):
            for p in range(25):
                for _ in range(3):groups.append(f'{b}:{p}');batch.append(b)
        d=dict(group=np.array(groups),batch=np.array(batch));ids=np.arange(len(groups));held=[]
        for _,train,test in block_splits(d,ids):
            self.assertFalse(set(d['group'][train])&set(d['group'][test]));held.extend(test)
        self.assertEqual(sorted(held),ids.tolist())


if __name__=='__main__':unittest.main()
