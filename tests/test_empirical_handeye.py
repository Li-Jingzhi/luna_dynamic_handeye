"""Joint Gaussian forward/inverse maps and saved pixel interface without arm inputs."""
import json
import unittest
import numpy as np
from dynamic_handeye.geometry import Assets,fixed_geometry_hashes
from dynamic_handeye.empirical import fit_fields,correct,base_points,predict_pixels


class EmpiricalTests(unittest.TestCase):
    def test_physical_point_groups_and_nested_holdouts(self):
        from reports.latest_handeye_20261030.calibrate import partitions
        rng=np.random.default_rng(9);groups=[];batch=[];heads=[];target=[];q7=[]
        for trajectory in ('A','B','C'):
            for point in range(12):
                pos=rng.uniform(-.2,.2,3);joints=rng.uniform(-60,60,7)
                for head in ('-10,-25','0,-20','10,-25'):
                    groups.append(f'{trajectory}:{point}');batch.append(trajectory);heads.append(head)
                    target.append(pos);q7.append(joints)
        data=dict(group=np.array(groups),batch=np.array(batch),heads=np.array(heads),target=np.array(target),q7=np.array(q7))
        ids=np.arange(len(groups))
        for kind in ('trajectory','spatial','configuration','head'):
            seen=[]
            for _,train,test in partitions(data,ids,kind):
                self.assertFalse(set(train)&set(test));seen.extend(test)
                if kind!='head':self.assertFalse(set(data['group'][train])&set(data['group'][test]))
                else:self.assertFalse(set(data['heads'][train])&set(data['heads'][test]))
                for _,inner_train,inner_test in partitions(data,train,kind,count=3):
                    self.assertFalse(set(test)&(set(inner_train)|set(inner_test)))
            self.assertEqual(sorted(seen),ids.tolist())

    def test_joint_affine_recovery_on_unseen_points(self):
        rng=np.random.default_rng(30)
        p=rng.uniform([-.2,-.2,.1],[.3,.2,.6],(90,3))
        test=rng.uniform([-.15,-.15,.15],[.25,.15,.55],(20,3))
        m=np.array([[1.01,.005,0],[0,.995,.003],[.002,0,1.005]])
        target=lambda x:x@m+[.003,-.002,.001]
        fields=fit_fields(p,target(p),.06,[1.,3.])
        for (kind,_),field in fields.items():
            e=correct(field,test)-target(test)
            self.assertLess(np.max(np.linalg.norm(e,axis=1)),.0001)
            if kind=='inverse':
                np.testing.assert_allclose(e,0,atol=1e-10)
                self.assertTrue(field['within_pro_scale_bounds'])

    def test_nonlinear_map_has_unseen_point_benefit(self):
        rng=np.random.default_rng(3);p=rng.uniform(-.1,.1,(150,3));test=rng.uniform(-.08,.08,(20,3))
        target=lambda x:x+np.c_[.004*np.sin(15*x[:,0]),.003*np.cos(15*x[:,1]),.002*np.sin(15*x[:,2])]
        fields=fit_fields(p,target(p),.1,[.01])
        for field in fields.values():
            self.assertLess(np.mean((correct(field,test)-target(test))**2),np.mean((test-target(test))**2)*.05)

    def test_saved_four_pixel_mapping_and_gates(self):
        a=Assets();point=np.array([.01,-.01,.4]);head=np.array([0.,-20.]);baseline=dict(x=[0]*12)
        base=base_points(baseline,a,point[None],head[None])[0]
        model=dict(schema='luna_empirical_handeye_v1',assets=a.hashes,fixed_geometry_hashes=fixed_geometry_hashes(a),
            candidate='synthetic',baseline=baseline,correction=dict(kind='geometry'),
            support=dict(head_deg=[[-10,-25],[10,-25],[10,-15],[-10,-15]],base_centers_m=[base.tolist()],radius_m=.05))
        pixels=[]
        for p in a.P:
            x=p@np.r_[point,1];pixels.extend(x[:2]/x[2])
        model=json.loads(json.dumps(model))
        got=predict_pixels(model,a,pixels,head,diagnostic=True)
        np.testing.assert_allclose(got['xyz_m'],base,atol=1e-10)
        self.assertTrue(got['position_inside_support']);self.assertTrue(got['head_inside_calibrated_hull'])
        with self.assertRaisesRegex(ValueError,'acceptance'):predict_pixels(model,a,pixels,head)
        with self.assertRaises(ValueError):predict_pixels(model,a,[np.nan]*4,head,diagnostic=True)
        model['assets']={}
        with self.assertRaisesRegex(ValueError,'assets changed'):predict_pixels(model,a,pixels,head,diagnostic=True)


if __name__=='__main__':unittest.main()
