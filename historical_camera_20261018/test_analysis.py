"""Check fixed-head SE3 scoring, held-out evaluation and rigid distance lower bound."""
import unittest
import numpy as np
from analyze import evaluate


class SyntheticAssets:
    def triangulate(self, pixels): return np.asarray(pixels[:3])


class AuditTests(unittest.TestCase):
    def test_rigid_recovery_and_block_separation(self):
        rng=np.random.default_rng(30)
        camera=rng.uniform([-.1,-.1,.2],[.1,.1,.6],(80,3))
        theta=.3
        r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        target=camera@r.T+[.03,-.02,.1]
        rows=[dict(waypoint_id=i,corrected_pixels=[*c,0],camera_xyz_m=c,
                   target_s_m=t,q7_deg=rng.uniform(-50,50,7),head_key='0,-25',session_id='synthetic')
              for i,(c,t) in enumerate(zip(camera,target))]
        result,points=evaluate(rows,SyntheticAssets())
        self.assertLess(result['training']['max_mm'],1e-8)
        for protocol in result['cv'].values():
            self.assertLess(protocol['pooled']['max_mm'],1e-8)
            tests=[]
            for fold in protocol['folds']:
                self.assertFalse(set(fold['train_waypoints'])&set(fold['test_waypoints']))
                tests.extend(fold['test_waypoints'])
            self.assertEqual(sorted(tests),list(range(80)))
        rows[-1]['target_s_m']=target[-1]+[.02,0,0]
        distorted,_=evaluate(rows,SyntheticAssets())
        self.assertGreater(distorted['rigid_max_error_lower_bound_mm'],1)
        self.assertLessEqual(distorted['rigid_max_error_lower_bound_mm'],distorted['training']['max_mm']+1e-8)


if __name__=='__main__':unittest.main()
