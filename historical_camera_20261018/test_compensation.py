import unittest
import numpy as np
from compensation import predict,folds


class CompensationTests(unittest.TestCase):
    def test_affine_recovery_and_rbf_unseen_points(self):
        rng=np.random.default_rng(31)
        c=rng.uniform(-.1,.1,(160,3));x=rng.uniform(-.08,.08,(30,3))
        matrix=np.array([[1.01,.01,0],[0,.99,.01],[.01,0,1.02]])
        def target(p):return p@matrix+[.02,-.01,.3]
        np.testing.assert_allclose(predict(c,target(c),x,('affine',0,0)),target(x),atol=1e-12)
        def nonlinear(p):return target(p)+np.c_[.004*np.sin(p[:,0]*20),.003*np.cos(p[:,1]*20),.002*np.sin(p[:,2]*20)]
        base=predict(c,nonlinear(c),x,('affine',0,0))
        corrected=predict(c,nonlinear(c),x,('affine_rbf',.1,.01))
        self.assertLess(np.mean((corrected-nonlinear(x))**2),np.mean((base-nonlinear(x))**2)*.2)

    def test_nested_groups_keep_outer_test_excluded(self):
        rng=np.random.default_rng(3);t=rng.normal(size=(80,3));q=rng.normal(size=(80,7))
        for protocol in ('trajectory','spatial','configuration'):
            for train,test in folds(np.arange(80),protocol,t,q,5):
                for inner_train,inner_test in folds(train,protocol,t,q,3):
                    self.assertFalse(set(test)&(set(inner_train)|set(inner_test)))
                    self.assertFalse(set(inner_train)&set(inner_test))


if __name__=='__main__':unittest.main()
