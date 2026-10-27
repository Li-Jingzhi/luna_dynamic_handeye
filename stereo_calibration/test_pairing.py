import unittest
import numpy as np
from stereo_pairing import StereoPairer
from capture_realtime import mapping_parameters


def frame(pts,arrival):
    return (None,pts*1_000_000,arrival*1_000_000,pts*1_000_000)


class PairingTests(unittest.TestCase):
    def test_both_gates_and_single_consumption(self):
        p=StereoPairer()
        p.push(0,frame(100,100));p.push(1,frame(110,110))
        self.assertIsNotNone(p.take_latest());self.assertIsNone(p.take_latest())
        p.push(0,frame(200,200));p.push(1,frame(201,211))
        self.assertIsNone(p.take_latest())
        self.assertEqual(p.stats['arrival_gate_drops'],[1,1])
        p.push(0,frame(300,300));p.push(1,frame(311,301))
        self.assertIsNone(p.take_latest())
        self.assertEqual(p.stats['unmatched_drops'][0],1)

    def test_pending_pair_drop_is_counted(self):
        p=StereoPairer()
        for t in (100,200):
            p.push(0,frame(t,t));p.push(1,frame(t,t))
        self.assertEqual(p.stats['unconsumed_pair_drops'],1)
        self.assertEqual(p.take_latest()[0][1],200_000_000)

    def test_half_size_mapping_preserves_translation(self):
        c={}
        k=np.array([[600.,0,640],[0,610,496],[0,0,1]])
        for i,eye in enumerate(('left','right'),1):
            c[f'{eye}_K']=k;c[f'{eye}_D']=np.zeros(4)
            c[f'rectify_R{i}']=np.eye(3)
            c[f'rectify_P{i}']=np.c_[k,[-36 if i==2 else 0,0,0]]
        params=mapping_parameters(c)
        np.testing.assert_allclose(params[1]['Kin'],[[300,0,320],[0,305,248]])
        np.testing.assert_allclose(params[1]['Kout'],params[1]['Kin'])
        np.testing.assert_allclose(params[0]['X'],np.c_[np.eye(3),np.zeros(3)])


if __name__=='__main__':unittest.main()
