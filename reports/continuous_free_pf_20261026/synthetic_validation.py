"""Synthetic unseen-head validation of free-pF geometry plus a known smooth bias."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,pose,apply,fixed_head_root,torso_from_arm
from dynamic_handeye.continuous_free_pf import fit,predict,add_residual,ball
from dynamic_handeye.calibration import metrics


def run():
    a=Assets();rng=np.random.default_rng(20261026)
    training_heads=np.array([(y,p) for y in (-30,-10,10,30) for p in (-25,0,25)],float)
    test_heads=np.array([(y,p) for y in (-20,0,20) for p in (-12.5,12.5)],float)
    q=np.concatenate([np.repeat(training_heads,25,axis=0),np.repeat(test_heads,20,axis=0)])
    n=len(training_heads)*25
    X=np.eye(4);X[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];X[:3,3]=[.08,.02,.06]
    pf=np.array([-.003,.005,.101])
    observed=rng.uniform([-.03,-.03,.32],[.03,.03,.45],(len(q),3))
    yaw,pitch=np.deg2rad(q).T
    local_bias=np.c_[.002*np.sin(yaw)*np.cos(pitch),.002*np.sin(pitch),.002*(observed[:,2]-.38)/.07]
    target_s=apply(fixed_head_root(a)@a.head(q)@X,observed+local_bias)
    target_m=apply(np.linalg.inv(torso_from_arm(a)),target_s)
    poses=np.array([pose(np.r_[rng.normal(0,.6,3),[0,0,0]]) for _ in q])
    poses[:,:3,3]=target_m-np.einsum('nij,j->ni',poses[:,:3,:3],pf)
    geo=fit(observed[:n],q[:n],poses[:n],a)
    corrected=add_residual(geo,observed[:n],q[:n],poses[:n],a,'additive',.1,45.,.1)
    out=dict(scope='Known synthetic smooth bias and unseen interpolated angles; not physical certification',train_heads=training_heads.tolist(),test_heads=test_heads.tolist(),results={})
    for name,m in [('geometry',geo),('additive',corrected)]:
        actual=predict(m,a,observed[n:],q[n:],diagnostic=True)
        fitted_target=apply(torso_from_arm(a),ball(poses[n:],m['pf_m']))
        out['results'][name]=dict(relative_to_fitted_fk=metrics(actual-fitted_target),relative_to_synthetic_truth=metrics(actual-target_s[n:]),pf_mm=(np.array(m['pf_m'])*1000).tolist())
    out['synthetic_target_passed']=out['results']['additive']['relative_to_synthetic_truth']['p95_mm']<1
    assert out['synthetic_target_passed'],out
    Path(__file__).with_name('synthetic_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['results'],indent=2))


if __name__=='__main__':run()
