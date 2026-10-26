"""Literal RA=Ry(feedback pitch) refit versus constant pitch-root hypotheses."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,apply,pose,parameters,fixed_head_root,torso_from_arm,digest
from dynamic_handeye.dataset import load_capture,arrays
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.optimizer import robust_optimize
from dynamic_handeye.continuous_free_pf import ball
from dynamic_handeye.calibration import metrics
from reports.fk_consistent_vision_20261026.analyze import rotation,space_kernel


def root_rotation(q,beta=None):
    return rotation([0,1,0],np.deg2rad(q[:,1]) if beta is None else np.full(len(q),beta))


def forward(a,c,q,t,x,beta=None):
    r=root_rotation(q,beta)@a.head(q)[:,:3,:3]
    return np.einsum('nij,nj->ni',r,apply(x,c))+t


def jacobian(fun,x):
    j=np.column_stack([(fun(x+np.eye(len(x))[i]*1e-5)-fun(x-np.eye(len(x))[i]*1e-5))/2e-5 for i in range(len(x))])
    sv=np.linalg.svd(j,compute_uv=False);r=fun(x)
    weight=np.minimum(1.,3/np.maximum(abs(r),1e-12))
    grad=float(np.max(abs(j.T@(weight*r))/np.maximum(np.linalg.norm(j,axis=0),1)))
    return dict(rank=int(sum(sv>sv[0]*1e-7)),singular_values=sv.tolist(),normalized_gradient=grad,optimizer_stationary=grad<.01)


def fit(a,c,q,y,kind):
    t=fixed_head_root(a)[:3,3];beta=None if kind=='feedback_q1' else np.deg2rad(25.)
    k=root_rotation(q,beta)@a.head(q)[:,:3,:3]
    local=np.einsum('nji,nj->ni',k,y-t)
    r,tx=rigid_fit(c,local);x=np.eye(4);x[:3,:3]=r;x[:3,3]=tx
    if kind=='feedback_q1':
        # Archived poses have only one nonzero-yaw pitch: one translation gauge remains.
        initial=np.r_[t[:2],parameters(x)]
        def unpack(z):return np.r_[z[:2],t[2]],pose(z[2:]),None
    elif kind=='constant_25deg':
        initial=np.r_[t,parameters(x)]
        def unpack(z):return z[:3],pose(z[3:]),np.deg2rad(25.)
    elif kind=='constant_fitted_beta':
        initial=np.r_[np.deg2rad(25.),t,parameters(x)]
        def unpack(z):return z[1:4],pose(z[4:]),z[0]
    else:raise ValueError('Unknown hypothesis')
    def residual(z):
        tt,xx,bb=unpack(z)
        return (forward(a,c,q,tt,xx,bb)-y).ravel()*1000
    z=robust_optimize(residual,initial,iterations=250);tt,xx,bb=unpack(z)
    diagnostics=jacobian(residual,z)
    if not diagnostics['optimizer_stationary']:raise ValueError('Root-constrained fit did not converge')
    model=dict(kind=kind,t_A_m=tt.tolist(),X_T=xx.tolist(),X_rotvec_deg=(parameters(xx)[:3]*180/np.pi).tolist(),
       beta_deg=None if bb is None else float(np.rad2deg(bb)),optimized_parameter_count=len(z),diagnostics=diagnostics,approved=False)
    if kind=='feedback_q1':
        full=np.r_[tt,parameters(xx)]
        def free_residual(v):return (forward(a,c,q,v[:3],pose(v[3:]),None)-y).ravel()*1000
        model['ungauged_9_parameter_diagnostics']=jacobian(free_residual,full)
        model['gauge']='t_A_z fixed to nominal 498.98mm; not independently measured'
    base=forward(a,c,q,tt,xx,bb)
    model['rbf_centers_m']=base.tolist()
    model['rbf_weights_m']=np.linalg.solve(space_kernel(base,base)+np.eye(len(c)),y-base).tolist()
    return model


def predict(a,m,c,q,compensated=False):
    beta=None if m['beta_deg'] is None else np.deg2rad(m['beta_deg'])
    base=forward(a,c,q,np.array(m['t_A_m']),np.array(m['X_T']),beta)
    if compensated:base+=space_kernel(base,np.array(m['rbf_centers_m']))@np.array(m['rbf_weights_m'])
    return base


def self_check():
    a=Assets();q=np.array([[0.,-25],[0,-20],[10,-25],[-10,-25]])
    k=root_rotation(q)@a.head(q)[:,:3,:3]
    np.testing.assert_allclose(k[:2],np.broadcast_to(np.eye(3),(2,3,3)),atol=1e-12)
    n=root_rotation(np.array([[0.,-25.]]))[0]@np.array([0,0,1.])
    np.testing.assert_allclose(np.einsum('nij,j->ni',k,n),np.tile(n,(len(q),1)),atol=1e-12)
    rng=np.random.default_rng(44);q=np.tile(q,(16,1));c=rng.uniform([-.05,-.04,.3],[.05,.04,.6],(len(q),3))
    t=np.array([-.02,.01,.49898]);x=pose([.2,-.3,.4,.1,.02,.03]);y=forward(a,c,q,t,x)
    m=fit(a,c,q,y,'feedback_q1')
    np.testing.assert_allclose(predict(a,m,c,q),y,atol=2e-6)
    assert m['diagnostics']['rank']==8
    assert m['ungauged_9_parameter_diagnostics']['rank']==8
    return dict(yaw_zero_pitch_cancellation=True,translation_gauge_confirmed=True,synthetic_8_parameter_recovery=True)


def main():
    checks=self_check();a=Assets();prior_path=ROOT/'reports/fk_consistent_vision_20261026/results.json'
    prior=json.loads(prior_path.read_text());assert prior['assets']==a.hashes
    d=load_capture([ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl',ROOT/'datasets/raw/ball_auto_20261018_visible_v1.jsonl'],a)
    assert d['sources']==prior['captures'];seen={}
    for row in d['rows']:
        key=(row['arm_side'],row['taught_path'],row['waypoint_id']);assert row['taught_path']
        if key in seen:np.testing.assert_allclose(row['q7_deg'],seen[key],atol=1e-6,rtol=0)
        seen[key]=row['q7_deg'];row['group_id']=str(key)
    c=np.array([a.triangulate(row['corrected_pixels']) for row in d['rows']]);_,_,q=arrays(d)
    p=a.flange(np.array([row['q7_deg'] for row in d['rows']]))
    g=np.array([row['group_id'] for row in d['rows']]);h=np.array([row['head_key'] for row in d['rows']]);ug=list(dict.fromkeys(g))
    mask=np.isin(g,ug[:int(.8*len(ug))]);folds=[('ball',np.flatnonzero(mask),np.flatnonzero(~mask))]
    for head in sorted(set(h)):
        folds += [('head:'+head,np.flatnonzero(h!=head),np.flatnonzero(h==head)),('joint:'+head,np.flatnonzero((h!=head)&mask),np.flatnonzero((h==head)&~mask))]
    out=dict(approved=False,interpretation='q1 is measured pitch, not +25deg constant; root translation expressed directly in S',
       prior_report_sha256=digest(prior_path),self_checks=checks,assets=a.hashes,folds={})
    for name,tr,te in folds:
        if not name.startswith('head:'):assert not set(g[tr])&set(g[te])
        if name!='ball':assert not set(h[tr])&set(h[te])
        old=prior['folds'][name];assert (len(tr),len(te))==(old['training_samples'],old['test_samples'])
        pf=old['pf_reference']['pf_m'];y=apply(torso_from_arm(a),ball(p,pf))
        result=dict(training_samples=len(tr),test_samples=len(te),frozen_pf_mm=(np.array(pf)*1000).tolist(),models={})
        for kind in ('feedback_q1','constant_25deg','constant_fitted_beta'):
            m=fit(a,c[tr],q[tr],y[tr],kind)
            result['models'][kind]=dict(train=metrics(predict(a,m,c[tr],q[tr])-y[tr]),test=metrics(predict(a,m,c[te],q[te])-y[te]),
               compensated_test=metrics(predict(a,m,c[te],q[te],True)-y[te]),parameters={k:v for k,v in m.items() if not k.startswith('rbf_')})
            if name=='ball':
                m.update(frozen_pf_m=pf,frame_id='S_MODEL',assets=a.hashes)
                Path(__file__).with_name(kind+'_train_model.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
        out['folds'][name]=result
        Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        print(name,{k:round(v['test']['p95_mm'],3) for k,v in result['models'].items()},flush=True)

if __name__=='__main__':main()
