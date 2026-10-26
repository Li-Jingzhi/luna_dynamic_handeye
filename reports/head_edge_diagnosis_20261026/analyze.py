"""Exploratory angle-trend ablation; no production model or acceptance claim."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,apply,torso_from_arm
from dynamic_handeye.dataset import load_capture,arrays
from dynamic_handeye.continuous_free_pf import fit,predict,ball
from dynamic_handeye.calibration import metrics


def features(q):
    return np.c_[np.ones(len(q)),q[:,0]/10,(q[:,1]+25)/5]


def spatial(c,d):
    return np.exp(-np.sum((c[:,None]-d[None])**2,axis=2)/(2*.05**2))


def trend_fit(q,residual):
    z=features(q)
    # Fixed weak prior prevents unsupported pitch direction fitting feedback noise.
    return np.linalg.solve(z.T@z+.01*np.eye(3),z.T@residual)


def affine_features(base,angles,center):
    return (features(angles)[:,:,None]*np.c_[np.ones(len(base)),(base-center)/.3][:,None,:]).reshape(len(base),12)


def self_check():
    q=np.array([[-10,-25],[0,-25],[0,-20],[10,-25.]])
    beta=np.arange(9).reshape(3,3)*.001
    recovered=trend_fit(np.tile(q,(100,1)),np.tile(features(q)@beta,(100,1)))
    np.testing.assert_allclose(features(q)@recovered,features(q)@beta,atol=3e-6)
    assert np.linalg.eigvalsh(spatial(q/100,q/100)).min()>-1e-10
    rng=np.random.default_rng(42);points=rng.normal(size=(400,3))*.2
    angles=np.tile(q,(100,1));center=points.mean(0)
    z=affine_features(points,angles,center);assert np.linalg.matrix_rank(z)==12
    coeff=rng.normal(size=(12,3))*.001
    np.testing.assert_allclose(np.linalg.lstsq(z,z@coeff,rcond=None)[0],coeff,atol=1e-12)


def main():
    self_check();a=Assets()
    d=load_capture([ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl',ROOT/'datasets/raw/ball_auto_20261018_visible_v1.jsonl'],a)
    seen={}
    for r in d['rows']:
        key=(r['arm_side'],r['taught_path'],r['waypoint_id'])
        assert r['taught_path']
        if key in seen:np.testing.assert_allclose(r['q7_deg'],seen[key],atol=1e-6,rtol=0)
        seen[key]=r['q7_deg'];r['group_id']=str(key)
    c,_,q=arrays(d);p=a.flange(np.array([r['q7_deg'] for r in d['rows']]))
    g=np.array([r['group_id'] for r in d['rows']]);h=np.array([r['head_key'] for r in d['rows']]);ug=list(dict.fromkeys(g))
    trainmask=np.isin(g,ug[:int(.8*len(ug))]);sm=torso_from_arm(a)
    out={'approved':False,'protocol':'Exploratory, fixed trend ridge .01; spatial length50mm/ridge1. No tuning on test folds. Prior test results already inspected; fresh validation required.', 'folds':{}}
    folds=[('ball',np.flatnonzero(trainmask),np.flatnonzero(~trainmask))]
    for head in sorted(set(h)):
        folds += [('head:'+head,np.flatnonzero(h!=head),np.flatnonzero(h==head)),('joint:'+head,np.flatnonzero((h!=head)&trainmask),np.flatnonzero((h==head)&~trainmask))]
    for name,tr,te in folds:
        if not name.startswith('head:'):assert not set(g[tr])&set(g[te])
        if name!='ball':assert not set(h[tr])&set(h[te])
        m=fit(c[tr],q[tr],p[tr],a)
        bt=predict(m,a,c[tr],q[tr],diagnostic=True);bv=predict(m,a,c[te],q[te],diagnostic=True)
        yt=apply(sm,ball(p[tr],m['pf_m']));yv=apply(sm,ball(p[te],m['pf_m']))
        beta=trend_fit(q[tr],yt-bt)
        trend=features(q[te])@beta
        w=np.linalg.solve(spatial(c[tr],c[tr])+np.eye(len(tr)),yt-bt-features(q[tr])@beta)
        center=bt.mean(0)
        zt=affine_features(bt,q[tr],center);zv=affine_features(bv,q[te],center)
        coeff=np.linalg.solve(zt.T@zt+.01*np.eye(12),zt.T@(yt-bt))
        wa=np.linalg.solve(spatial(c[tr],c[tr])+np.eye(len(tr)),yt-bt-zt@coeff)
        preds={'geometry':bv,'angle_linear':bv+trend,'angle_linear_spatial':bv+trend+spatial(c[te],c[tr])@w,
               'angle_affine':bv+zv@coeff,'angle_affine_spatial':bv+zv@coeff+spatial(c[te],c[tr])@wa}
        out['folds'][name]={'pf_mm':(np.array(m['pf_m'])*1000).tolist(),'trend_coefficients_mm':(beta*1000).tolist(),'head_feature_singular_values':np.linalg.svd(features(q[tr]),compute_uv=False).tolist(),'metrics':{k:metrics(v-yv) for k,v in preds.items()}}
        if name=='ball':out['ball_per_head']={head:{k:metrics(v[h[te]==head]-yv[h[te]==head]) for k,v in preds.items()} for head in sorted(set(h))}
        print(name,{k:round(metrics(v-yv)['p95_mm'],3) for k,v in preds.items()},flush=True)
    Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
