"""Expanded grouped development benchmark; never changes FK labels or live calibration."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports.algorithm_search_20261029 import compare as old
from reports.new_data_refit_20261029.analyze import CAPTURES,clamp_encoder_quantization
from dynamic_handeye.geometry import Assets,apply,pose,fixed_head_root,torso_from_arm
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.calibration import metrics
from dynamic_handeye.optimizer import robust_optimize

OUT=Path(__file__).resolve().parent


def flexible_head(q,extra):
    """Yaw about Z, pitch about a tilted/offset line; encoder scale corrections regularized in fit."""
    cx,cz,ux,uz,sy,sp=np.asarray(extra)
    axis=np.array([ux,1.,uz]);axis/=np.linalg.norm(axis)
    skew=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    theta=-np.deg2rad(q[:,1])*(1+sp)
    co=np.cos(theta)[:,None,None];si=np.sin(theta)[:,None,None]
    r=co*np.eye(3)+(1-co)*np.outer(axis,axis)+si*skew
    h=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
    h[:,:3,:3]=r;h[:,:3,3]=np.einsum('nij,j->ni',np.eye(3)-r,[cx,0,cz])
    rz=np.broadcast_to(np.eye(4),h.shape).copy();y=np.deg2rad(q[:,0])*(1+sy)
    rz[:,0,0]=np.cos(y);rz[:,0,1]=-np.sin(y);rz[:,1,0]=np.sin(y);rz[:,1,1]=np.cos(y)
    return rz@h


def fit_flexible(c,q,group,seed,assets):
    _,inv,count=np.unique(group,return_inverse=True,return_counts=True)
    def residual(x,prior=True):
        p=apply(flexible_head(q,x[6:])@pose(x[:6]),c)
        mean=np.array([np.bincount(inv,weights=p[:,a])/count for a in range(3)]).T
        r=((p-mean[inv])*1000).ravel()
        return np.r_[r,x[6:]/[.03,.03,.15,.15,.03,.03]] if prior else r
    x0=np.r_[seed['x'][6:],np.zeros(6)]
    fitted=robust_optimize(residual,x0,iterations=120)
    # Report data-only rank; priors must not disguise unobservable physical parameters.
    eps=1e-5;J=np.column_stack([(residual(fitted+np.eye(12)[i]*eps,False)-residual(fitted-np.eye(12)[i]*eps,False))/(2*eps) for i in range(12)])
    singular=np.linalg.svd(J,compute_uv=False)
    return dict(kind='flexible_relative',x=fitted.tolist(),optimizer='existing_Huber_damped_least_squares',
                data_jacobian_singular_values=singular.tolist(),data_rank=int(sum(singular>singular[0]*1e-7)))


def base_predict(model,c,q,assets):
    if model['kind']!='flexible_relative':return old.base_predict(model,c,q,assets)
    x=np.array(model['x'])
    return apply(fixed_head_root(assets)@flexible_head(q,x[6:])@pose(x[:6]),c)


def distances(a,b):
    return np.maximum(np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None]-2*a@b.T,0)


def fit_predict(kind,btrain,qtrain,target,btest,qtest,length=.06,ridge=1.,head_length=30.,neighbors=64):
    center=btrain.mean(0);f=old.features(btrain,center,1);ft=old.features(btest,center,1)
    residual=target-btrain
    if kind=='local_affine':
        index=np.argsort(distances(btest,btrain),axis=1)[:,:min(neighbors,len(btrain))]
        answer=[]
        for point,ix in zip(btest,index):
            z=(btrain[ix]-point)/.2;phi=np.c_[np.ones(len(ix)),z]
            d=np.sum(z*z,axis=1);weights=np.exp(-d/max(float(np.max(d)),1e-9))
            penalty=np.diag([1e-9,ridge,ridge,ridge])
            coeff=np.linalg.solve(phi.T@(weights[:,None]*phi)+penalty,phi.T@(weights[:,None]*residual[ix]))
            answer.append(point+coeff[0])
        return np.array(answer)
    def kernel(a,b,qa,qb):
        d=distances(a,b)/(length*length)
        if kind=='matern32':
            r=np.sqrt(3*d);return (1+r)*np.exp(-r)
        k=np.exp(-.5*d)
        if kind=='head_additive':return k*(1+np.exp(-.5*distances(qa,qb)/head_length**2))
        return k
    k=kernel(btrain,btrain,qtrain,qtrain)
    inv=np.linalg.solve(k+ridge*np.eye(len(k)),np.c_[f,residual])
    coeff=np.linalg.solve(f.T@inv[:,:4]+.01*np.eye(4),f.T@inv[:,4:])
    weights=inv[:,4:]-inv[:,:4]@coeff
    return btest+ft@coeff+kernel(btest,btrain,qtest,qtrain)@weights


def folds_for(target,q7,trajectory,group,heads):
    groups=sorted(set(group));centers=np.array([target[group==g].mean(0) for g in groups])
    order=np.array(groups)[np.argsort(centers[:,np.argmax(np.ptp(centers,axis=0))])]
    folds=[('A_to_B',trajectory=='A',trajectory=='B'),('B_to_A',trajectory=='B',trajectory=='A')]
    for i,block in enumerate(np.array_split(order,5)):
        test=np.isin(group,block);folds.append((f'spatial_{i}',~test,test))
    gq=np.array([q7[group==g].mean(0) for g in groups]);pc=np.linalg.svd(gq-gq.mean(0),full_matrices=False)[2][0]
    order=np.array(groups)[np.argsort((gq-gq.mean(0))@pc)]
    for i,block in enumerate(np.array_split(order,5)):
        test=np.isin(group,block);folds.append((f'configuration_{i}',~test,test))
    for h in sorted(set(heads)):
        test=heads==h;folds.append((f'head_{h}',~test,test))
    return folds


def main():
    assets=Assets();started=time.monotonic()
    paths=CAPTURES+[ROOT/f'datasets/raw/luna_v2_validationB_yaw{y}_pitchm{abs(p)}_20261029.jsonl' for y,p in [(0,-25),(0,-20),(10,-25),(10,-20)]]
    data=load_capture(paths,assets);rows=data['rows']
    c=np.array([assets.triangulate(r['corrected_pixels']) for r in rows]);q=clamp_encoder_quantization([r['head_angles_deg'] for r in rows])
    target=np.array([r['target_s_m'] for r in rows]);q7=np.array([r['q7_deg'] for r in rows])
    trajectory=np.array(['B' if 'validationB' in r['taught_path'] else 'A' for r in rows])
    group=np.array([f'{t}:{r["waypoint_id"]}' for t,r in zip(trajectory,rows)]);heads=np.array([r['head_key'] for r in rows])
    previous=json.loads((ROOT/'reports/algorithm_search_20261029/comparison.json').read_text())
    assert data['sources']==previous['captures'],'Prior results are from different source files'
    specs=[]
    for length in (.03,.06,.10):
        for ridge in (.1,1.,10.):specs.append(dict(kind='matern32',length=length,ridge=ridge))
    for length in (.06,.10):
        for head_length in (10.,30.):
            for ridge in (.1,1.,10.):specs.append(dict(kind='head_additive',length=length,ridge=ridge,head_length=head_length))
    for neighbors in (32,64,128):
        for ridge in (.01,.1):specs.append(dict(kind='local_affine',neighbors=neighbors,ridge=ridge))
    flexible_specs=[dict(kind='geometry'),dict(kind='gaussian',length=.06,ridge=1.),dict(kind='matern32',length=.06,ridge=1.),dict(kind='head_additive',length=.06,ridge=1.,head_length=30.)]
    report=dict(approved=False,pf_m=[0,0,.101],frozen_fk=True,assets=assets.hashes,sources=data['sources'],folds={},specifications={},
                prior_selected=previous['selected'],note='Exploratory development only; current acquisition excluded from fitting and candidate selection.')
    for name,tr,te in folds_for(target,q7,trajectory,group,heads):
        if not name.startswith('head_'):assert not set(group[tr])&set(group[te])
        assert (int(tr.sum()),int(te.sum()))==(previous['folds'][name]['train'],previous['folds'][name]['test'])
        seed=old.fit_base(c[tr],q[tr],target[tr],assets)
        relative=old.fit_relative(c[tr],q[tr],group[tr],seed,assets)
        flexible=fit_flexible(c[tr],q[tr],group[tr],seed,assets)
        scores={}
        for family,model,choices in [('relative',relative,specs),('flexible',flexible,flexible_specs)]:
            btrain=base_predict(model,c[tr],q[tr],assets);btest=base_predict(model,c[te],q[te],assets)
            for spec in choices:
                key=family+':'+','.join(f'{k}={v}' for k,v in spec.items())
                pred=btest if spec['kind']=='geometry' else fit_predict(**spec,btrain=btrain,qtrain=q[tr],target=target[tr],btest=btest,qtest=q[te])
                scores[key]=metrics(pred-target[te]);report['specifications'][key]=dict(family=family,**spec)
        report['folds'][name]=dict(train=int(tr.sum()),test=int(te.sum()),scores=scores,flexible_diagnostic=flexible)
        print(name,'new_best',min(scores,key=lambda k:scores[k]['p95_mm']),round(min(v['p95_mm'] for v in scores.values()),3),flush=True)
    summary={key:dict(worst_p95_mm=max(f['scores'][key]['p95_mm'] for f in report['folds'].values()),
                     mean_p95_mm=float(np.mean([f['scores'][key]['p95_mm'] for f in report['folds'].values()]))) for key in scores}
    report['new_summary']=summary
    combined={**previous['summary'],**summary}
    selected=min(combined,key=lambda k:(combined[k]['worst_p95_mm'],combined[k]['mean_p95_mm']))
    report.update(selected=selected,selected_summary=combined[selected],total_candidates=len(combined),all_folds_pass_1mm=combined[selected]['worst_p95_mm']<=1,elapsed_s=time.monotonic()-started)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SELECTED',selected,combined[selected],flush=True)


if __name__=='__main__':main()
