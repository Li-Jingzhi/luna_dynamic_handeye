"""Same Luna observations: train-only free pF, rigid/head-chain and RBF comparison."""
from pathlib import Path
import sys, json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets, apply, fixed_head_root, torso_from_arm, parameters, pose
from dynamic_handeye.dataset import load_capture, arrays
from dynamic_handeye.calibration import fit_geometry, predict, metrics
from dynamic_handeye.optimizer import robust_optimize
from pf_variable_projection import fit_model


def target(poses,pf):
    return poses[:,:3,3]+np.einsum('nij,j->ni',poses[:,:3,:3],pf)


def baseline(model,camera,heads):
    out=np.empty_like(camera)
    for h in set(heads):
        t=model['camera_from_mdh_by_head'][h]
        out[heads==h]=(camera[heads==h]-t['t_m'])@np.array(t['R'])
    return out


def rbf(x,y,length):
    return np.exp(-np.sum((x[:,None]-y[None])**2,axis=2)/(2*length**2))


def kernel(kind,length,camera,base,heads,ids,centers):
    shared=rbf(base[ids],base[centers],length) if kind in ('shared','additive') else 0.
    local=(rbf(camera[ids],camera[centers],length)*(heads[ids,None]==heads[None,centers])
           if kind in ('per_head','additive') else 0.)
    return shared+local


def predict_selected(camera,heads,model):
    """Offline diagnostic prediction: camera points/head keys only, no q7 or pF."""
    camera,heads=np.asarray(camera,float),np.asarray(heads)
    base=baseline(model['rigid'],camera,heads)
    kind,length,_=model['spec']
    if kind=='rigid':return base
    known_camera=np.asarray(model['centers_camera_m'])
    known_base=np.asarray(model['centers_base_m'])
    known_heads=np.asarray(model['heads'])
    shared=rbf(base,known_base,length) if kind in ('shared','additive') else 0.
    local=rbf(camera,known_camera,length)*(heads[:,None]==known_heads[None]) if kind in ('per_head','additive') else 0.
    return base+(shared+local)@np.asarray(model['weights_m'])


def splits(groups,poses,q,ids):
    unique=list(dict.fromkeys(groups[ids]));first=np.array([ids[np.flatnonzero(groups[ids]==g)[0]] for g in unique])
    centered=q[first]-q[first].mean(0)
    cfg=centered@np.linalg.svd(centered,full_matrices=False)[2][0]
    xyz=poses[first,:3,3];axis=int(np.argmax(np.ptp(xyz,axis=0)))
    layouts={'trajectory':np.arange(len(unique)), 'spatial':np.argsort(xyz[:,axis]),'configuration':np.argsort(cfg)}
    result=[]
    for name,order in layouts.items():
        for fold,held in enumerate(np.array_split(order,5)):
            te=ids[np.isin(groups[ids],np.array(unique)[held])];tr=np.setdiff1d(ids,te)
            assert not set(groups[tr])&set(groups[te])
            result.append((name,fold,tr,te))
    return result


def mixed_pf(camera,poses,angles,assets,tr,te):
    """Only X(6) and pF(3); fixed hybrid geometry remains unchanged."""
    initial=fit_geometry(camera[tr],target(poses[tr],assets.pf),angles[tr],assets)
    head=np.linalg.inv(torso_from_arm(assets))@fixed_head_root(assets)@assets.head(angles)
    def residual(x):return ((apply(head[tr]@pose(x[:6]),camera[tr])-target(poses[tr],x[6:]))*1000).ravel()
    x=robust_optimize(residual,np.r_[parameters(initial['head_tip_from_rectified_camera_T']),assets.pf],iterations=200)
    err=apply(head[te]@pose(x[:6]),camera[te])-target(poses[te],x[6:])
    jac=np.column_stack([(residual(x+np.eye(9)[i]*1e-5)-residual(x-np.eye(9)[i]*1e-5))/2e-5 for i in range(9)])
    singular=np.linalg.svd(jac,compute_uv=False)
    return dict(pf_mm=(x[6:]*1000).tolist(),pf_norm_mm=float(np.linalg.norm(x[6:])*1000),
                rank=int(sum(singular>singular[0]*1e-7)),parameter_count=9,
                heldout=metrics(err),camera_X=pose(x[:6]).tolist(),head_geometry_fixed=True)


def main():
    assets=Assets();data=load_capture([ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl'],assets)
    camera,_,angles=arrays(data);q=np.array([r['q7_deg'] for r in data['rows']]);poses=assets.flange(q)
    groups=np.array([r['group_id'] for r in data['rows']]);heads=np.array([r['head_key'] for r in data['rows']])
    unique=list(dict.fromkeys(groups));cut=int(.8*len(unique));train=np.flatnonzero(np.isin(groups,unique[:cut]));test=np.setdiff1d(np.arange(len(camera)),train)
    specs=[('rigid',0.,0.)]+[(kind,l,a) for kind in ('shared','per_head','additive') for l in (.02,.05,.1) for a in (.1,1.,10.)]
    scores={str(spec):[] for spec in specs};pf_folds=[];mixed_scores=[]
    for name,fold,tr,te in splits(groups,poses,q,train):
        check=mixed_pf(camera,poses,angles,assets,tr,te)
        mixed_scores.append(dict(method=name,fold=fold,p95=check['heldout']['p95_mm'],pf_mm=check['pf_mm']))
        model=fit_model(poses[tr],camera[tr],heads[tr]);pf=np.array(model['pf_m']);pf_folds.append(pf.tolist())
        base=baseline(model,camera,heads);labels=target(poses,pf);error=labels[tr]-base[tr]
        scores[str(specs[0])].append(dict(method=name,fold=fold,p95=metrics(base[te]-labels[te])['p95_mm']))
        for kind in ('shared','per_head','additive'):
            for length in (.02,.05,.1):
                K=kernel(kind,length,camera,base,heads,tr,tr);Kt=kernel(kind,length,camera,base,heads,te,tr)
                eig,U=np.linalg.eigh(K); projected=U.T@error
                for alpha in (.1,1.,10.):
                    W=U@(projected/(eig[:,None]+alpha))
                    err=base[te]+Kt@W-labels[te]
                    scores[str((kind,length,alpha))].append(dict(method=name,fold=fold,p95=metrics(err)['p95_mm']))
        print(f'fold {name}/{fold} done',flush=True)
    candidates=[dict(spec=list(s),worst_p95_mm=max(v['p95'] for v in scores[str(s)]),
                     mean_p95_mm=float(np.mean([v['p95'] for v in scores[str(s)]])),folds=scores[str(s)]) for s in specs]
    minimax=min(candidates,key=lambda c:(c['worst_p95_mm'],c['mean_p95_mm']))
    winner=min([c for c in candidates if c['worst_p95_mm']<=1.05*minimax['worst_p95_mm']],key=lambda c:(c['mean_p95_mm'],c['worst_p95_mm']))
    model=fit_model(poses[train],camera[train],heads[train]);pf=np.array(model['pf_m']);base=baseline(model,camera,heads);labels=target(poses,pf)
    results=[];saved={}
    for s in specs:
        if s[0]=='rigid':delta=np.zeros_like(camera[test])
        else:
            K=kernel(s[0],s[1],camera,base,heads,train,train)
            W=np.linalg.solve(K+s[2]*np.eye(len(train)),labels[train]-base[train])
            delta=kernel(s[0],s[1],camera,base,heads,test,train)@W
            if list(s)==winner['spec']:saved=dict(centers_camera_m=camera[train].tolist(),centers_base_m=base[train].tolist(),heads=heads[train].tolist(),weights_m=W.tolist())
        results.append(dict(spec=list(s),heldout=metrics(base[test]+delta-labels[test])))
    mixed=mixed_pf(camera,poses,angles,assets,train,test)
    result=dict(scope='Archived same-session, observed-head diagnostic; no production model change',
      observations=len(camera),groups=len(unique),train_groups=cut,test_groups=len(unique)-cut,train_observations=len(train),test_observations=len(test),
      selection='15 training-only blocks (trajectory/spatial/configuration); min mean P95 within 5% of minimax; no test-based choice',
      candidates=candidates,strict_minimax=minimax,selected=winner,heldout_candidates=results,
      pf_mm=(pf*1000).tolist(),pf_shift_from_measured_mm=float(np.linalg.norm(pf-assets.pf)*1000),
      pf_training_folds_mm=(np.array(pf_folds)*1000).tolist(),pf_identifiability=model['pf_projected_jacobian_condition'],
      mixed_head_plus_free_pf=mixed,mixed_head_training_cv=mixed_scores,approved=False,limitations=['No unseen-head or cross-session acceptance.',
      'Each candidate uses the same train-only shared effective pF; it changes FK ball labels versus measured pF.',
      'RBF is two-stage after pF/rigid fitting, not jointly fitted affine/pF/kernel.',
      'All 28 test scores are retrospective diagnostics; selected model was chosen only on training CV.'])
    folder=Path(__file__).parent
    (folder/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    trained=dict(schema='diagnostic_free_pf_discrete_rbf_v1',approved=False,
       spec=winner['spec'],pf_m=pf.tolist(),rigid=model,source_hashes=data['assets'],**saved)
    selected_error=predict_selected(camera[test],heads[test],trained)-labels[test]
    expected=next(r['heldout'] for r in results if r['spec']==winner['spec'])
    np.testing.assert_allclose(metrics(selected_error)['p95_mm'],expected['p95_mm'],atol=1e-10)
    (folder/'selected_train_model.json').write_text(json.dumps(trained,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('selected','pf_mm','mixed_head_plus_free_pf')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
