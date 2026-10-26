"""Train-only continuous-model selection; whole-head and joint head/position holdouts."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,apply,torso_from_arm
from dynamic_handeye.dataset import load_capture,arrays
from dynamic_handeye.calibration import metrics
from dynamic_handeye.continuous_free_pf import fit,predict,ball,add_residual


def main():
    a=Assets();d=load_capture([ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl', ROOT/'datasets/raw/ball_auto_20261018_visible_v1.jsonl'],a)
    # Same trajectory and identical q7 at matching archived waypoints.
    seen={}
    for row in d['rows']:
        key=(row['arm_side'],row['taught_path'],row['waypoint_id'])
        if not row['taught_path']:raise ValueError('Missing trajectory provenance')
        if key in seen:np.testing.assert_allclose(row['q7_deg'],seen[key],atol=1e-6,rtol=0)
        seen[key]=row['q7_deg']
        row['group_id']=str(key)
    body=np.array([r['body_q_deg'] for r in d['rows']])
    if np.ptp(body,axis=0).max()>.3:raise ValueError('Body changed across sessions')
    c,_,q=arrays(d);arm=np.array([r['q7_deg'] for r in d['rows']]);p=a.flange(arm)
    g=np.array([r['group_id'] for r in d['rows']]);h=np.array([r['head_key'] for r in d['rows']]);ug=list(dict.fromkeys(g))
    train=np.flatnonzero(np.isin(g,ug[:int(.8*len(ug))]));test=np.setdiff1d(np.arange(len(g)),train)
    specs=[('geometry',.05,15.,1.)]+[(k,l,angle,r) for k in ('spatial','conditional','additive') for l in (.02,.05,.1)
       for angle in ([15.] if k=='spatial' else [5.,15.,45.]) for r in (.1,1.,10.)]
    scores={str(s):[] for s in specs}
    def make(tr,s,geometry=None):
        if geometry is None:geometry=fit(c[tr],q[tr],p[tr],a)
        return geometry if s[0]=='geometry' else add_residual(geometry,c[tr],q[tr],p[tr],a,*s)
    def error(m,te):return predict(m,a,c[te],q[te],diagnostic=True)-apply(torso_from_arm(a),ball(p[te],m['pf_m']))
    u=list(dict.fromkeys(g[train]));first=np.array([train[np.flatnonzero(g[train]==x)[0]] for x in u])
    pos=p[first,:3,3];ax=int(np.argmax(np.ptp(pos,axis=0)));ac=arm[first]-arm[first].mean(0);cfg=ac@np.linalg.svd(ac,full_matrices=False)[2][0]
    for name,order in [('trajectory',np.arange(len(u))),('spatial',np.argsort(pos[:,ax])),('configuration',np.argsort(cfg))]:
        for fold,held in enumerate(np.array_split(order,3)):
            te=train[np.isin(g[train],np.array(u)[held])];tr=np.setdiff1d(train,te)
            assert not set(g[tr])&set(g[te])
            geo=fit(c[tr],q[tr],p[tr],a)
            for s in specs:scores[str(s)].append(metrics(error(make(tr,s,geo),te))['p95_mm'])
            print(name,fold,flush=True)
    ranked=[dict(spec=list(s),mean_p95_mm=float(np.mean(scores[str(s)])),worst_p95_mm=max(scores[str(s)]),folds=scores[str(s)]) for s in specs]
    best=min(ranked,key=lambda r:(r['worst_p95_mm'],r['mean_p95_mm']))
    pool=[r for r in ranked if r['worst_p95_mm']<=1.05*best['worst_p95_mm']]
    selected=min(pool,key=lambda r:(r['mean_p95_mm'],r['worst_p95_mm']))
    # Family representatives are selected exclusively by training folds.
    families={}
    for kind in ('geometry','spatial','conditional','additive'):
        family=[r for r in ranked if r['spec'][0]==kind]
        mm=min(r['worst_p95_mm'] for r in family)
        families[kind]=min([r for r in family if r['worst_p95_mm']<=1.05*mm],key=lambda r:r['mean_p95_mm'])
    out=dict(target_p95_mm=1.,selection='9 training-only trajectory/spatial/configuration blocks; 5% minimax tolerance',
      whole_head_selection='Nested: only outer-training data, 3 inner grouped trajectory folds per model family',
      samples=len(c),groups=len(ug),candidate_count=len(specs),selected=selected,candidates=ranked,families={},
      source_hashes=a.hashes,source_captures=d['sources'],head_settings=sorted(set(h)),approved=False,notes=['Four archived head settings; no new untouched-session acceptance.',
      'Free effective pF changes the FK ball labels; not an external physical accuracy measurement.',
      'Fourth head [0,-25] lies on the previous triangle boundary; folds have different interpolation/extrapolation status.'])
    local_cache={}
    def local_choices(outer_train):
        key=tuple(outer_train)
        if key in local_cache:return local_cache[key]
        local={str(s):[] for s in specs}
        names=list(dict.fromkeys(g[outer_train]))
        for held in np.array_split(np.array(names),3):
            va=outer_train[np.isin(g[outer_train],held)];tr=np.setdiff1d(outer_train,va)
            assert not set(g[tr])&set(g[va])
            gm=fit(c[tr],q[tr],p[tr],a)
            for spec in specs:local[str(spec)].append(metrics(error(make(tr,spec,gm),va))['p95_mm'])
        choices={}
        for kind in ('geometry','spatial','conditional','additive'):
            rows=[dict(spec=list(s),mean=float(np.mean(local[str(s)])),worst=max(local[str(s)])) for s in specs if s[0]==kind]
            limit=min(x['worst'] for x in rows)*1.05
            choices[kind]=min([x for x in rows if x['worst']<=limit],key=lambda x:x['mean'])['spec']
        local_cache[key]=choices
        return choices
    geo=fit(c[train],q[train],p[train],a)
    for kind,choice in families.items():
        s=choice['spec'];m=make(train,s,geo)
        result=dict(selection=choice,pf_mm=(np.array(m['pf_m'])*1000).tolist(),trajectory_holdout=metrics(error(m,test)),head_holdouts={},joint_head_position_holdouts={})
        for head in sorted(set(h)):
            tr=np.flatnonzero(h!=head);te=np.flatnonzero(h==head)
            chosen=local_choices(tr)[kind]
            gm=fit(c[tr],q[tr],p[tr],a);hm=make(tr,chosen,gm)
            result['head_holdouts'][head]=dict(metrics(error(hm,te)),selected_spec=chosen)
            tr=np.flatnonzero((h!=head)&np.isin(g,ug[:int(.8*len(ug))]));te=test[h[test]==head]
            assert not set(g[tr])&set(g[te]) and head not in h[tr]
            chosen=local_choices(tr)[kind]
            gm=fit(c[tr],q[tr],p[tr],a);hm=make(tr,chosen,gm)
            result['joint_head_position_holdouts'][head]=dict(metrics(error(hm,te)),selected_spec=chosen)
        worst=max(v['p95_mm'] for v in result['head_holdouts'].values())
        result['observed_tests_all_below_target']=bool(result['trajectory_holdout']['p95_mm']<1 and worst<1 and all(v['p95_mm']<1 for v in result['joint_head_position_holdouts'].values()))
        out['families'][kind]=result
        if s==selected['spec']:
            Path(__file__).with_name('selected_train_model.json').write_text(json.dumps(m,indent=2)+'\n')
        print('family',kind,'done',flush=True)
    out['arbitrary_head_target_demonstrated']=False
    out['acceptance_reason']='Four sparse archived head settings cannot certify arbitrary head poses; head and acquisition session remain confounded.'
    Path(__file__).with_name('comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:dict(spec=v['selection']['spec'],trajectory=v['trajectory_holdout']['p95_mm'],head=[x['p95_mm'] for x in v['head_holdouts'].values()],joint=[x['p95_mm'] for x in v['joint_head_position_holdouts'].values()]) for k,v in out['families'].items()},indent=2))


if __name__=='__main__':main()
