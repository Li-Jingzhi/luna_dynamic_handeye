"""Fixed-head affine/RBF search; frozen FK labels and nested grouped validation."""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from dynamic_handeye.geometry import Assets,digest
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.calibration import metrics

SPECS=[('rigid',0.,0.),('affine',0.,0.)]+[(kind,length,ridge)
    for kind in ('rigid_rbf','affine_rbf') for length in (.02,.05,.1,.2) for ridge in (.01,.1,1.,10.)]


def predict(c,t,x,spec):
    kind,length,ridge=spec
    if kind.startswith('rigid'):
        r,shift=rigid_fit(c,t);base=c@r.T+shift;test=x@r.T+shift
    else:
        center=c.mean(0)
        design=np.c_[(c-center)/.3,np.ones(len(c))]
        mapping=np.linalg.lstsq(design,t,rcond=None)[0]
        base=design@mapping;test=np.c_[(x-center)/.3,np.ones(len(x))]@mapping
    if kind.endswith('rbf'):
        def kernel(a,b):return np.exp(-np.sum((a[:,None]-b[None])**2,axis=2)/(2*length**2))
        weights=np.linalg.solve(kernel(c,c)+ridge*np.eye(len(c)),t-base)
        test+=kernel(x,c)@weights
    return test


def folds(ids,protocol,t,q,count):
    if protocol=='trajectory':order=ids
    elif protocol=='spatial':
        axis=np.argmax(np.ptp(t[ids],axis=0));order=ids[np.argsort(t[ids,axis])]
    else:
        centered=q[ids]-q[ids].mean(0)
        pc=centered@np.linalg.svd(centered,full_matrices=False)[2][0]
        order=ids[np.argsort(pc)]
    for test in np.array_split(order,count):
        train=ids[(ids<test.min()-2)|(ids>test.max()+2)] if protocol=='trajectory' else np.setdiff1d(ids,test)
        if len(train)<18:raise ValueError('Too few independent training positions')
        assert not set(train)&set(test)
        yield train,test


def select(c,t,q,ids,protocols):
    totals=np.zeros(len(SPECS));n=0
    for protocol in protocols:
        for train,test in folds(ids,protocol,t,q,3):
            for i,spec in enumerate(SPECS):
                error=predict(c[train],t[train],c[test],spec)-t[test]
                totals[i]+=float(np.sum(error**2))
            n+=len(test)
    scores=np.sqrt(totals/n)*1000
    return scores


def run_group(rows,a):
    rows=sorted(rows,key=lambda r:r['waypoint_id'])
    c=np.array([a.triangulate(r['corrected_pixels']) for r in rows])
    t=np.array([r['target_s_m'] for r in rows]);q=np.array([r['q7_deg'] for r in rows]);ids=np.arange(len(c))
    protocols=('trajectory','spatial','configuration')
    inner=select(c,t,q,ids,protocols);best=int(np.argmin(inner))
    full=[dict(spec=spec,inner_rms_mm=float(inner[i]),training=metrics(predict(c,t,c,spec)-t)) for i,spec in enumerate(SPECS)]
    result=dict(head=rows[0]['head_key'],session=rows[0]['session_id'],n=len(c),all_data_selected=full[best],
                minimum_training_p95_diagnostic=min(full,key=lambda r:r['training']['p95_mm']),candidates=full,protocols={})
    for protocol in protocols:
        errors={kind:[] for kind in ('rigid','affine','rigid_rbf','affine_rbf','selected')};details=[]
        for train,test in folds(ids,protocol,t,q,5):
            inner_scores=select(c,t,q,train,[protocol])
            winners={kind:min((i for i,s in enumerate(SPECS) if s[0]==kind),key=lambda i:inner_scores[i])
                     for kind in errors if kind!='selected'}
            winners['selected']=int(np.argmin(inner_scores))
            fold=dict(train_waypoints=[rows[i]['waypoint_id'] for i in train],
                      test_waypoints=[rows[i]['waypoint_id'] for i in test],models={})
            for kind,index in winners.items():
                delta=predict(c[train],t[train],c[test],SPECS[index])-t[test]
                errors[kind].extend(delta)
                fold['models'][kind]=dict(spec=SPECS[index],inner_rms_mm=float(inner_scores[index]),test=metrics(delta))
            details.append(fold)
        pooled={kind:dict(metrics=metrics(e),below_1mm_fraction=float(np.mean(np.linalg.norm(e,axis=1)<.001))) for kind,e in errors.items()}
        result['protocols'][protocol]=dict(pooled=pooled,folds=details)
        print(rows[0]['head_key'],protocol,[(k,round(v['metrics']['p95_mm'],3)) for k,v in pooled.items()],flush=True)
    return result


def main():
    snapshot=HERE/'snapshot'
    manifest=json.loads((HERE/'RESTORE_MANIFEST.json').read_text())
    for f in manifest['files']:
        if digest(snapshot/f['path'])!=f['sha256']:raise ValueError('Archived file changed')
    a=Assets(snapshot)
    assert np.array_equal(a.pf,[0,0,.101])
    d=load_capture([snapshot/'datasets/raw/ball_point_head_20261018_v1.jsonl',snapshot/'datasets/raw/ball_auto_20261018_visible_v1.jsonl'],a)
    groups={}
    for row in d['rows']:groups.setdefault((row['session_id'],row['head_key']),[]).append(row)
    report=dict(source_commit=manifest['source_commit'],assets=a.hashes,pf_m=a.pf.tolist(),joint_zero_correction_deg=[0]*7,
        candidate_count=len(SPECS),selection='minimum inner pooled RMS; 5 outer blocks, 3 inner blocks; trajectory purge 2 positions',
        note='Exploratory historical fixed-head analysis. RBF only uses camera XYZ; q7 only makes FK labels and configuration splits. No external/new-session acceptance.',groups=[])
    for _,rows in sorted(groups.items()):report['groups'].append(run_group(rows,a))
    (HERE/'compensation_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
