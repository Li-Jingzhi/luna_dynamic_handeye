"""Latest current-robot development calibration; fixed FK, zero corrections and pF."""
from pathlib import Path
import json
import sys
import time
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,digest,fixed_geometry_hashes
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.calibration import metrics
from dynamic_handeye.empirical import base_points,fit_fields,correct,predict_pixels
from reports.algorithm_search_20261029.compare import fit_base,fit_relative
from reports.new_data_refit_20261029.analyze import CAPTURES,clamp_encoder_quantization

LENGTHS=(.03,.06,.1)
RIDGES=(.1,1.,3.)


def specs(short=False):
    result=[]
    for base in ('fk','relative'):
        result.extend([(base,'geometry',0.,0.),(base,'affine',0.,0.)])
        if short:result.extend([(base,'forward',.06,1.),(base,'inverse',.06,3.)])
        else:result.extend((base,kind,l,r) for l in LENGTHS for r in RIDGES for kind in ('forward','inverse'))
    return result


def key(spec):return f'{spec[0]}:{spec[1]}_l{spec[2]*1000:g}_r{spec[3]:g}'


def read_data():
    a=Assets();paths=[('A',p) for p in CAPTURES]
    paths += [('B',ROOT/f'datasets/raw/luna_v2_validationB_yaw{y}_pitchm{abs(p)}_20261029.jsonl') for y,p in [(0,-25),(0,-20),(10,-25),(10,-20)]]
    paths += [('C',p) for p in sorted((ROOT/'datasets/new_capture_20261029/shared_fast_grid_20261029').glob('head_0?.jsonl'))]
    rows=[];sources=[];batches=[];rejects=[]
    for batch,path in paths:
        d=load_capture([path],a);rows.extend(d['rows']);batches.extend([batch]*len(d['rows']))
        sources.append(dict(batch=batch,path=str(path.relative_to(ROOT)),sha256=digest(path),valid=len(d['rows']),rejected=d['rejected']))
    c=np.array([a.triangulate(r['corrected_pixels']) for r in rows]);q=clamp_encoder_quantization([r['head_angles_deg'] for r in rows])
    target=np.array([r['target_s_m'] for r in rows]);q7=np.array([r['q7_deg'] for r in rows])
    batch=np.array(batches);group=np.array([f'{b}:{r["waypoint_id"]}' for b,r in zip(batch,rows)])
    # Same physical waypoint across heads must share measured arm configuration.
    for g in set(group):
        if np.ptp(q7[group==g],axis=0).max()>.01:raise ValueError('Physical group joint mismatch')
    return a,dict(camera=c,head=q,target=target,q7=q7,batch=batch,group=group,
                  heads=np.array([r['head_key'] for r in rows]),rows=rows,sources=sources)


def partitions(d,ids,kind,count=5):
    if kind=='trajectory':
        for label in sorted(set(d['batch'][ids])):
            test=ids[d['batch'][ids]==label];train=np.setdiff1d(ids,test)
            if len(train):yield 'trajectory_'+label,train,test
    elif kind=='head':
        for label in sorted(set(d['heads'][ids])):
            test=ids[d['heads'][ids]==label];train=np.setdiff1d(ids,test)
            if len(train):yield 'head_'+label,train,test
    else:
        groups=sorted(set(d['group'][ids]));g=np.array(groups)
        centers=np.array([d['target'][ids[d['group'][ids]==x]].mean(0) for x in groups])
        if kind=='spatial':v=centers[:,np.argmax(np.ptp(centers,axis=0))]
        else:
            q=np.array([d['q7'][ids[d['group'][ids]==x]].mean(0) for x in groups]);q-=q.mean(0)
            v=q@np.linalg.svd(q,full_matrices=False)[2][0]
        for fold,block in enumerate(np.array_split(g[np.argsort(v)],count)):
            test=ids[np.isin(d['group'][ids],block)];train=np.setdiff1d(ids,test)
            yield f'{kind}_{fold}',train,test


def fit_models(d,a,train,choices):
    c=d['camera'][train];q=d['head'][train];t=d['target'][train]
    seed=fit_base(c,q,t,a)
    bases=dict(fk=seed,relative=fit_relative(c,q,d['group'][train],seed,a))
    models={}
    for family,baseline in bases.items():
        base=base_points(baseline,a,c,q)
        selected=[s for s in choices if s[0]==family]
        fields={}
        for length in sorted({s[2] for s in selected if s[2]}):
            ridges=sorted({s[3] for s in selected if s[2]==length})
            fields[length]=fit_fields(base,t,length,ridges)
        for spec in selected:
            _,kind,length,ridge=spec
            if kind=='geometry':field=dict(kind='geometry')
            elif kind=='affine':
                center=base.mean(0);f=np.c_[np.ones(len(base)),(base-center)/.2]
                b=np.linalg.lstsq(f,t-base,rcond=None)[0]
                field=dict(kind='affine',center_m=center.tolist(),trend=b.tolist())
            else:field=fields[length][(kind,ridge)]
            models[key(spec)]=dict(baseline=baseline,correction=field)
    return models


def predictions(model,d,a,ids):
    base=base_points(model['baseline'],a,d['camera'][ids],d['head'][ids])
    return correct(model['correction'],base)


def inner_select(d,a,train,kind):
    choices=specs(short=True);scores={key(s):[] for s in choices};valid={key(s):True for s in choices}
    inner=[]
    for label,tr,te in partitions(d,train,kind,count=3):
        models=fit_models(d,a,tr,choices)
        fold={}
        for name,model in models.items():
            ok=model['correction'].get('within_pro_scale_bounds',True)
            valid[name]&=ok
            score=metrics(predictions(model,d,a,te)-d['target'][te]);scores[name].append(score['p95_mm']);fold[name]=score
        inner.append(dict(label=label,train=int(len(tr)),test=int(len(te)),scores=fold))
    eligible=[name for name in scores if valid[name]]
    chosen=min(eligible,key=lambda n:(max(scores[n]),float(np.mean(scores[n]))))
    return chosen,dict(criterion='minimum worst inner P95, then mean P95; 8 prespecified candidates',selected=chosen,
                       scores=scores,valid_pro_scale=valid,folds=inner)


def main():
    a,d=read_data();ids=np.arange(len(d['camera']));started=time.monotonic()
    assert np.array_equal(a.pf,[0,0,.101])
    report=dict(approved=False,independently_verified=False,assets=a.hashes,sources=d['sources'],
        n=len(ids),physical_waypoints=len(set(d['group'])),head_support=sorted(set(d['heads'])),
        batch_counts={b:int(sum(d['batch']==b)) for b in sorted(set(d['batch']))},
        fixed_pf_m=[0,0,.101],joint_zero_correction_deg=[0]*7,
        candidates=[key(s) for s in specs()],nested_candidates=[key(s) for s in specs(short=True)],folds={},
        note='All A/B/C now development data. Candidate design reused historical research; nested scores reduce current-fold selection leakage but are not untouched acceptance.')
    selected_errors=[];baseline_errors=[];valid_all={key(s):True for s in specs()}
    for kind in ('trajectory','spatial','configuration','head'):
        for label,train,test in partitions(d,ids,kind):
            if kind!='head':assert not set(d['group'][train])&set(d['group'][test])
            print('START',label,len(train),len(test),flush=True)
            chosen,inner=inner_select(d,a,train,kind)
            models=fit_models(d,a,train,specs());scores={};scales={}
            for name,model in models.items():
                scores[name]=metrics(predictions(model,d,a,test)-d['target'][test])
                ok=model['correction'].get('within_pro_scale_bounds',True)
                valid_all[name]&=ok;scales[name]=ok
            selected_errors.extend(predictions(models[chosen],d,a,test)-d['target'][test])
            baseline_errors.extend(predictions(models['fk:geometry_l0_r0'],d,a,test)-d['target'][test])
            report['folds'][label]=dict(kind=kind,train=int(len(train)),test=int(len(test)),
                train_groups=sorted(set(d['group'][train])),test_groups=sorted(set(d['group'][test])),
                test_heads=sorted(set(d['heads'][test])),scores=scores,valid_pro_scale=scales,
                nested_selected=chosen,nested_test=scores[chosen],inner=inner)
            print('DONE',label,'nested',chosen,round(scores[chosen]['p95_mm'],3),flush=True)
            (OUT/'validation.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    summary={name:dict(worst_p95_mm=max(f['scores'][name]['p95_mm'] for f in report['folds'].values()),
             mean_p95_mm=float(np.mean([f['scores'][name]['p95_mm'] for f in report['folds'].values()])),
             valid_pro_scale=valid_all[name]) for name in report['candidates']}
    selected=min([n for n in summary if valid_all[n]],key=lambda n:(summary[n]['worst_p95_mm'],summary[n]['mean_p95_mm']))
    report.update(summary=summary,selected=selected,nested_pooled=metrics(selected_errors),
                  pooled_note='Pooled multi-protocol predictions repeat positions across protocols; use per-protocol/fold results for assessment.')
    final_models=fit_models(d,a,ids,specs());final=final_models[selected]
    base=base_points(final['baseline'],a,d['camera'],d['head']);fit=correct(final['correction'],base)
    report['training']=metrics(fit-d['target']);report['training_candidates']={n:metrics(predictions(m,d,a,ids)-d['target']) for n,m in final_models.items()}
    report['elapsed_s']=time.monotonic()-started
    model=dict(schema='luna_empirical_handeye_v1',approved=False,independently_verified=False,
        moving_head_validated=False,frame_id='S_MODEL',candidate=selected,assets=a.hashes,
        fixed_geometry_hashes=fixed_geometry_hashes(a),fixed_pf_m=[0,0,.101],joint_zero_correction_deg=[0]*7,
        source_captures=d['sources'],baseline=final['baseline'],correction=final['correction'],
        support=dict(head_deg=np.unique(d['head'],axis=0).tolist(),base_centers_m=base.tolist(),radius_m=.05),
        validation=dict(worst_p95_mm=summary[selected]['worst_p95_mm'],training=report['training'],
                        acceptance='not passed; current data are development only'))
    (OUT/'model.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Serialization + actual four-pixel inference is verified against full-data predictions.
    restored=json.loads((OUT/'model.json').read_text());errors=[]
    for i in np.linspace(0,len(ids)-1,25,dtype=int):
        got=predict_pixels(restored,a,d['rows'][i]['corrected_pixels'],d['rows'][i]['head_angles_deg'],diagnostic=True)
        errors.append(float(np.linalg.norm(np.array(got['xyz_m'])-fit[i])*1000))
    if max(errors)>1e-6:raise AssertionError('Saved pixel-interface mismatch')
    (OUT/'verification.json').write_text(json.dumps(dict(pixel_interface_roundtrip_max_mm=max(errors),samples=len(errors),
        asset_hashes_verified=True,source_files=len(d['sources'])),indent=2)+'\n')
    from reports.latest_handeye_20261030.fixed_view import main as fixed_view
    fixed_view()
    from reports.latest_handeye_20261030.summarize import summarize
    summarize(a,d)
    (OUT/'validation.partial.json').unlink()
    print('SELECTED',selected,summary[selected],'TRAIN',report['training'],flush=True)


if __name__=='__main__':main()
