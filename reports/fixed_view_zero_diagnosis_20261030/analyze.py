"""Fixed-view cause audit: distinguish original FK from calibrated-FK chain closure."""
from pathlib import Path
import json
import sys
import time
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from reports.latest_handeye_20261030.calibrate import read_data
from dynamic_handeye.geometry import apply,torso_from_arm,pose,fixed_geometry_hashes
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.optimizer import robust_optimize
from dynamic_handeye.calibration import metrics
from dynamic_handeye.empirical import fit_fields,correct


def target_fk(a,q,offsets):
    return apply(torso_from_arm(a)@a.flange(q+offsets),a.pf)


def pose_maps(camera,target,groups,scale):
    models={};prediction=np.empty_like(camera)
    for group in sorted(set(groups)):
        ix=groups==group;r,t=rigid_fit(camera[ix]*scale,target[ix]);prediction[ix]=camera[ix]*scale@r.T+t
        models[group]=dict(R=r.tolist(),t_m=t.tolist())
    return prediction,models


def offsets_scale(x,kind,free):
    offsets=np.zeros(7);n=len(free) if kind in ('zeros','zeros_scale') else 0
    if n:offsets[free]=2*np.tanh(x[:n])
    scale=1+.02*np.tanh(x[n]) if kind in ('scale','zeros_scale') else 1.
    return offsets,float(scale)


def fit(a,c,q,groups,kind,free=(1,2,3,4,5,6),initial=None):
    free=np.array(free,int);n=(len(free) if kind in ('zeros','zeros_scale') else 0)+(kind in ('scale','zeros_scale'))
    def residual(x,prior=True):
        offsets,scale=offsets_scale(x,kind,free);target=target_fk(a,q,offsets)
        predicted,_=pose_maps(c,target,groups,scale)
        data=((predicted-target)*1000).ravel()
        penalties=[]
        if kind in ('zeros','zeros_scale'):penalties.extend(offsets[free]/.5)
        if kind in ('scale','zeros_scale'):penalties.append((scale-1)/.005)
        return np.r_[data,penalties] if prior else data
    x=robust_optimize(residual,np.zeros(n) if initial is None else np.asarray(initial,float),iterations=220) if n else np.empty(0)
    offsets,scale=offsets_scale(x,kind,free);target=target_fk(a,q,offsets);pred,models=pose_maps(c,target,groups,scale)
    result=dict(kind=kind,effective_offsets_deg=offsets.tolist(),camera_scale=scale,views=models,
        training_corrected_fk=metrics(pred-target),training_original_fk=metrics(pred-target_fk(a,q,np.zeros(7))),
        optimizer_parameters=x.tolist(),free_joint_indices_0based=free.tolist())
    if n:
        eps=1e-5;j=np.column_stack([(residual(x+np.eye(n)[i]*eps,False)-residual(x-np.eye(n)[i]*eps,False))/(2*eps) for i in range(n)])
        singular=np.linalg.svd(j,compute_uv=False)
        result['data_only_jacobian']=dict(rank=int(sum(singular>singular[0]*1e-7)),singular_values=singular.tolist(),
            condition_number=float(singular[0]/singular[-1]))
    return result


def predict(model,c,groups):
    pred=np.empty_like(c)
    for group in sorted(set(groups)):
        m=model['views'][group];ix=groups==group
        pred[ix]=c[ix]*model['camera_scale']@np.array(m['R']).T+m['t_m']
    return pred


def block_splits(d,ids,count=5):
    # Hold all head observations of a physical waypoint together, with two valid-point neighbours purged.
    per_batch={}
    for batch in sorted(set(d['batch'][ids])):
        groups=sorted(set(d['group'][ids[d['batch'][ids]==batch]]),key=lambda x:int(x.split(':')[1]))
        per_batch[batch]=(groups,np.array_split(np.arange(len(groups)),count))
    for fold in range(count):
        held=[];purged=[]
        for batch,(groups,blocks) in per_batch.items():
            block=blocks[fold];held.extend(groups[i] for i in block)
            purged.extend(groups[i] for i in range(max(0,block[0]-2),min(len(groups),block[-1]+3)))
        test=ids[np.isin(d['group'][ids],held)];train=ids[~np.isin(d['group'][ids],purged)]
        assert not set(d['group'][train])&set(d['group'][test])
        yield str(fold),train,test


def evaluate(a,d,ids,kind,group_labels,splitter):
    corrected=[];original=[];folds=[]
    for label,train,test in splitter:
        covered=np.isin(group_labels[test],group_labels[train]);valid_test=test[covered]
        if not len(valid_test):continue
        model=fit(a,d['camera'][train],d['q7'][train],group_labels[train],kind)
        pred=predict(model,d['camera'][valid_test],group_labels[valid_test])
        target=target_fk(a,d['q7'][valid_test],np.array(model['effective_offsets_deg']))
        delta=pred-target;error_original=pred-d['target'][valid_test]
        corrected.extend(delta);original.extend(error_original)
        fold=dict(label=label,train=len(train),test=len(test),covered_test=len(valid_test),
                  omitted_unseen_views=int(sum(~covered)),effective_offsets_deg=model['effective_offsets_deg'],camera_scale=model['camera_scale'],
                  corrected_fk=metrics(delta),original_fk=metrics(error_original),
                  train_groups=sorted(set(d['group'][train])),test_groups=sorted(set(d['group'][test])))
        folds.append(fold)
    return dict(corrected_fk=metrics(corrected),original_fk=metrics(original),folds=folds)


def original_affine_cv(a,d,ids,groups):
    results={kind:dict(errors=[],folds=[]) for kind in ('affine','pro_gaussian')}
    for label,tr,te in block_splits(d,ids):
        predictions={kind:np.empty((len(te),3)) for kind in results}
        for group in sorted(set(groups[te])):
            train=tr[groups[tr]==group];test=np.flatnonzero(groups[te]==group)
            c=d['camera'][train];target=d['target'][train]
            center=c.mean(0);f=np.c_[np.ones(len(c)),(c-center)/.2]
            mapping=np.linalg.lstsq(f,target,rcond=None)[0]
            predictions['affine'][test]=np.c_[np.ones(len(test)),(d['camera'][te[test]]-center)/.2]@mapping
            r,t=rigid_fit(c,target);base=c@r.T+t
            field=fit_fields(base,target,.06,[3.])[('inverse',3.)]
            predictions['pro_gaussian'][test]=correct(field,d['camera'][te[test]]@r.T+t)
        for kind in results:
            delta=predictions[kind]-d['target'][te];results[kind]['errors'].extend(delta)
            results[kind]['folds'].append(dict(label=label,metrics=metrics(delta)))
    return {kind:dict(original_fk=metrics(v['errors']),folds=v['folds']) for kind,v in results.items()}


def residual_field_cv(a,d,ids,groups,kind='zeros',splitter=None):
    errors=[];original=[];folds=[]
    for label,tr,test in (block_splits(d,ids) if splitter is None else splitter):
        te=test[np.isin(groups[test],groups[tr])]
        model=fit(a,d['camera'][tr],d['q7'][tr],groups[tr],kind)
        train_pred=predict(model,d['camera'][tr],groups[tr]);test_pred=predict(model,d['camera'][te],groups[te])
        ttrain=target_fk(a,d['q7'][tr],np.array(model['effective_offsets_deg']))
        corrected=np.empty_like(test_pred)
        for group in sorted(set(groups[te])):
            ti=groups[tr]==group;ei=groups[te]==group
            field=fit_fields(train_pred[ti],ttrain[ti],.06,[3.])[('inverse',3.)]
            corrected[ei]=correct(field,test_pred[ei])
        target=target_fk(a,d['q7'][te],np.array(model['effective_offsets_deg']))
        delta=corrected-target;orig=corrected-d['target'][te]
        errors.extend(delta);original.extend(orig)
        folds.append(dict(label=label,effective_offsets_deg=model['effective_offsets_deg'],camera_scale=model['camera_scale'],
            test=len(test),covered_test=len(te),omitted_unseen_views=len(test)-len(te),corrected_fk=metrics(delta),original_fk=metrics(orig)))
    return dict(corrected_fk=metrics(errors),original_fk=metrics(original),folds=folds)


def main():
    started=time.monotonic();a,d=read_data();all_ids=np.arange(len(d['camera']));primary=all_ids[d['heads']=='0,-25']
    report=dict(assets=a.hashes,sources=d['sources'],fixed_pf_m=a.pf.tolist(),approved=False,
        primary_fixed_head_deg=[0,-25],primary_n=len(primary),n=len(all_ids),
        note='Effective software geometry hypotheses only. Corrected-FK closure changes labels; original-FK errors are retained. No encoder reset or controller write.',scopes={})
    for name,ids in [('fixed_0m25',primary),('all_fixed_views',all_ids)]:
        result=dict(n=len(ids),fits={},blocked={},cross_trajectory={})
        for kind in ('nominal','scale','zeros','zeros_scale'):
            model=fit(a,d['camera'][ids],d['q7'][ids],d['heads'][ids],kind)
            result['fits'][kind]=model
            result['blocked'][kind]=evaluate(a,d,ids,kind,d['heads'],block_splits(d,ids))
            transfers=[]
            for batch in sorted(set(d['batch'][ids])):
                te=ids[d['batch'][ids]==batch];tr=np.setdiff1d(ids,te);transfers.append((batch,tr,te))
            result['cross_trajectory'][kind]=evaluate(a,d,ids,kind,d['heads'],transfers)
            print(name,kind,'fit',round(model['training_corrected_fk']['p95_mm'],3),
                  'block',round(result['blocked'][kind]['corrected_fk']['p95_mm'],3),
                  'transfer',round(result['cross_trajectory'][kind]['corrected_fk']['p95_mm'],3),
                  'scale',model['camera_scale'],'offsets',np.round(model['effective_offsets_deg'],3),flush=True)
        result['original_fk_visual_compensation']=original_affine_cv(a,d,ids,d['heads'])
        for kind in ('zeros','zeros_scale'):
            result[kind+'_plus_pro_gaussian']=residual_field_cv(a,d,ids,d['heads'],kind)
            transfers=[]
            for batch in sorted(set(d['batch'][ids])):
                te=ids[d['batch'][ids]==batch];tr=np.setdiff1d(ids,te);transfers.append((batch,tr,te))
            result[kind+'_plus_pro_gaussian_transfer']=residual_field_cv(a,d,ids,d['heads'],kind,transfers)
        report['scopes'][name]=result
        (OUT/'results.partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Additional pose freedom tests whether session/view drift is dominating the zero hypothesis.
    labels=np.array([f'{b}|{h}' for b,h in zip(d['batch'],d['heads'])])
    report['per_capture_pose_diagnostic']={}
    for kind in ('nominal','zeros','zeros_scale'):
        report['per_capture_pose_diagnostic'][kind]=fit(a,d['camera'],d['q7'],labels,kind)
    # Primary zero stability: remove each axis and re-fit all others, not just set a fitted component to zero.
    report['primary_zero_ablations']={}
    for missing in range(1,7):
        report['primary_zero_ablations'][f'J{missing+1}_forced_zero']=fit(a,d['camera'][primary],d['q7'][primary],d['heads'][primary],
            'zeros',free=tuple(i for i in range(1,7) if i!=missing))
    report['elapsed_s']=time.monotonic()-started
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    model=report['scopes']['fixed_0m25']['fits']['zeros']
    fixed=dict(schema='luna_fixed_view_calibrated_fk_v1',approved=False,independently_verified=False,assets=a.hashes,fixed_geometry_hashes=fixed_geometry_hashes(a),
        fixed_head_deg=[0,-25],fixed_head_tolerance_deg=.02,reference_frame='S_MODEL',pf_m=a.pf.tolist(),
        camera_to_torso=model['views']['0,-25'],effective_joint_offsets_deg=model['effective_offsets_deg'],
        camera_scale=model['camera_scale'],source_captures=d['sources'],
        reference_note='Camera-only prediction compared to FK(q+effective_offsets)*pF. This does NOT achieve the same errors against original FK(q).')
    (OUT/'fixed_head_candidate.json').write_text(json.dumps(fixed,indent=2,allow_nan=False)+'\n')
    candidates={}
    for kind in ('zeros','zeros_scale'):
        for use_field in (False,True):
            name=kind+('_plus_pro_gaussian' if use_field else '')
            block=report['scopes']['fixed_0m25'][name] if use_field else report['scopes']['fixed_0m25']['blocked'][kind]
            transfer=report['scopes']['fixed_0m25'][name+'_transfer'] if use_field else report['scopes']['fixed_0m25']['cross_trajectory'][kind]
            worst=max(f['corrected_fk']['p95_mm'] for f in block['folds']+transfer['folds'])
            candidates[name]=dict(worst_dev_fold_p95_mm=worst,blocked_p95_mm=block['corrected_fk']['p95_mm'],
                                 transfer_p95_mm=transfer['corrected_fk']['p95_mm'])
    selected=min(candidates,key=lambda n:candidates[n]['worst_dev_fold_p95_mm'])
    selected_kind='zeros_scale' if selected.startswith('zeros_scale') else 'zeros'
    chosen=report['scopes']['fixed_0m25']['fits'][selected_kind]
    candidate=dict(fixed,camera_to_torso=chosen['views']['0,-25'],camera_scale=chosen['camera_scale'],
                  effective_joint_offsets_deg=chosen['effective_offsets_deg'],candidate=selected,validation=candidates[selected])
    if 'gaussian' in selected:
        b=predict(chosen,d['camera'][primary],d['heads'][primary]);t=target_fk(a,d['q7'][primary],np.array(chosen['effective_offsets_deg']))
        candidate['correction']=fit_fields(b,t,.06,[3.])[('inverse',3.)]
    else:candidate['correction']=dict(kind='geometry')
    report['primary_selection']=dict(selected=selected,candidates=candidates,
        note='Four prespecified diagnostic variants, selected by worst developed fold P95; new acceptance still required.')
    # J7 offset is exactly interchangeable with an effective tool-point vector if that vector is free.
    from dynamic_handeye.geometry import mdh_poses
    flange=mdh_poses(a.rows('R.A1',7)[-1:],np.zeros((1,1)))[0]
    delta=np.deg2rad(chosen['effective_offsets_deg'][6]);rz=np.eye(4);rz[:2,:2]=[[np.cos(delta),-np.sin(delta)],[np.sin(delta),np.cos(delta)]]
    equivalent=apply(np.linalg.inv(flange)@rz@flange,a.pf)
    offset=np.zeros(7);offset[6]=chosen['effective_offsets_deg'][6]
    lhs=target_fk(a,d['q7'][primary],offset)
    rhs=apply(torso_from_arm(a)@a.flange(d['q7'][primary]),equivalent)
    report['J7_tool_point_exact_ambiguity']=dict(J7_deg=offset[6],equivalent_pf_m=equivalent.tolist(),
        original_pf_m=a.pf.tolist(),equivalence_max_mm=float(np.max(np.linalg.norm(lhs-rhs,axis=1))*1000),
        note='If pF direction/origin is not independently known, data cannot distinguish this last-axis zero from the corresponding tool-point change.')
    (OUT/'fixed_head_best_candidate.json').write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    from reports.fixed_view_zero_diagnosis_20261030.summarize import summarize
    summarize(a,d)
    (OUT/'results.partial.json').unlink()


if __name__=='__main__':main()
