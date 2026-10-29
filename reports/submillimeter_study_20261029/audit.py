"""Freeze-source audit and prospective prefix diagnostics; never trains on live capture."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from reports.submillimeter_study_20261029 import experiment as ex
from reports.algorithm_search_20261029 import compare as old
from reports.new_data_refit_20261029.analyze import CAPTURES,clamp_encoder_quantization,predict_selected
from dynamic_handeye.geometry import Assets,apply,pose,fixed_head_root,digest
from dynamic_handeye.dataset import load_capture
from dynamic_handeye.calibration import metrics

OUT=Path(__file__).resolve().parent


def describe(values):
    a=np.asarray(values,float)
    return dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))


def gravity_diagnostic(body):
    import mujoco
    import xml.etree.ElementTree as ET
    from dynamic_handeye.geometry import parameters
    path=ROOT/'reference/LunaUdpTool/model/gento_luna/gento_luna_contact_filtered.xml'
    tree=ET.parse(path);tree.getroot().find('compiler').set('meshdir',str(path.parent/'meshes'))
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));data=mujoco.MjData(model)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'Gento Luna_Leg_J6_Link')
    rotations=[];result={}
    for key in ('A','new_prefix'):
        for i,value in enumerate(body[key]['mean_deg'],1):
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'Gento Luna_Leg_J{i}_Joint')
            if j<0:raise ValueError('Missing torso joint in nominal model')
            data.qpos[model.jnt_qposadr[j]]=np.deg2rad(value)
        mujoco.mj_forward(model,data);r=data.xmat[bid].reshape(3,3).copy();rotations.append(r)
        result[key]=dict(world_position_m=data.xpos[bid].tolist(),gravity_direction_in_torso=(r.T@[0,0,-1]).tolist())
    rel=np.eye(4);rel[:3,:3]=rotations[0].T@rotations[1]
    result['relative_torso_rotation_deg']=float(np.linalg.norm(parameters(rel)[:3])*180/np.pi)
    return result


def main():
    a=Assets();report=json.loads((OUT/'comparison.json').read_text())
    paths=CAPTURES+[ROOT/f'datasets/raw/luna_v2_validationB_yaw{y}_pitchm{abs(p)}_20261029.jsonl' for y,p in [(0,-25),(0,-20),(10,-25),(10,-20)]]
    data=load_capture(paths,a);rows=data['rows'];assert data['sources']==report['sources']
    c=np.array([a.triangulate(r['corrected_pixels']) for r in rows]);qh=clamp_encoder_quantization([r['head_angles_deg'] for r in rows]);target=np.array([r['target_s_m'] for r in rows])
    tr=np.array(['B' if 'validationB' in r['taught_path'] else 'A' for r in rows]);groups=np.array([f'{t}:{r["waypoint_id"]}' for t,r in zip(tr,rows)])
    selected=report['selected'];spec=report['specifications'][selected];seed=old.fit_base(c,qh,target,a)
    base=old.fit_relative(c,qh,groups,seed,a) if spec['family']=='relative' else ex.fit_flexible(c,qh,groups,seed,a)
    b=ex.base_predict(base,c,qh,a);kwargs={k:v for k,v in spec.items() if k!='family'}
    full=ex.fit_predict(**kwargs,btrain=b,qtrain=qh,target=target,btest=b,qtest=qh)
    snapshot=ROOT/'datasets/new_capture_20261029/analysis_snapshot_head00/head_00.jsonl'
    prefix=load_capture([snapshot],a);pr=prefix['rows']
    pc=np.array([a.triangulate(r['corrected_pixels']) for r in pr]);pq=clamp_encoder_quantization([r['head_angles_deg'] for r in pr]);pt=np.array([r['target_s_m'] for r in pr])
    previous=json.loads((ROOT/'reports/algorithm_search_20261029/development_model.json').read_text())
    oldpred=old.correct(previous['correction'],old.base_predict(previous['baseline'],pc,pq,a))
    newpred=ex.fit_predict(**kwargs,btrain=b,qtrain=qh,target=target,btest=ex.base_predict(base,pc,pq,a),qtest=pq)
    prefix_scores=dict(previous_AB_selected=metrics(oldpred-pt),new_AB_selected=metrics(newpred-pt))
    for name in ('fixed6','learned12','selected'):
        model=json.loads((ROOT/f'reports/new_data_refit_20261029/{name}_model.json').read_text())
        prefix_scores['frozen_A_'+name]=metrics(predict_selected(model,a,pc,pq)-pt)
    raw=[json.loads(s) for s in snapshot.read_text().splitlines() if s.strip()];valid=[r for r in raw if r.get('capture_valid')]
    fx=a.K[0][0,0];baseline=float(np.linalg.norm(a.stereo['right_from_left_T_m']))
    budget=[dict(depth_m=z,disparity_px=fx*baseline/z,depth_error_mm_per_01px=1000*z*z/(fx*baseline)*.1,
                 disparity_bias_for_1mm_px=.001*fx*baseline/(z*z)) for z in (.4,.6,.8,1.)]
    learned=json.loads((ROOT/'reports/new_data_refit_20261029/learned12_model.json').read_text())
    nominal=fixed_head_root(a);fitted=pose(np.array(learned['x'])[:6]);rel=np.linalg.inv(nominal)@fitted
    from dynamic_handeye.geometry import parameters
    body={}
    for key,rr in [('A',[r for i,r in enumerate(rows) if tr[i]=='A']),('B',[r for i,r in enumerate(rows) if tr[i]=='B']),('new_prefix',pr)]:
        vals=np.array([r['body_q_deg'] for r in rr]);body[key]=dict(mean_deg=vals.mean(0).tolist(),range_deg=np.ptp(vals,axis=0).tolist())
    nearest=np.sqrt(ex.distances(ex.base_predict(base,pc,pq,a),b).min(1))*1000
    findings=dict(approved=False,pf_m=[0,0,.101],selected=selected,full_AB_training_fit=metrics(full-target),
        snapshot_file=str(snapshot),snapshot_sha256=digest(snapshot),snapshot_records=len(raw),snapshot_valid=len(valid),snapshot_head=[0,-25],snapshot_complete=False,
        prefix_scores=prefix_scores,body_feedback=body,nominal_body_gravity_diagnostic=gravity_diagnostic(body),nearest_training_base_distance_mm=describe(nearest),
        static_window_3d_p95_jitter_mm=describe([r['sphere_center_p95_jitter_mm'] for r in valid]),
        half_window_shift_mm=describe([r['sphere_center_half_window_shift_mm'] for r in valid]),
        disparity_jitter_px=describe([r['disparity_jitter_p95_px'] for r in valid]),
        corrected_epipolar_abs_px=describe([abs(r['sphere_center_corrected_stereo_pixels'][1]-r['sphere_center_corrected_stereo_pixels'][3]) for r in valid]),
        depths_m=describe(pc[:,2]),sphere_diameter_m=valid[0]['yellow_ball_diameter_m'],
        exposure_time_valid_count=sum(r.get('exposure_time_valid') is True for r in valid),
        controls=valid[0]['camera_window_frames'][0]['camera_controls'],
        image_evidence_present=sum(bool(f.get('image_evidence')) for r in valid for f in r['camera_window_frames']),
        focal_px=float(fx),baseline_m=baseline,depth_error_budget=budget,
        effective_A_vs_nominal=dict(translation_delta_mm=(1000*(fitted[:3,3]-nominal[:3,3])).tolist(),
                                   translation_norm_mm=float(np.linalg.norm(fitted[:3,3]-nominal[:3,3])*1000),rotation_deg=float(np.linalg.norm(parameters(rel)[:3])*180/np.pi)),
        note='Snapshot prefix never used for fitting or hyperparameter selection. One incomplete head and changed body pose do not prove arbitrary-head performance.')
    (OUT/'audit.json').write_text(json.dumps(findings,indent=2,allow_nan=False)+'\n')
    payload=dict(approved=False,input='corrected stereo pixels and head angles; no arm joint input',assets=a.hashes,fixed_pf_m=[0,0,.101],baseline=base,
                 specification=kwargs,training_base_m=b.tolist(),training_head_deg=qh.tolist(),training_target_s_m=target.tolist(),source_hashes=[r['sha256'] for r in data['sources']],
                 note='Offline research payload, not deployed; use fit_predict to reproduce.')
    (OUT/'experimental_model.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected=selected,prefix=prefix_scores,nominal_difference=findings['effective_A_vs_nominal'],body=body),indent=2))


if __name__=='__main__':main()
