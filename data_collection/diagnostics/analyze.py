"""Analyze captured diagnostics without fitting new geometry or changing frozen models."""
import json
from pathlib import Path
import numpy as np
from .core import digest
from .collect import write_json
from dynamic_handeye.geometry import Assets,apply,torso_from_arm
from dynamic_handeye.calibration import metrics
from reports.fixed_view_zero_diagnosis_20261030.predict import predict_pixels
from reports.acquisition_error_audit_20261029.audit import mono_sphere


def spread(points):
    p=np.array(points);center=p.mean(0)
    result=metrics(p-center)
    result['pairwise_max_mm']=float(np.max(np.linalg.norm(p[:,None]-p[None],axis=2))*1000)
    return result


def analyze(directory):
    directory=Path(directory)
    if (directory/'suite.json').exists():
        suite=json.loads((directory/'suite.json').read_text())
        report=dict(simulated=suite['simulated'],model_sha256=suite['model_sha256'],experiments={})
        for name in ('optical','hold','return','acceptance'):
            if (directory/name/'manifest.json').exists():report['experiments'][name]=analyze(directory/name)
        if any(v['model_sha256']!=suite['model_sha256'] for v in report['experiments'].values()):raise ValueError('Suite frozen models differ')
        write_json(directory/'analysis.json',report);return report
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,value in manifest['asset_files'].items():
        if digest(directory/'assets'/name)!=value:raise ValueError('Session asset modified: '+name)
    if digest(directory/'frozen_model.json')!=manifest['model_sha256']:raise ValueError('Frozen model modified')
    a=Assets(directory/'assets');model=json.loads((directory/'frozen_model.json').read_text())
    path=directory/'observations.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    if (directory/'completed.json').exists():
        complete=json.loads((directory/'completed.json').read_text())
        if digest(path)!=complete['sha256']:raise ValueError('Completed observations modified')
    accepted=[r for r in rows if r.get('capture_valid')]
    if len({r['task']['id'] for r in accepted})!=len(accepted):raise ValueError('Repeated completed physical task')
    all_sources=[set(),set()]
    for row in accepted:
        if len(row['frames'])!=manifest['plan']['samples_per_window'] or row['model_sha256']!=manifest['model_sha256']:raise ValueError('Frame/model contract mismatch')
        for frame in row['frames']:
            m=frame['metadata']
            for eye in (0,1):
                identity=(m['process_id'],m['source_clock_ns'][eye])
                if identity in all_sources[eye]:raise ValueError('Source eye reused across windows')
                all_sources[eye].add(identity)
            for item in frame['image_files'].values():
                p=(directory/item['path']).resolve()
                if not p.is_relative_to(directory.resolve()) or digest(p)!=item['sha256']:raise ValueError('Image evidence missing/tampered')
    report=dict(simulated=manifest['simulated'],experiment=manifest['experiment'],planned=len(manifest['tasks']),valid=len(accepted),
        full_protocol=manifest['full_protocol'],model_sha256=manifest['model_sha256'],model_refitted=False,
        independent_absolute_scale_verified=False,mechanical_fixed_center_verified=False,
        note='SIMULATION ONLY' if manifest['simulated'] else 'Static diagnostic evidence; single yellow ball does not identify absolute tool offsets or true encoder zeros.')
    if not accepted:report['reason']='No completed valid windows';write_json(directory/'analysis.json',report);return report
    q=np.array([r['q7_deg'] for r in accepted]);c=np.array([r['camera_xyz_m'] for r in accepted]);offsets=np.array(model['effective_joint_offsets_deg'])
    fk=apply(torso_from_arm(a)@a.flange(q),a.pf);cal=apply(torso_from_arm(a)@a.flange(q+offsets),a.pf)
    visual=np.array([predict_pixels(model,a,r['corrected_pixels'],r['head_deg'],diagnostic=True)['xyz_m'] for r in accepted])
    report['visual_vs_original_fk']=metrics(visual-fk);report['visual_vs_calibrated_fk']=metrics(visual-cal)
    report['head_deg']=manifest['plan']['head_deg']
    if manifest['experiment']=='optical':
        diagnostics=[]
        for row in accepted:
            measurements=[]
            for frame in row['frames']:
                point,_=mono_sphere(frame['measurement']['left'],a.K[0],manifest['plan']['ball_diameter_m']/2)
                measurements.append(np.linalg.norm(point-np.array(frame['measurement']['center_left_camera_m']))*1000)
            diagnostics.append(dict(task=row['task']['id'],camera_xyz_m=row['camera_xyz_m'],
                nominal_diameter_mono_vs_stereo_p95_mm=float(np.percentile(measurements,95)),
                jitter_p95_mm=row['xyz_jitter_p95_mm'],nominal_diameter_is_ground_truth=False))
        report['optical_consistency']=diagnostics
    elif manifest['experiment']=='hold':
        report['sites']={}
        sites=np.array([r['task']['site'] for r in accepted])
        for site in sorted(set(sites)):
            ix=sites==site
            report['sites'][str(site)]=dict(poses=int(sum(ix)),camera_spread=spread(c[ix]),
                original_fk_spread=spread(fk[ix]),calibrated_fk_spread=spread(cal[ix]),
                joint_span_deg=np.ptp(q[ix],axis=0).tolist(),constraint='visually held, not independently fixed mechanical center')
    elif manifest['experiment']=='return':
        report['return_comparisons']=[]
        for site,axis in sorted({(r['task']['site'],r['task']['axis']) for r in accepted}):
            sides={sign:np.array([i for i,r in enumerate(accepted) if r['task']['site']==site and r['task']['axis']==axis and r['task']['sign']==sign]) for sign in (-1,1)}
            if any(not len(x) for x in sides.values()):continue
            lo,hi=sides[-1],sides[1];residual=visual-cal
            report['return_comparisons'].append(dict(site=int(site),axis_1based=int(axis+1),
                repeats_from_below=len(lo),repeats_from_above=len(hi),
                visual_return_difference_mm=float(np.linalg.norm(visual[lo].mean(0)-visual[hi].mean(0))*1000),
                actual_feedback_difference_deg=(q[lo].mean(0)-q[hi].mean(0)).tolist(),
                calibrated_fk_difference_mm=float(np.linalg.norm(cal[lo].mean(0)-cal[hi].mean(0))*1000),
                visual_minus_fk_direction_difference_mm=float(np.linalg.norm(residual[lo].mean(0)-residual[hi].mean(0))*1000)))
    elif manifest['experiment']=='acceptance':
        errors=np.linalg.norm(visual-cal,axis=1)*1000
        report.update(fraction_over_or_equal_1mm=float(np.mean(errors>=1)),
            all_valid_positions_below_1mm=bool(errors.max()<1),
            calibrated_fk_consistency_passed=bool(not manifest['simulated'] and manifest['full_protocol'] and len(accepted)==len(manifest['tasks']) and errors.max()<1),
            original_fk_consistency_passed=bool(not manifest['simulated'] and manifest['full_protocol'] and len(accepted)==len(manifest['tasks']) and np.max(np.linalg.norm(visual-fk,axis=1))*1000<1),
            acceptance_scope='Only current fixed view, physical support and calibrated FK; no dynamic-head or external absolute accuracy claim')
    write_json(directory/'analysis.json',report)
    return report
