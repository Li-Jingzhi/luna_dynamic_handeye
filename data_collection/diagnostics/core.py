"""Hardware-independent plans, guarded visual-servo math and measurement checks."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SERVICE='luna_diagnostic_camera_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_plan():
    return dict(schema='luna_auto_diagnostics_v1',head_deg=[0,-25],samples_per_window=50,
        control_samples=3,window_timeout_s=90.,head_tolerance_deg=.02,
        max_joint_step_deg=.3,max_total_joint_excursion_deg=12.,joint_limit_margin_deg=2.,
        max_camera_excursion_m=.05,servo_tolerance_m=.0003,servo_max_iterations=80,
        joint_target_tolerance_deg=.01,approach_deg=1.5,settle_s=.7,
        max_window_joint_span_deg=.02,hold_tolerance_m=.0006,
        optical_offsets_m=[[x,y,z] for z in (-.02,0.,.02) for x,y in ((0,0),(-.015,-.01),(.015,-.01),(-.015,.01),(.015,.01))],
        hold_offsets_m=[[0,0,0],[.015,0,0],[-.015,.01,0]],
        posture_axes_0based=[2,3,5,6],posture_delta_deg=4.,hold_posture_deltas_deg=[2.,4.],return_repeats=5,
        acceptance_offsets_m=[[x,y,z] for z in (-.015,.005,.025) for x,y in ((-.012,-.006),(.008,-.006),(-.012,.012),(.008,.012),(-.02,0),(.02,0),(0,-.015),(0,.015),(-.008,.002),(.015,.005))],
        acceptance_min_joint_novelty_deg=1.,ball_diameter_m=.04,
        ball_diameter_independently_measured=False,
        note='Only an existing yellow ball. Optical audit is consistency, not known-distance ground truth. Visual hold is not a mechanical pivot fixture.')


def validate_plan(p):
    if p.get('schema')!='luna_auto_diagnostics_v1' or p.get('samples_per_window')!=50:
        raise ValueError('Expected diagnostic plan and exactly fifty frames')
    if p.get('ball_diameter_m')!=.04:raise ValueError('Current detector contract assumes a nominal40mm sphere')
    if not np.allclose(p['head_deg'],[0,-25],atol=0,rtol=0):raise ValueError('Stage one requires fixed head [0,-25]')
    for key in ('window_timeout_s','settle_s','max_camera_excursion_m','servo_tolerance_m','hold_tolerance_m',
                'joint_target_tolerance_deg','max_window_joint_span_deg','posture_delta_deg','approach_deg'):
        if not np.isfinite(p[key]) or p[key]<=0:raise ValueError('Invalid '+key)
    if not 0<p['max_joint_step_deg']<=.5 or not 0<p['max_total_joint_excursion_deg']<=15:
        raise ValueError('Joint movement guard exceeds supported local range')
    if not 0<p['max_camera_excursion_m']<=.06:raise ValueError('Camera movement guard exceeds local range')
    for key in ('control_samples','servo_max_iterations','return_repeats'):
        if type(p[key]) is not int or p[key]<1:raise ValueError('Invalid '+key)
    axes=p['posture_axes_0based']
    if len(set(axes))!=len(axes) or any(type(i)is not int or i not in range(7) for i in axes):raise ValueError('Invalid joint axes')
    if not p['hold_posture_deltas_deg'] or any(not np.isfinite(x) or not 0<x<=8 for x in p['hold_posture_deltas_deg']):raise ValueError('Invalid hold posture amplitudes')
    for key in ('optical_offsets_m','hold_offsets_m','acceptance_offsets_m'):
        x=np.asarray(p[key],float)
        if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all() or np.max(np.linalg.norm(x,axis=1))>p['max_camera_excursion_m']:
            raise ValueError('Invalid offsets: '+key)
    return p


def tasks(plan,experiment):
    if experiment=='optical':return [dict(id=f'optical_{i:02}',kind='optical',offset_m=v) for i,v in enumerate(plan['optical_offsets_m'])]
    if experiment=='hold':return [dict(id=f'hold_{s:02}_{axis}_{sign:+}_a{amplitude:g}',kind='hold',site=s,axis=axis,sign=sign,amplitude_deg=amplitude,offset_m=v)
        for s,v in enumerate(plan['hold_offsets_m']) for amplitude in plan['hold_posture_deltas_deg'] for axis in plan['posture_axes_0based'] for sign in (-1,1)]
    if experiment=='return':return [dict(id=f'return_{s:02}_{axis}_{sign:+}_{repeat:02}',kind='return',site=s,axis=axis,sign=sign,repeat=repeat,offset_m=v)
        for s,v in enumerate(plan['hold_offsets_m']) for axis in plan['posture_axes_0based'] for repeat in range(plan['return_repeats']) for sign in (-1,1)]
    if experiment=='acceptance':return [dict(id=f'acceptance_{i:02}',kind='acceptance',offset_m=v) for i,v in enumerate(plan['acceptance_offsets_m'])]
    raise ValueError('Unknown experiment')


def servo_step(jac,error,q,posture,step_limit):
    if np.asarray(jac).shape!=(3,7):raise ValueError('Expected camera position Jacobian (m/degree)')
    j=np.asarray(jac,float);error=np.asarray(error,float)
    pinv=j.T@np.linalg.inv(j@j.T+(2e-4)**2*np.eye(3))
    null=np.eye(7)-pinv@j
    dq=pinv@error
    if posture is not None:dq+=.35*null@(np.asarray(posture)-q)
    magnitude=float(np.max(abs(dq)))
    return dq*min(1.,step_limit/max(magnitude,1e-12)),null


def guard_q(q,origin,limits,plan):
    q=np.asarray(q,float)
    if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Invalid joint target')
    if np.max(abs(q-origin))>plan['max_total_joint_excursion_deg']:raise ValueError('Total joint excursion exceeded')
    margin=plan['joint_limit_margin_deg']
    if np.any(q<limits[:,0]+margin) or np.any(q>limits[:,1]-margin):raise ValueError('Live joint limit margin exceeded')


def validate_metadata(m,expected,after_ns=0):
    if m.get('service')!=SERVICE:raise ValueError('Wrong camera service')
    for k in ('source_calibration_sha256','stereo_model_sha256','controls_sha256'):
        if m.get(k)!=expected[k]:raise ValueError('Camera asset mismatch: '+k)
    if m.get('image_size')!=[1280,992] or not m.get('controls_verified'):raise ValueError('Image/control contract failed')
    if m.get('exposure_time_valid') is not False:raise ValueError('Unverified exposure must be labelled false')
    for k in ('source_clock_ns','arrival_monotonic_ns'):
        a=np.asarray(m.get(k),float)
        if a.shape!=(2,) or not np.isfinite(a).all() or np.min(a)<=0 or np.ptp(a)>10_000_000:raise ValueError('Dual 10ms gate failed: '+k)
    if min(m['arrival_monotonic_ns'])<=after_ns:raise ValueError('Frame predates stationary capture barrier')
    if not isinstance(m.get('process_id'),str) or not m['process_id']:raise ValueError('Missing camera process identity')
    return (m['process_id'],*m['source_clock_ns'])


def window_summary(frames,plan):
    if len(frames)!=50:raise ValueError('Exactly fifty accepted frames required')
    c=np.array([f['measurement']['center_left_camera_m'] for f in frames])
    pixels=np.array([f['measurement']['corrected_stereo_pixels'] for f in frames])
    q=np.array([f['feedback']['q7_deg'] for f in frames])
    h=np.array([f['feedback']['head_deg'] for f in frames])
    if not np.isfinite(c).all() or not np.isfinite(q).all():raise ValueError('Nonfinite observation')
    if np.max(np.ptp(q,axis=0))>plan['max_window_joint_span_deg'] or np.max(np.ptp(h,axis=0))>.02:
        raise ValueError('Robot moved within fusion window')
    if np.max(abs(h-np.array(plan['head_deg'])))>plan['head_tolerance_deg']:raise ValueError('Head drift')
    corrected_disparity=pixels[:,0]-pixels[:,2]
    jitter=float(np.percentile(abs(corrected_disparity-np.median(corrected_disparity)),95))
    if jitter>.1:raise ValueError('Corrected disparity jitter exceeds 0.1px')
    center=np.median(c,axis=0)
    return dict(camera_xyz_m=center.tolist(),corrected_pixels=np.median(pixels,axis=0).tolist(),
        q7_deg=np.median(q,axis=0).tolist(),head_deg=np.median(h,axis=0).tolist(),
        disparity_jitter_p95_px=jitter,xyz_jitter_p95_mm=float(np.percentile(np.linalg.norm(c-center,axis=1)*1000,95)),
        joint_span_deg=np.ptp(q,axis=0).tolist(),frames=frames)
