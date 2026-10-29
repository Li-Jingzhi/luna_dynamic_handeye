"""Audit stationary acquisition evidence; no changes to live camera or robot."""
from pathlib import Path
import gzip,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,apply,torso_from_arm
from dynamic_handeye.registration import rigid_fit
from dynamic_handeye.calibration import metrics

OUT=Path(__file__).resolve().parent


def stats(values):
    x=np.asarray(values,float)
    return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))


def mono_sphere(detection,K,radius=.02):
    """Known-radius sphere from the silhouette conic; diagnostic, not a replacement detector."""
    e=detection['ellipse'];theta=np.deg2rad(e['angle_deg'])
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    A=rot@np.diag([4/e['major_px']**2,4/e['minor_px']**2])@rot.T
    center=np.array(detection['center']);Q=np.empty((3,3));Q[:2,:2]=A
    Q[:2,2]=-A@center;Q[2,:2]=Q[:2,2];Q[2,2]=center@A@center-1
    eig,vec=np.linalg.eigh(K.T@Q@K)
    if eig[0]>=0 or eig[1]<=0:raise ValueError('Not a physical ellipse cone')
    direction=vec[:,0]*np.sign(vec[2,0]);distance=radius*np.sqrt(1-np.mean(eig[1:])/eig[0])
    return direction*distance,float(abs(eig[1]-eig[2])/np.mean(eig[1:]))


def evaluate(rows,K,flange=None,mount=None):
    valid=[r for r in rows if r.get('capture_valid')];camera=np.array([r['sphere_center_left_camera_m'] for r in valid])
    if flange is None:
        fk=np.array([np.array(r['flange_rotation'])@[0,0,.101]+r['flange_position_m'] for r in valid])
    else:fk=apply(mount@flange([r['q7_deg'] for r in valid]),[0,0,.101])
    rotation,translation=rigid_fit(camera,fk);res=camera@rotation.T+translation-fk
    dc=np.linalg.norm(camera[:,None]-camera[None],axis=-1);df=np.linalg.norm(fk[:,None]-fk[None],axis=-1)
    lengths=1000*abs(dc-df)[np.triu_indices(len(valid),1)]
    result=dict(total=len(rows),valid=len(valid),rigid_fit=metrics(res),pair_distance_disagreement_mm=stats(lengths),
                lower_bound_on_rigid_max_error_mm=float(lengths.max()/2),lags={},source_frame_duplicates=[0,0])
    for lag in (-2,-1,0,1,2):
        c=camera[max(0,-lag):len(camera)-max(0,lag)];t=fk[max(0,lag):len(fk)-max(0,-lag)]
        R,T=rigid_fit(c,t);result['lags'][str(lag)]=metrics(c@R.T+T-t)
    fk_window=[];final_vs_median=[];retri=[];ages=[];skews=[];q_range=[];mono=[];isotropy=[]
    seen=[set(),set()];missing=0
    for r in valid:
        points=np.array(r.get('sphere_center_framewise_m',[]));q=np.array(r.get('robot_window_q7_deg',[]))
        frames=r.get('camera_window_frames',[])
        if flange is not None and len(q)==50:
            p=apply(mount@flange(q),[0,0,.101]);mid=np.median(p,axis=0)
            fk_window.append(1000*np.max(np.linalg.norm(p-mid,axis=1)))
            final_vs_median.append(1000*np.linalg.norm(apply(mount@flange([r['q7_deg']]),[0,0,.101])[0]-mid))
            stamps=r['robot_window_feedback_monotonic_ns']
            ages.extend([(feedback-f['camera_timestamp_monotonic_ns'])/1e6 for feedback,f in zip(stamps,frames)])
        if len(q):q_range.append(float(np.ptp(q,axis=0).max()))
        for f in frames:
            identity=f.get('source_timestamp_ns',f.get('source_pts_ns'))
            if identity is None:missing+=1
            else:
                for eye in (0,1):
                    result['source_frame_duplicates'][eye]+=identity[eye] in seen[eye];seen[eye].add(identity[eye])
            skews.append(f['pair_dequeue_skew_ms'])
        estimates=[];iso=[]
        for f in frames:
            if 'left' not in f:continue
            point,error=mono_sphere(f['left'],K)
            estimates.append(point);iso.append(error)
        if estimates:
            mono.append(1000*np.linalg.norm(np.median(estimates,axis=0)-r['sphere_center_left_camera_m']))
            isotropy.append(float(np.median(iso)))
    result.update(static_3d_jitter_p95_mm=stats([r['sphere_center_p95_jitter_mm'] for r in valid]),
        missing_source_identity_frames=missing,pair_dequeue_skew_ms=stats(skews))
    for name,value in [('window_FK_max_motion_mm',fk_window),('final_feedback_vs_window_median_FK_mm',final_vs_median),
                       ('feedback_minus_camera_proxy_ms',ages),('window_q_range_deg',q_range),
                       ('monocular40mm_vs_stereo_mm',mono),('monocular_cone_isotropy_relative',isotropy)]:
        if value:result[name]=stats(value)
    return result


def main():
    a=Assets();report=dict(note='Evidence audit only. Known40mm sphere is an assumption; monocular disagreement does not uniquely identify camera/detector error.',datasets={})
    for name,path in [('Luna_A_0m25',ROOT/'datasets/raw/luna_v2_transferred_marvin360_fixed25_20261028.jsonl'),
                      ('Luna_B_0m25',ROOT/'datasets/raw/luna_v2_validationB_yaw0_pitchm25_20261029.jsonl'),
                      ('Luna_new_0m25',ROOT/'datasets/new_capture_20261029/shared_fast_grid_20261029/head_00.jsonl')]:
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        report['datasets'][name]=evaluate(rows,a.K[0],a.flange,torso_from_arm(a))
    pro=ROOT.parent/'head_eye_pro';c=json.loads((pro/'configs/orin_pro_rectified_stereo_20261019.json').read_text())
    rows=[json.loads(s) for s in gzip.decompress((pro/'datasets/drag_replay_20261020_091535_901340/replay_capture.jsonl.gz').read_bytes()).splitlines()]
    report['datasets']['Pro_right_288_fixed101']=evaluate(rows,np.array(c['left_K']))
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,data in report['datasets'].items():print(name,json.dumps(data),flush=True)


if __name__=='__main__':main()
