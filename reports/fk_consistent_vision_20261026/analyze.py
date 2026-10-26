"""Offline FK-reference calibration with train-only pF and camera/head-only inference."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,apply,pose,parameters,fixed_head_root,torso_from_arm,vector,fixed_geometry_hashes
from dynamic_handeye.dataset import load_capture,arrays
from dynamic_handeye.calibration import fit_geometry,metrics
from dynamic_handeye.continuous_free_pf import ball,kernel
from dynamic_handeye.optimizer import robust_optimize
from dynamic_handeye.registration import rigid_fit


def rotation(axis,angle):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    angle=np.asarray(angle,float)
    w=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(angle)[:,None,None]*w+(1-np.cos(angle))[:,None,None]*(w@w)


def revolute(axis,point,angle):
    r=rotation(axis,angle);t=np.broadcast_to(np.eye(4),(len(r),4,4)).copy()
    t[:,:3,:3]=r;t[:,:3,3]=point-np.einsum('nij,j->ni',r,point)
    return t


def poe_pose(x,q,origin):
    # Gauge: axis points intersect local z=0 (yaw) and local y=0 (pitch).
    yaw=np.array([x[6],x[7],1.]);pitch=np.array([x[10],-1.,x[11]])
    yp=origin+np.array([x[8],x[9],0]);pp=origin+np.array([x[12],0,x[13]])
    return revolute(yaw,yp,np.deg2rad(q[:,0]))@revolute(pitch,pp,np.deg2rad(q[:,1]+25))@pose(x[:6])


def pitch_first_head(q):
    t=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
    t[:,:3,:3]=rotation([0,-1,0],np.deg2rad(q[:,1]))@rotation([0,0,1],np.deg2rad(q[:,0]))
    return t


def linear_features(c,q,center):
    z=np.c_[np.ones(len(q)),q[:,0]/10,(q[:,1]+25)/5]
    u=np.c_[np.ones(len(c)),(c-center)/.3]
    return (z[:,:,None]*u[:,None,:]).reshape(len(q),12)


def space_kernel(c,d):
    return np.exp(-np.sum((c[:,None]-d[None])**2,axis=2)/(2*.05**2))


def fit_models(c,q,target_s,a):
    nominal=fit_geometry(c,apply(np.linalg.inv(torso_from_arm(a)),target_s),q,a)
    root=np.array(nominal['torso_from_head_root_T']);mount=np.array(nominal['head_tip_from_rectified_camera_T'])
    # Calibrate effective endpoints; CAD is initialization only, no D=640 or fixed root constraint.
    initial=np.r_[parameters(root),parameters(mount)]
    k=a.head(q)
    def endpoint_error(x):return (apply(pose(x[:6])@k@pose(x[6:]),c)-target_s).ravel()*1000
    x=robust_optimize(endpoint_error,initial,iterations=250)
    def rank(fun,x,objective=None):
        j=np.column_stack([(fun(x+np.eye(len(x))[i]*1e-5)-fun(x-np.eye(len(x))[i]*1e-5))/2e-5 for i in range(len(x))])
        s=np.linalg.svd(j,compute_uv=False)
        objective=objective or fun;error=objective(x)
        jo=np.column_stack([(objective(x+np.eye(len(x))[i]*1e-5)-objective(x-np.eye(len(x))[i]*1e-5))/2e-5 for i in range(len(x))])
        weight=np.minimum(1.,3/np.maximum(abs(error),1e-12))
        grad=float(np.max(abs(jo.T@(weight*error))/np.maximum(np.linalg.norm(jo,axis=0),1)))
        return dict(rank=int(sum(s>s[0]*1e-7)),singular_values=s.tolist(),objective_normalized_gradient=grad,
                    optimizer_stationary=bool(grad<.01))
    models={'fixed6':dict(kind='endpoints',x=initial.tolist(),rank={'rank':6}),
            'learned12':dict(kind='endpoints',x=x.tolist(),rank=rank(endpoint_error,x))}
    # Alternate order is diagnostic only; the old V2 runtime order is unchanged.
    kp=pitch_first_head(q)
    rr,tt=rigid_fit(c,apply(np.linalg.inv(root@kp),target_s))
    mp=np.eye(4);mp[:3,:3]=rr;mp[:3,3]=tt
    def alternate_error(z):return (apply(root@kp@pose(z),c)-target_s).ravel()*1000
    xa=robust_optimize(alternate_error,parameters(mp),iterations=150)
    models['pitch_first6_diagnostic']=dict(kind='pitch_first',root_T=root.tolist(),mount=xa.tolist(),rank=rank(alternate_error,xa))
    # POE allows nonorthogonal, nonintersecting axes; home is at [yaw=0,pitch=-25].
    origin=root[:3,3];home=root@a.head(np.array([[0.,-25.]]))[0]@mount
    start=np.r_[parameters(home),np.zeros(8)]
    def poe_error(z):return (apply(poe_pose(z,q,origin),c)-target_s).ravel()*1000
    # Weak explicit prior for poorly excited axis directions; not an independent measurement.
    scales=np.array([.35,.35,.10,.10,.35,.35,.10,.10])
    def regularized(z):return np.r_[poe_error(z),z[6:]/scales]
    xp=robust_optimize(regularized,start,iterations=250)
    models['poe14']=dict(kind='poe',x=xp.tolist(),origin_m=origin.tolist(),rank=rank(poe_error,xp,regularized),axis_prior_scales=scales.tolist())
    center=c.mean(0);z=linear_features(c,q,center)
    # Fit residual from nominal base; extrapolation retains the known rigid baseline.
    base=apply(pose(initial[:6])@k@pose(initial[6:]),c)
    b=np.linalg.solve(z.T@z+.01*np.eye(12),z.T@(target_s-base))
    models['nominal_linear_residual']=dict(kind='linear',baseline=models['fixed6'],center_m=center.tolist(),weights=b.tolist())
    b=np.linalg.solve(z.T@z+.01*np.eye(12),z.T@target_s)
    models['direct_linear']=dict(kind='direct',center_m=center.tolist(),weights=b.tolist())
    for name in ('fixed6','learned12','poe14','direct_linear'):
        m=models[name];res=target_s-predict(m,a,c,q,diagnostic=True)
        weights=np.linalg.solve(space_kernel(c,c)+np.eye(len(c)),res)
        models[name+'_rbf']=dict(kind='rbf',baseline=m,centers_m=c.tolist(),weights_m=weights.tolist())
    effective=models['learned12'];xx=np.array(effective['x'])
    poses=pose(xx[:6])@k@pose(xx[6:]);base=apply(poses,c)
    local_residual=apply(np.linalg.inv(poses),target_s)-c
    models['learned12_camera_local_rbf']=dict(kind='camera_local_rbf',baseline=effective,centers_m=c.tolist(),
       weights_m=np.linalg.solve(space_kernel(c,c)+np.eye(len(c)),local_residual).tolist())
    models['learned12_torso_rbf']=dict(kind='torso_rbf',baseline=effective,centers_m=base.tolist(),
       weights_m=np.linalg.solve(space_kernel(base,base)+np.eye(len(c)),target_s-base).tolist())
    poe_base=predict(models['poe14'],a,c,q,diagnostic=True)
    models['poe14_torso_rbf']=dict(kind='torso_rbf',baseline=models['poe14'],centers_m=poe_base.tolist(),
       weights_m=np.linalg.solve(space_kernel(poe_base,poe_base)+np.eye(len(c)),target_s-poe_base).tolist())
    alt_base=predict(models['pitch_first6_diagnostic'],a,c,q,diagnostic=True)
    models['pitch_first6_torso_rbf_diagnostic']=dict(kind='torso_rbf',baseline=models['pitch_first6_diagnostic'],centers_m=alt_base.tolist(),
       weights_m=np.linalg.solve(space_kernel(alt_base,alt_base)+np.eye(len(c)),target_s-alt_base).tolist())
    nominal_base=predict(models['fixed6'],a,c,q,diagnostic=True)
    ks=dict(kind='conditional',length_m=.1,angle_length_deg=5.)
    models['fixed6_conditional']=dict(kind='conditional',baseline=models['fixed6'],centers_m=c.tolist(),angles_deg=q.tolist(),
       weights_m=np.linalg.solve(kernel(c,q,c,q,ks)+.1*np.eye(len(c)),target_s-nominal_base).tolist())
    for m in models.values():m.update(approved=False,frame_id='S_MODEL',camera_assets=a.hashes,fixed_geometry_hashes=fixed_geometry_hashes(a))
    return models


def predict(model,a,c,q,*,diagnostic=False):
    if not diagnostic:raise ValueError('Independent arbitrary-head acceptance is required')
    c=np.asarray(c,float);q=np.asarray(q,float)
    vector(c,(len(c),3),'camera points');vector(q,(len(c),2),'head angles')
    if model.get('camera_assets',a.hashes)!=a.hashes:raise ValueError('Camera/FK reference assets changed')
    if model.get('fixed_geometry_hashes',fixed_geometry_hashes(a))!=fixed_geometry_hashes(a):
        raise ValueError('Nominal reference configuration changed')
    k=a.head(q)  # finite values, V2 yaw/pitch order and physical limit checks
    kind=model['kind']
    if kind=='endpoints':
        x=np.array(model['x']);return apply(pose(x[:6])@k@pose(x[6:]),c)
    if kind=='pitch_first':return apply(np.array(model['root_T'])@pitch_first_head(q)@pose(model['mount']),c)
    if kind=='poe':return apply(poe_pose(np.array(model['x']),q,np.array(model['origin_m'])),c)
    if kind=='camera_local_rbf':
        xx=np.array(model['baseline']['x']);poses=pose(xx[:6])@k@pose(xx[6:])
        delta=space_kernel(c,np.array(model['centers_m']))@np.array(model['weights_m'])
        return apply(poses,c+delta)
    if kind=='torso_rbf':
        base=predict(model['baseline'],a,c,q,diagnostic=True)
        return base+space_kernel(base,np.array(model['centers_m']))@np.array(model['weights_m'])
    if kind=='conditional':
        ks=dict(kind='conditional',length_m=.1,angle_length_deg=5.)
        delta=kernel(c,q,np.array(model['centers_m']),np.array(model['angles_deg']),ks)@np.array(model['weights_m'])
        return predict(model['baseline'],a,c,q,diagnostic=True)+delta
    if kind=='rbf':return predict(model['baseline'],a,c,q,diagnostic=True)+space_kernel(c,np.array(model['centers_m']))@np.array(model['weights_m'])
    delta=linear_features(c,q,np.array(model['center_m']))@np.array(model['weights'])
    if kind=='direct':return delta
    if kind=='linear':return predict(model['baseline'],a,c,q,diagnostic=True)+delta
    raise ValueError('Unknown research model')


def predict_pixels(model,a,pixels,head_angles,*,diagnostic=False):
    """Four rectified corresponding pixels + head feedback -> torso XYZ; no q7/pF input."""
    if not diagnostic:raise ValueError('Independent arbitrary-head acceptance is required')
    return predict(model,a,a.triangulate(pixels)[None],np.asarray(head_angles,float)[None],diagnostic=True)[0]


def self_check():
    a=Assets();rng=np.random.default_rng(1026)
    q=np.array([(y,p) for y in (-20,0,20) for p in (-25,-15,-5)]*8,float)
    c=rng.uniform([-.08,-.06,.3],[.08,.06,.65],(len(q),3))
    root=pose([.03,-.06,.02,-.02,.01,.49]);mount=pose([1.1,-.7,1.1,.08,.02,.05])
    target=apply(root@a.head(q)@mount,c)
    models=fit_models(c,q,target,a)
    np.testing.assert_allclose(predict(models['learned12'],a,c,q,diagnostic=True),target,atol=2e-6)
    assert models['learned12']['rank']['rank']==12
    cross=np.array([[-10,-25],[0,-25],[10,-25],[0,-20.]])
    b=np.eye(4);b[:3,:3]=rotation([0,1,0],np.array([np.deg2rad(25)]))[0]
    np.testing.assert_allclose(pitch_first_head(cross),b@a.head(cross)@np.linalg.inv(b),atol=1e-12)
    assert np.max(abs(pitch_first_head(np.array([[10.,-15.]]))-b@a.head(np.array([[10.,-15.]]))@np.linalg.inv(b)))>.01
    ref=np.array([0.,-25.]);home=root@a.head(ref[None])[0]@mount
    # A pure POE reference must equal its home at zero relative joint angles.
    x=np.r_[parameters(home),np.zeros(8)]
    np.testing.assert_allclose(poe_pose(x,ref[None],root[:3,3])[0],home,atol=1e-12)
    simple={'kind':'endpoints','x':np.r_[parameters(root),parameters(mount)].tolist()}
    point=np.array([.01,-.01,.5]);uv=[]
    for P in a.P:
        p=P@np.r_[point,1];uv.extend(p[:2]/p[2])
    np.testing.assert_allclose(predict_pixels(simple,a,uv,ref,diagnostic=True),apply(root@a.head(ref[None])[0]@mount,point),atol=1e-9)
    try:predict_pixels(simple,a,uv,ref)
    except ValueError:pass
    else:raise AssertionError('Unaccepted model enabled')
    try:predict(simple,a,point[None],np.array([[0,-26]]),diagnostic=True)
    except ValueError:pass
    else:raise AssertionError('Physical joint limits bypassed')
    error=predict(models['learned12'],a,c,q,diagnostic=True)-target
    simple['fixed_geometry_hashes']={'changed':True}
    try:predict_pixels(simple,a,uv,ref,diagnostic=True)
    except ValueError:pass
    else:raise AssertionError('Changed FK/torso reference configuration accepted')
    origin=fixed_head_root(a)[:3,3]
    truth=np.r_[parameters(home),[.06,-.02,.01,-.005,.03,.05,.02,.01]]
    target_poe=apply(poe_pose(truth,q,origin),c)
    fitted=fit_models(c,q,target_poe,a)['poe14']
    unseen=np.array([[-12,-21],[7,-12],[14,-8]],float);points=c[:3]
    delta=predict(fitted,a,points,unseen,diagnostic=True)-apply(poe_pose(truth,unseen,origin),points)
    max_poe_mm=float(1000*np.max(np.linalg.norm(delta,axis=1)))
    assert max_poe_mm<.2, max_poe_mm
    return {'learned12_recovery_max_mm':float(1000*np.max(np.linalg.norm(error,axis=1))),
            'poe14_nonorthogonal_separated_axis_unseen_max_mm':max_poe_mm,'pixel_interface_no_arm_inputs':True}


def main():
    a=Assets();checks=self_check();print('self checks passed',flush=True)
    spec=importlib.util.spec_from_file_location('pf_reference',ROOT/'reports/free_pf_comparison_20261026/pf_variable_projection.py')
    pfmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(pfmod)
    d=load_capture([ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl',ROOT/'datasets/raw/ball_auto_20261018_visible_v1.jsonl'],a)
    seen={}
    for r in d['rows']:
        key=(r['arm_side'],r['taught_path'],r['waypoint_id']);assert r['taught_path']
        if key in seen:np.testing.assert_allclose(r['q7_deg'],seen[key],atol=1e-6,rtol=0)
        seen[key]=r['q7_deg'];r['group_id']=str(key)
    assert np.ptp([r['body_q_deg'] for r in d['rows']],axis=0).max()<.3
    c,_,q=arrays(d)
    pixels=np.array([r['corrected_pixels'] for r in d['rows']]);pixel_camera=np.array([a.triangulate(uv) for uv in pixels])
    # Calibration and evaluation use exactly the declared pixel interface.
    input_difference=metrics(pixel_camera-c);c=pixel_camera
    p=a.flange(np.array([r['q7_deg'] for r in d['rows']]))
    g=np.array([r['group_id'] for r in d['rows']]);h=np.array([r['head_key'] for r in d['rows']]);ug=list(dict.fromkeys(g))
    mask=np.isin(g,ug[:int(.8*len(ug))]);tr0=np.flatnonzero(mask);te0=np.flatnonzero(~mask)
    folds=[('ball',tr0,te0)]
    for head in sorted(set(h)):
        folds += [('head:'+head,np.flatnonzero(h!=head),np.flatnonzero(h==head)),('joint:'+head,np.flatnonzero((h!=head)&mask),np.flatnonzero((h==head)&~mask))]
    out=dict(approved=False,capture_robot_version_confirmation=dict(robot_version='luna_v2_0',order='yaw_then_pitch',basis='Explicit user confirmation in this conversation, 2026-10-26; raw logs omit robot_version'),goal='Vision agrees with frozen nominal arm FK in S_MODEL, not external absolute truth',
      selection='Fixed prespecified models; exploratory reused test data; no test-driven model promotion',target_p95_mm=1.,
      self_checks=checks,pixel_vs_archived_camera=input_difference,captures=d['sources'],assets=a.hashes,folds={})
    for name,tr,te in folds:
        if not name.startswith('head:'):assert not set(g[tr])&set(g[te])
        if name!='ball':assert not set(h[tr])&set(h[te])
        pf=pfmod.fit_model(p[tr],c[tr],h[tr]);target=apply(torso_from_arm(a),ball(p,pf['pf_m']))
        models=fit_models(c[tr],q[tr],target[tr],a)
        result=dict(pf_reference=pf,training_samples=len(tr),test_samples=len(te),models={})
        for key,m in models.items():
            trainpred=predict(m,a,c[tr],q[tr],diagnostic=True);pred=predict(m,a,c[te],q[te],diagnostic=True)
            result['models'][key]=dict(train=metrics(trainpred-target[tr]),test=metrics(pred-target[te]),
               by_head={head:metrics(pred[h[te]==head]-target[te][h[te]==head]) for head in sorted(set(h[te]))},rank=m.get('rank',m.get('baseline',{}).get('rank')))
            if name=='ball':
                m.update(pf_reference_training_only=pf['pf_m'],input='rectified [uL,vL,uR,vR] and [yaw,pitch] degrees',reference='nominal fixed arm FK',approved=False)
                Path(__file__).with_name(key+'_train_model.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n')
        out['folds'][name]=result
        Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        print(name,{key:round(v['test']['p95_mm'],3) for key,v in result['models'].items()},flush=True)
    out['models_all_tested_folds_below_1mm']={key:all(f['models'][key]['test']['p95_mm']<1 for f in out['folds'].values()) for key in models}
    out['arbitrary_head_target_demonstrated']=False  # Four reused heads cannot provide acceptance.
    Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
