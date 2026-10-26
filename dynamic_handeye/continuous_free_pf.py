"""Research-only continuous head/free-pF geometry and smooth conditional residuals."""
import copy
import numpy as np
from .geometry import apply, pose, parameters, torso_from_arm, torso_camera_poses, vector
from .calibration import fit_geometry
from .optimizer import robust_optimize


def ball(poses,pf):
    return poses[:,:3,3]+np.einsum('nij,j->ni',poses[:,:3,:3],pf)


def fit(camera,angles,poses,assets,kind='geometry',length=.05,angle_length=15.,ridge=1.):
    camera=np.asarray(camera,float);angles=np.asarray(angles,float);poses=np.asarray(poses,float)
    if poses.shape!=(len(camera),4,4) or angles.shape!=(len(camera),2):
        raise ValueError('Expected paired camera/head/flange arrays')
    if kind not in ('geometry','spatial','conditional','additive') or min(length,angle_length,ridge)<=0:
        raise ValueError('Invalid residual specification')
    sm=torso_from_arm(assets)
    geometric=fit_geometry(camera,ball(poses,assets.pf),angles,assets)
    initial=np.r_[parameters(geometric['head_tip_from_rectified_camera_T']),assets.pf]
    def residual(x):
        m=dict(geometric,head_tip_from_rectified_camera_T=pose(x[:6]).tolist())
        return (apply(torso_camera_poses(m,assets,angles),camera)-apply(sm,ball(poses,x[6:]))).ravel()*1000
    x=robust_optimize(residual,initial,iterations=200)
    if np.linalg.norm(x[6:])>.2:raise ValueError('Effective pF exceeds 200 mm')
    jac=np.column_stack([(residual(x+np.eye(9)[i]*1e-5)-residual(x-np.eye(9)[i]*1e-5))/2e-5 for i in range(9)])
    singular=np.linalg.svd(jac,compute_uv=False)
    rank=int(sum(singular>singular[0]*1e-7))
    if rank<9:raise ValueError('Free-pF geometry is not identifiable')
    geometric['head_tip_from_rectified_camera_T']=pose(x[:6]).tolist()
    model=dict(schema='luna_continuous_free_pf_research_v1',approved=False,geometry=geometric,pf_m=x[6:].tolist(),
       geometry_parameter_count=9,rank=rank,kind=kind,length_m=length,angle_length_deg=angle_length,ridge=ridge,
       centers_camera_m=camera.tolist(),centers_angles_deg=angles.tolist())
    if kind!='geometry':
        base=apply(torso_camera_poses(geometric,assets,angles),camera)
        target=apply(sm,ball(poses,x[6:]))
        K=kernel(camera,angles,camera,angles,model)
        model['weights_m']=np.linalg.solve(K+ridge*np.eye(len(K)),target-base).tolist()
    return model


def kernel(camera,angles,centers,known_angles,model):
    spatial=np.exp(-np.sum((camera[:,None]-centers[None])**2,axis=2)/(2*model['length_m']**2))
    # Sin/cos embedding is continuous across angular wrap, without a head-ID lookup.
    def features(q):
        a=np.deg2rad(q)
        return np.c_[np.sin(a),np.cos(a)]
    delta=features(angles)[:,None]-features(known_angles)[None]
    angular=np.exp(-np.sum(delta**2,axis=2)/(2*np.deg2rad(model['angle_length_deg'])**2))
    if model['kind']=='spatial':return spatial
    if model['kind']=='conditional':return spatial*angular
    if model['kind']=='additive':return spatial+spatial*angular
    raise ValueError('Geometry has no residual kernel')


def predict(model,assets,camera,angles,*,diagnostic=False):
    if not diagnostic:raise ValueError('Research model lacks independent arbitrary-head/session acceptance')
    camera=np.asarray(camera,float);angles=np.asarray(angles,float)
    vector(camera,(len(camera),3),'camera points');vector(angles,(len(camera),2),'head angles')
    base=apply(torso_camera_poses(model['geometry'],assets,angles),camera)
    if model['kind']!='geometry':
        base+=kernel(camera,angles,np.array(model['centers_camera_m']),np.array(model['centers_angles_deg']),model)@np.array(model['weights_m'])
    return base


def add_residual(geometry,camera,angles,poses,assets,kind,length,angle_length,ridge):
    """Reuse only a training-fitted geometry while selecting residual hyperparameters."""
    model=copy.deepcopy(geometry)
    model.update(kind=kind,length_m=length,angle_length_deg=angle_length,ridge=ridge)
    base=predict(geometry,assets,camera,angles,diagnostic=True)
    target=apply(torso_from_arm(assets),ball(poses,model['pf_m']))
    K=kernel(camera,angles,camera,angles,model)
    model['weights_m']=np.linalg.solve(K+ridge*np.eye(len(K)),target-base).tolist()
    return model
