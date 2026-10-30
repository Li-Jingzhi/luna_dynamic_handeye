"""Fixed-FK empirical hand-eye mapping. Camera/head inputs only; acceptance is separate."""
import numpy as np
import cv2
from .geometry import apply,pose,vector,fixed_geometry_hashes


def kernel(x,y,length):
    return np.exp(-np.maximum(np.sum(x*x,1)[:,None]+np.sum(y*y,1)[None]-2*x@y.T,0)/(2*length**2))


def base_points(baseline,assets,camera,head):
    x=vector(baseline['x'],(12,),'effective head-chain parameters')
    return apply(pose(x[:6])@assets.head(np.asarray(head,float))@pose(x[6:]),camera)


def fit_fields(base,target,length,ridges):
    """Joint forward and Pro-style inverse affine/Gaussian fits; frozen target labels."""
    center=base.mean(0);target_center=target.mean(0);scale=.2
    f=np.c_[np.ones(len(base)),(base-center)/scale]
    d=np.c_[np.ones(len(target)),(target-target_center)/scale]
    residual=target-base
    rhs=np.c_[f,residual,d,base]
    k=kernel(base,base,length)
    output={}
    for ridge in ridges:
        inv=np.linalg.solve(k+ridge*np.eye(len(base)),rhs)
        # Forward global affine residual and Gaussian field are estimated jointly.
        b=np.linalg.solve(f.T@inv[:,:4]+.01*np.eye(4),f.T@inv[:,4:7])
        w=inv[:,4:7]-inv[:,:4]@b
        common=dict(length_m=length,ridge=ridge,center_m=center.tolist(),centers_m=base.tolist())
        output[('forward',ridge)]=dict(kind='forward',trend=b.tolist(),weights=w.tolist(),**common)
        # Pro equation: D(target) M = base + K W; invert only the learned affine block.
        m=np.linalg.solve(d.T@inv[:,7:11],d.T@inv[:,11:14])
        wi=inv[:,7:11]@m-inv[:,11:14]
        matrix=m[1:]/scale
        sv=np.linalg.svd(matrix,compute_uv=False)
        output[('inverse',ridge)]=dict(kind='inverse',mapping=m.tolist(),weights=wi.tolist(),
            target_center_m=target_center.tolist(),affine_singular_values=sv.tolist(),
            within_pro_scale_bounds=bool(sv.min()>=.9 and sv.max()<=1.1),**common)
    return output


def correct(field,base):
    kind=field['kind']
    if kind=='geometry':return base
    f=np.c_[np.ones(len(base)),(base-np.array(field['center_m']))/.2]
    if kind=='affine':return base+f@np.array(field['trend'])
    k=kernel(base,np.array(field['centers_m']),field['length_m'])
    if kind=='forward':return base+f@np.array(field['trend'])+k@np.array(field['weights'])
    if kind=='inverse':
        m=np.array(field['mapping']);p=base+k@np.array(field['weights'])-m[0]
        return np.linalg.solve(m[1:].T,p.T).T*.2+np.array(field['target_center_m'])
    raise ValueError('Unknown empirical correction')


def support(model,base,head):
    hull=cv2.convexHull(np.asarray(model['support']['head_deg'],np.float32))
    head_ok=cv2.contourArea(hull)>1e-6 and cv2.pointPolygonTest(hull,tuple(map(float,head)),False)>=0
    distance=float(np.sqrt(np.min(np.sum((np.array(model['support']['base_centers_m'])-base)**2,axis=1))))
    return dict(head_inside_calibrated_hull=bool(head_ok),nearest_training_position_mm=distance*1000,
                position_inside_support=bool(distance<=model['support']['radius_m']))


def predict_pixels(model,assets,pixels,head_angles,*,diagnostic=False):
    if model.get('schema')!='luna_empirical_handeye_v1':raise ValueError('Wrong empirical model schema')
    if model.get('assets')!=assets.hashes or model.get('fixed_geometry_hashes')!=fixed_geometry_hashes(assets):
        raise ValueError('Model camera/FK/reference assets changed')
    if not diagnostic:raise ValueError('Independent acceptance required; this model is diagnostic only')
    head=vector(head_angles,(2,),'measured head angles').copy()
    # Match the logged encoder quantization tolerance, not a general extrapolation policy.
    if -25.01<=head[1]<-25:head[1]=-25
    camera=assets.triangulate(vector(pixels,(4,),'corrected sphere-center/corresponding-point pixels'))
    base=base_points(model['baseline'],assets,camera[None],head[None])[0]
    corrected=correct(model['correction'],base[None])[0]
    return dict(xyz_m=corrected.tolist(),frame_id='S_MODEL',diagnostic=True,
                independently_verified=False,candidate=model['candidate'],**support(model,base,head))
