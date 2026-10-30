"""Diagnostic fixed-head camera mapping; no arm feedback is used for prediction."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,vector,fixed_geometry_hashes
from dynamic_handeye.empirical import correct


def predict_pixels(model,assets,pixels,head,*,diagnostic=False):
    if model.get('schema')!='luna_fixed_view_calibrated_fk_v1':raise ValueError('Wrong fixed-view model')
    if model['assets']!=assets.hashes:raise ValueError('Camera/FK assets changed')
    if model.get('fixed_geometry_hashes')!=fixed_geometry_hashes(assets):raise ValueError('Fixed geometry configuration changed')
    if not diagnostic:raise ValueError('Independent acceptance required before non-diagnostic use')
    qh=vector(head,(2,),'measured head feedback')
    if np.max(abs(qh-np.array(model['fixed_head_deg'])))>model['fixed_head_tolerance_deg']:
        raise ValueError('Head outside calibrated fixed view')
    c=assets.triangulate(vector(pixels,(4,),'corrected sphere-center/corresponding point pixels'))
    pose=model['camera_to_torso'];base=c*model['camera_scale']@np.array(pose['R']).T+pose['t_m']
    field=model.get('correction',dict(kind='geometry'));position=correct(field,base[None])[0]
    nearest=None
    if field.get('centers_m'):
        nearest=float(np.min(np.linalg.norm(np.array(field['centers_m'])-base,axis=1))*1000)
    return dict(xyz_m=position.tolist(),frame_id='S_MODEL',diagnostic=True,independently_verified=False,
        reference='FK(q + effective_joint_offsets_deg) * fixed pF; original controller FK differs',
        effective_reference_offsets_deg=model['effective_joint_offsets_deg'],nearest_training_position_mm=nearest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(OUT/'fixed_head_best_candidate.json'))
    p.add_argument('--pixels',required=True,nargs=4,type=float)
    p.add_argument('--head',required=True,nargs=2,type=float)
    p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();model=json.loads(Path(a.model).read_text())
    print(json.dumps(predict_pixels(model,Assets(),a.pixels,a.head,diagnostic=a.diagnostic),indent=2,allow_nan=False))


if __name__=='__main__':main()
