"""Offline joint-space transfer only; never imports SDK or commands a robot."""
import configparser
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dynamic_handeye.geometry import Assets,mdh_poses,digest
OUT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'Data/marvin_pro_right_arm_teaching_trajectory_360points_20261020.json'
SOURCE_MDH=ROOT.parent/'head_eye_pro/configs/1004.ini'
MODEL=ROOT/'reference/LunaUdpTool/model/gento_luna/gento_luna_contact_filtered.xml'


def interpolate(points, max_step=1.):
    for index,(a,b) in enumerate(zip(points[:-1],points[1:])):
        count=max(1,int(np.ceil(np.max(np.abs(b-a))/max_step)))
        for step in range(count):yield index,a+(b-a)*step/count
    yield len(points)-1,points[-1]


def main():
    source=json.loads(SOURCE.read_text());snapshot=json.loads((OUT/'controller_snapshot.json').read_text())
    assert source['schema']=='head_eye_pro_taught_path_v1' and source['complete'] and source['arm_side']=='right'
    assert digest(SOURCE_MDH)==source['mdh_sha256'],'Source MDH file hash differs from recorded trajectory'
    q=np.array([w['q7_deg'] for w in source['waypoints']],float)
    assert q.shape==(360,7) and np.isfinite(q).all()
    limits=np.array(snapshot['arm_limits_deg']);assets=Assets()
    config=configparser.ConfigParser();config.read(SOURCE_MDH)
    sections=[*(f'R.A1.L{i}.DH' for i in range(7)),'R.A1.FLANGE']
    old=np.array([[float(config[s][k]) for k in ('A','Alpha','D','Theta')] for s in sections])
    new=assets.rows('R.A1',7)
    angle_definitions_equal=bool(np.allclose(old[:,[1,3]],new[:,[1,3]],atol=1e-9))
    source_fk=mdh_poses(old,q);target_fk=mdh_poses(new,q)
    if angle_definitions_equal:assert np.allclose(source_fk[:,:3,:3],target_fk[:,:3,:3],atol=1e-12)
    delta=target_fk[:,:3,3]-source_fk[:,:3,3]
    distances=1000*np.linalg.norm(delta,axis=1)
    bad=np.argwhere((q<limits[:,0])|(q>limits[:,1]))
    tree=ET.parse(MODEL);tree.getroot().find('compiler').set('meshdir',str(MODEL.parent/'meshes'))
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'));data=mujoco.MjData(model)
    feedback=snapshot['feedback']
    def set_joints(names,values):
        for name,value in zip(names,values):
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            assert j>=0,name
            data.qpos[model.jnt_qposadr[j]]=np.deg2rad(value)
    set_joints([f'Gento Luna_Leg_J{i}_Joint' for i in range(1,7)],feedback['body']['fb_pos'])
    set_joints([f'Arm_L{i}_Joint' for i in range(1,8)],feedback['arms'][0]['fb']['fb_pos'])
    set_joints(['Head_J1_joint','Head_J2_Joint'],feedback['head']['fb_pos'][:2])
    def collisions(points):
        checked=0;colliding=0;minimum=0.;pairs={};examples=[]
        for index,point in interpolate(np.array(points)):
            set_joints([f'Arm_R{i}_Joint' for i in range(1,8)],point)
            mujoco.mj_forward(model,data);checked+=1;hit=False
            for contact in data.contact[:data.ncon]:
                names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or 'world' for g in (contact.geom1,contact.geom2)]
                if not any(name.startswith('Arm_R') for name in names) or contact.dist>=0:continue
                hit=True;minimum=min(minimum,float(contact.dist));key=' / '.join(sorted(names))
                pairs[key]=min(pairs.get(key,0.),float(contact.dist)*1000)
            if hit:
                colliding+=1
                if len(examples)<10:examples.append(dict(segment=index,q7_deg=point.tolist()))
        return dict(samples=checked,colliding_samples=colliding,maximum_penetration_mm=-minimum*1000,pairs_min_distance_mm=pairs,first_collision_samples=examples)
    # Positive control ensures enabled contacts can actually detect right-arm intersections.
    random_points=np.random.default_rng(28).uniform(limits[:,0],limits[:,1],(10,7))
    positive_control=collisions(random_points)
    assert positive_control['colliding_samples']>0,'Collision detector positive control failed'
    current=np.array(feedback['arms'][1]['fb']['fb_pos'])
    approach=collisions([current,q[-1]])
    replay=collisions(q[::-1])
    candidate=dict(schema='luna_offline_joint_transfer_v1',executable=False,approved=False,
                   robot_version='luna_v2_0',arm_side='right',head_target_deg=[0,-20],
                   mapping='joint_angle_identity_candidate_not_cartesian_retargeting',
                   source_trajectory=str(SOURCE),source_trajectory_sha256=digest(SOURCE),
                   source_mdh_sha256=digest(SOURCE_MDH),target_mdh_sha256=assets.hashes['mdh'],
                   fixed_body_q_deg=feedback['body']['fb_pos'],
                   waypoints=[dict(source_waypoint_index=i,q7_deg=row.tolist()) for i,row in enumerate(q)])
    result=dict(points=len(q),angle_definitions_equal=angle_definitions_equal,
                mdh_rows_source_mm_deg=old.tolist(),mdh_rows_luna_mm_deg=new.tolist(),
                limit_violations=bad.tolist(),minimum_joint_limit_margin_deg=float(np.minimum(q-limits[:,0],limits[:,1]-q).min()),
                maximum_adjacent_joint_step_deg=float(np.abs(np.diff(q,axis=0)).max()),
                fk_difference_under_numerically_aligned_mdh_parents_mm=dict(mean=float(distances.mean()),p95=float(np.percentile(distances,95)),maximum=float(distances.max())),
                approach_to_reverse_start=approach,reverse_replay=replay,
                collision_positive_control=dict(samples=positive_control['samples'],colliding_samples=positive_control['colliding_samples']),
                source_sha256=digest(SOURCE),model_sha256=digest(MODEL),executable=False,
                limitations=['MDH parent frames numerically aligned for FK comparison; physical base registration is not measured.',
                    'CAD/controller head-chain mapping is unverified; head collision geometry is approximate.',
                    'Convex mesh contacts can be conservative; missing attachments, ball rod, table and people are not represented.',
                    'No physical replay authorization is derived from this offline check.'])
    assert np.array_equal(np.array([w['q7_deg'] for w in candidate['waypoints']]),q)
    check=np.array([row for _,row in interpolate(np.array([[0.]*7,[2.5]*7]))])
    assert np.max(np.abs(np.diff(check,axis=0)))<=1.
    (OUT/'luna_joint_candidate_360.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (OUT/'offline_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['mdh_rows_source_mm_deg','mdh_rows_luna_mm_deg']},indent=2))


if __name__=='__main__':main()
