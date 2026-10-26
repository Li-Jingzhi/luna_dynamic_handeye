"""Local MuJoCo frame inspection, without SDK, network, or physics stepping."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from .geometry import Assets, ROOT, body_fixed, mdh_poses, transform, vector

MODEL = ROOT/'reference/LunaUdpTool/model/gento_luna/gento_luna_contact_filtered.xml'
DIAGNOSTIC = ROOT/'reports/luna_v2_hybrid_6dof_20261026/model.json'
# Lowest planar surface of Leg_J6_Link.STL; a display reference, not a controller frame.
CAD_BOTTOM_Z = .182
DEFAULT_VISIBLE = {1: True, 2: True, 3: False, 4: False, 5: False, 6: True}
COLORS = np.array([[.9, .06, .04, 1.], [.02, .58, .12, 1.], [.04, .22, .95, 1.]])


def load_scene():
    import mujoco
    tree = ET.parse(MODEL)
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(MODEL.parent/'meshes'))
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1600', offheight='1200')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5', diffuse='.65 .65 .65', specular='.1 .1 .1')
    ET.SubElement(root.find('asset'), 'texture', name='inspection_background', type='skybox',
                  builtin='gradient', rgb1='.94 .96 .99', rgb2='.82 .87 .93', width='512', height='3072')
    for geom in root.findall('.//geom'):
        geom.set('rgba', '.35 .43 .55 .35')
    ET.SubElement(root.find('worldbody'), 'geom', name='inspection_floor', type='plane',
                  size='2 2 .01', rgba='.82 .85 .89 1', contype='0', conaffinity='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def frames(assets, model, data, angles):
    """Nominal MDH bases assume the first axis matches its CAD joint frame at zero."""
    s = np.eye(4)
    s[:3, :3] = data.body('Gento Luna_Leg_J6_Link').xmat.reshape(3, 3)
    s[:3, 3] = data.body('Gento Luna_Leg_J6_Link').xpos
    torso = ET.parse(MODEL).getroot().find(".//body[@name='Gento Luna_Leg_J6_Link']")
    out = [dict(name='S_TORSO', kind='model', group=1, size=.10, T=s)]
    for side, arm in [('R', 'right'), ('L', 'left')]:
        installed = body_fixed(torso.find(f"body[@name='Arm_{side}1_Link']"))
        first_row = assets.rows('R.A1' if side == 'R' else 'R.A0', 7)[:1]
        offset = mdh_poses(first_row, np.zeros((1, 1)))[0]
        sm = installed @ np.linalg.inv(offset)
        out.append(dict(name=f'M_{side} [NOMINAL]', kind='nominal_first_joint_alignment', group=2, size=.085, T=s@sm, parent='S_TORSO'))
    fitted = json.loads(DIAGNOSTIC.read_text())
    if fitted['assets'] != assets.hashes or fitted['chain'] not in ('luna_v2_hybrid', 'controller_mdh'):
        raise ValueError('Diagnostic model does not match controller/camera assets')
    if fitted['schema'] == 'luna_dynamic_head_model_v2':
        h0 = s @ transform(fitted['torso_from_head_root_T'])
    else:
        h0 = out[1]['T'] @ transform(fitted['mdh_from_head_root_T'])
    h2 = h0 @ assets.head(np.array([angles]), fitted['chain'])[0]
    left = h2 @ transform(fitted['head_tip_from_rectified_camera_T'])
    right_from_left = np.eye(4)
    right_from_left[:3, :3] = assets.stereo['right_from_left_R']
    right_from_left[:3, 3] = assets.stereo['right_from_left_T_m']
    out.extend([
        dict(name='H0_CTRL [FIT]', kind='unapproved_diagnostic', group=3, size=.09, T=h0),
        dict(name='H2_CTRL [FIT]', kind='unapproved_diagnostic', group=3, size=.075, T=h2),
        dict(name='C_L [FIT]', kind='unapproved_diagnostic', group=4, size=.045, T=left),
        dict(name='C_R [FIT]', kind='unapproved_diagnostic', group=4, size=.045, T=left@np.linalg.inv(right_from_left)),
    ])
    interface = np.eye(4)
    interface[2, 3] = CAD_BOTTOM_Z
    out.append(dict(name='S_INTERFACE [CAD REF]', kind='cad_surface_reference_only',
                    group=6, size=.10, T=s@interface, parent='S_TORSO'))
    for index in (1, 2):
        body = data.body(f'Head_J{index}_Link')
        pose = np.eye(4)
        pose[:3, :3] = body.xmat.reshape(3, 3)
        pose[:3, 3] = body.xpos
        out.append(dict(name=f'JH{index} [CAD]', kind='reference_mesh_frame', group=6,
                        size=.10 if index == 1 else .065, T=pose,
                        parent='S_TORSO' if index == 1 else 'JH1 [CAD]'))
    return out


def draw(scene, entries, visible):
    import mujoco

    def geom(kind, size, pos, color, label=''):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Insufficient inspection scene capacity')
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(), np.asarray(color, float))
        g.emission = 1.0
        g.label = label
        scene.ngeom += 1
        return g

    for item in entries:
        if not visible.get(item['group'], False):
            continue
        t, length = item['T'], item['size']
        origin = t[:3, 3]
        label_offsets = {'S_TORSO': [.08, -.16, -.08], 'M_R [NOMINAL]': [0, -.22, -.01],
                         'M_L [NOMINAL]': [0, .22, .02], 'H0_CTRL [FIT]': [-.04, -.12, -.04],
                         'H2_CTRL [FIT]': [0, -.18, .08], 'C_L [FIT]': [0, -.19, .19],
                         'C_R [FIT]': [0, .19, .16], 'S_INTERFACE [CAD REF]': [0, -.25, .0],
                         'JH1 [CAD]': [0, -.23, .13], 'JH2 [CAD]': [0, .23, .18]}
        label_pos = origin+label_offsets[item['name']]
        display_name = item['name'].replace('S_TORSO', 'S_MODEL (Leg_J6)').replace('H0_CTRL [FIT]', 'H_REF [FIXED NOMINAL]').replace('H2_CTRL [FIT]', 'H1 [FIXED FK]')
        geom(mujoco.mjtGeom.mjGEOM_LABEL, [0, 0, 0], label_pos, [.1, .1, .1, 1], display_name)
        leader = geom(mujoco.mjtGeom.mjGEOM_LINE, [1.5, 0, 0], origin, [.25, .28, .33, .85])
        mujoco.mjv_connector(leader, mujoco.mjtGeom.mjGEOM_LINE, 1.5, origin, label_pos)
        geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.004, 0, 0], origin, [.08, .08, .08, 1])
        for i, axis in enumerate('XYZ'):
            tip = origin+length*t[:3, i]
            # A full-length shaft keeps XYZ labels attached to a visible positive axis.
            shaft = geom(mujoco.mjtGeom.mjGEOM_LINE, [3, 0, 0], origin, COLORS[i])
            mujoco.mjv_connector(shaft, mujoco.mjtGeom.mjGEOM_LINE, 3, origin, tip)
            head_start = tip-.018*t[:3, i]
            arrow = geom(mujoco.mjtGeom.mjGEOM_ARROW, [.004, .004, .018], head_start, COLORS[i])
            mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .004, head_start, tip)
            short = item['name'].split()[0].replace('M_', '').replace('_CTRL', '').replace('_TORSO', '')
            geom(mujoco.mjtGeom.mjGEOM_LABEL, [0, 0, 0], tip, COLORS[i], f'{short}.{axis}')
    by_name = {e['name']: e for e in entries}
    for item in entries:
        parent = by_name.get(item.get('parent'))
        if parent is None or not visible.get(item['group']) or not visible.get(parent['group']):
            continue
        start, end = parent['T'][:3, 3], item['T'][:3, 3]
        if np.linalg.norm(end-start) > 1e-8:
            edge = geom(mujoco.mjtGeom.mjGEOM_ARROW, [.002, 0, 0], start, [.45, .22, .55, .7])
            mujoco.mjv_connector(edge, mujoco.mjtGeom.mjGEOM_ARROW, .002, start, end)
    camera = [r['T'][:3, 3] for r in entries if r['name'].startswith('C_')]
    if visible.get(4, False):
        bar = geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [.002, 0, 0], camera[0], [.7, .45, .02, 1])
        mujoco.mjv_connector(bar, mujoco.mjtGeom.mjGEOM_CAPSULE, .002, *camera)


def camera_setup(camera):
    camera.lookat[:] = [.02, .0, 1.15]
    camera.distance = 1.95
    camera.azimuth = 145
    camera.elevation = -10


def screenshot(model, data, entries, filename, angles):
    import cv2
    import mujoco
    camera = mujoco.MjvCamera()
    camera_setup(camera)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    with mujoco.Renderer(model, height=1200, width=1600) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        draw(renderer.scene, entries, DEFAULT_VISIBLE)
        rgb = renderer.render()
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lines = ['CAD frame audit | RED=X GREEN=Y BLUE=Z; purple arrows=parent -> child',
             'S_MODEL = Leg_J6 origin; S_INTERFACE = CAD bottom plane (+182 mm), reference only',
             'JH1/JH2 = MJCF head frames, SAME origin; not controller head frames',
             'Diagnostic head/camera fits are hidden; keys 3/4 reveal them. Key 6: CAD references.']
    for i, line in enumerate(lines):
        cv2.putText(image, line, (25, 35+29*i), cv2.FONT_HERSHEY_SIMPLEX, .60, (35, 40, 50), 2, cv2.LINE_AA)
    panel = np.full((1200, 860, 3), 248, dtype=np.uint8)
    hierarchy = [
        'MODEL TREE (MJCF, neutral pose)', '',
        'S_MODEL = Leg_J6_Link',
        '  |-> Arm_R1_Link: [0,-214.6,391.5] mm',
        '  |     Rx(+90 deg) -> right arm chain',
        '  |-> Arm_L1_Link: [0,+214.6,391.5] mm',
        '  |     Rx(-90 deg) -> left arm chain',
        '  |-> Head_J1_Link: [-13.496,0,498.98] mm',
        '        Rx(+90 deg) -> joint 1 (local Z)',
        '        |-> Head_J2_Link: [0,0,0] mm',
        '              Rx(-90 deg) -> joint 2 (local Z)', '',
        'JH1 and JH2 origins coincide in this file.',
        'Joint axis locations cannot be read from shell edges.',
        'No camera body/site exists in the supplied MJCF.', '',
        'REFERENCE ORIGIN (display only)',
        'S_MODEL -> S_INTERFACE: [0,0,182] mm, R=I',
        '182 mm = lowest torso STL surface, not SDK data.',
        'Using this origin subtracts 182 mm from table Z.', '',
        'CONTROLLER CHAIN (authoritative for calibration)',
        'S -> H_base [fixed mounting, needs alignment]',
        'H_base -> H_after_yaw -> H_after_pitch',
        'H_after_pitch -> C_L [fixed extrinsic to calibrate]',
        'C_L -> C_R [known stereo relative transform]', '',
        'H_base is an MDH parent, not the first motor center.',
        'The fitted H_base/cameras are NOT verified mounts.',
        'M_R/M_L in 3D: derived nominal MDH parent frames.',
    ]
    for i, line in enumerate(hierarchy):
        cv2.putText(panel, line, (20, 38+35*i), cv2.FONT_HERSHEY_SIMPLEX, .53,
                    (35, 40, 50), 1, cv2.LINE_AA)
    image = np.concatenate([image, panel], axis=1)
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError('Could not save MuJoCo screenshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head', nargs=2, type=float, default=[0., -25.], metavar=('YAW', 'PITCH'))
    parser.add_argument('--check', action='store_true', help='Compile model and verify frame geometry without opening a window')
    parser.add_argument('--snapshot', help='Save rendered PNG before opening the window')
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    import mujoco
    angles = vector(args.head, (2,), 'head feedback')
    assets = Assets()
    model, data = load_scene()
    entries = frames(assets, model, data, angles)
    for item in entries:
        transform(item['T'])
    positions = [item['T'][:3, 3] for item in entries if item['name'].startswith('C_')]
    np.testing.assert_allclose(np.linalg.norm(positions[0]-positions[1]),
                               np.linalg.norm(assets.stereo['right_from_left_T_m']), atol=1e-10)
    report = dict(head_angles_deg=angles.tolist(), hardware_access=False,
                  notes=['M_L/M_R are nominal first-joint alignments, not measured installation transforms.',
                         'Head root is fixed nominal geometry; cameras use the unapproved six-parameter model.',
                         'JH1/JH2 are coincident MJCF origins, not controller head frames.',
                         'S_INTERFACE is a CAD bottom-surface reference at +182 mm, not a verified controller frame.'],
                  frames=[dict(name=e['name'], kind=e['kind'], parent=e.get('parent'), world_from_frame_T=e['T'].tolist()) for e in entries])
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        return
    if args.snapshot:
        screenshot(model, data, entries, args.snapshot, angles)
    if args.snapshot_only:
        return
    import mujoco.viewer
    visible = DEFAULT_VISIBLE.copy()

    def key_callback(key):
        if ord('1') <= key <= ord('6'):
            group = key-ord('0')
            visible[group] = not visible[group]

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            camera_setup(viewer.cam)
            viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw(viewer.user_scn, entries, visible)
            viewer.sync()
            time.sleep(.03)


if __name__ == '__main__':
    main()
