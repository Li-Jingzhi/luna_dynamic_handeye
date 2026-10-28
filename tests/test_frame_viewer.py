"""MuJoCo compilation checks; skipped when optional viewer dependency is absent."""
import json
from dynamic_handeye.view_frames import DIAGNOSTIC
import importlib.util
import unittest
import numpy as np
from dynamic_handeye.geometry import Assets, body_fixed, mdh_poses
from dynamic_handeye.view_frames import frames, load_scene, draw, COLORS, MODEL, CAD_BOTTOM_Z, DEFAULT_VISIBLE
import xml.etree.ElementTree as ET


@unittest.skipUnless(json.loads(DIAGNOSTIC.read_text())['assets'] == Assets().hashes, 'Archived frame fit belongs to retired camera; refit required')
@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Optional MuJoCo viewer environment required')
class FrameViewerTests(unittest.TestCase):
    def test_nominal_bases_close_first_mdh_and_stereo_baseline(self):
        assets = Assets()
        model, data = load_scene()
        entries = frames(assets, model, data, [0., -25.])
        by_name = {r['name']: r for r in entries}
        s = by_name['S_TORSO']['T']
        torso = ET.parse(MODEL).getroot().find(".//body[@name='Gento Luna_Leg_J6_Link']")
        for side, prefix in [('R', 'R.A1'), ('L', 'R.A0')]:
            first = mdh_poses(assets.rows(prefix, 7)[:1], np.zeros((1, 1)))[0]
            m = by_name[f'M_{side} [NOMINAL]']['T']
            installed = body_fixed(torso.find(f"body[@name='Arm_{side}1_Link']"))
            np.testing.assert_allclose(m @ first, s @ installed, atol=1e-12)
            self.assertEqual(by_name[f'M_{side} [NOMINAL]']['kind'], 'nominal_first_joint_alignment')
        l = by_name['C_L [FIT]']['T']; r = by_name['C_R [FIT]']['T']
        self.assertAlmostEqual(np.linalg.norm(l[:3, 3]-r[:3, 3]), .05953835532631367)
        self.assertTrue(all('unapproved' in e['kind'] for e in entries if e['group'] in (3, 4)))
        np.testing.assert_allclose(data.qpos, np.zeros(model.nq))
        np.testing.assert_allclose(s[:3, 3], [0., 0., .865], atol=1e-5)

    def test_nominal_mdh_roots_match_all_seven_cad_joint_origins_and_axes(self):
        import mujoco
        assets = Assets()
        model, data = load_scene()
        entries = frames(assets, model, data, [0., -25.])
        bases = {r['name']: r['T'] for r in entries}
        for q in (np.zeros(7), np.array([-20., 15., -10., -35., 12., 8., -20.])):
            for side, prefix in [('R', 'R.A1'), ('L', 'R.A0')]:
                for i in range(7):
                    joint = model.joint(f'Arm_{side}{i+1}_Joint')
                    data.qpos[joint.qposadr[0]] = np.deg2rad(q[i])
                mujoco.mj_forward(model, data)
                base = bases[f'M_{side} [NOMINAL]']
                for i in range(7):
                    predicted = base @ mdh_poses(assets.rows(prefix, 7)[:i+1], q[None, :i+1])[0]
                    body = data.body(f'Arm_{side}{i+1}_Link')
                    np.testing.assert_allclose(predicted[:3, 3], body.xpos, atol=2e-5)
                    np.testing.assert_allclose(predicted[:3, 2], body.xmat.reshape(3, 3)[:, 2], atol=3e-5)

    def test_cad_references_preserve_source_origins_without_promoting_fit(self):
        model, data = load_scene()
        entries = frames(Assets(), model, data, [0., -25.])
        by_name = {r['name']: r for r in entries}
        s = by_name['S_TORSO']['T']
        relative = np.linalg.inv(s) @ by_name['S_INTERFACE [CAD REF]']['T']
        np.testing.assert_allclose(relative[:3, 3], [0, 0, CAD_BOTTOM_Z], atol=1e-12)
        raw = (MODEL.parent/'meshes/Gento Luna_Leg_J6_Link.STL').read_bytes()
        triangles = np.frombuffer(raw, dtype=[('normal', '<f4', 3), ('vertices', '<f4', (3, 3)),
                                             ('attribute', '<u2')], offset=84)
        self.assertAlmostEqual(float(triangles['vertices'][:, :, 2].min()), CAD_BOTTOM_Z, places=7)
        np.testing.assert_allclose(by_name['JH1 [CAD]']['T'][:3, 3],
                                   by_name['JH2 [CAD]']['T'][:3, 3], atol=1e-12)
        self.assertEqual(by_name['JH2 [CAD]']['parent'], 'JH1 [CAD]')
        self.assertEqual(by_name['JH1 [CAD]']['parent'], 'S_TORSO')
        self.assertFalse(DEFAULT_VISIBLE[3])
        self.assertFalse(DEFAULT_VISIBLE[4])
        self.assertTrue(DEFAULT_VISIBLE[6])

    def test_positive_axis_shafts_reach_xyz_label_positions(self):
        import mujoco
        assets = Assets()
        model, data = load_scene()
        entries = frames(assets, model, data, [0., -25.])
        scene = mujoco.MjvScene(model, maxgeom=1000)
        draw(scene, entries, {1: True})
        origin = entries[0]['T'][:3, 3]
        for axis, letter in enumerate('XYZ'):
            shafts = [g for g in scene.geoms[:scene.ngeom] if g.type == mujoco.mjtGeom.mjGEOM_LINE
                      and np.allclose(g.rgba, COLORS[axis], atol=1e-6)]
            labels = [g for g in scene.geoms[:scene.ngeom] if g.label == f'S.{letter}']
            self.assertEqual(len(shafts), 1)
            self.assertEqual(len(labels), 1)
            shaft = shafts[0]
            tip = shaft.pos+shaft.mat.reshape(3, 3)[:, 2]*shaft.size[2]
            np.testing.assert_allclose(shaft.pos, origin, atol=1e-7)
            np.testing.assert_allclose(tip, labels[0].pos, atol=1e-7)
            self.assertEqual(labels[0].type, mujoco.mjtGeom.mjGEOM_LABEL)


if __name__ == '__main__':
    unittest.main()
