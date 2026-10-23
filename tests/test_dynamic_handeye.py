import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dynamic_handeye.geometry import Assets, ROOT, apply, pose, mdh_poses
from dynamic_handeye.calibration import fit_geometry, predict, evaluate
from dynamic_handeye.dataset import load_capture, write_json
from dynamic_handeye.runtime import predict_stopped, predict_moving, pose_at_exposure
from dynamic_handeye.capture import load_plan


class DynamicHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = Assets()
        cls.grid = np.array([(y, p) for y in (-10, 0, 10) for p in (-25, -20, -15)], float)
        cls.b = pose([.1, -.15, .2, .03, -.02, .05])
        cls.a = np.linalg.inv(cls.assets.head(np.array([[0., -20.]]))[0] @ cls.b)
        cls.truth = dict(schema='luna_dynamic_head_model_v1', robot='gento_luna', assets=cls.assets.hashes,
            chain='controller_mdh', mjcf_mapping=None, mdh_from_head_root_T=cls.a.tolist(),
            head_tip_from_rectified_camera_T=cls.b.tolist(), frame_id='right_arm_mdh_parent',
            approved=True, independently_verified=True, head_coverage_qualified=True,
            moving_head_validated=False, head_support_deg=cls.grid.tolist())
        rng = np.random.default_rng(23)
        targets = rng.uniform([-.06, -.04, .32], [.06, .04, .48], (15, 3))
        cls.angles = np.tile(cls.grid, (len(targets), 1))
        cls.target = np.repeat(targets, len(cls.grid), axis=0)
        transforms = cls.a @ cls.assets.head(cls.angles) @ cls.b
        cls.camera = apply(np.linalg.inv(transforms), cls.target)
        cls.truth['camera_support_m'] = cls.camera.tolist()
        cls.fitted = fit_geometry(cls.camera, cls.target, cls.angles, cls.assets)

    def project(self, camera):
        return np.concatenate([(p @ np.r_[camera, 1])[:2]/(p @ np.r_[camera, 1])[2] for p in self.assets.P])

    def test_recovers_unseen_angles_without_q7_or_pf_at_inference(self):
        angles = np.array([[3., -22.], [-4., -18.]])
        camera = np.array([[.02, -.01, .4], [-.03, .01, .38]])
        np.testing.assert_allclose(predict(self.fitted, self.assets, camera, angles),
                                   predict(self.truth, self.assets, camera, angles), atol=1e-6)
        self.assertEqual(self.fitted['jacobian_rank'], 12)
        self.assertFalse(self.fitted['approved'])

    def test_camera_signed_baseline_and_raw_roundtrip(self):
        import cv2
        camera = np.array([.02, -.01, .4]); uv = self.project(camera)
        self.assertGreater(uv[0]-uv[2], 0)
        np.testing.assert_allclose(self.assets.triangulate(uv), camera, atol=1e-10)
        raw = self.assets.raw
        left = np.array(raw['rectify_R1']).T @ camera
        right = np.array(raw['right_from_left_R']) @ left + raw['right_from_left_T_m']
        pixels = []
        for eye, point in zip(('left', 'right'), (left, right)):
            image, _ = cv2.fisheye.projectPoints(point.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                           np.array(raw[f'{eye}_K']), np.array(raw[f'{eye}_D']))
            pixels.extend(image.ravel())
        np.testing.assert_allclose(self.assets.triangulate(pixels, 'raw'), camera, atol=1e-10)
        with self.assertRaises(ValueError): self.assets.triangulate([float('nan'), 0, 0, 0])
        with self.assertRaises(ValueError): self.assets.triangulate(uv[[2, 3, 0, 1]])

    def test_controller_fk_includes_fixed_flange(self):
        rows = self.assets.rows('R.A1', 7)
        self.assertEqual(rows[2, 2], 264.)
        self.assertEqual(rows[4, 2], 264.)
        self.assertEqual(rows[-1, 2], 87.)
        q = np.array([[5, -10, 15, -40, 10, 20, -5]])
        full = self.assets.flange(q)[0]
        partial = mdh_poses(rows[:-1], q)[0]
        self.assertAlmostEqual(np.linalg.norm(full[:3, 3]-partial[:3, 3]), .087)

    def test_controller_is_authoritative_and_insufficient_excitation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Two-dimensional'):
            fit_geometry(self.camera, self.target, np.tile([0., -20.], (len(self.camera), 1)), self.assets)
        with self.assertRaisesRegex(ValueError, 'authoritative'):
            self.assets.head(self.grid, 'mjcf')
        mapping = dict(verified=True, feedback_indices=[1, 0], signs=[1, 1])
        with self.assertRaisesRegex(ValueError, 'authoritative'):
            self.assets.head(np.array([[0., 0.]]), 'mjcf', mapping)

    def test_stopped_guards_and_approved_synthetic_prediction(self):
        model = copy.deepcopy(self.truth)
        index = 4
        camera = self.camera[index]; q = self.angles[index]
        kwargs = dict(head_velocity_deg_s=[0, 0], head_window_range_deg=[0, 0])
        result = predict_stopped(model, self.assets, self.project(camera), q, **kwargs)
        from dynamic_handeye.geometry import torso_from_arm
        self.assertEqual(result['frame_id'], 'S_MODEL')
        np.testing.assert_allclose(result['xyz_m'], apply(torso_from_arm(self.assets), self.target[index]), atol=1e-10)
        for field, value in [('approved', False), ('assets', {}), ('head_coverage_qualified', False)]:
            bad = copy.deepcopy(model); bad[field] = value
            with self.assertRaises(ValueError): predict_stopped(bad, self.assets, self.project(camera), q, **kwargs)
        with self.assertRaises(ValueError): predict_stopped(model, self.assets, self.project(camera), [50, -20], **kwargs)
        with self.assertRaises(ValueError): predict_stopped(model, self.assets, self.project(camera), q,
                                      head_velocity_deg_s=[1, 0], head_window_range_deg=[0, 0])

    def test_asynchronous_exposure_rays_recover_static_world_point(self):
        from dynamic_handeye.runtime import camera_pose
        model = copy.deepcopy(self.truth)
        model.update(moving_head_validated=True, dynamic_speed_limit_deg_s=[30, 30], timing_radius_m=.7)
        history = [dict(timestamp_ns=1_000_000_000, clock_id='sensor', head_angles_deg=[0, -20]),
                   dict(timestamp_ns=1_020_000_000, clock_id='sensor', head_angles_deg=[.4, -20])]
        obs = dict(exposure_time_valid=True, clock_mapping_verified=True, clock_id='sensor',
                   left_exposure_timestamp_ns=1_005_000_000, right_exposure_timestamp_ns=1_015_000_000,
                   timestamp_uncertainty_ns=1000)
        from dynamic_handeye.geometry import torso_from_arm
        xyz = apply(torso_from_arm(self.assets), self.target[4])
        uv, support = [], []
        rl = np.eye(4); rl[:3, 3] = self.assets.stereo['right_from_left_T_m']
        for i, eye in enumerate(('left', 'right')):
            q, _ = pose_at_exposure(history, obs[f'{eye}_exposure_timestamp_ns'], 'sensor')
            t = camera_pose(model, self.assets, q)
            support.append(apply(np.linalg.inv(t), xyz))
            if i: t = t @ np.linalg.inv(rl)
            p = self.assets.K[i] @ apply(np.linalg.inv(t), xyz)
            uv.extend(p[:2]/p[2])
        model['camera_support_m'].extend(np.array(support).tolist())
        result = predict_moving(model, self.assets, uv, obs, history)
        np.testing.assert_allclose(result['xyz_m'], xyz, atol=1e-9)
        bad = dict(obs, exposure_time_valid=False)
        with self.assertRaises(ValueError): predict_moving(model, self.assets, uv, bad, history)
        bad = dict(obs, timestamp_uncertainty_ns=100_000_000)
        with self.assertRaises(ValueError): predict_moving(model, self.assets, uv, bad, history)
        with self.assertRaises(ValueError): pose_at_exposure(history, 0, 'sensor')
        with self.assertRaises(ValueError): pose_at_exposure(history, obs['left_exposure_timestamp_ns'], 'pc')

    def test_capture_plan_and_existing_real_log_contract(self):
        plan = load_plan(ROOT/'configs/capture_plan.json', self.assets)
        self.assertEqual(len(plan['head_angles_deg']), 9)
        dataset = load_capture([ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl'], self.assets)
        self.assertEqual(len(dataset['rows']), 884)
        self.assertEqual(sum(dataset['rejected'].values()), 85)
        self.assertEqual(len({r['head_key'] for r in dataset['rows']}), 3)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'bad.jsonl'
            original = json.loads((ROOT/'datasets/raw/ball_point_head_20261018_v1.jsonl').read_text().splitlines()[0])
            original['sensor_ids'] = ['left:csi://8', 'right:csi://0']
            p.write_text(json.dumps(original)+'\n')
            with self.assertRaisesRegex(ValueError, 'sensor_ids'): load_capture([p], self.assets)
            original['sensor_ids'] = self.assets.stereo['sensor_ids']
            original['camera_window_frames'] = [dict(camera_timestamp_monotonic_ns=1, frame_number=1)]*50
            p.write_text(json.dumps(original)+'\n')
            with self.assertRaisesRegex(ValueError, 'reused source frames'): load_capture([p], self.assets)
            original['camera_window_frames'] = [dict(camera_timestamp_monotonic_ns=i, frame_number=i) for i in range(50)]
            original['robot_window_head_deg'] = [[0, -25]]*49+[[.1, -25]]
            original['robot_window_q7_deg'] = [original['q7_deg']]*50
            p.write_text(json.dumps(original)+'\n')
            with self.assertRaisesRegex(ValueError, 'not stationary'): load_capture([p], self.assets)
            write_json(Path(temp)/'model.json', {'ok': True})
            with self.assertRaises(FileExistsError): write_json(Path(temp)/'model.json', {'ok': False})

    def test_grouped_validation_keeps_entire_ball_positions_together(self):
        rows = []
        for i, (camera, target, angle) in enumerate(zip(self.camera, self.target, self.angles)):
            rows.append(dict(camera_xyz_m=camera.tolist(), target_m=target.tolist(), head_angles_deg=angle.tolist(),
                             group_id=f'ball:{i//9}', head_key=','.join(map(str, angle)), session_id='synthetic'))
        dataset = dict(rows=rows, rejected={}, sources=[dict(sha256='synthetic')])
        model, report = evaluate(dataset, self.assets)
        for fold in [report['trajectory_holdout'], *report['spatial_holdouts']]:
            self.assertEqual(fold['status'], 'diagnostic')
            self.assertFalse(set(fold['train_groups']) & set(fold['test_groups']))
            self.assertLess(fold['metrics']['p95_mm'], .001)
        self.assertTrue(model['head_coverage_qualified'])
        self.assertFalse(model['approved'])


if __name__ == '__main__':
    unittest.main()
