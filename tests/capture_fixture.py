"""Synthetic stationary observation using the installed camera contract."""
import numpy as np


def capture_row(assets, kind='interactive_ball_point_head'):
    xyz = np.array([.02, -.01, .6])
    pixels = np.concatenate([(p @ np.r_[xyz, 1])[:2] / (p @ np.r_[xyz, 1])[2] for p in assets.P])
    return dict(capture_valid=True, capture_kind=kind, arm_side='right', waypoint_index=0,
                mdh_sha256=assets.hashes['mdh'], stereo_model_sha256=assets.hashes['stereo'],
                source_calibration_sha256=assets.hashes['raw'], sensor_ids=assets.stereo['sensor_ids'],
                image_size=assets.stereo['image_size'], pixel_domain='ideal_pinhole',
                sphere_center_method='sphere_center_projection_stereo_v1', valid_frames=50,
                head_angles_deg=[0, -25], q7_deg=[0]*7, body_q_deg=[0]*6,
                sphere_center_left_camera_m=xyz.tolist(), sphere_center_corrected_stereo_pixels=pixels.tolist(),
                disparity_jitter_p95_px=.01, pair_capture_skew_ms=1.,
                **({'target_head_deg': [0, -25]} if kind == 'interactive_ball_point_head' else {}))
