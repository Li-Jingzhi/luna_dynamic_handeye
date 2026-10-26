"""Continuous head-angle inference; moving mode requires separate acceptance."""
import cv2
import numpy as np
from .geometry import apply, transform, vector, torso_camera_poses, check_version


def camera_pose(model, assets, angles):
    return torso_camera_poses(model, assets, np.array([angles]))[0]


def check_model(model, assets, moving=False):
    check_version(model)
    if model.get('schema') not in ('luna_dynamic_head_model_v1', 'luna_dynamic_head_model_v2') or model.get('robot') != 'gento_luna':
        raise ValueError('Wrong model schema or robot')
    if model.get('assets') != assets.hashes:
        raise ValueError('Model uses different camera/MDH/MJCF assets')
    if (model.get('approved') is not True or model.get('independently_verified') is not True or
            model.get('head_coverage_qualified') is not True):
        raise ValueError('Model has not passed independent head/position/session acceptance')
    if moving and model.get('moving_head_validated') is not True:
        raise ValueError('Moving-head exposure synchronization and accuracy not validated')


def check_support(model, angles, camera=None):
    angle = vector(angles, (2,), 'head angles')
    support = np.asarray(model['head_support_deg'], np.float32)
    hull = cv2.convexHull(support)
    if cv2.contourArea(hull) < 1e-6 or cv2.pointPolygonTest(hull, tuple(angle), False) < 0:
        raise ValueError('Head angle outside calibrated hull')
    if camera is not None:
        known = np.asarray(model['camera_support_m'])
        if np.min(np.linalg.norm(known-camera, axis=1)) > .03:
            raise ValueError('Point outside calibrated spatial support (30mm)')


def predict_stopped(model, assets, pixels, angles, *, head_velocity_deg_s, head_window_range_deg):
    check_model(model, assets)
    speed = vector(head_velocity_deg_s, (2,), 'head speed')
    span = vector(head_window_range_deg, (2,), 'head fusion-window range')
    if np.max(abs(speed)) > .02 or np.min(span) < 0 or np.max(span) > .02:
        raise ValueError('Stopped head and stable fusion window required')
    camera = assets.triangulate(pixels)
    check_support(model, angles, camera)
    return dict(xyz_m=apply(camera_pose(model, assets, angles), camera).tolist(),
                frame_id='S_MODEL', mode='stationary', independently_verified=True)


def pose_at_exposure(history, timestamp_ns, clock_id):
    if not clock_id or len(history) < 2 or any(r.get('clock_id') != clock_id for r in history):
        raise ValueError('Head feedback and exposure must share verified clock')
    stamps = np.array([r['timestamp_ns'] for r in history], dtype=np.int64)
    q = vector([r['head_angles_deg'] for r in history], (len(history), 2), 'head history')
    if np.any(np.diff(stamps) <= 0) or timestamp_ns < stamps[0] or timestamp_ns > stamps[-1]:
        raise ValueError('Unordered head history or exposure extrapolation')
    i = int(np.clip(np.searchsorted(stamps, timestamp_ns, side='right')-1, 0, len(stamps)-2))
    dt = int(stamps[i+1]-stamps[i])
    if dt > 20_000_000:
        raise ValueError('Head feedback bracket exceeds 20ms')
    fraction = (timestamp_ns-int(stamps[i]))/dt
    return q[i]+fraction*(q[i+1]-q[i]), (q[i+1]-q[i])/(dt/1e9)


def predict_moving(model, assets, pixels, observation, history):
    check_model(model, assets, moving=True)
    if observation.get('exposure_time_valid') is not True or observation.get('clock_mapping_verified') is not True:
        raise ValueError('Verified exposure time and clock mapping required')
    uv = vector(pixels, (4,), 'corrected stereo pixels')
    if any(not (0 <= u < 1280 and 0 <= v < 992) for u, v in uv.reshape(2, 2)):
        raise ValueError('Pixels outside calibrated image')
    uncertainty = float(observation['timestamp_uncertainty_ns'])
    if not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError('Invalid exposure time uncertainty')
    right_from_left = np.eye(4)
    right_from_left[:3, :3] = assets.stereo['right_from_left_R']
    right_from_left[:3, 3] = assets.stereo['right_from_left_T_m']
    origins, directions, poses, angles = [], [], [], []
    for i, eye in enumerate(('left', 'right')):
        q, speed = pose_at_exposure(history, observation[f'{eye}_exposure_timestamp_ns'], observation['clock_id'])
        check_support(model, q)
        speed_limit = vector(model['dynamic_speed_limit_deg_s'], (2,), 'validated speed limit')
        radius = float(model['timing_radius_m'])
        if np.any(speed_limit <= 0) or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Invalid certified motion limits')
        if np.any(abs(speed) > speed_limit) or radius*np.deg2rad(np.linalg.norm(speed))*uncertainty/1e9 > .0002:
            raise ValueError('Head speed or 0.2mm timing budget exceeded')
        t = camera_pose(model, assets, q)
        if i == 1:
            t = t @ np.linalg.inv(right_from_left)
        ray = np.linalg.solve(assets.K[i], np.r_[uv[2*i:2*i+2], 1.])
        ray = t[:3, :3] @ ray; ray /= np.linalg.norm(ray)
        origins.append(t[:3, 3]); directions.append(ray); poses.append(t); angles.append(q)
    a = np.column_stack((directions[0], -directions[1]))
    distances, _, rank, singular = np.linalg.lstsq(a, origins[1]-origins[0], rcond=None)
    if rank < 2 or singular[-1] < 1e-4 or np.min(distances) <= 0:
        raise ValueError('Degenerate or negative-depth moving stereo rays')
    ends = np.array(origins)+distances[:, None]*directions
    xyz = ends.mean(0)
    for i in (0, 1):
        local = apply(np.linalg.inv(poses[i]), xyz)
        projection = assets.K[i] @ local
        if local[2] <= 0 or np.linalg.norm(projection[:2]/projection[2]-uv[2*i:2*i+2]) > .5:
            raise ValueError('Moving stereo reprojection exceeds 0.5px')
        check_support(model, angles[i], apply(np.linalg.inv(camera_pose(model, assets, angles[i])), xyz))
    return dict(xyz_m=xyz.tolist(), frame_id='S_MODEL', mode='moving_asynchronous_rays',
                ray_gap_mm=float(np.linalg.norm(ends[0]-ends[1])*1000), independently_verified=True)
