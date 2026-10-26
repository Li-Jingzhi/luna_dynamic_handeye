"""Import unchanged Luna stationary capture logs without importing device SDKs."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .geometry import apply, digest, vector, torso_from_arm


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load_capture(paths, assets, *, left_ball_offset=None):
    rows, rejected, seen, sources = [], Counter(), set(), []
    for filename in paths:
        path = Path(filename)
        source_hash = digest(path)
        manifest_path = path.with_suffix('.completed.json')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('complete') is not True or manifest.get('sha256') != source_hash:
                raise ValueError('Completed capture manifest mismatch: '+str(path))
        session = path.stem
        sources.append(dict(file=str(path), sha256=source_hash))
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('robot_version', 'luna_v2_0') != 'luna_v2_0':
                raise ValueError('Capture belongs to another Luna version')
            if r.get('capture_valid') is not True:
                rejected[r.get('rejection_reason', 'capture_valid_false')] += 1
                continue
            expected = dict(mdh_sha256=assets.hashes['mdh'], stereo_model_sha256=assets.hashes['stereo'],
                source_calibration_sha256=assets.hashes['raw'], sensor_ids=assets.stereo['sensor_ids'],
                image_size=assets.stereo['image_size'], pixel_domain='ideal_pinhole',
                sphere_center_method='sphere_center_projection_stereo_v1')
            for k, v in expected.items():
                if r.get(k) != v:
                    raise ValueError(f'Wrong Luna contract {k} at {path}:{line_number}')
            kind = r.get('capture_kind')
            if kind not in ('interactive_ball_point_head', 'interactive_ball_auto'):
                raise ValueError('Expected stationary point/head capture')
            arm = r.get('arm_side')
            if arm not in ('left', 'right'):
                raise ValueError('Expected left or right arm_side')
            if arm == 'left' and left_ball_offset is None:
                raise ValueError('Left arm needs explicitly measured left_ball_offset in metres')
            pf = assets.pf if arm == 'right' else vector(left_ball_offset, (3,), 'left ball offset')
            count = r.get('valid_frames', 0)
            if not isinstance(count, int) or count < assets.contract['valid_frames_per_point']:
                raise ValueError('Insufficient unique fused frames')
            head = vector(r['head_angles_deg'], (2,), 'head feedback')
            target = vector(r.get('target_head_deg', [0., -25.] if kind == 'interactive_ball_auto' else None), (2,), 'target head')
            q = vector(r['q7_deg'], (7,), 'arm feedback')
            body = vector(r['body_q_deg'], (6,), 'body feedback')
            xyz = vector(r['sphere_center_left_camera_m'], (3,), 'sphere center')
            pixels = vector(r['sphere_center_corrected_stereo_pixels'], (4,), 'corrected sphere pixels')
            jitter = r.get('disparity_jitter_p95_px', float('nan'))
            skew = r.get('pair_capture_skew_ms', float('nan'))
            if (not np.isfinite([jitter, skew]).all() or jitter < 0 or skew < 0 or
                    jitter > assets.contract['max_point_disparity_jitter_p95_px'] or
                    skew > assets.contract['max_pair_skew_ms']):
                raise ValueError('Fused observation violates original quality gates')
            if np.max(abs(head-target)) > .25:
                raise ValueError('Head did not reach stationary target')
            has_window = 'camera_window_frames' in r
            if has_window:
                frames = r['camera_window_frames']
                identities = [(f.get('camera_timestamp_monotonic_ns'), str(f.get('frame_number'))) for f in frames]
                if len(frames) != count or len(set(identities)) != count or any(t is None for t, _ in identities):
                    raise ValueError('Incomplete or reused source frames in fusion window')
                h_window = vector(r.get('robot_window_head_deg'), (count, 2), 'head fusion window')
                q_window = vector(r.get('robot_window_q7_deg'), (count, 7), 'arm fusion window')
                if max(np.ptp(h_window, axis=0).max(), np.ptp(q_window, axis=0).max()) > .02:
                    raise ValueError('Fusion window robot feedback is not stationary')
            reprojection = assets.reprojection(xyz, pixels)
            triangulated = assets.triangulate(pixels)
            key = ','.join(f'{x:g}' for x in target)
            identity = (session, arm, r['waypoint_index'], key)
            if identity in seen:
                raise ValueError('Duplicate session/waypoint/head observation')
            seen.add(identity)
            fk = assets.flange(q, arm)[0]
            point_s = apply(torso_from_arm(assets, arm), apply(fk, pf))
            point_mr = apply(np.linalg.inv(torso_from_arm(assets)), point_s)
            rows.append(dict(session_id=session, waypoint_id=r['waypoint_index'],
                group_id=f'{session}:{arm}:{r["waypoint_index"]}', head_key=key, head_angles_deg=head.tolist(),
                camera_xyz_m=xyz.tolist(), corrected_pixels=pixels.tolist(), q7_deg=q.tolist(),
                target_m=point_mr.tolist(), target_s_m=point_s.tolist(), arm_side=arm, flange_to_ball_center_m=pf.tolist(), body_q_deg=body.tolist(),
                source_line=line_number, source_sha256=source_hash, capture_kind=kind, taught_path=r.get('taught_path'),
                measurement_mode='stationary_fused_with_window' if has_window else 'legacy_stationary_fused',
                exposure_time_valid=r.get('exposure_time_valid') is True,
                reprojection_px=reprojection, fused_vs_triangulated_mm=float(1000*np.linalg.norm(xyz-triangulated))))
    if not rows:
        raise ValueError('No compatible valid observations')
    for session in {r['session_id'] for r in rows}:
        body = np.array([r['body_q_deg'] for r in rows if r['session_id'] == session])
        if np.ptp(body, axis=0).max() > .3:
            raise ValueError('Body moved during legacy stationary session')
    return dict(schema='luna_dynamic_dataset_v1', assets=assets.hashes, sources=sources,
                robot_version='luna_v2_0', target_frame='right_arm_mdh_parent',
                provenance='legacy_stationary_fused_without_complete_per_frame_evidence',
                rejected=dict(rejected), rows=rows)


def arrays(dataset):
    rows = dataset['rows']
    return (np.array([r['camera_xyz_m'] for r in rows]), np.array([r['target_m'] for r in rows]),
            np.array([r['head_angles_deg'] for r in rows]))
