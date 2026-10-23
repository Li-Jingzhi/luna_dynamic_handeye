"""Luna controller MDH is authoritative; MJCF assets are structural references."""
import configparser
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'Expected finite {name}, shape {shape}')
    return a


def transform(value):
    t = vector(value, (4, 4), 'rigid transform')
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-7) or
            not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-7)):
        raise ValueError('Transform must be SE(3), not affine')
    return t


def pose(parameters):
    p = vector(parameters, (6,), 'rotation-vector and translation')
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(p[:3])[0]
    t[:3, 3] = p[3:]
    return t


def parameters(t):
    t = transform(t)
    return np.r_[cv2.Rodrigues(t[:3, :3])[0].ravel(), t[:3, 3]]


def apply(t, points):
    return np.einsum('...ij,...j->...i', t[..., :3, :3], points) + t[..., :3, 3]


def mdh_poses(rows, angles):
    # Same Rx(alpha) Tx(a) Rz(theta+q) Tz(d) convention as Luna capture FK.
    q = np.asarray(angles, float)
    if q.ndim != 2 or not np.isfinite(q).all() or q.shape[1] > len(rows):
        raise ValueError('Invalid joint angle matrix')
    out = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    for i, (a, alpha, d, theta) in enumerate(rows):
        ca, sa = np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha))
        ct = np.cos(np.deg2rad(theta + (q[:, i] if i < q.shape[1] else 0)))
        st = np.sin(np.deg2rad(theta + (q[:, i] if i < q.shape[1] else 0)))
        step = np.zeros_like(out)
        step[:, 0, 0] = ct; step[:, 0, 1] = -st; step[:, 0, 3] = a / 1000
        step[:, 1, 0] = ca*st; step[:, 1, 1] = ca*ct; step[:, 1, 2] = -sa; step[:, 1, 3] = -d*sa/1000
        step[:, 2, 0] = sa*st; step[:, 2, 1] = sa*ct; step[:, 2, 2] = ca; step[:, 2, 3] = d*ca/1000
        step[:, 3, 3] = 1
        out = out @ step
    return out


def body_fixed(body):
    w, x, y, z = vector(list(map(float, body.get('quat', '1 0 0 0').split())), (4,), 'MJCF quaternion')
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        raise ValueError('Zero MJCF quaternion')
    w, x, y, z = np.array([w, x, y, z])/norm
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = vector(list(map(float, body.get('pos', '0 0 0').split())), (3,), 'MJCF translation')
    return transform(t)


class Assets:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        paths = dict(mdh='configs/robot/luna_new.ini', raw='configs/camera/luna_raw_fisheye_stereo.json',
                     stereo='configs/camera/luna_rectified_stereo.json', ball='configs/robot/ball_offset.json',
                     mjcf='reference/LunaUdpTool/model/gento_luna/gento_luna_contact_filtered.xml')
        self.hashes = {k: digest(self.root/v) for k, v in paths.items()}
        source = json.loads((self.root/'sources.json').read_text())
        recorded = {r['destination']: r['sha256'] for r in source['files']}
        if any(self.hashes[k] != recorded[v] for k, v in paths.items()):
            raise ValueError('Calibration/model asset differs from source manifest')
        contract_path = 'configs/camera/yellow_measurement_source.json'
        if digest(self.root/contract_path) != recorded[contract_path]:
            raise ValueError('Camera quality contract differs from source manifest')
        self.raw = json.loads((self.root/paths['raw']).read_text())
        self.stereo = json.loads((self.root/paths['stereo']).read_text())
        self.ball = json.loads((self.root/paths['ball']).read_text())
        self.contract = json.loads((self.root/'configs/camera/yellow_measurement_source.json').read_text())
        if (self.stereo['source_calibration_sha256'] != self.hashes['raw'] or
                self.ball['mdh_sha256'] != self.hashes['mdh'] or self.ball['stereo_sha256'] != self.hashes['stereo'] or
                self.stereo['sensor_ids'] != ['head_left:csi://0', 'head_right:csi://1'] or
                self.stereo['image_size'] != [1280, 992] or self.stereo['pixel_domain'] != 'ideal_pinhole'):
            raise ValueError('Incompatible Luna camera/robot/ball contract')
        self.ini = configparser.ConfigParser()
        self.ini.read(self.root/paths['mdh'])
        self.head_rows = self.rows('R.H', 2)
        self.pf = vector(self.ball['flange_to_ball_center_m'], (3,), 'measured pF')
        self.K = [vector(self.stereo[f'{s}_K'], (3, 3), 'camera K') for s in ('left', 'right')]
        self.P = [self.K[0] @ np.c_[np.eye(3), np.zeros(3)],
                  self.K[1] @ np.c_[self.stereo['right_from_left_R'], self.stereo['right_from_left_T_m']]]

    def rows(self, prefix, count):
        sections = [f'{prefix}.L{i}.DH' for i in range(count)]
        if count == 7:
            sections.append(f'{prefix}.FLANGE')
        return np.array([[float(self.ini[s][k]) for k in ('A', 'Alpha', 'D', 'Theta')] for s in sections])

    def flange(self, q, arm='right'):
        if arm not in ('left', 'right'):
            raise ValueError('Unknown arm')
        return mdh_poses(self.rows('R.A1' if arm == 'right' else 'R.A0', 7), np.atleast_2d(q))

    def head(self, angles, chain='controller_mdh', mapping=None):
        q = np.asarray(angles, float)
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError('Expected Nx2 measured head angles in degrees')
        if chain != 'controller_mdh':
            raise ValueError('Controller MDH is authoritative; MJCF is reference only')
        return mdh_poses(self.head_rows, q)

    def triangulate(self, pixels, pixel_domain='ideal_pinhole'):
        uv = vector(pixels, (4,), 'stereo pixels')
        size = self.raw['image_size'] if pixel_domain == 'raw' else self.stereo['image_size']
        if any(not (0 <= u < size[0] and 0 <= v < size[1]) for u, v in uv.reshape(2, 2)):
            raise ValueError('Pixels outside calibrated image')
        if pixel_domain == 'raw':
            uv = np.concatenate([cv2.fisheye.undistortPoints(uv[2*i:2*i+2].reshape(1, 1, 2),
                 np.array(self.raw[f'{eye}_K']), np.array(self.raw[f'{eye}_D']),
                 R=np.array(self.raw[f'rectify_R{i+1}']), P=self.K[i]).ravel() for i, eye in enumerate(('left', 'right'))])
        elif pixel_domain != 'ideal_pinhole':
            raise ValueError('Use raw or ideal_pinhole pixel domain')
        if abs(uv[1]-uv[3]) > self.contract['max_epipolar_dv_px']:
            raise ValueError('Epipolar mismatch')
        h = cv2.triangulatePoints(*self.P, uv[:2].reshape(2, 1), uv[2:].reshape(2, 1)).ravel()
        if not np.isfinite(h).all() or abs(h[3]) < 1e-12:
            raise ValueError('Degenerate stereo rays')
        xyz = h[:3]/h[3]
        self.reprojection(xyz, uv)
        return xyz

    def reprojection(self, xyz, pixels):
        errors = []
        for i, p in enumerate(self.P):
            image = p @ np.r_[xyz, 1.]
            if image[2] <= 0:
                raise ValueError('Negative stereo depth')
            errors.append(np.linalg.norm(image[:2]/image[2]-pixels[2*i:2*i+2]))
        if max(errors) > .5:
            raise ValueError('Stereo reprojection exceeds 0.5px')
        return float(max(errors))


def torso_from_arm(assets):
    """Nominal CAD torso output; does not assert controller head-root alignment."""
    mounts = json.loads((assets.root/'configs/robot/torso_mounts.json').read_text())
    if digest(assets.root/mounts['source_model']) != mounts['source_model_sha256']:
        raise ValueError('Torso mounting source model changed')
    rows = [r for r in mounts['relations'] if r['parent'] == 'S_MODEL'
            and r['child'] == 'right_arm_mdh_parent']
    if len(rows) != 1:
        raise ValueError('Expected one torso-to-right-arm mounting transform')
    return transform(rows[0]['parent_from_child_T'])
