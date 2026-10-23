"""Runtime passive stereo XYZ query, always in robot_base and metres."""
import json
from pathlib import Path
import cv2
import numpy as np

from .geometry import dynamic_projections, ideal_pixels, project, triangulate


class Mapping:
    def __init__(self, model):
        if model.get("schema") != "head_2dof_capt_v1" or model.get("stage") not in ("reference", "dynamic"):
            raise ValueError("Unsupported calibration schema")
        if model.get("units") != "m" or model.get("frame_id") != "robot_base":
            raise ValueError("Unrecognized XYZ coordinate contract")
        if model.get("motion_enabled") is not False:
            raise ValueError("Control-enabled model forbidden in passive mapping")
        self.data = model
        self.Pl = np.asarray(model["P_left_reference"], float)
        self.Pr = np.asarray(model["P_right_reference"], float)
        if any(P.shape != (3,4) or not np.isfinite(P).all() or np.linalg.matrix_rank(P) != 3 for P in (self.Pl,self.Pr)):
            raise ValueError("Invalid 3x4 projection matrices")
        self.q0 = np.asarray(model["reference_head_angles_rad"], float)
        if self.q0.shape != (2,):
            raise ValueError("Invalid reference head")
        self.dynamic = model["stage"] == "dynamic"
        if self.dynamic:
            self.params = np.asarray(model["head_parameters"], float)
            self.hull = np.asarray(model["head_angle_hull_rad"], np.float32).reshape(-1, 1, 2)
            if self.params.shape != (12,) or len(self.hull) < 3:
                raise ValueError("Incomplete dynamic two-axis model")
    @classmethod
    def load(cls, filename):
        return cls(json.loads(Path(filename).read_text(encoding="utf-8")))

    def projections(self, angles_rad):
        q = np.asarray(angles_rad, float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("Measured head yaw/pitch required in radians")
        if not self.dynamic:
            if np.max(np.abs(q-self.q0)) > np.radians(.5):
                raise ValueError("Reference-only model rejects moved head")
            return self.Pl, self.Pr
        if cv2.pointPolygonTest(self.hull, (float(q[0]),float(q[1])), True) < -1.e-5:
            raise ValueError("Head outside calibrated 2D yaw/pitch range")
        return dynamic_projections(self.Pl, self.Pr, self.params, q, self.q0)

    def query(self, pixels, head_angles_rad, *, sensor_ids=None, image_size=None,
              pixel_domain=None):
        d = self.data
        if sensor_ids is not None and sensor_ids != d["sensor_ids"]:
            raise ValueError("Stereo camera order changed")
        if image_size is not None and image_size != d["image_size"]:
            raise ValueError("Image resolution changed")
        if pixel_domain is not None and pixel_domain != d["input_pixel_domain"]:
            raise ValueError("Pixel domain changed")
        raw = np.asarray(pixels, float)
        if raw.shape != (4,) or not np.isfinite(raw).all():
            raise ValueError("Four stereo pixels required")
        w,h = d["image_size"]
        if (np.any(raw[[0,2]]<0) or np.any(raw[[0,2]]>=w) or
                np.any(raw[[1,3]]<0) or np.any(raw[[1,3]]>=h)):
            raise ValueError("Pixel coordinates out of bounds")
        ideal = ideal_pixels(raw.reshape(1,4),d.get("lens"))[0]
        Pl,Pr = self.projections(head_angles_rad)
        xyz = triangulate(Pl,Pr,ideal)
        ep = max(np.linalg.norm(project(P,[xyz])[0]-obs)
                 for P,obs in ((Pl,ideal[:2]),(Pr,ideal[2:])))
        if ep > d["max_reprojection_error_px"]:
            raise ValueError("Excess stereo reprojection residual")
        bounds = np.asarray(d["xyz_bounds_m"], float)
        if bounds.shape != (2,3) or np.any(xyz<bounds[0]) or np.any(xyz>bounds[1]):
            raise ValueError("XYZ outside fitted workspace")
        return dict(valid=True, frame_id="robot_base", units="m", xyz_m=xyz.tolist(),
                    reprojection_error_px=float(ep),
                    head_angles_rad=np.asarray(head_angles_rad,float).tolist(),
                    calibration_id=d["calibration_id"],
                    independently_verified=False, research_only=True, motion_enabled=False)
