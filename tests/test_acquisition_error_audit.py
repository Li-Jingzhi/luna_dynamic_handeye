import unittest
import numpy as np
from dynamic_handeye.geometry import Assets,apply,torso_from_arm
from reports.acquisition_error_audit_20261029.joint_zero_probe import corrected_fk
from reports.acquisition_error_audit_20261029.audit import mono_sphere

class AuditTests(unittest.TestCase):
    def test_joint_correction_zero_preserves_reference(self):
        a=Assets();q=np.zeros((2,7))
        np.testing.assert_allclose(corrected_fk(q,np.zeros(12),a),apply(torso_from_arm(a)@a.flange(q),a.pf))

    def test_monocular_known_radius_centered_sphere(self):
        radius=.02;depth=.4;f=400.;pixel_radius=f*radius/np.sqrt(depth**2-radius**2)
        detection=dict(center=[640,496],ellipse=dict(major_px=2*pixel_radius,minor_px=2*pixel_radius,angle_deg=0))
        point,error=mono_sphere(detection,np.array([[f,0,640],[0,f,496],[0,0,1]]))
        np.testing.assert_allclose(point,[0,0,depth],atol=1e-10)
        self.assertLess(error,1e-10)
