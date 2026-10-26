"""Effective geometry recovery and the final pixel-only research interface."""
import unittest
from reports.fk_consistent_vision_20261026.analyze import self_check


class FKReferenceTests(unittest.TestCase):
    def test_uncertain_geometry_and_unseen_pose(self):
        result=self_check()
        self.assertLess(result['learned12_recovery_max_mm'],.002)
        self.assertLess(result['poe14_nonorthogonal_separated_axis_unseen_max_mm'],.2)
        self.assertTrue(result['pixel_interface_no_arm_inputs'])
