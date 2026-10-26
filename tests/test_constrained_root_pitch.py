"""Literal pitch-root cancellation, gauge and constrained parameter recovery."""
import unittest
from reports.constrained_root_pitch_20261026.analyze import self_check


class ConstrainedRootPitchTests(unittest.TestCase):
    def test_cancellation_gauge_and_recovery(self):
        self.assertTrue(all(self_check().values()))
