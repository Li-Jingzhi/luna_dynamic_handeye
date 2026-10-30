import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dynamic_handeye.geometry import Assets
from data_collection.diagnostics.cli import MODEL
from data_collection.diagnostics.core import default_plan
from data_collection.diagnostics.collect import Collector,run_session,skippable_window_error
from data_collection.diagnostics.simulation import SimRobot,SimCamera

class RejectedWindowTests(unittest.TestCase):
    def test_only_optical_quality_failures_can_be_skipped(self):
        self.assertTrue(skippable_window_error(ValueError('Corrected disparity jitter exceeds 0.1px')))
        self.assertTrue(skippable_window_error(RuntimeError('Only 12 of 50 unique qualified frames; window preserved')))
        for message in ('Robot moved within fusion window','Measured ball left local camera workspace','Visual servo diverged','Joint limit exceeded'):
            self.assertFalse(skippable_window_error(RuntimeError(message)))

    def test_failed_window_retained_and_never_counted_as_valid(self):
        assets=Assets();model=json.loads(MODEL.read_text());plan=default_plan();plan['continue_measurement_rejects']=True
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(temp)/'capture'
            with patch.object(Collector,'window',side_effect=ValueError('Corrected disparity jitter exceeds 0.1px')):
                result=run_session(assets,MODEL,plan,'optical',folder,lambda:SimRobot(assets,model,plan),
                    lambda robot:SimCamera(robot,assets),simulated=True,limit=1)
            self.assertTrue(result['complete'])
            row=json.loads((folder/'observations.jsonl').read_text());self.assertFalse(row['capture_valid'])
            summary=json.loads((folder/'completed.json').read_text())
            self.assertEqual(summary['valid_windows'],0);self.assertEqual(summary['rejected_windows'],1)
            self.assertFalse(summary['all_windows_valid'])
