import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acquisition'))
from orin_rectified_stream.capture_progress import read_progress


class CaptureProgressTests(unittest.TestCase):
    def test_counts_rejected_points_and_detects_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'capture.jsonl';trajectory=root/'path.json'
            self.assertIn('waiting',read_progress(path))
            trajectory.write_text(json.dumps(dict(waypoints=[{}, {}, {}])))
            rows=[dict(taught_path=str(trajectory),waypoint_index=2,capture_valid=True),
                  dict(taught_path=str(trajectory),waypoint_index=1,capture_valid=False)]
            path.write_text('\n'.join(map(json.dumps,rows))+'\n')
            self.assertIn('2/3 (66.7%) | valid 1 | rejected 1',read_progress(path))
            path.write_text('\n'.join(map(json.dumps,rows+[rows[0]])))
            with self.assertRaises(ValueError):read_progress(path)
