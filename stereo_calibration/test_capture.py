"""Direct-CSI desktop contracts and automatic pose gating without hardware."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from auto_capture_orin_pro import PoseGate
from capture_settings import SETTINGS, load_settings
from orin_pro_calibration_gui import save_pair


class CaptureTests(unittest.TestCase):
    def test_stability_and_duplicate_gate(self):
        gate = PoseGate(); point = np.zeros((10, 2))
        self.assertFalse(gate.ready(point, 0))
        self.assertFalse(gate.ready(point, .4))
        self.assertTrue(gate.ready(point, .51))
        gate.accepted(point, .51)
        self.assertFalse(gate.ready(point, 2))
        self.assertFalse(gate.ready(point+10, 3))
        self.assertTrue(gate.ready(point+10, 3.51))
        self.assertFalse(gate.ready(None, 4))
        self.assertFalse(gate.ready(point+20, 4.1))

    def test_pair_metadata_and_non_overwriting_resume(self):
        image = np.zeros((1984, 2560, 3), np.uint8)
        metadata = dict(locked=True, pair_skew_ms=10, pair_source_clock_skew_ms=10, sensors=SETTINGS['sensors'],
                        service='luna_stereo_direct_csi_v2')
        def archive():
            data = io.BytesIO()
            with zipfile.ZipFile(data, 'w') as z:
                z.writestr('left.png', b'left'); z.writestr('right.png', b'right')
                z.writestr('metadata.json', json.dumps(metadata))
            return data.getvalue()
        with tempfile.TemporaryDirectory() as tmp, patch('orin_pro_calibration_gui.cv2.imdecode', return_value=image), patch('orin_pro_calibration_gui.cv2.findChessboardCornersSB', return_value=(True,None)):
            root = Path(tmp); (root/'0005_left.png').write_bytes(b'existing')
            self.assertEqual(save_pair(archive(), root, (11,8), 35), 6)
            self.assertEqual((root/'0005_left.png').read_bytes(), b'existing')
            metadata['sensors'] = [99,98]
            with self.assertRaisesRegex(ValueError, 'sensor order'): save_pair(archive(), root, (11,8), 35)
            metadata['sensors'] = SETTINGS['sensors']; metadata['locked'] = False
            with self.assertRaisesRegex(ValueError, 'timing gate'): save_pair(archive(), root, (11,8), 35)
            metadata['locked'] = True; metadata['pair_skew_ms'] = 21
            with self.assertRaisesRegex(ValueError, 'timing gate'): save_pair(archive(), root, (11,8), 35)

    def test_invalid_sensor_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            path.write_text(json.dumps(dict(SETTINGS, sensors=[0,0])))
            with self.assertRaisesRegex(ValueError, 'distinct'): load_settings(path)


if __name__ == '__main__': unittest.main()
