import unittest
import threading
import cv2
import numpy as np

from live_rectified_preview import make_maps, LatestReceiver


class PreviewMapsTest(unittest.TestCase):
    def test_latest_receiver_does_not_wait_for_other_eye(self):
        class Capture:
            def __init__(self):
                self.ready = threading.Event()
                self.released = threading.Event()
            def read(self):
                self.ready.wait(1)
                return False, None
            def release(self):
                self.released.set()
        slow, fast = Capture(), Capture()
        receivers = [LatestReceiver(slow), LatestReceiver(fast)]
        try:
            fast.ready.set()
            self.assertTrue(fast.released.wait(1))
            self.assertFalse(slow.released.is_set())
            self.assertIsNotNone(receivers[1].error)
        finally:
            slow.ready.set()
            for receiver in receivers:
                receiver.stop.set()
                receiver.thread.join(2)

    def test_scaled_maps_and_aspect_ratio(self):
        K = np.array([[980., 0, 1280], [0, 980, 992], [0, 0, 1]])
        P = np.array([[800., 0, 1280, 0], [0, 800, 992, 0], [0, 0, 1, 0]])
        c = dict(distortion_model='opencv_fisheye', image_size=[2560, 1984])
        for eye, index in (('left', 1), ('right', 2)):
            c.update({f'{eye}_K': K.tolist(), f'{eye}_D': [0, 0, 0, 0],
                      f'rectify_R{index}': np.eye(3).tolist(), f'rectify_P{index}': P.tolist()})
        for mapping in make_maps(c):
            x, y = cv2.convertMaps(*mapping, cv2.CV_32FC1)
            self.assertEqual(x.shape, (496, 640))
            self.assertAlmostEqual(float(x[248, 320]), 320)
            self.assertAlmostEqual(float(y[248, 320]), 248)
            expected = 320 + 245 * np.arctan((400 - 320) / 200)
            self.assertAlmostEqual(float(x[248, 400]), expected, delta=.04)
        with self.assertRaises(ValueError):
            make_maps(c, (640, 480))


if __name__ == '__main__':
    unittest.main()
