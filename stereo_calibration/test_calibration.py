"""Synthetic stereo recovery and holdout contract tests; no robot or ROS needed."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import calibrate
import evaluate


class CalibrationTests(unittest.TestCase):
    def test_recovery_and_independent_evaluation(self):
        rng = np.random.default_rng(42)
        pattern = (8, 6); square = .025; size = (1280, 992)
        grid = np.zeros((1, 48, 3), np.float64)
        grid[0, :, :2] = np.mgrid[:8, :6].T.reshape(-1, 2) * square
        k = np.array([[650., 0, 640], [0, 655., 496], [0, 0, 1]])
        d = np.array([-.03, .002, -.001, .0001])
        baseline = .06
        manifest = dict(pixel_domain="raw", sensor_ids=["left", "right"],
                        board_inner_corners=list(pattern), square_size_m=square)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train = root / "train"; test = root / "test"
            train.mkdir(); test.mkdir(); points = {}
            for directory, count in ((train, 24), (test, 3)):
                (directory / "capture_manifest.json").write_text(json.dumps(manifest))
                for i in range(count):
                    rotation = rng.uniform(-.45, .45, 3)
                    translation = np.array([rng.uniform(-.22, .05), rng.uniform(-.16, .03), rng.uniform(.45, .85)])
                    for eye, shift in (("left", 0), ("right", -baseline)):
                        path = directory / f"{i:04d}_{eye}.png"
                        path.write_bytes(f"{directory.name}/{i}/{eye}".encode())
                        points[str(path)] = cv2.fisheye.projectPoints(
                            grid, rotation, translation + [shift, 0, 0], k, d)[0]
            def detect(path, board):
                self.assertEqual(tuple(board), pattern)
                return size, points[str(path)]
            output = root / "model.json"
            args = ["calibrate", "--images", str(train), "--output", str(output),
                    "--corners", "8", "6", "--square-mm", "25"]
            with patch.object(sys, "argv", args), patch.object(calibrate, "detect", detect), contextlib.redirect_stdout(io.StringIO()):
                calibrate.main()
            model = json.loads(output.read_text())
            self.assertAlmostEqual(model["baseline_m"], baseline, places=5)
            self.assertLess(model["stereo_rms_px"], .001)
            np.testing.assert_allclose(model["left_K"], k, atol=.02)
            self.assertEqual(len(model["training_image_sha256"]), 48)
            report = root / "evaluation.json"
            args = ["evaluate", "--calibration", str(output), "--images", str(test), "--output", str(report)]
            with patch.object(sys, "argv", args), patch.object(evaluate, "detect", detect), contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            result = json.loads(report.read_text())
            self.assertLess(result["metrics"]["rigid_board_error_mm"]["max"], .01)
            self.assertLess(result["metrics"]["edge_error_mm"]["max"], .01)
            self.assertFalse(result["approved"])
            args[args.index(str(test))] = str(train)
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "training image reused"):
                evaluate.main()
            manifest["pixel_domain"] = "ideal_pinhole"
            (train / "capture_manifest.json").write_text(json.dumps(manifest))
            with patch.object(sys, "argv", ["calibrate", "--images", str(train), "--output", str(output)]), self.assertRaisesRegex(ValueError, "pixel domain"):
                calibrate.main()

    def test_real_corner_detection(self):
        board = np.full((420, 540), 127, np.uint8)
        for row in range(7):
            for col in range(9):
                board[35+row*50:35+(row+1)*50, 45+col*50:45+(col+1)*50] = 255 if (row+col) % 2 else 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            cv2.imwrite(str(path), board)
            size, corners = calibrate.detect(path, (8, 6))
            self.assertEqual(size, (540, 420))
            self.assertEqual(corners.shape, (1, 48, 2))


if __name__ == "__main__": unittest.main()
