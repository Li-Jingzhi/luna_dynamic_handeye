import unittest
import json
from pathlib import Path
import numpy as np

from reports.new_data_refit_20261029.analyze import FIXED_PF_M, clamp_encoder_quantization


class EncoderQuantizationTest(unittest.TestCase):
    def test_clamps_only_small_limit_rounding(self):
        np.testing.assert_allclose(clamp_encoder_quantization([[10, -25.001]]), [[10, -25]])
        with self.assertRaises(ValueError):
            clamp_encoder_quantization([[10, -25.02]])

    def test_generated_result_keeps_model_diagnostic(self):
        root = Path(__file__).resolve().parent
        result = json.loads((root / "results.json").read_text())
        blocks = [v for k, v in result["folds"].items() if k.startswith("block_")]
        expected = min(result["folds"]["all"]["models"], key=lambda model: (
            max(f["models"][model]["test"]["p95_mm"] for f in blocks),
            np.mean([f["models"][model]["test"]["p95_mm"] for f in blocks]), model))
        self.assertEqual(result["selection"]["model"], expected)
        self.assertFalse(result["approved"])
        self.assertFalse(result["validation_summary"][result["selection"]["model"]]["all_p95_below_1mm"])
        self.assertEqual(set(result["head_counts"]), {"0,-25", "0,-20", "10,-25", "10,-20"})
        np.testing.assert_allclose(result["flange_to_ball_center_m"], FIXED_PF_M, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
