import unittest
import numpy as np
from metrics import scene_scores, macro_average


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(0).uniform(0.1, 0.9, (24, 28, 5))

    def test_identical(self):
        r = scene_scores(self.image, self.image, scale=4)["metrics"]
        self.assertAlmostEqual(r["MPSNR_dB"], 120.0)
        self.assertAlmostEqual(r["MSSIM_Wang"], 1.0)
        self.assertLess(r["SAM_degrees"], 1e-5)
        self.assertEqual(r["ERGAS"], 0.0)

    def test_psnr_is_mean_of_band_decibels(self):
        image = np.full((20, 20, 2), 0.5)
        estimate = image + np.array([0.01, 0.1])
        self.assertAlmostEqual(scene_scores(image, estimate, scale=4)["metrics"]["MPSNR_dB"], 30.0)

    def test_zero_prediction_angle(self):
        r = scene_scores(self.image, np.zeros_like(self.image), scale=4)
        self.assertEqual(r["metrics"]["SAM_degrees"], 90.0)

    def test_mask_and_nonfinite(self):
        mask = np.ones(self.image.shape[:2], bool)
        mask[:2] = False
        r = scene_scores(self.image, self.image, scale=4, region_mask=mask)
        self.assertEqual(r["diagnostics"]["valid_pixels"], 22*28)
        broken = self.image.copy(); broken[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            scene_scores(self.image, broken, scale=4)

    def test_macro_uses_equal_scene_weights(self):
        a = scene_scores(self.image, self.image+0.01, scale=4)
        b = scene_scores(self.image[:16, :16], self.image[:16, :16]+0.02, scale=4)
        result = macro_average([a, b])
        for key in result:
            self.assertAlmostEqual(result[key], (a["metrics"][key]+b["metrics"][key])/2)

    def test_ergas_spatial_ratio(self):
        a = scene_scores(self.image, self.image+0.01, scale=2)["metrics"]["ERGAS"]
        b = scene_scores(self.image, self.image+0.01, scale=4)["metrics"]["ERGAS"]
        self.assertAlmostEqual(a, 2*b)


if __name__ == "__main__":
    unittest.main()
