import unittest

import numpy as np

from meghprahari import ctt as C


def cold_blob(cy, cx, n=20, base=280.0, depth=60.0, r=3.0):
    """A warm background with one cold spot (like a storm cloud) centred at (cy, cx)."""
    yy, xx = np.mgrid[0:n, 0:n]
    return base - depth * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))


class TestCttMin(unittest.TestCase):
    def test_coldest_value_and_missing(self):
        frames = np.stack([cold_blob(10, 10, depth=40), cold_blob(10, 10, depth=60), cold_blob(10, 10, depth=20)])
        m = C.ctt_min(frames)
        self.assertAlmostEqual(float(m[10, 10]), 220.0, delta=0.5)          # 280 - 60 (the coldest of the 3 frames)
        self.assertTrue(np.isfinite(m).all())

    def test_all_missing_is_nan(self):
        frames = np.full((3, 4, 4), np.nan)
        self.assertTrue(np.isnan(C.ctt_min(frames)).all())

    def test_partial_missing_ignored(self):
        frames = np.stack([cold_blob(2, 2), cold_blob(2, 2)])
        frames[0, 2, 2] = np.nan
        self.assertAlmostEqual(float(C.ctt_min(frames)[2, 2]), frames[1, 2, 2], places=5)


class TestCttDropRate(unittest.TestCase):
    def test_steady_cooling_detected(self):
        base = cold_blob(5, 5, depth=30)
        frames = np.stack([base - 5.0 * k for k in range(6)])              # cooling 5 K per step
        rate = C.ctt_drop_rate(frames, dt_h=3.0)                            # 5 K per 3 h step
        self.assertAlmostEqual(float(rate[5, 5]), -5.0 / 3.0, delta=0.05)
        self.assertLess(float(rate[5, 5]), 0)

    def test_warming_is_positive(self):
        base = cold_blob(5, 5, depth=30)
        frames = np.stack([base + 4.0 * k for k in range(6)])
        self.assertGreater(float(C.ctt_drop_rate(frames, 1.0)[5, 5]), 0)

    def test_too_few_frames_is_nan(self):
        frames = np.stack([cold_blob(3, 3), cold_blob(3, 3)])
        self.assertTrue(np.isnan(C.ctt_drop_rate(frames, 1.0)).all())


class TestColdCloudArea(unittest.TestCase):
    def test_area_matches_pixel_count(self):
        frame = np.full((20, 20), 280.0)
        frame[5:8, 5:8] = 200.0                                             # a 3x3 = 9 pixel cold patch
        area = C.cold_cloud_area_km2(frame, thr_k=221.0, pixel_km2=64.0)
        self.assertAlmostEqual(float(area[6, 6]), 9 * 64.0)
        self.assertEqual(float(area[0, 0]), 0.0)                            # warm pixel: not part of any cold patch

    def test_missing_pixels_excluded(self):
        frame = np.full((10, 10), 280.0)
        frame[4:6, 4:6] = 210.0
        frame[4, 4] = np.nan
        area = C.cold_cloud_area_km2(frame, 221.0, 4.0)
        self.assertAlmostEqual(float(area[5, 5]), 3 * 4.0)                  # 4 cold cells minus 1 missing
        self.assertTrue(np.isnan(area[4, 4]) == False)                       # missing pixel itself gets area 0, not NaN
        self.assertEqual(float(area[4, 4]), 0.0)

    def test_two_separate_patches_not_merged(self):
        frame = np.full((20, 20), 280.0)
        frame[2, 2] = 200.0
        frame[15, 15] = 200.0
        area = C.cold_cloud_area_km2(frame, 221.0, 1.0)
        self.assertAlmostEqual(float(area[2, 2]), 1.0)
        self.assertAlmostEqual(float(area[15, 15]), 1.0)


if __name__ == "__main__":
    unittest.main()
