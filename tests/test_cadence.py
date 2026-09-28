"""Rain features must mean the same at 30-min (HEM/IMERG) and 60-min (MERA) cadence."""
import unittest

import numpy as np

from meghprahari import pipeline as P


def storm(dt_min, n_min=300):
    """Gaussian cell moving 8 km/h east on 4-km pixels, peak rain rising 6 mm/h per hour, sampled every dt_min."""
    yy, xx = np.mgrid[0:48, 0:64]
    return np.array([(10 + 6 * m / 60) * np.exp(-((yy - 24) ** 2 + (xx - 10 - 2 * m / 60) ** 2) / 50.0)
                     for m in range(0, n_min + 1, dt_min)])


class TestCadence(unittest.TestCase):
    def test_frames_for(self):
        self.assertEqual(P.frames_for(P.HISTORY_MIN, 30), 6)          # unchanged 30-min behaviour
        self.assertEqual(P.frames_for(P.HISTORY_MIN, 60), 3)
        self.assertEqual(P.frames_for(P.MAX_WINDOW_MIN, 30), 3)
        self.assertEqual(P.frames_for(P.MAX_WINDOW_MIN, 60), 2)
        with self.assertRaises(ValueError):
            P.frames_for(60, 90)

    def test_same_physics_at_30_and_60_min(self):
        f30 = P.hem_fields(storm(30)[-P.frames_for(P.HISTORY_MIN, 30):], 30, 4.0, tile=64)
        f60 = P.hem_fields(storm(60)[-P.frames_for(P.HISTORY_MIN, 60):], 60, 4.0, tile=64)
        cy, cx = 24, 20                                                # cell centre at the anchor (t = 5 h)
        for f in (f30, f60):
            self.assertAlmostEqual(float(f["rain_now"][cy, cx]), 40.0, delta=0.5)
            self.assertAlmostEqual(float(f["rain_max_1h"][cy, cx]), 40.0, delta=0.5)
            self.assertGreater(float(f["accel"][cy, cx]), 0)             # intensifying at the centre
            self.assertAlmostEqual(float(np.nanmedian(f["speed"])), 8.0, delta=1.5)
        self.assertAlmostEqual(float(f30["accel"][cy, cx]), float(f60["accel"][cy, cx]), delta=2.0)

    def test_rejects_unusable_cadence(self):
        with self.assertRaises(ValueError):
            P.hem_fields(storm(45)[-4:], 45, 4.0)                        # 45 min does not divide the 60-min motion lag
        with self.assertRaises(ValueError):
            P.hem_fields(storm(60)[-2:], 60, 4.0)                        # too few frames


if __name__ == "__main__":
    unittest.main()
