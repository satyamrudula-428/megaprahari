import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from meghprahari import baselines as B, verify as V

T0 = datetime(2025, 6, 30, 19, 0, tzinfo=timezone.utc)


def blob(cy, cx, n=40, amp=20.0, r=3.0):
    yy, xx = np.mgrid[0:n, 0:n]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))


class TestFSS(unittest.TestCase):
    def test_perfect_and_no_overlap(self):
        o = blob(20, 20)
        self.assertAlmostEqual(V.fss(o, o, 5.0, 1), 1.0)
        far = blob(20, 35)
        self.assertLess(V.fss(far, o, 5.0, 1), 0.05)                     # displaced: no pixel-scale skill

    def test_skill_grows_with_neighbourhood(self):
        o, f = blob(20, 20), blob(20, 24)                                   # 4-pixel displacement
        scores = [V.fss(f, o, 5.0, w) for w in (1, 5, 9, 15)]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(scores, scores[1:])))
        self.assertGreater(scores[-1], 0.8)

    def test_missing_and_undefined(self):
        o = blob(20, 20)
        f = o.copy()
        f[:, :10] = np.nan                                                   # missing forecast pixels are ignored
        self.assertAlmostEqual(V.fss(f, o, 5.0, 1), 1.0)
        self.assertTrue(np.isnan(V.fss(np.zeros((5, 5)), np.zeros((5, 5)), 1.0, 3)))
        with self.assertRaises(ValueError):
            V.fss(o, o, 5.0, 4)
        stack = np.stack([o, o])
        self.assertAlmostEqual(V.fss(stack, stack, 5.0, 3), 1.0)
        self.assertAlmostEqual(V.fss_useful(0.1), 0.55)


class TestLeadTime(unittest.TestCase):
    def test_first_warning_counts(self):
        onset = T0 + timedelta(hours=3)
        times = [T0 + timedelta(minutes=30 * k) for k in range(8)]
        warned = [False, True, False, True, True, True, True, True]
        self.assertEqual(V.warning_lead_min(times, warned, onset), 150.0)
        self.assertIsNone(V.warning_lead_min(times, [False] * 8, onset))
        late = [t + timedelta(hours=4) for t in times]                    # all issued after onset
        self.assertIsNone(V.warning_lead_min(late, [True] * 8, onset))
        self.assertEqual(V.warning_lead_min(times, warned, onset, max_lead_min=60), 60.0)   # earlier ones too early

    def test_storm_onset_needs_quiet_period(self):
        ts = [T0 + timedelta(minutes=30 * k) for k in range(20)]
        peak = [12.0] * 3 + [2.0] * 8 + [15.0] * 9                        # old storm, 4 h dry, new storm at k=11
        self.assertEqual(V.storm_onset(ts, peak, 10.0, after=T0), ts[11])  # k=0..2 lack a preceding quiet period
        self.assertIsNone(V.storm_onset(ts, peak, 10.0, after=T0, quiet_h=5.0))
        self.assertIsNone(V.storm_onset(ts, [2.0] * 20, 10.0, after=T0))
        gap = ts[:1] + ts[9:]                                             # data gap: quiet period not covered
        self.assertIsNone(V.storm_onset(gap, [2.0] + peak[9:], 10.0, after=T0 + timedelta(hours=5)))

    def test_summary(self):
        s = V.lead_time_summary([30.0, None, 90.0])
        self.assertEqual((s["events"], s["hits"]), (3, 2))
        self.assertAlmostEqual(s["hit_rate"], 2 / 3)
        self.assertEqual(s["median_lead_min"], 60.0)
        self.assertTrue(np.isnan(V.lead_time_summary([None])["median_lead_min"]))


class TestBaselines(unittest.TestCase):
    def test_persistence(self):
        f = blob(20, 20)
        p = B.persistence(f, 4)
        self.assertEqual(p.shape, (4, 40, 40))
        self.assertTrue(np.array_equal(p[3], f))

    def test_motion_and_extrapolation(self):
        a, b = blob(20, 10), blob(20, 14)                                   # 4 px east in 30 min
        vy, vx = B.mean_motion(a, b, 30, pixel_km=4.0)
        self.assertAlmostEqual(vx, 32.0, delta=3.0)                         # 16 km / 0.5 h
        self.assertAlmostEqual(vy, 0.0, delta=3.0)
        ex = B.extrapolate(b, vy, vx, 30, 2, pixel_km=4.0)
        self.assertEqual(int(np.nanargmax(ex[1])) % 40, 22)                  # 2 more steps x 4 px east
        self.assertTrue(np.isnan(ex[1][:, :5]).all())                       # advected in from outside: unknown
        self.assertEqual(B.mean_motion(np.zeros((40, 40)), np.zeros((40, 40)), 30, 4.0), (0.0, 0.0))

    def test_window_max(self):
        st = np.arange(12, dtype=float)[:, None, None] * np.ones((12, 2, 2))
        self.assertEqual(B.window_max(st, 30, (0, 120))[0, 0], 3.0)        # leads 30..120 -> frames 0..3
        self.assertEqual(B.window_max(st, 30, (120, 240))[0, 0], 7.0)
        st[4:8] = np.nan
        self.assertTrue(np.isnan(B.window_max(st, 30, (120, 240))[0, 0]))
        with self.assertRaises(ValueError):
            B.window_max(st, 30, (400, 500))


if __name__ == "__main__":
    unittest.main()
