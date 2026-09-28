"""Time (C10) and space (C11) alignment on synthetic grids with the shapes/orientations of the real inputs:
IMERG 0.1 deg ascending lat (30-min), ERA5 0.25 deg descending lat (3-hourly), MERA 0.0375 deg descending lat."""
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from meghprahari import pipeline as P

UTC = timezone.utc
T0 = datetime(2025, 6, 30, 0, tzinfo=UTC)


class TestTimeAlignment(unittest.TestCase):
    def setUp(self):
        self.met = [T0 + timedelta(hours=h) for h in range(0, 24, 3)]           # 3-hourly analyses

    def test_holds_latest_past_analysis(self):
        now, prev = P.select_met_times(self.met, T0 + timedelta(hours=19, minutes=30), max_age_h=6)
        self.assertEqual(now, T0 + timedelta(hours=18))                          # never the 21 UTC (future) one
        self.assertEqual(prev, T0 + timedelta(hours=15))

    def test_exact_match_and_first_time(self):
        now, prev = P.select_met_times(self.met, T0 + timedelta(hours=3), max_age_h=6)
        self.assertEqual((now, prev), (T0 + timedelta(hours=3), T0))
        now, prev = P.select_met_times(self.met, T0 + timedelta(minutes=30), max_age_h=6)
        self.assertEqual(now, T0)
        self.assertIsNone(prev)                                                  # nothing 3 h earlier

    def test_max_age(self):
        met = [T0]
        self.assertEqual(P.select_met_times(met, T0 + timedelta(hours=6), 6)[0], T0)
        self.assertEqual(P.select_met_times(met, T0 + timedelta(hours=6, minutes=30), 6), (None, None))
        self.assertEqual(P.select_met_times([], T0, 6), (None, None))

    def test_prev_tolerance(self):
        met = [T0, T0 + timedelta(hours=1, minutes=30), T0 + timedelta(hours=4)]
        now, prev = P.select_met_times(met, T0 + timedelta(hours=4), 6)
        self.assertEqual(prev, T0 + timedelta(hours=1, minutes=30))              # 2.5 h back beats 4 h back
        now, prev = P.select_met_times([T0, T0 + timedelta(hours=4)], T0 + timedelta(hours=4), 6)
        self.assertEqual(prev, T0)                                               # 4 h back is within 3 +- 1 h
        now, prev = P.select_met_times([T0, T0 + timedelta(hours=4)], T0 + timedelta(hours=4), 6, prev_tol_h=0.5)
        self.assertIsNone(prev)

    def test_hourly_rain_window(self):
        frames = {T0 + timedelta(hours=k): np.full((2, 2), float(k)) for k in range(8)}   # MERA-like hourly
        w = P.pick_window(frames, T0 + timedelta(hours=7), n=6, step_min=60)
        self.assertEqual(list(w[:, 0, 0]), [2, 3, 4, 5, 6, 7])


class TestSpatialAlignment(unittest.TestCase):
    def test_assemble_on_real_grid_orientations(self):
        rain_lat = np.round(np.arange(30.05, 33.0, 0.1), 2)                     # IMERG crop, ascending
        rain_lon = np.round(np.arange(76.05, 78.0, 0.1), 2)
        met_lat = np.array([33.0, 32.75, 32.5, 32.25, 32.0, 31.75, 31.5, 31.25, 31.0, 30.75, 30.5, 30.25, 30.0])
        met_lon = 76.0 + 0.25 * np.arange(9)                                     # ERA5: 13 x 9, lat descending
        ry, rx = np.meshgrid(rain_lat, rain_lon, indexing="ij")
        my, mx = np.meshgrid(met_lat, met_lon, indexing="ij")
        code = lambda y, x: np.round(y, 2) * 1000 + np.round(x, 2)              # value encodes its own location
        hem = {k: code(ry, rx) for k in ("rain_now", "rain_max_1h", "accel", "growth", "speed", "stall")}
        met = {k: code(my, mx) for k in ("iwv", "mfc_low", "omfi_low", "shear_low_500", "kindex", "totals")}
        catch = pd.DataFrame({"lat": [31.61, 31.72, 34.0], "lon": [77.11, 76.93, 77.0], "slope_mean": 0.5,
                              "hand_mean": 100.0, "tc_min": 30.0, "area_up_km2": 5.0}, index=[1, 2, 3])
        f = P.assemble(catch, hem, met, None, rain_lat, rain_lon, met_lat, met_lon)
        self.assertAlmostEqual(f.at[1, "hem_rain_now"], code(31.65, 77.15), places=3)   # nearest 0.1-deg centre
        self.assertAlmostEqual(f.at[2, "hem_rain_now"], code(31.75, 76.95), places=3)
        self.assertAlmostEqual(f.at[1, "met_iwv"], code(31.5, 77.0), places=3)          # nearest 0.25-deg point
        self.assertAlmostEqual(f.at[2, "met_iwv"], code(31.75, 77.0), places=3)
        self.assertTrue(f.loc[3, ["hem_rain_now", "met_iwv"]].isna().all())            # outside both grids
        self.assertTrue(f["met_d_iwv_3h"].isna().all())                                # no previous analysis


class TestFeaturesAt(unittest.TestCase):
    def test_same_features_as_manual_chain(self):
        from test_pipeline_ingest import hem_stack, met_state
        frames = {T0 + timedelta(minutes=30 * k): f for k, f in enumerate(hem_stack())}
        lat, lon = 32.0 - 0.036 * np.arange(64), 76.0 + 0.036 * np.arange(64)
        m = met_state()
        met = {T0: m, T0 + timedelta(hours=3): met_state(1.05)}
        catch = pd.DataFrame({"lat": [lat[33]], "lon": [lon[27]], "slope_mean": 0.2, "hand_mean": 3.0, "tc_min": 25.0,
                              "area_up_km2": 12.0}, index=[10])
        rc = dict(cadence_min=30, pixel_km=4.0, growth_thr_mm_h=10.0, cell_thr_mm_h=10.0)
        t = max(frames)                                                  # 03:30 -> met now 03:00, prev 00:00
        f, why = P.features_at(t, frames, (lat, lon), list(met), met.__getitem__, catch, rc, 6)
        self.assertIsNone(why)
        self.assertEqual(list(f.columns), P.FEATURES_V1)
        stack = P.pick_window(frames, t, 6, 30)
        manual = P.assemble(catch, P.hem_fields(stack, 30, 4.0, 10.0, 10.0), P.met_fields(met[T0 + timedelta(hours=3)]),
                            P.met_fields(met[T0]), lat, lon, m.lat, m.lon)
        pd.testing.assert_frame_equal(f, manual)
        self.assertGreater(f.at[10, "met_d_iwv_3h"], 0)
        f, why = P.features_at(T0 + timedelta(minutes=60), frames, (lat, lon), list(met), met.__getitem__, catch, rc, 6)
        self.assertIsNone(f)                                             # fewer than 6 frames before 01:00
        self.assertIn("incomplete", why)


if __name__ == "__main__":
    unittest.main()
