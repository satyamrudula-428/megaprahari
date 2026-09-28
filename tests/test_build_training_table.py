import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from meghprahari import pipeline as P
from scripts import build_training_table as B
from test_pipeline_ingest import met_state

T0 = datetime(2021, 7, 1, tzinfo=timezone.utc)


class TestTrainingTable(unittest.TestCase):
    def test_thresholds(self):
        self.assertIsNone(B.label_thresholds(dict(ts_mm_h=10, cb_mm_h=50, ff_mm="CHANGE_ME"))["ff_mm"])
        self.assertEqual(B.label_thresholds(dict(ts_mm_h=10, cb_mm_h=50, ff_mm=40))["ff_mm"], 40.0)

    def test_accum_thresholds(self):
        thr = B.label_thresholds(dict(ts_mm_h=10, cb_mm_h=50, cb_rule="accum", cb_accum_mm=70, cb_accum_h=3))
        self.assertEqual((thr["cb_accum_mm"], thr["cb_accum_min"]), (70.0, 180))
        with self.assertRaises(ValueError):
            B.label_thresholds(dict(ts_mm_h=10, cb_mm_h=50, cb_rule="accum", cb_accum_mm="CHANGE_ME", cb_accum_h=3))
        with self.assertRaises(ValueError):
            B.label_thresholds(dict(ts_mm_h=10, cb_mm_h=50, cb_rule="bogus"))

    def test_accum_labels_sum_per_pixel_before_neighbourhood(self):
        # two DIFFERENT pixels (4 px apart) each get 30 mm/h in different hours: max-then-sum would reach 60 mm,
        # the correct sum-then-max gives only 30 mm per pixel
        fut = np.zeros((6, 11, 11))
        fut[0, 5, 3] = 30.0
        fut[1, 5, 7] = 30.0
        hist = np.zeros((2, 11, 11))
        thr = dict(ts=10.0, cb=50.0, ff_mm=np.inf, cb_accum_mm=50.0, cb_accum_min=180)
        wf = np.array([1])
        lab = P.make_labels(fut, np.array([5]), np.array([5]), thr, wf, 60, radius_px=3, history=hist)
        self.assertEqual(lab.at[0, "y_cb_0_2"], 0.0)
        fut[1, 5, 3] = 30.0                                                # now the SAME pixel gets 60 mm in 2 h
        lab = P.make_labels(fut, np.array([5]), np.array([5]), thr, wf, 60, radius_px=3, history=hist)
        self.assertEqual(lab.at[0, "y_cb_0_2"], 1.0)
        self.assertEqual(lab.at[0, "y_ts_0_2"], 1.0)                       # ts: peak 30 mm/h within 3 px

    def test_accum_uses_history_before_anchor(self):
        fut = np.zeros((6, 5, 5))
        fut[0, 2, 2] = 25.0                                                # +1 h
        hist = np.zeros((2, 5, 5))
        hist[:, 2, 2] = 25.0                                               # the 2 h up to the anchor
        thr = dict(ts=10.0, cb=50.0, ff_mm=np.inf, cb_accum_mm=70.0, cb_accum_min=180)
        lab = P.make_labels(fut, np.array([2]), np.array([2]), thr, np.array([1]), 60, history=hist)
        self.assertEqual(lab.at[0, "y_cb_0_2"], 1.0)                       # 75 mm in the 3 h ending at +1 h
        self.assertEqual(lab.at[0, "y_cb_2_4"], 0.0)
        with self.assertRaises(ValueError):
            P.make_labels(fut, np.array([2]), np.array([2]), thr, np.array([1]), 60, history=None)

    def test_seq_catchments_and_write(self):
        import os, tempfile
        links = list(range(100))
        with tempfile.TemporaryDirectory() as d:
            vp = os.path.join(d, "v.csv")
            pd.DataFrame(dict(name=["a", "b"], link_id=[7, 999])).to_csv(vp, index=False)
            s = B.seq_catchments(links, vp, 5, 0)
            self.assertIn(7, s)                                           # village catchment always included
            self.assertNotIn(999, s)                                      # not in the atlas: ignored
            self.assertEqual(len(s), 6)
            self.assertEqual(s, B.seq_catchments(links, vp, 5, 0))        # deterministic
            w = pd.DataFrame({"f1": [1.0, 2.0], "y_ts_0_2": [0.0, np.nan]}, index=[3, 4])
            p = os.path.join(d, "seq.npz")
            B.write_sequences(p, [(20.0, w * 2), (10.0, w)], ["f1"], ["y_ts_0_2"])
            with np.load(p) as z:                                         # close the file (Windows locks it)
                self.assertEqual(z["t"].tolist(), [10.0, 20.0])           # sorted by time
                self.assertEqual(z["X"].shape, (2, 2, 1))
                self.assertEqual(z["X"][1, 1, 0], 4.0)
                self.assertTrue(np.isnan(z["Y"][0, 1, 0]))
                self.assertEqual(z["targets"].tolist(), ["ts_0_2"])

    def test_exclusion(self):
        w = [(datetime(2025, 6, 29, tzinfo=timezone.utc), datetime(2025, 7, 2, 23, 59, tzinfo=timezone.utc))]
        h = timedelta(hours=6)
        self.assertTrue(B.touches_excluded(datetime(2025, 6, 28, 20, tzinfo=timezone.utc), w, h, h))   # label window
        self.assertTrue(B.touches_excluded(datetime(2025, 7, 3, 4, tzinfo=timezone.utc), w, h, h))     # history
        self.assertFalse(B.touches_excluded(datetime(2025, 7, 3, 12, tzinfo=timezone.utc), w, h, h))

    def test_neighbourhood_max(self):
        f = np.zeros((1, 11, 11))
        f[0, 5, 8] = 60.0
        self.assertEqual(P.neighbourhood_max(f, 0)[0, 5, 5], 0.0)
        self.assertEqual(P.neighbourhood_max(f, 3)[0, 5, 5], 60.0)        # 3 px away: inside radius 3
        self.assertEqual(P.neighbourhood_max(f, 2)[0, 5, 5], 0.0)

    def test_rows_at_labels_and_weights(self):
        lat, lon = 32.0 - 0.036 * np.arange(40), 76.0 + 0.036 * np.arange(40)
        frames = {T0 + timedelta(hours=k): np.zeros((40, 40)) for k in range(12)}
        frames[T0 + timedelta(hours=5)][20, 20] = 60.0                    # cloudburst-level rain 3 h after the anchor
        met = {T0: met_state()}
        catch = pd.DataFrame({"lat": [lat[20], lat[2]], "lon": [lon[20], lon[2]], "slope_mean": 0.5, "hand_mean": 5.0,
                              "tc_min": 30.0, "area_up_km2": 5.0}, index=[1, 2])
        rc = dict(cadence_min=60, pixel_km=4.0, growth_thr_mm_h=10.0, cell_thr_mm_h=10.0)
        thr = dict(ts=10.0, cb=50.0, ff_mm=None)
        df, why = B.rows_at(T0 + timedelta(hours=2), frames, (lat, lon), list(met), met.__getitem__, catch, rc, 6,
                            P.FEATURES_V1, thr, 0, np.random.default_rng(0), 1.0)
        self.assertIsNone(why)
        self.assertEqual(df.at[1, "y_cb_2_4"], 1.0)                        # peak at +3 h falls in the 2-4 h bin
        self.assertEqual(df.at[1, "y_cb_0_2"], 0.0)
        self.assertEqual(df.at[2, "y_ts_2_4"], 0.0)
        self.assertTrue(df[[c for c in df if c.startswith("y_ff")]].isna().all().all())   # no invented ff threshold
        self.assertTrue((df["w"] == 1.0).all())
        none, why = B.rows_at(T0 + timedelta(hours=8), frames, (lat, lon), list(met), met.__getitem__, catch, rc, 12,
                              P.FEATURES_V1, thr, 0, np.random.default_rng(0), 1.0)
        self.assertIsNone(none)
        self.assertIn("future", why)


if __name__ == "__main__":
    unittest.main()
