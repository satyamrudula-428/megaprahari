import os
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from meghprahari import alerts as A, ingest as I, model as MD, pipeline as P

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def hem_stack():
    yy, xx = np.mgrid[0:64, 0:64]
    frames = []
    for k in range(8):
        cy, cx = 30 + 0.5 * k, 20 + 1.0 * k
        frames.append((10 + 2 * k) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0))
    return np.array(frames, np.float32)


def met_state(scale=1.0):
    lat = 32.0 - 0.11 * np.arange(25)
    lon = 76.0 + 0.11 * np.arange(25)
    p = np.array([1000.0, 925.0, 850.0, 700.0, 500.0])
    prof = lambda a: np.array(a)[:, None, None] * np.ones((5, 25, 25))
    lonm = np.broadcast_to(lon[None, :], (25, 25))
    return P.MetState(NOW, lat, lon, p, prof([0.014, 0.012, 0.010, 0.005, 0.001]) * scale, prof([300, 296, 292, 282, 258]),
                      prof([8, 10, 12, 16, 20]) * (1 + 0 * lonm), prof([1, 1, 2, 2, 3]) * (1 + 0 * lonm),
                      np.full((25, 25), 1000.0), 100.0 + 300.0 * (lonm - 76.0)).validate()


class TestPipeline(unittest.TestCase):
    def test_nearest_idx(self):
        self.assertEqual(list(P.nearest_idx([8, 9, 10], [8.2, 9.6, 10.4])), [0, 2, 2])
        self.assertEqual(list(P.nearest_idx([10, 9, 8], [9.1, 12, 7.9], 0.6)), [1, -1, 2])
        self.assertEqual(list(P.nearest_idx([10, 9, 8], [9.1, 7.9])), [1, 2])

    def test_metstate_validation(self):
        m = met_state()
        m.q = m.q * 100
        with self.assertRaises(ValueError):
            m.validate()
        m2 = met_state()
        m2.p_hpa = m2.p_hpa[::-1]
        with self.assertRaises(ValueError):
            m2.validate()

    def test_assemble_end_to_end(self):
        hem_lat, hem_lon = 32.0 - 0.036 * np.arange(64), 76.0 + 0.036 * np.arange(64)
        hem = P.hem_fields(hem_stack(), tile=64)
        now_f, prev_f = P.met_fields(met_state()), P.met_fields(met_state(0.9))
        m = met_state()
        catch = pd.DataFrame({"lat": [hem_lat[33], hem_lat[5], 40.0], "lon": [hem_lon[27], hem_lon[5], 100.0],
                              "slope_mean": [0.2, 0.3, 0.1], "hand_mean": [3.0, 8.0, 1.0], "tc_min": [25.0, 40.0, 10.0],
                              "area_up_km2": [12.0, 30.0, 5.0]}, index=[10, 11, 12])
        f = P.assemble(catch, hem, now_f, prev_f, hem_lat, hem_lon, m.lat, m.lon)
        self.assertEqual(list(f.columns), P.FEATURES_V1)
        self.assertGreater(f.at[10, "hem_rain_now"], 10)
        self.assertLess(f.at[11, "hem_rain_now"], 1e-6)
        self.assertGreater(f.at[10, "hem_rain_accel"], 0)
        self.assertTrue(np.isfinite(f.at[10, "hem_stall_h"]))
        self.assertTrue(np.isnan(f.at[11, "hem_stall_h"]))
        self.assertAlmostEqual(f.at[10, "hem_speed_kmh"], np.hypot(2, 1) * 4.0, delta=2.0)
        self.assertGreater(f.at[10, "met_d_iwv_3h"], 0)
        self.assertGreater(f.at[10, "met_iwv"], 30)
        self.assertGreater(f.at[10, "met_kindex"], 0)
        self.assertTrue(f.loc[12, [c for c in P.FEATURES_V1 if c.startswith(("hem_", "met_"))]].isna().all())
        self.assertEqual(f.at[12, "ter_tc_min"], 10.0)
        self.assertTrue(np.isfinite(f.at[10, "met_omfi_low"]))
        # unit check of IWV: five-level profile integrates to a plausible column value
        self.assertLess(f.at[10, "met_iwv"], 60)

    def test_ctt_fields(self):
        yy, xx = np.mgrid[0:16, 0:16]
        cold = 260.0 - 40.0 * np.exp(-((yy - 8) ** 2 + (xx - 8) ** 2) / 6.0)
        stack = np.stack([cold + 6.0 * k for k in range(4)]).astype(np.float32)   # warming with time (k=0 coldest)
        ctt = P.ctt_fields(stack[::-1], dt_h=3.0)                                 # reversed: coldest last -> cooling trend
        self.assertAlmostEqual(ctt["ctt_min"][8, 8], cold[8, 8], places=3)
        self.assertLess(ctt["ctt_drop_rate"][8, 8], 0)                            # cooling into the anchor
        self.assertGreater(ctt["cold_area_km2"][8, 8], 0)
        self.assertEqual(ctt["cold_area_km2"][0, 0], 0)

    def test_labels(self):
        fut = np.zeros((12, 1, 4), np.float32)
        fut[2, 0, 0] = 60.0
        fut[5, 0, 1] = 12.0
        fut[:, 0, 3] = np.nan
        L = P.make_labels(fut, np.zeros(4, int), np.arange(4), dict(ts=10.0, cb=50.0, ff_mm=15.0), np.full(4, 2))
        self.assertEqual(list(L["y_ts_0_2"].iloc[:3]), [1, 0, 0])
        self.assertEqual(list(L["y_cb_0_2"].iloc[:3]), [1, 0, 0])
        self.assertEqual(list(L["y_ff_0_2"].iloc[:3]), [1, 0, 0])
        self.assertEqual(list(L["y_ts_2_4"].iloc[:3]), [0, 1, 0])
        self.assertEqual(list(L["y_cb_2_4"].iloc[:3]), [0, 0, 0])
        self.assertEqual(list(L["y_ff_2_4"].iloc[:3]), [0, 0, 0])
        self.assertTrue(L.iloc[3].isna().all())
        self.assertEqual(list(P.window_frames_from_tc([15, 31, np.nan, 1000])), [1, 2, 4, 12])

    def test_score_and_alert_drafts(self):
        rng = np.random.default_rng(0)
        n = 30000
        X = pd.DataFrame(rng.normal(size=(n, len(P.FEATURES_V1))), columns=P.FEATURES_V1)
        X["hem_rain_now"] = rng.exponential(2.0, n)
        X["met_iwv"] = rng.normal(50, 8, n)
        X.loc[rng.random(n) < 0.05, "met_mfc_low"] = np.nan
        logit = -4 + 0.25 * X.hem_rain_now + 0.05 * (X.met_iwv - 50)
        X["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        X["t"] = np.arange(n)
        tm = MD.train_target(X, P.FEATURES_V1, "y", "t", 18000, 24000, target="ff_0_2")
        models = {"ff_0_2": tm}
        feats = pd.DataFrame({c: [np.nan, np.nan] for c in P.FEATURES_V1}, index=[10, 11])
        feats.loc[10, ["hem_rain_now", "met_iwv"]] = [20.0, 55.0]
        feats.loc[11, ["hem_rain_now", "met_iwv"]] = [0.0, 45.0]
        probs = P.score_all(feats, models)
        self.assertTrue(((probs["ff_0_2"] >= 0) & (probs["ff_0_2"] <= 1)).all())
        self.assertGreater(probs.at[10, "ff_0_2"], 0.6)
        self.assertLess(probs.at[11, "ff_0_2"], 0.1)
        v1 = A.VillageCtx(1, "Alpha", 30.0, 77.0, 2, 10, 400, 90, 20, catchment_id=10)
        v2 = A.VillageCtx(2, "Beta", 30.1, 77.1, 2, 10, catchment_id=11)
        cfg = dict(walk_speed_kmh=3.0, detour_factor=1.5, dissemination_delay_min=15, lag_fraction=0.0, horizon_min=180)
        states = {}
        first = P.evaluate_villages(probs, feats, [v1, v2], states, models, cfg, NOW, True, "s-1", "S")
        self.assertEqual(first, [])                                     # ACT_NOW needs two consecutive cycles
        second = P.evaluate_villages(probs, feats, [v1, v2], states, models, cfg, NOW, True, "s-1", "S")
        self.assertEqual(len(second), 1)
        d = second[0]
        self.assertEqual((d["village_id"], d["level"], d["hazard"], d["required_approvals"]), (1, 3, "ff", 2))
        self.assertTrue(d["shadow"])
        self.assertIn("Exercise", d["cap_xml"])
        ET.fromstring(d["cap_xml"].split("\n", 1)[1])
        self.assertTrue(d["drivers"])
        self.assertEqual([i["name"] for i in d["explanation"]["ingredients"]],
                         ["moisture", "instability", "lift", "storm_signal", "terrain_response"])
        third = P.evaluate_villages(probs, feats, [v1, v2], states, models, cfg, NOW, True, "s-1", "S")
        self.assertEqual(third, [])                                     # no repeat draft at the same level
        tm.calibrated = False
        with self.assertRaises(MD.UncalibratedModelError):
            P.score_all(feats, models)


class TestIngest(unittest.TestCase):
    def test_scan_stable_new_files(self):
        with tempfile.TemporaryDirectory() as d:
            old, new = os.path.join(d, "a.h5"), os.path.join(d, "b.h5")
            for p, txt in ((old, b"aaa"), (new, b"bbb")):
                with open(p, "wb") as f:
                    f.write(txt)
            os.utime(old, (time.time() - 100, time.time() - 100))
            cache = {}
            got = I.scan_new_files(d, "*.h5", set(), min_age_s=10, cache=cache)
            self.assertEqual([os.path.basename(p) for p, _ in got], ["a.h5"])
            sha = got[0][1]
            self.assertEqual(I.scan_new_files(d, "*.h5", {sha}, min_age_s=10, cache=cache), [])
            self.assertEqual(len(I.scan_new_files(d, "*.h5", set(), min_age_s=0, cache=cache)), 2)

    def test_time_parse(self):
        t = I.parse_time_from_name("X_20AUG2024_1200_Y.h5", r"(\d{2}[A-Z]{3}\d{4}_\d{4})", "%d%b%Y_%H%M")
        self.assertEqual(t, datetime(2024, 8, 20, 12, 0, tzinfo=UTC))
        with self.assertRaises(I.IngestError):
            I.parse_time_from_name("nothing.h5", r"(\d{8})", "%Y%m%d")

    def test_qc(self):
        raw = np.array([[-999.0, 10.0, 20.0, 9999.0]])
        a, frac = I.qc_rain_frame(raw, [-999.0], scale=0.1, max_mm_h=500, max_missing_frac=0.6)
        self.assertTrue(np.isnan(a[0, 0]) and np.isnan(a[0, 3]))
        self.assertAlmostEqual(float(a[0, 1]), 1.0, places=5)
        self.assertAlmostEqual(frac, 0.5)
        with self.assertRaises(I.IngestError):
            I.qc_rain_frame(raw, [-999.0], scale=0.1, max_missing_frac=0.4)

    def test_cadence(self):
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        ts = [t0, t0 + timedelta(minutes=30), t0 + timedelta(minutes=90)]
        self.assertEqual(I.check_cadence(ts)[0][2], 60.0)
        self.assertEqual(I.check_cadence(ts[:2]), [])


if __name__ == "__main__":
    unittest.main()


class TestWindow(unittest.TestCase):
    def test_pick_window(self):
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        frames = {t0 + timedelta(minutes=30 * k): np.full((2, 2), float(k)) for k in range(8)}
        w = P.pick_window(frames, t0 + timedelta(minutes=210))
        self.assertEqual(w.shape, (6, 2, 2))
        self.assertEqual(list(w[:, 0, 0]), [2, 3, 4, 5, 6, 7])
        del frames[t0 + timedelta(minutes=120)]
        self.assertIsNone(P.pick_window(frames, t0 + timedelta(minutes=210)))
        self.assertIsNone(P.pick_window({}, t0))
