"""Integration tests on the REAL pilot files (tasks O1, O2). They run only where data/ holds the 2025 Mandi files
(git-ignored, so CI skips them). They prove the plumbing from real files to a schema-valid CAP draft and replay
reproducibility; the model used here is trained on SYNTHETIC data inside the test and says nothing about skill."""
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, time, timezone

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
HAVE = all(os.path.exists(p) for p in (os.path.join(DATA, "atlas", "catchments.csv"),
                                       os.path.join(DATA, "landing", "era5_pl", "era5_pl_2025_jun_jul.nc"),
                                       os.path.join(DATA, "landing", "era5_sl", "era5_sl_2025_jun_jul.nc"))) \
    and len(glob.glob(os.path.join(DATA, "landing", "mera", "mera_20250630*.nc"))) >= 20
sys.path.insert(0, os.path.join(REPO, "tools"))


@unittest.skipUnless(HAVE, "real pilot data not present (data/ is git-ignored)")
class TestRealFilesToAlert(unittest.TestCase):
    def test_real_files_to_schema_valid_cap(self):
        from meghprahari import alerts as A, ingest as I, model as MD, pipeline as P
        from evaluate import load_frames
        from inspect_inputs import resolve
        cfg = yaml.safe_load(open(os.path.join(REPO, "settings", "settings.yaml"), encoding="utf-8"))
        Mc, rain = cfg["met"], cfg[cfg["rain_source"]]
        start, end = datetime(2025, 6, 30, 12, tzinfo=timezone.utc), datetime(2025, 6, 30, 20, tzinfo=timezone.utc)
        frames, axes = load_frames(rain, DATA, start, end)
        ter = np.load(resolve(Mc["terrain_file"], DATA))
        idx = I.MetIndex(Mc, ter["h"], resolve(Mc["ps_file"], DATA), ter["lat"], ter["lon"])
        idx.add_file(os.path.join(DATA, "landing", "era5_pl", "era5_pl_2025_jun_jul.nc"))
        catch = pd.read_csv(os.path.join(DATA, "atlas", "catchments.csv")).set_index("link_id")
        feats = P.FEATURE_SETS[cfg["features"]]
        f, why = P.features_at(max(frames), frames, axes, idx.times(), idx.get, catch, rain, Mc["max_age_h"], feats)
        self.assertIsNone(why)
        self.assertEqual(len(f), len(catch))
        self.assertEqual(float(f["met_cape"].isna().mean()), 0.0)                     # CAPE defined everywhere
        # a calibrated model trained on SYNTHETIC data, only to exercise scoring -> alert drafting
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(6000, len(feats))), columns=feats)
        X["t"] = np.arange(6000.0)
        X["y"] = (X["hem_rain_now"] + rng.normal(size=6000) > 1.5).astype(int)
        tm = MD.train_target(X, feats, "y", "t", 3000, 4500, target="cb_0_2")
        self.assertTrue(tm.calibrated)
        probs = P.score_all(f, {"cb_0_2": tm})
        v = pd.read_csv(os.path.join(DATA, "inputs", "villages.csv")).iloc[0]
        vctx = A.VillageCtx(1, v["name"], float(v["lat"]), float(v["lon"]), 1, 10, 400, 180, 30, int(v["link_id"]))
        d = A.draft_alert(vctx, 2, float(probs.at[vctx.catchment_id, "cb_0_2"]), "cb", 0, 120, [], cfg["alerts"],
                          datetime(2025, 6, 30, 20, tzinfo=timezone.utc), True, cfg["sender"], cfg["sender_name"])
        self.assertIn("<status>Exercise</status>", d["cap_xml"])                        # shadow mode: never Actual
        self.assertIn('hi-IN', d["cap_xml"])
        if importlib.util.find_spec("xmlschema"):
            import xmlschema
            xmlschema.XMLSchema(os.path.join(REPO, "tests", "fixtures", "CAP-v1.2.xsd")).validate(d["cap_xml"])

    def test_replay_export_is_reproducible(self):
        outs = []
        with tempfile.TemporaryDirectory() as d:
            for k in range(2):
                o = os.path.join(d, str(k))
                r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "export_replay.py"), "--event",
                                    "mandi_2025_06_30", "--out-dir", o], cwd=REPO, capture_output=True, text=True,
                                   env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "backend")))
                self.assertEqual(r.returncode, 0, r.stderr[-1500:])
                b = json.load(open(os.path.join(o, "mandi_2025_06_30.json"), encoding="utf-8"))
                b["meta"].pop("generated")
                outs.append(b)
        self.assertEqual(outs[0], outs[1])                                                # identical except timestamp


if __name__ == "__main__":
    unittest.main()
