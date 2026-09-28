import json
import os
import tempfile
import unittest
from datetime import date

import numpy as np

from meghprahari import model as MD
from scripts import export_replay as X
from test_alerts_governance_model import synth


class TestExportReplay(unittest.TestCase):
    def test_encode_rain(self):
        self.assertEqual(X.encode_rain(np.array([[1.26, np.nan], [0.0, 40.0]])), [13, -1, 0, 400])

    def test_compact_card(self):
        card = {"summary": "S", "ingredients": [{"name": "lift", "status": "strong", "text": "T", "evidence": {"a": 1}}]}
        self.assertEqual(X.compact_card(card), {"s": "S", "i": [["lift", "strong", "T"]]})

    def test_ledger_rows_filters_dates_and_coords(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "events.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("date,place,lat,lon,hazard,what_happened,deaths_missing,source_name\n"
                        "2025-06-30,A,31.6,77.0,cb,x,1,S\n2025-06-30,B,,,ff,y,,S\n2024-07-31,C,32.0,76.9,cb,z,,S\n")
            rows = X.ledger_rows(p, date(2025, 6, 30), date(2025, 7, 1))
            self.assertEqual([r["place"] for r in rows], ["A"])
            self.assertEqual(X.ledger_rows(os.path.join(d, "missing.csv"), date(2025, 1, 1), date(2025, 12, 31)), [])

    def test_load_models_checks_hash_and_calibration(self):
        df = synth(6000)
        tm = MD.train_target(df, ["x1", "x2", "x3"], "y", "t", 3000, 4500, target="ts_0_2")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ts_0_2_v2_x.joblib")
            sha = MD.save_model(tm, path)
            json.dump({"target": "ts_0_2", "sha256": sha}, open(path + ".metrics.json", "w"))
            models, notes = X.load_models(d)
            self.assertEqual(list(models), ["ts_0_2"] if tm.calibrated else [])
            json.dump({"target": "ts_0_2", "sha256": "0" * 64}, open(path + ".metrics.json", "w"))
            with self.assertRaises(MD.ModelIntegrityError):              # tampered file / wrong hash is refused
                X.load_models(d)


if __name__ == "__main__":
    unittest.main()
