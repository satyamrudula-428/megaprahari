import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from scripts import evaluate_models as E

T0 = datetime(2025, 6, 30, tzinfo=timezone.utc)


class TestEvaluateModels(unittest.TestCase):
    def test_persistence_uses_label_from_two_hours_earlier(self):
        rows = []
        for h in range(4):
            for link in (1, 2):
                rows.append(dict(anchor=pd.Timestamp(T0 + timedelta(hours=h)), link_id=link,
                                 y_ts_0_2=1.0 if (h == 0 and link == 1) else 0.0, y_cb_0_2=0.0, y_ff_0_2=np.nan))
        t = pd.DataFrame(rows)
        p = E.persistence_probs(t)
        at2 = (t["anchor"] == pd.Timestamp(T0 + timedelta(hours=2))).to_numpy()
        self.assertEqual(p.loc[at2 & (t["link_id"] == 1).to_numpy(), "ts_4_6"].tolist(), [1.0])   # same for every bin
        self.assertEqual(p.loc[at2 & (t["link_id"] == 2).to_numpy(), "ts_0_2"].tolist(), [0.0])
        self.assertTrue(np.isnan(p.loc[(t["anchor"] == pd.Timestamp(T0)).to_numpy(), "ts_0_2"]).all())   # no past
        self.assertTrue(np.isnan(p["ff_0_2"]).all())

    def test_scores(self):
        s = E.scores([0, 0, 1, 1, np.nan], [0.05, 0.2, 0.8, 0.02, 0.9], 0.1)
        self.assertEqual((s["n"], s["pos"]), (4, 2))
        self.assertAlmostEqual(s["POD"], 0.5)
        self.assertAlmostEqual(s["FAR"], 0.5)
        self.assertIn("auc", s)
        self.assertNotIn("auc", E.scores([0, 0], [0.1, 0.2], 0.1))

    def test_first_warning_lead(self):
        times = [T0 + timedelta(hours=h) for h in range(6)]
        self.assertEqual(E.first_warning_lead(times, [0, 0.05, 0.3, 0.4, 0, 0], 0.1, T0 + timedelta(hours=5)), 180.0)
        self.assertIsNone(E.first_warning_lead(times, [0] * 6, 0.1, T0 + timedelta(hours=5)))


if __name__ == "__main__":
    unittest.main()
