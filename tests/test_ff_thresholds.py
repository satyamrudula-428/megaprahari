import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from scripts import build_training_table as B, ff_thresholds as F

T0 = datetime(2021, 7, 1, tzinfo=timezone.utc)


class TestFFThresholds(unittest.TestCase):
    def test_rolling_totals_respect_gaps(self):
        times = [T0 + timedelta(hours=h) for h in (0, 1, 2, 5, 6)]           # gap between 02 and 05
        s = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        tot = F.rolling_totals(s, times, 2, 60)
        self.assertEqual(sorted(tot[:, 0].tolist()), [3.0, 5.0, 9.0])          # 1+2, 2+3, 4+5 (never 3+4 across the gap)
        self.assertEqual(F.rolling_totals(s, times, 4, 60).shape[0], 0)

    def test_thresholds_per_window(self):
        times = [T0 + timedelta(hours=h) for h in range(100)]
        s = np.column_stack([np.arange(100.0), np.arange(100.0)])
        thr = F.thresholds(s, times, np.array([1, 2]), 60, 1.0)
        self.assertEqual(thr[1], 99.0)
        self.assertEqual(thr[2], 98.0 + 99.0)

    def test_aligned_to_catchments(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ff.csv")
            pd.DataFrame(dict(link_id=[5, 3], window_frames=[1, 2], ff_mm=[14.7, 27.9])).to_csv(p, index=False)
            catch = pd.DataFrame(index=pd.Index([3, 5], name="link_id"))
            self.assertEqual(B.ff_thresholds_for(catch, p).tolist(), [27.9, 14.7])
            with self.assertRaises(ValueError):
                B.ff_thresholds_for(pd.DataFrame(index=pd.Index([3, 9], name="link_id")), p)


if __name__ == "__main__":
    unittest.main()
