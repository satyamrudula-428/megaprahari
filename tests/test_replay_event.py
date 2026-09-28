import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from scripts import replay_event as R
from test_pipeline_ingest import hem_stack, met_state

T0 = datetime(2025, 6, 30, tzinfo=timezone.utc)


class TestReplay(unittest.TestCase):
    def test_replay_rows_and_skips(self):
        frames = {T0 + timedelta(minutes=30 * k): f for k, f in enumerate(hem_stack())}
        lat, lon = 32.0 - 0.036 * np.arange(64), 76.0 + 0.036 * np.arange(64)
        met = {T0: met_state()}
        catch = pd.DataFrame({"lat": [lat[33], lat[5]], "lon": [lon[27], lon[5]], "slope_mean": 0.2, "hand_mean": 3.0,
                              "tc_min": 25.0, "area_up_km2": 12.0}, index=[7, 8])
        rc = dict(cadence_min=30, pixel_km=4.0, growth_thr_mm_h=10.0, cell_thr_mm_h=10.0)
        df, skipped = R.replay(sorted(frames), frames, (lat, lon), list(met), met.__getitem__, catch, rc, 6)
        self.assertEqual(df["anchor"].nunique(), 3)                  # 8 frames -> anchors 6..8 have a full window
        self.assertEqual(len(df), 6)                                  # 3 anchors x 2 catchments
        self.assertEqual(sum(skipped.values()), 5)
        empty, sk = R.replay([T0], frames, (lat, lon), [], met.__getitem__, catch, rc, 6)
        self.assertTrue(empty.empty and sum(sk.values()) == 1)


if __name__ == "__main__":
    unittest.main()
