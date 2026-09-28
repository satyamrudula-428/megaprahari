import unittest
from datetime import datetime, timezone

import pandas as pd

from scripts import train as T


class TestTrainScript(unittest.TestCase):
    def test_to_epoch(self):
        self.assertEqual(T.to_epoch("1600000000"), 1600000000.0)
        self.assertEqual(T.to_epoch("2023-09-30"),
                         datetime(2023, 9, 30, 23, 59, 59, 999999, tzinfo=timezone.utc).timestamp())

    def test_split_counts(self):
        df = pd.DataFrame(dict(t=[1, 2, 3, 4, 5, 6], y_ts_0_2=[0, 1, 0, 1, None, 1]))
        self.assertEqual(T.split_counts(df, "y_ts_0_2", 2, 4), {"train": (2, 1), "cal": (2, 1), "test": (1, 1)})

    def test_targets(self):
        self.assertEqual(len(T.ALL_TARGETS), 9)
        self.assertIn("cb_2_4", T.ALL_TARGETS)


if __name__ == "__main__":
    unittest.main()
