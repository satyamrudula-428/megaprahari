import os
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np

from scripts import fetch_arco_era5 as F
from test_readers import write_era5


class TestFetchHelpers(unittest.TestCase):
    def test_orography(self):
        z = np.full((3, 2, 2), 9.80665 * 1500.0)
        self.assertTrue(np.allclose(F.orography_m(z), 1500.0))
        self.assertTrue(np.allclose(F.orography_m(z[0]), 1500.0))

    def test_like_grid(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "pl.nc")
            lat, lon = np.array([32.5, 32.25, 32.0]), np.array([76.5, 76.75, 77.0])
            write_era5(p, [0, 3], lat, lon, np.array([1000.0, 850.0, 700.0, 500.0, 300.0]))
            times, la, lo = F.like_grid(p)
            self.assertEqual(times[1], np.datetime64(datetime(2025, 6, 30, 3), "ns"))
            self.assertTrue(np.array_equal(la, lat) and np.array_equal(lo, lon))

    def test_only_cin_may_be_undefined(self):
        self.assertEqual(set(F.UNDEFINED_OK), {"cin"})
        self.assertEqual({s for s, _ in F.FIELDS.values()}, {"sp", "z", "cape", "cin", "tcwv", "t2m", "d2m"})


if __name__ == "__main__":
    unittest.main()
