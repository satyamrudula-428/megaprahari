import os
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone

from scripts import fetch_era5_cds as F


class TestCdsRequests(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(F.parse_range("2021-2024"), [2021, 2022, 2023, 2024])
        self.assertEqual(F.parse_range("6,8"), [6, 8])
        self.assertEqual(F.parse_range("7"), [7])

    def test_month_requests_fit_the_cost_limit(self):
        pls, sl = F.month_requests(2024, 7)                    # 31 days: the most expensive month
        self.assertEqual([r["variable"] for r in pls], F.PL_GROUPS)                  # two variables per request
        self.assertEqual(len(pls[0]["day"]), 31)
        self.assertEqual(pls[0]["area"], [33, 76, 30, 78])
        for r in pls + [sl]:
            self.assertLessEqual(F.cost(r), F.COST_LIMIT)
        self.assertNotIn("pressure_level", sl)
        self.assertEqual(len(F.month_requests(2024, 2)[0][0]["day"]), 29)   # leap year

    def test_cost_matches_observed_cds_value(self):
        req = dict(variable=["a"] * 6, pressure_level=["x"] * 37, day=["d"] * 8, time=["t"] * 24)
        self.assertEqual(F.cost(req), 255744)                  # the cost CDS reported for our first request
        self.assertGreater(F.cost(dict(req, variable=["a"] * 4, pressure_level=["x"] * 13, day=["d"] * 31,
                                       time=["t"] * 8)), F.COST_LIMIT)   # why variables are split into groups

    def test_merge_month(self):
        import netCDF4
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            parts = []
            for name, val in (("t", 280.0), ("q", 0.01)):
                p = os.path.join(d, f"{name}.nc")
                with netCDF4.Dataset(p, "w") as f:
                    for c, n in (("valid_time", 2), ("pressure_level", 3), ("latitude", 2), ("longitude", 2)):
                        f.createDimension(c, n)
                        f.createVariable(c, "f8", (c,), fill_value=np.nan)[:] = np.arange(n)   # CDS files carry _FillValue
                    f["valid_time"].units = "seconds since 1970-01-01"
                    v = f.createVariable(name, "f4", ("valid_time", "pressure_level", "latitude", "longitude"))
                    v[:] = val
                    v.units = "K" if name == "t" else "kg kg**-1"
                parts.append(p)
            out = os.path.join(d, "era5_pl_2021_06.nc")
            F.merge_month(parts, out)
            with netCDF4.Dataset(out) as f:
                self.assertEqual(sorted(k for k in f.variables if k in ("t", "q")), ["q", "t"])
                self.assertAlmostEqual(float(f["q"][1, 2, 1, 1]), 0.01, places=6)
                self.assertEqual(f["t"].units, "K")
                self.assertEqual(f["valid_time"].units, "seconds since 1970-01-01")
            with netCDF4.Dataset(parts[1], "a") as f:
                f["latitude"][:] = [5, 6]
            with self.assertRaises(RuntimeError):
                F.merge_month(parts, os.path.join(d, "bad.nc"))

    def test_match_finished_exact_only(self):
        pls, sl = F.month_requests(2021, 6)
        wanted = {"a.nc": pls[0], "b.nc": sl}
        same = dict(pls[0], day=list(reversed(pls[0]["day"])), labels={"x": 1})     # order and extras do not matter
        other = dict(pls[0], month=["07"])
        got = F.match_finished([(same, "http://h/1"), (other, "http://h/2")], wanted)
        self.assertEqual(got, [("a.nc", "http://h/1")])

    def test_queue_full_is_retryable(self):
        self.assertTrue(F.is_queue_full(RuntimeError("The job has been rejected; Number queued requests for this dataset exceeds the limit")))
        self.assertFalse(F.is_queue_full(RuntimeError("400 Bad Request: invalid variable")))

    def test_replay_months_are_refused(self):
        w = [(datetime(2025, 6, 29, tzinfo=timezone.utc), datetime(2025, 7, 2, 23, 59, tzinfo=timezone.utc))]
        self.assertTrue(F.overlaps(2025, 6, w) and F.overlaps(2025, 7, w))
        self.assertFalse(F.overlaps(2025, 8, w) or F.overlaps(2024, 7, w))

    def test_unzip_single_nc(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.part")
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("data_stream-oper_stepType-instant.nc", b"CDF-content")
            F.unzip_if_needed(p)
            self.assertEqual(open(p, "rb").read(), b"CDF-content")
            F.unzip_if_needed(p)                                   # plain file: left unchanged
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("a.nc", b"1")
                z.writestr("b.nc", b"2")
            with self.assertRaises(RuntimeError):
                F.unzip_if_needed(p)


if __name__ == "__main__":
    unittest.main()
