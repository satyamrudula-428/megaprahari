import unittest
from datetime import datetime, timedelta, timezone

from meghprahari import monitor as M

NOW = datetime(2025, 6, 30, 20, tzinfo=timezone.utc)
CFG = {"rain_source": "mera", "mera": {"cadence_min": 60}, "met": {"max_age_h": 6}}


class TestMonitor(unittest.TestCase):
    def test_limits(self):
        self.assertEqual(M.expected_max_age(CFG), {"mera": 120.0, "met": 360.0})

    def test_status(self):
        lim = M.expected_max_age(CFG)
        ok = M.data_status({"mera": NOW - timedelta(minutes=70), "met": NOW - timedelta(hours=4)}, lim, NOW)
        self.assertFalse(ok["degraded"])
        self.assertEqual(ok["message"], "all inputs current")
        bad = M.data_status({"mera": NOW - timedelta(hours=3), "met": None}, lim, NOW)
        self.assertTrue(bad["degraded"])
        self.assertEqual([r["stale"] for r in bad["sources"]], [True, True])
        self.assertIn("mera", bad["message"])
        self.assertIsNone(bad["sources"][1]["age_min"])

    def test_as_utc(self):
        self.assertIsNone(M.as_utc(datetime(2025, 1, 1)))
        self.assertIsNone(M.as_utc(None))
        self.assertEqual(M.as_utc(NOW), NOW)


if __name__ == "__main__":
    unittest.main()
