import os
import tempfile
import unittest
from datetime import date, datetime, timezone

from meghprahari.settings import ConfigError, load_region, replay_exclusion_windows

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOOD = """
region:
  id: r
  name: R
  bbox: {lat_min: 31.5, lat_max: 32.0, lon_min: 76.8, lon_max: 77.3}
  met_margin_deg: 0.5
grid: {dem_arcsec: 1.0, max_dem_cells: 4000000}
replay_events:
  - {id: e1, name: E1, status: confirmed, hazards: [cb, ff], event_start: 2025-06-30, event_end: 2025-07-01,
     download_start: 2025-06-29, download_end: 2025-07-02}
"""


class TestRegion(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "region.yaml")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            return load_region(p)

    def test_good_region(self):
        cfg = self.load(GOOD)
        self.assertEqual(cfg["region"]["bbox"]["lat_min"], 31.5)
        e = cfg["replay_events"][0]
        self.assertEqual((e["event_start"], e["download_end"]), (date(2025, 6, 30), date(2025, 7, 2)))
        (start, end), = replay_exclusion_windows(cfg)
        self.assertEqual(start, datetime(2025, 6, 29, tzinfo=timezone.utc))
        self.assertEqual((end.date(), end.hour, end.minute), (date(2025, 7, 2), 23, 59))   # whole last day excluded

    def test_rejects_bad_values(self):
        cases = {
            "inverted bbox": GOOD.replace("lat_min: 31.5, lat_max: 32.0", "lat_min: 32.0, lat_max: 31.5"),
            "too many DEM cells": GOOD.replace("max_dem_cells: 4000000", "max_dem_cells: 1000"),
            "unknown hazard": GOOD.replace("hazards: [cb, ff]", "hazards: [cb, hail]"),
            "event outside download window": GOOD.replace("download_start: 2025-06-29", "download_start: 2025-07-01"),
            "bad status": GOOD.replace("status: confirmed", "status: maybe"),
            "bad date": GOOD.replace("event_end: 2025-07-01", "event_end: 1-Jul"),
            "CHANGE_ME": GOOD.replace("name: R", "name: CHANGE_ME"),
            "duplicate id": GOOD + GOOD.split("replay_events:")[1],
            "missing key": GOOD.replace("grid: {dem_arcsec: 1.0, max_dem_cells: 4000000}\n", ""),
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(ConfigError):
                self.load(text)

    def test_repo_region_file_is_valid(self):
        cfg = load_region(os.path.join(REPO, "settings", "region.yaml"))
        self.assertEqual(cfg["region"]["bbox"], dict(lat_min=31.5, lat_max=32.0, lon_min=76.8, lon_max=77.3))
        self.assertIn("mandi_2025_06_30", [e["id"] for e in cfg["replay_events"]])
        self.assertEqual(len(replay_exclusion_windows(cfg)), len(cfg["replay_events"]))


if __name__ == "__main__":
    unittest.main()
