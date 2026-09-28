import unittest
from datetime import datetime, timezone
from unittest import mock

from scripts import live_gfs as G


class TestFilterQuery(unittest.TestCase):
    def test_builds_expected_query(self):
        url = G.filter_query(datetime(2026, 9, 24).date(), "06", G.AREA)
        self.assertIn("dir=%2Fgfs.20260924%2F06%2Fatmos", url)
        self.assertIn("file=gfs.t06z.pgrb2.0p25.f000", url)
        self.assertIn("var_CAPE=on", url)
        self.assertIn("var_CIN=on", url)
        self.assertIn("lev_1000_mb=on", url)
        self.assertIn("lev_surface=on", url)
        self.assertIn("leftlon=76&rightlon=78&toplat=33&bottomlat=30", url)


class TestLatestRun(unittest.TestCase):
    def test_picks_newest_published_run(self):
        now = datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc)

        class FakeResp:
            def __init__(self, status):
                self.status_code = status

        def fake_head(url, timeout):
            # only the 06z run of 2026-09-24 is "published"
            return FakeResp(200 if "gfs.20260924/06/" in url else 404)

        session = mock.Mock()
        session.head.side_effect = fake_head
        d, hh = G.latest_run(session, now=now)
        self.assertEqual((d, hh), (now.date(), "06"))

    def test_raises_when_nothing_published(self):
        session = mock.Mock()
        session.head.return_value = mock.Mock(status_code=404)
        with self.assertRaises(RuntimeError):
            G.latest_run(session, now=datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc), max_age_days=1)


if __name__ == "__main__":
    unittest.main()
