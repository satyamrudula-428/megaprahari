import os
import tempfile
import unittest
from unittest import mock

from scripts import live_imerg as L


def cmr_entry(name, has_data_link=True):
    links = [{"href": "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/2026/265/" + name}] \
        if has_data_link else [{"href": "https://disc.gsfc.nasa.gov/datacollection/GPM_3IMERGHHE_07.html"}]
    return {"links": links}


class TestParseGranules(unittest.TestCase):
    def test_extracts_filename_and_url(self):
        name = "3B-HHR-E.MS.MRG.3IMERG.20260922-S000000-E002959.0000.V07C.HDF5"
        out = L.parse_granules({"feed": {"entry": [cmr_entry(name)]}})
        self.assertEqual(out, [(name, "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/2026/265/" + name)])

    def test_skips_entries_without_a_data_link(self):
        out = L.parse_granules({"feed": {"entry": [cmr_entry("x.HDF5", has_data_link=False)]}})
        self.assertEqual(out, [])

    def test_empty_feed(self):
        self.assertEqual(L.parse_granules({"feed": {"entry": []}}), [])


class TestDownload(unittest.TestCase):
    def test_download_writes_via_temp_file_then_renames(self):
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "out.HDF5")

            class FakeResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

                def raise_for_status(self):
                    pass

                def iter_content(self, n):
                    yield b"hello "
                    yield b"world"

            with mock.patch.object(L.requests, "get", return_value=FakeResp()):
                L.download("https://example.test/x.HDF5", dst)
            self.assertFalse(os.path.exists(dst + ".part"))
            with open(dst, "rb") as f:
                self.assertEqual(f.read(), b"hello world")


if __name__ == "__main__":
    unittest.main()
