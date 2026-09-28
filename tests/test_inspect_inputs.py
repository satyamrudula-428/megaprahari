import os
import unittest

import numpy as np

from scripts import inspect_inputs as S


class TestInspectHelpers(unittest.TestCase):
    def test_resolve(self):
        self.assertEqual(S.resolve("/data/landing/mera", "data"), os.path.join("data", "landing/mera"))
        self.assertEqual(S.resolve("local/file.nc", "data"), "local/file.nc")
        self.assertIsNone(S.resolve(None, "data"))

    def test_unset(self):
        c = {"var": "Rainfall", "lat_var": "CHANGE_ME", "vars": {"q": "q", "ps": "CHANGE_ME"}}
        self.assertEqual(S.unset(c, ["var", "lat_var", "lon_var", "vars.q", "vars.ps"]), ["lat_var", "lon_var", "vars.ps"])

    def test_value_summary(self):
        s = S.value_summary([1.0, np.nan, 3.0, 5.0])
        self.assertEqual((s["min"], s["mean"], s["max"]), (1.0, 3.0, 5.0))
        self.assertAlmostEqual(s["nan_frac"], 0.25)
        self.assertIsNone(S.value_summary([np.nan])["min"])
        self.assertEqual(S.fmt(S.value_summary([np.nan])), "all missing")


if __name__ == "__main__":
    unittest.main()
