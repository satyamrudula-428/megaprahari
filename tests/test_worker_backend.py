import importlib.util
import unittest

HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None


@unittest.skipUnless(HAS_PSYCOPG, "psycopg not installed")
class TestBackendSwitch(unittest.TestCase):
    def test_backend(self):
        from meghprahari import worker as W
        self.assertEqual(W.backend({}), "gbm")                               # gradient boosting stays the default
        self.assertEqual(W.backend({"model": {"backend": "transformer"}}), "transformer")
        with self.assertRaises(ValueError):
            W.backend({"model": {"backend": "magic"}})

    def test_met_section(self):
        from meghprahari import worker as W
        cfg = {"met": {"max_age_h": 6}, "met_nrt": {"max_age_h": 1}}
        self.assertEqual(W.met_section(cfg), ("met", {"max_age_h": 6}))                           # default: ERA5
        self.assertEqual(W.met_section(dict(cfg, met_source="met_nrt")), ("met_nrt", {"max_age_h": 1}))


if __name__ == "__main__":
    unittest.main()
