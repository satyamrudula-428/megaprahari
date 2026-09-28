import unittest

from scripts.load_villages import opt_int, opt_str


class TestLoadVillagesHelpers(unittest.TestCase):
    def test_opt_int(self):
        self.assertIsNone(opt_int(float("nan")))            # blank CSV cell read by pandas
        self.assertIsNone(opt_int(""))
        self.assertIsNone(opt_int(None))
        self.assertEqual(opt_int(1250), 1250)
        self.assertEqual(opt_int(1250.0), 1250)
        self.assertEqual(opt_int("1,250"), 1250)
        with self.assertRaises(ValueError):
            opt_int(-5)

    def test_opt_str(self):
        self.assertIsNone(opt_str(float("nan")))
        self.assertIsNone(opt_str("  "))
        self.assertEqual(opt_str(" Mandi "), "Mandi")


if __name__ == "__main__":
    unittest.main()
