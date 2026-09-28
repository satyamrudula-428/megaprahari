import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from scripts import model_card as M


class TestModelCard(unittest.TestCase):
    def test_render_from_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump(dict(target="cb_0_2", calibrated=True, n_test=100, positives_test=7, brier=0.0512, bss=0.21,
                           auc=0.83, ece=0.02, sha256="a" * 64, features=["hem_rain_now"], splits={"train": [10, 1]}),
                      open(os.path.join(d, "cb_0_2_v2_x.joblib.metrics.json"), "w"))
            json.dump(dict(target="mtl", calibrated=True, seq_len=6, sha256="b" * 64, train_range=["2021", "2022"],
                           per_target={"ts_0_2": dict(n_test=50, positives_test=5, brier=0.1, bss=0.1, auc=0.7, ece=0.03,
                                                      temperature=1.4), "cb_4_6": dict(n_test=50, positives_test=0,
                                                                                       note="test split lacks both classes")}),
                      open(os.path.join(d, "mtl_x.pt.metrics.json"), "w"))
            text = M.render(M.load(d), datetime(2026, 9, 23, tzinfo=timezone.utc))
        self.assertIn("| cb_0_2 | True | 100 | 7 | 0.0512 | 0.210 | 0.830 | 0.020 | `aaaaaaaaaaaa` |", text)
        self.assertIn("## Multi-task Transformer", text)
        self.assertIn("test split lacks both classes", text)
        self.assertIn("## Limitations", text)

    def test_no_models(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("No trained models found yet.", M.render(M.load(d), datetime(2026, 9, 23, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()
