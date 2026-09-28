import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import train_transformer as TT
from meghprahari import mtl_backend as T

HAS_TORCH = importlib.util.find_spec("torch") is not None
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestHelpers(unittest.TestCase):
    def test_valid_anchor_indices(self):
        t = np.array([0, 3600, 7200, 10800, 21600, 25200, 28800]) * 1.0
        self.assertEqual(TT.valid_anchor_indices(t, 3).tolist(), [2, 3, 6])     # 4 and 5 sit right after a gap

    def test_reorder_targets(self):
        Y = np.zeros((1, 1, 2), np.float32)
        Y[..., 1] = 1.0
        out = TT.reorder_targets(Y, ["cb_0_2", "ts_0_2"])
        self.assertEqual(out[0, 0, T.TARGETS.index("ts_0_2")], 1.0)
        self.assertTrue(np.isnan(out[0, 0, T.TARGETS.index("ff_4_6")]))

    def test_gather(self):
        Z = np.arange(5 * 2 * 1, dtype=np.float32).reshape(5, 2, 1)
        Y = np.zeros((5, 2, 9), np.float32)
        x, y = TT.gather(Z, Y, np.array([[4, 1]]), 3)
        self.assertEqual(x[0, :, 0].tolist(), [5.0, 7.0, 9.0])                  # anchors 2, 3, 4 of catchment 1
        self.assertEqual(y.shape, (1, 3, 3))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestEndToEnd(unittest.TestCase):
    def test_trains_calibrates_and_saves(self):
        rng = np.random.default_rng(0)
        A, C = 240, 6
        t = 1.6e9 + 3600.0 * np.arange(A)
        feats = ["hem_rain_now", "hem_rain_accel", "met_cape", "met_cin", "ter_tc_min"]
        X = rng.normal(size=(A, C, len(feats))).astype(np.float32)
        X[..., 3][rng.random((A, C)) < 0.3] = np.nan
        signal = X[..., 0] + 0.5 * np.nan_to_num(X[..., 2])
        Y = np.stack([(signal + rng.normal(scale=0.5, size=(A, C)) > 1.0).astype(np.float32)] * 9, axis=-1)
        targets = np.array(T.TARGETS)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "seq.npz")
            np.savez_compressed(p, t=t, links=np.arange(C), X=X, Y=Y, features=np.array(feats), targets=targets)
            out = os.path.join(d, "models")
            r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "train_transformer.py"), "--seq", p,
                                "--train-end", str(t[120]), "--cal-end", str(t[180]), "--out-dir", out, "--epochs", "25",
                                "--samples-per-epoch", "700", "--batch", "32", "--patience", "6", "--seq-len", "4", "--model-config", os.path.join(REPO, "settings", "model_v03.yaml")],
                               capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "backend")))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            pt = [f for f in os.listdir(out) if f.endswith(".pt")][0]
            import json
            meta = json.load(open(os.path.join(out, pt + ".metrics.json")))
            tm, net = T.load_transformer(os.path.join(out, pt), meta["sha256"])
            self.assertTrue(tm.calibrated)
            self.assertEqual(tm.mask_features, ["met_cin"])
            self.assertGreater(meta["per_target"]["ts_0_2"]["auc"], 0.7)        # learns the synthetic signal


if __name__ == "__main__":
    unittest.main()
