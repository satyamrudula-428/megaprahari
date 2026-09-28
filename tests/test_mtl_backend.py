import importlib.util
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from meghprahari import mtl_backend as T

HAS_TORCH = importlib.util.find_spec("torch") is not None
FEATS = ["hem_rain_now", "hem_rain_accel", "met_cape", "met_cin", "ter_tc_min"]


def frame(n=6, seed=0, cin_nan=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATS))) * [5, 2, 800, 50, 30] + [3, 0, 1200, 80, 60], columns=FEATS,
                      index=pd.Index(range(100, 100 + n), name="link_id"))
    if cin_nan:
        df.loc[df.index[:2], "met_cin"] = np.nan
    return df


class TestNumpyParts(unittest.TestCase):
    def test_standardize_uses_training_stats_and_flags_missing(self):
        tr = frame(200)
        mean, std, mask = T.standardize_fit(tr, FEATS)
        self.assertEqual(mask, ["met_cin"])
        x = T.standardize_apply(tr, FEATS, mean, std, mask)
        self.assertEqual(x.shape, (200, len(FEATS) + 1))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[:2, -1].tolist(), [1.0, 1.0])                  # missing flag set, value -> 0 (training mean)
        self.assertEqual(x[:2, FEATS.index("met_cin")].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(np.nanmean(x[:, 0])), 0.0, places=5)
        other = T.standardize_apply(frame(50, seed=9) + 100, FEATS, mean, std, mask)   # never refits on new data
        self.assertGreater(float(other[:, 0].mean()), 5)

    def test_groups(self):
        groups, names = T.input_groups(FEATS, ["met_cin"])
        self.assertEqual(groups["sat"], (0, 1))
        self.assertEqual(groups["met"], (2, 3, 5))                         # flag joins the met group
        self.assertEqual(groups["terrain"], (4,))
        self.assertEqual(names[-1], "missing_met_cin")
        with self.assertRaises(ValueError):
            T.input_groups(["hem_rain_now", "met_cape"], [])

    def test_sequences_pad_and_align(self):
        tr = frame(100)
        mean, std, mask = T.standardize_fit(tr, FEATS)
        h = [frame(6, seed=s, cin_nan=False) for s in range(3)]
        x = T.sequences(h, FEATS, mean, std, mask, seq_len=5)
        self.assertEqual(x.shape, (6, 5, len(FEATS) + 1))
        self.assertTrue(np.array_equal(x[:, 0], x[:, 2]))                  # left-padded with the oldest step
        self.assertTrue(np.array_equal(x[:, -1], T.standardize_apply(h[-1], FEATS, mean, std, mask)))

    def test_fit_temperature(self):
        rng = np.random.default_rng(0)
        z_true = rng.normal(scale=2.0, size=20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-z_true))).astype(float)
        self.assertAlmostEqual(T.fit_temperature(3.0 * z_true, y), 3.0, delta=0.3)   # overconfident logits -> T ~ 3
        self.assertEqual(T.fit_temperature([1.0, 2.0], [1.0, 1.0]), 1.0)
        self.assertEqual(T.fit_temperature([1.0, 2.0], [np.nan, np.nan]), 1.0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTorchParts(unittest.TestCase):
    def make(self):
        import torch
        tr = frame(100)
        mean, std, mask = T.standardize_fit(tr, FEATS)
        groups, names = T.input_groups(FEATS, mask)
        cfg = dict(n_features=len(names), sat_idx=groups["sat"], met_idx=groups["met"], terrain_idx=groups["terrain"],
                   d_model=16, n_heads=2, n_layers=1, ff_dim=32, max_seq_len=4)
        torch.manual_seed(0)
        tm = T.TransformerModel(cfg, FEATS, mean, std, mask, 4)
        tm.state_dict = T.build_network(tm).state_dict()
        return tm

    def test_save_load_integrity_and_predict(self):
        tm = self.make()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "mtl.pt")
            sha = T.save_transformer(tm, p)
            tm2, net = T.load_transformer(p, sha)
            hist = [frame(6, seed=s) for s in range(4)]
            with self.assertRaises(T.UncalibratedTransformerError):
                T.score_transformer(tm2, net, hist)
            tm2.temperatures = {t: 1.5 for t in T.TARGETS}
            probs = T.score_transformer(tm2, net, hist)
            self.assertEqual(list(probs.columns), T.TARGETS)
            self.assertEqual(list(probs.index), list(hist[-1].index))
            self.assertTrue(((probs >= 0) & (probs <= 1)).all().all())
            with open(p, "ab") as f:
                f.write(b"tampered")
            with self.assertRaises(T.TransformerIntegrityError):
                T.load_transformer(p, sha)


if __name__ == "__main__":
    unittest.main()
