import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from meghprahari.mtl_transformer import MTLConfig, PhysicsGuidedMTLTransformer, multitask_bce


def small_cfg():
    return MTLConfig(n_features=12, sat_idx=(0, 1, 2), met_idx=(3, 4, 5, 6, 7, 8), terrain_idx=(9, 10, 11),
                     d_model=32, n_heads=4, n_layers=1, ff_dim=64, max_seq_len=6)


@unittest.skipUnless(HAS_TORCH, "torch not installed (pip install -r requirements-ai.txt)")
class TestMTLTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shapes_and_probabilities(self):
        model = PhysicsGuidedMTLTransformer(small_cfg())
        y = model(torch.randn(2, 6, 12))
        self.assertEqual(tuple(y.shape), (2, 3, 3))                  # batch x hazards x lead bins
        p = model.probabilities(torch.randn(2, 6, 12))
        self.assertTrue(bool(torch.all((p >= 0) & (p <= 1))))

    def test_rejects_bad_inputs(self):
        model = PhysicsGuidedMTLTransformer(small_cfg())
        with self.assertRaises(ValueError):
            model(torch.randn(2, 12))                                 # missing time axis
        with self.assertRaises(ValueError):
            model(torch.randn(2, 6, 11))                              # wrong feature count
        with self.assertRaises(ValueError):
            model(torch.randn(2, 7, 12))                              # longer than max_seq_len

    def test_masked_bce_ignores_nan_labels(self):
        logits = torch.zeros(1, 3, 3)
        targets = torch.full((1, 3, 3), float("nan"))
        targets[0, 0, 0] = 1.0
        loss = multitask_bce(logits, targets)
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.0))), places=5)   # BCE(0 logit, 1) = ln 2
        self.assertEqual(float(multitask_bce(logits, torch.full((1, 3, 3), float("nan")))), 0.0)


if __name__ == "__main__":
    unittest.main()
