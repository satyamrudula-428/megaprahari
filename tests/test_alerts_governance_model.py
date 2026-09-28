import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from meghprahari import alerts as A, governance as G, model as MD
from meghprahari.settings import ConfigError, load_settings

NS = {"c": A.CAP_NS}
CFG = dict(walk_speed_kmh=3.0, detour_factor=1.5, dissemination_delay_min=15, lag_fraction=0.0, horizon_min=180)
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


class TestAlerts(unittest.TestCase):
    def test_cost_loss(self):
        self.assertAlmostEqual(A.cost_loss_threshold(1, 10), 0.1)
        with self.assertRaises(ValueError):
            A.cost_loss_threshold(10, 1)
        t = A.level_thresholds(2, 10)
        self.assertAlmostEqual(t[1], 0.1)
        self.assertAlmostEqual(t[2], 0.2)
        self.assertAlmostEqual(t[3], 0.6)
        self.assertGreaterEqual(A.level_thresholds(9, 10)[3], 0.9)

    def test_state_machine_hysteresis(self):
        thr = A.level_thresholds(2, 10)
        s = A.VillageState()
        A.step(s, 0.3, thr)
        self.assertEqual(s.level, 0)            # one cycle at p=0.3 is not enough for WARNING
        A.step(s, 0.3, thr)
        self.assertEqual(s.level, 2)            # needs two consecutive cycles for WARNING
        A.step(s, 0.1, thr)
        self.assertEqual(s.level, 2)            # no immediate drop
        for _ in range(4):
            A.step(s, 0.1, thr)                 # 0.1 < 0.8*0.2
        self.assertEqual(s.level, 1)
        for _ in range(3):
            A.step(s, 0.07, thr)
        self.assertEqual(s.level, 1)
        A.step(s, 0.07, thr)                    # 0.07 < 0.8*0.1
        self.assertEqual(s.level, 0)
        s = A.VillageState()
        A.step(s, 0.15, thr)
        self.assertEqual(s.level, 1)            # WATCH escalates after one cycle
        A.step(s, 0.9, thr)
        self.assertEqual(s.level, 1)
        A.step(s, 0.9, thr)
        self.assertEqual(s.level, 3)

    def test_margin_and_walk(self):
        self.assertAlmostEqual(A.walk_time_min(500, 3.0, 1.5), 15.0)
        self.assertAlmostEqual(A.actionable_margin_min(120, 15, 15), 90.0)
        self.assertAlmostEqual(A.actionable_margin_min(0, 15, 15), -30.0)
        self.assertAlmostEqual(A.actionable_margin_min(0, 15, 15, tc_min=40, lag_fraction=0.5), -10.0)
        self.assertEqual(A.compass(90), "east")
        self.assertEqual(A.compass(359), "north")

    def test_cap_structure_shadow_and_live(self):
        v = A.VillageCtx(1, "Testpur & Sons <village>", 30.1, 78.2, 2, 10, refuge_dist_m=500, refuge_bearing_deg=90, tc_min=25)
        d = A.draft_alert(v, 2, 0.42, "ff", 0, 120, ["iwv high (97th percentile of training)"], CFG, NOW, True, "sender-1", "Sender")
        root = ET.fromstring(d["cap_xml"].split("\n", 1)[1])
        self.assertEqual(root.tag, f"{{{A.CAP_NS}}}alert")
        get = lambda path: root.find(path, NS).text
        self.assertEqual(get("c:status"), "Exercise")
        self.assertEqual(get("c:scope"), "Restricted")
        self.assertIsNotNone(root.find("c:note", NS))
        self.assertTrue(get("c:sent").endswith("+05:30"))
        self.assertEqual(get("c:info/c:urgency"), "Expected")
        self.assertEqual(get("c:info/c:severity"), "Severe")
        self.assertEqual(get("c:info/c:category"), "Met")
        self.assertIn("east", get("c:info/c:instruction"))
        self.assertIn("shorter than the time", get("c:info/c:description"))   # margin is negative at lead 0
        self.assertEqual(get("c:info/c:area/c:circle"), "30.10000,78.20000 3")
        self.assertEqual(d["required_approvals"], 1)
        self.assertLess(d["margin_min"], 0)
        d3 = A.draft_alert(v, 3, 0.9, "cb", 60, 120, [], CFG, NOW, False, "sender-1", "Sender")
        r3 = ET.fromstring(d3["cap_xml"].split("\n", 1)[1])
        self.assertEqual(r3.find("c:status", NS).text, "Actual")
        self.assertEqual(r3.find("c:scope", NS).text, "Public")
        self.assertIsNone(r3.find("c:note", NS))
        self.assertEqual(d3["required_approvals"], 2)
        self.assertAlmostEqual(d3["margin_min"], 60 - 15 - 15)
        with self.assertRaises(ValueError):
            A.build_cap(identifier="x", sender="s", sent=datetime(2026, 1, 1), status="Actual", scope="Public", event="e",
                        urgency="Future", severity="Minor", certainty="Possible", expires=NOW, sender_name="n",
                        headline="h", description="d", instruction="i", area_desc="a", lat=1, lon=1, radius_km=1)


class TestGovernance(unittest.TestCase):
    def _chain(self, n=4):
        rows, prev = [], G.GENESIS
        for i in range(n):
            body = G.canonical_body(f"2026-09-20T12:00:0{i}+00:00", "alice", f"act{i}", {"i": i, "x": 1.5})
            h = G.chain_hash(prev, body)
            rows.append(dict(prev_hash=prev, body=body, hash=h, actor="alice", action=f"act{i}"))
            prev = h
        return rows

    def test_audit_chain(self):
        rows = self._chain()
        self.assertEqual(G.verify_chain(rows), (True, None))
        bad = [dict(r) for r in rows]
        bad[2]["body"] = bad[2]["body"].replace("act2", "actX")
        self.assertEqual(G.verify_chain(bad), (False, 2))
        self.assertEqual(G.verify_chain(rows[:1] + rows[2:]), (False, 1))
        bad = [dict(r) for r in rows]
        bad[1]["actor"] = "mallory"
        self.assertEqual(G.verify_chain(bad), (False, 1))

    def test_password(self):
        h = G.hash_password("correct horse")
        self.assertTrue(G.verify_password("correct horse", h))
        self.assertFalse(G.verify_password("wrong", h))
        self.assertFalse(G.verify_password("x", "garbage"))

    def test_jwt(self):
        secret = "s" * 40
        tok = G.issue_token(secret, "alice", "approver", ttl_s=60)
        self.assertEqual(G.decode_token(secret, tok)["role"], "approver")
        import jwt
        with self.assertRaises(jwt.ExpiredSignatureError):
            G.decode_token(secret, G.issue_token(secret, "a", "approver", ttl_s=10, now=1000))
        with self.assertRaises(jwt.InvalidTokenError):
            G.decode_token("t" * 40, tok)
        with self.assertRaises(ValueError):
            G.issue_token("short", "a", "approver")
        none_tok = jwt.encode({"sub": "a", "role": "admin", "exp": 9999999999}, None, algorithm="none")
        with self.assertRaises(jwt.InvalidTokenError):
            G.decode_token(secret, none_tok)

    def test_rbac_and_approvals(self):
        self.assertTrue(G.allowed("approver", "alert:approve"))
        self.assertFalse(G.allowed("volunteer", "alert:approve"))
        self.assertFalse(G.allowed("scientist", "alert:publish"))
        self.assertFalse(G.allowed("nobody", "alert:read"))
        self.assertTrue(G.is_approved(2, [(1, "approver")]))
        self.assertFalse(G.is_approved(3, [(1, "approver")]))
        self.assertFalse(G.is_approved(3, [(1, "approver"), (1, "approver")]))
        self.assertFalse(G.is_approved(3, [(1, "approver"), (2, "scientist")]))
        self.assertTrue(G.is_approved(3, [(1, "approver"), (2, "approver")]))


class TestSettings(unittest.TestCase):
    def test_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.yaml")
            with open(p, "w") as f:
                f.write("a: 1\nb:\n  c: CHANGE_ME\n")
            with self.assertRaises(ConfigError):
                load_settings(p)
            with open(p, "w") as f:
                f.write("a: 1\nb:\n  c: ok\n  d: [1, 2]\n")
            self.assertEqual(load_settings(p)["b"]["c"], "ok")
            with open(p, "w") as f:
                f.write("a: 1\nb:\n  c: CHANGE_ME\n")
            self.assertEqual(load_settings(p, only=("a",))["a"], 1)      # unrelated CHANGE_ME does not block a partial check
            with self.assertRaises(ConfigError):
                load_settings(p, only=("b",))
            with self.assertRaises(ConfigError):
                load_settings(p, only=("zzz",))


def synth(n=30000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(-3.0 + 1.5 * X[:, 0] + 0.8 * X[:, 1])))
    return pd.DataFrame(dict(t=np.arange(n), x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=(rng.random(n) < p).astype(int), p_true=p))


class TestModel(unittest.TestCase):
    def test_contingency_scores(self):
        c = MD.contingency([1, 1, 1, 0, 0, 0, 0], [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1], 0.5)
        s = MD.skill_scores(c)
        self.assertAlmostEqual(s["POD"], 2 / 3)
        self.assertAlmostEqual(s["FAR"], 1 / 3)
        self.assertAlmostEqual(s["CSI"], 0.5)
        self.assertTrue(np.isnan(MD.skill_scores(MD.contingency([0, 0], [0.1, 0.1], 0.5))["POD"]))
        self.assertAlmostEqual(MD.brier([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(MD.brier_skill([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]), 0.0)

    def test_train_calibrate_verify(self):
        df = synth()
        tm = MD.train_target(df, ["x1", "x2", "x3"], "y", "t", 18000, 24000, target="synthetic")
        self.assertTrue(tm.calibrated)
        m = tm.metrics
        self.assertGreater(m["auc"], 0.75)
        self.assertGreater(m["bss"], 0.03)
        self.assertLess(abs(np.mean(MD.predict(tm, df[df.t > 24000])) - m["base_rate_test"]), 0.01)
        self.assertGreater(tm.importance["x1"], tm.importance["x3"])
        row = df.iloc[0].copy()
        row["x1"] = 4.0
        self.assertTrue(MD.top_drivers(tm, row)[0].startswith("x1 high"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.joblib")
            sha = MD.save_model(tm, path)
            back = MD.load_model(path, sha)
            self.assertTrue(np.allclose(MD.predict(back, df.head(50)), MD.predict(tm, df.head(50))))
            with open(path, "ab") as f:
                f.write(b"tamper")
            with self.assertRaises(MD.ModelIntegrityError):
                MD.load_model(path, sha)

    def test_weights_undo_subsampling(self):
        df = synth(60000, seed=2)
        keep = (df.y == 1) | (np.random.default_rng(3).random(len(df)) < 0.2)
        sub = df[keep].copy()
        sub["w"] = np.where(sub.y == 1, 1.0, 5.0)
        tm = MD.train_target(sub, ["x1", "x2", "x3"], "y", "t", 36000, 48000, w_col="w", target="sub")
        test_full = df[df.t > 48000]
        self.assertLess(abs(np.mean(MD.predict(tm, test_full)) - test_full.y.mean()), 0.012)

    def test_uncalibrated_refused_and_bad_split(self):
        df = synth(8000)
        tm = MD.train_target(df, ["x1", "x2", "x3"], "y", "t", 5000, 6500, min_pos_cal=10 ** 6)
        self.assertFalse(tm.calibrated)
        with self.assertRaises(MD.UncalibratedModelError):
            MD.predict(tm, df)
        self.assertEqual(len(MD.predict(tm, df, require_calibrated=False)), len(df))
        with self.assertRaises(ValueError):
            MD.train_target(df.assign(y=0), ["x1"], "y", "t", 5000, 6500)


if __name__ == "__main__":
    unittest.main()
