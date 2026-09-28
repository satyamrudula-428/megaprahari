"""Per-target probability models (scikit-learn only): train, calibrate, verify, explain, persist.
Fail-closed: an uncalibrated model can be used for research scoring but never to trigger alerts."""
import hashlib
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


class UncalibratedModelError(RuntimeError):
    pass


class ModelIntegrityError(RuntimeError):
    pass


def contingency(y, p, thr, w=None):
    y = np.asarray(y).astype(bool)
    f = np.asarray(p) >= thr
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return dict(hits=w[y & f].sum(), misses=w[y & ~f].sum(), fa=w[~y & f].sum(), cn=w[~y & ~f].sum())


def skill_scores(c):
    H, M, F = c["hits"], c["misses"], c["fa"]
    d = lambda a, b: float(a / b) if b > 0 else float("nan")
    return dict(POD=d(H, H + M), FAR=d(F, H + F), CSI=d(H, H + M + F))


def brier(y, p, w=None):
    return float(np.average((np.asarray(p, float) - np.asarray(y, float)) ** 2, weights=w))


def brier_skill(y, p, w=None):
    y = np.asarray(y, float)
    ref = brier(y, np.full(len(y), np.average(y, weights=w)), w)
    return float("nan") if ref == 0 else 1.0 - brier(y, p, w) / ref


def reliability(y, p, bins=10, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return [(float(np.average(p[idx == b], weights=w[idx == b])), float(np.average(y[idx == b], weights=w[idx == b])),
             float(w[idx == b].sum())) for b in range(bins) if (idx == b).any()]


def ece(y, p, bins=10, w=None):
    rows = reliability(y, p, bins, w)
    tot = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[0] - r[1]) for r in rows) / tot)


@dataclass
class TargetModel:
    target: str
    features: list
    clf: object
    iso: object
    quantiles: dict
    importance: dict
    metrics: dict
    calibrated: bool
    train_range: tuple = field(default=(None, None))


def train_target(df, features, y_col, time_col, train_end, cal_end, w_col=None, min_pos_cal=30, seed=0, target=None):
    """Time-ordered split: train <= train_end < calibration <= cal_end < test. Weights let you undo negative
    subsampling (weight = 1/sampling fraction) so probabilities stay valid on the natural base rate."""
    d = df.sort_values(time_col)
    parts = {"train": d[d[time_col] <= train_end],
             "cal": d[(d[time_col] > train_end) & (d[time_col] <= cal_end)],
             "test": d[d[time_col] > cal_end]}
    for name, part in parts.items():
        if len(part) == 0 or part[y_col].nunique() < 2:
            raise ValueError(f"'{name}' split lacks both classes; adjust the split dates or gather more events")
    wt = {k: (v[w_col].to_numpy(float) if w_col else None) for k, v in parts.items()}
    X = {k: v[features].to_numpy(float) for k, v in parts.items()}
    y = {k: v[y_col].to_numpy(int) for k, v in parts.items()}
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                         random_state=seed)
    clf.fit(X["train"], y["train"], sample_weight=wt["train"])
    raw_cal = clf.predict_proba(X["cal"])[:, 1]
    iso = None
    if int(y["cal"].sum()) >= min_pos_cal:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_cal, y["cal"], sample_weight=wt["cal"])
    p_test = clf.predict_proba(X["test"])[:, 1]
    if iso is not None:
        p_test = iso.predict(p_test)
    base = float(np.average(y["test"], weights=wt["test"]))
    metrics = dict(n_test=int(len(y["test"])), positives_test=int(y["test"].sum()), base_rate_test=base,
                   brier=brier(y["test"], p_test, wt["test"]), bss=brier_skill(y["test"], p_test, wt["test"]),
                   auc=float(roc_auc_score(y["test"], p_test, sample_weight=wt["test"])),
                   ece=ece(y["test"], p_test, 10, wt["test"]),
                   reliability=reliability(y["test"], p_test, 10, wt["test"]),
                   scores={str(t): skill_scores(contingency(y["test"], p_test, t, wt["test"])) for t in (0.05, 0.1, 0.2, 0.5)},
                   calibrated=iso is not None, min_pos_cal=min_pos_cal)
    pi = permutation_importance(clf, X["cal"], y["cal"], scoring="neg_brier_score", n_repeats=3, random_state=seed,
                                sample_weight=wt["cal"])
    grid = np.linspace(0, 1, 21)
    q = {f: np.nanquantile(parts["train"][f].to_numpy(float), grid).tolist() for f in features}
    tm = TargetModel(target or y_col, list(features), clf, iso, q,
                     {f: float(m) for f, m in zip(features, pi.importances_mean)}, metrics, iso is not None,
                     (str(parts["train"][time_col].min()), str(parts["train"][time_col].max())))
    return tm


def predict(tm, X, require_calibrated=True):
    if require_calibrated and not tm.calibrated:
        raise UncalibratedModelError(f"model '{tm.target}' is not calibrated; it may not drive alerts")
    missing = [f for f in tm.features if f not in X.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    raw = tm.clf.predict_proba(X[tm.features].to_numpy(float))[:, 1]
    return tm.iso.predict(raw) if tm.iso is not None else raw


def top_drivers(tm, row, k=3):
    """Plain-language anomaly drivers: importance-weighted distance of each feature from the training median.
    This is NOT SHAP; it says which important inputs are unusual, not their causal contribution."""
    grid = np.linspace(0, 1, len(next(iter(tm.quantiles.values()))))
    scored = []
    for f in tm.features:
        v = row[f]
        if not np.isfinite(v):
            continue
        pct = float(np.interp(v, tm.quantiles[f], grid))
        scored.append((max(tm.importance.get(f, 0.0), 0.0) * abs(pct - 0.5), f, pct))
    if scored and max(s[0] for s in scored) == 0:
        scored = [(abs(p - 0.5), f, p) for _, f, p in scored]
    scored.sort(reverse=True)
    return [f"{f} {'high' if p > 0.5 else 'low'} ({p * 100:.0f}th percentile of training)" for _, f, p in scored[:k]]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(tm, path):
    joblib.dump(tm, path)
    return sha256_file(path)


def load_model(path, expected_sha256):
    """joblib/pickle can execute code: only load files whose hash matches the one recorded at approval."""
    if sha256_file(path) != expected_sha256:
        raise ModelIntegrityError(f"{path} does not match its registered SHA-256")
    return joblib.load(path)
