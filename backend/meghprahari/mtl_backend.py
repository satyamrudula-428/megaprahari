"""Operational wrapper for the multi-task Transformer (tasks G4, G6, G7, D8): normalisation from the training split only,
missing-value indicators, temperature-scaling calibration, SHA-256-checked save/load and sequence building.

The pure-numpy parts (standardize_fit/apply, fit_temperature, sequences) need no torch; save/load/predict import torch
lazily, so the gradient-boosting path keeps working without it (hard rule 7).
Inputs are feature tables with FEATURES_V* columns (units as in pipeline.py); outputs are probabilities in [0, 1]."""
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mtl_transformer import HAZARDS, LEADS

TARGETS = [f"{h}_{lead}" for h in HAZARDS for lead in LEADS]      # order of the model's 3 x 3 output grid
GROUP_PREFIX = {"sat": "hem_", "met": "met_", "terrain": "ter_"}


class UncalibratedTransformerError(RuntimeError):
    pass


class TransformerIntegrityError(RuntimeError):
    pass


def standardize_fit(X, features):
    """(mean, std, mask_features) from the TRAINING split only. mask_features = features that have missing values in
    training; each gets a 0/1 'is missing' input so undefined values (e.g. CIN) are not confused with average values."""
    A = X[features].to_numpy(float)
    mean = np.nanmean(A, axis=0)
    std = np.nanstd(A, axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    mask_features = [f for f, m in zip(features, np.isnan(A).any(axis=0)) if m]
    return mean, std, mask_features


def standardize_apply(X, features, mean, std, mask_features):
    """(N, F + M) float32 inputs: standardised features with missing -> 0 (the training mean), then M missing flags."""
    A = (X[features].to_numpy(float) - mean) / std
    flags = [X[f].isna().to_numpy(float)[:, None] for f in mask_features]
    A = np.nan_to_num(A, nan=0.0)
    return np.hstack([A] + flags).astype(np.float32) if flags else A.astype(np.float32)


def input_groups(features, mask_features):
    """Column indices of the satellite / meteorology / terrain modalities in the standardised input (missing flags join
    their feature's group)."""
    names = list(features) + [f"missing_{f}" for f in mask_features]
    base = list(features) + list(mask_features)
    groups = {g: tuple(i for i, f in enumerate(base) if f.startswith(p)) for g, p in GROUP_PREFIX.items()}
    for g, idx in groups.items():
        if not idx:
            raise ValueError(f"no input features for the {g} modality")
    return groups, names


def sequences(history, features, mean, std, mask_features, seq_len):
    """[N, T, F'] float32 input from a list of feature DataFrames (oldest -> newest, same catchment index; the last is
    the anchor). Shorter histories are left-padded by repeating the oldest available step."""
    if not history:
        raise ValueError("empty feature history")
    steps = [standardize_apply(h.reindex(history[-1].index), features, mean, std, mask_features)
             for h in history[-seq_len:]]
    while len(steps) < seq_len:
        steps.insert(0, steps[0])
    return np.stack(steps, axis=1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits, y, grid=np.exp(np.linspace(math.log(0.05), math.log(20.0), 241))):
    """Temperature T > 0 minimising the log loss of sigmoid(logits / T) on CALIBRATION data (labels 0/1, NaN ignored).
    Returns 1.0 when the calibration data lack both classes."""
    z, t = np.asarray(logits, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(z)
    z, t = z[ok], t[ok]
    if t.size == 0 or t.min() == t.max():
        return 1.0
    eps = 1e-7
    nll = [-np.mean(t * np.log(np.clip(_sigmoid(z / T), eps, 1)) + (1 - t) * np.log(np.clip(1 - _sigmoid(z / T), eps, 1)))
           for T in grid]
    return float(grid[int(np.argmin(nll))])


@dataclass
class TransformerModel:
    """Everything needed to run the Transformer reproducibly. `state_dict` holds tensors; the rest is plain data."""
    config: dict
    features: list
    mean: np.ndarray
    std: np.ndarray
    mask_features: list
    seq_len: int
    temperatures: dict = field(default_factory=dict)       # {target: T}; calibrated only when all 9 are fitted
    metrics: dict = field(default_factory=dict)
    state_dict: dict = None

    @property
    def calibrated(self):
        return set(self.temperatures) == set(TARGETS)


def build_network(tm):
    from .mtl_transformer import MTLConfig, PhysicsGuidedMTLTransformer
    return PhysicsGuidedMTLTransformer(MTLConfig(**tm.config))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_transformer(tm, path):
    """Write a torch file (tensors + plain Python types only) and return its SHA-256."""
    import torch
    torch.save(dict(config=tm.config, features=list(tm.features), mean=tm.mean.tolist(), std=tm.std.tolist(),
                    mask_features=list(tm.mask_features), seq_len=int(tm.seq_len),
                    temperatures={k: float(v) for k, v in tm.temperatures.items()}, metrics=tm.metrics,
                    state_dict=tm.state_dict), path)
    return sha256_file(path)


def load_transformer(path, expected_sha256):
    """Load only if the file's SHA-256 equals the registered value, and with torch's weights_only safe loader.
    Returns (TransformerModel, network in eval mode)."""
    if sha256_file(path) != expected_sha256:
        raise TransformerIntegrityError(f"{path} does not match its registered SHA-256")
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    tm = TransformerModel(d["config"], d["features"], np.asarray(d["mean"]), np.asarray(d["std"]), d["mask_features"],
                          d["seq_len"], d["temperatures"], d["metrics"], d["state_dict"])
    net = build_network(tm)
    net.load_state_dict(tm.state_dict)
    net.eval()
    return tm, net


def predict_transformer(tm, net, x, require_calibrated=True):
    """DataFrame-ready {target: probabilities} for input x [N, T, F'] (from sequences()). Uncalibrated models are
    refused for alerting, like the gradient-boosting path."""
    if require_calibrated and not tm.calibrated:
        raise UncalibratedTransformerError("transformer is not temperature-calibrated; it may not drive alerts")
    import torch
    with torch.no_grad():
        z = net(torch.as_tensor(x, dtype=torch.float32)).reshape(len(x), -1).numpy()
    return {t: _sigmoid(z[:, j] / tm.temperatures.get(t, 1.0)) for j, t in enumerate(TARGETS)}


def score_transformer(tm, net, history, require_calibrated=True):
    """Probabilities per catchment (DataFrame indexed like history[-1], columns = target names), the Transformer
    counterpart of pipeline.score_all."""
    x = sequences(history, tm.features, tm.mean, tm.std, tm.mask_features, tm.seq_len)
    return pd.DataFrame(predict_transformer(tm, net, x, require_calibrated), index=history[-1].index)
