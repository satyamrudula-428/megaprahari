#!/usr/bin/env python3
"""Train a tiny synthetic MTL Transformer and print an efficiency report.

This proves the architecture executes end-to-end. It is NOT a weather-skill
benchmark and must not be presented as one.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from meghprahari.mtl_transformer import MTLConfig, PhysicsGuidedMTLTransformer, multitask_bce, parameter_count

torch.manual_seed(7)
np.random.seed(7)

# Feature groups: satellite dynamics, atmospheric predictors, terrain.
N_FEATURES = 24
SAT = tuple(range(0, 8))
MET = tuple(range(8, 20))
TERRAIN = tuple(range(20, 24))

cfg = MTLConfig(
    n_features=N_FEATURES,
    sat_idx=SAT,
    met_idx=MET,
    terrain_idx=TERRAIN,
    d_model=64,
    n_heads=4,
    n_layers=2,
    ff_dim=128,
    max_seq_len=12,
)
model = PhysicsGuidedMTLTransformer(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

B, T = 128, 12
X = torch.randn(B, T, N_FEATURES)
# Synthetic labels are generated from deterministic feature combinations only
# to exercise all 9 outputs. They are not meteorological labels.
score = 0.7 * X[:, -1, 8] + 0.5 * X[:, -1, 0] - 0.3 * X[:, -1, 21]
Y = torch.stack([
    torch.sigmoid(score + j * 0.08) > 0.55 for j in range(9)
], dim=1).float().reshape(B, 3, 3)

model.train()
for _ in range(30):
    optimizer.zero_grad()
    logits = model(X)
    loss = multitask_bce(logits, Y)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    start = time.perf_counter()
    for _ in range(50):
        probs = model.probabilities(X[:1])
    elapsed = (time.perf_counter() - start) / 50 * 1000

print("MeghPrahari v0.3 Transformer smoke test")
print(f"Trainable parameters : {parameter_count(model):,}")
print(f"Input shape          : {tuple(X[:1].shape)}")
print(f"Output shape         : {tuple(probs.shape)}  # hazards x lead windows")
print(f"CPU batch-1 latency  : {elapsed:.2f} ms")
print("Hazards              : thunderstorm, cloudburst, flash flood")
print("Lead windows         : 0-2h, 2-4h, 4-6h")
print("NOTE                 : synthetic smoke test; no forecast-skill claim.")
