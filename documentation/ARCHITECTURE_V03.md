# MeghPrahari v0.3 — Physics-Guided MTL Transformer

## What is new

v0.3 adds an actual lightweight multimodal spatiotemporal Transformer rather
than describing one without implementing it.

### Model

1. Satellite token: HEM/QPE intensity, acceleration, growth, motion and stall.
2. Meteorology token: IWV, ΔIWV, CAPE, CIN, MFC, shear, K-index, Total Totals,
   and related thermodynamic predictors when available.
3. Terrain token: slope, HAND, upstream area and time of concentration.
4. Explicit cross-modal attention fuses satellite and atmospheric signals.
5. A small temporal Transformer encodes the recent sequence.
6. One shared representation feeds three hazard heads:
   - severe thunderstorm
   - cloudburst
   - flash flood
7. Each head emits three lead-time probabilities:
   - 0–2 h
   - 2–4 h
   - 4–6 h

This is multi-task learning: the hazards share the expensive representation
while retaining separate outputs.

## Why this is designed for low cost

The prototype deliberately uses a small 64-dimensional model, four attention
heads and two Transformer layers. Static terrain features are precomputed
rather than recomputed for every inference. A production deployment can use
risk-adaptive compute: inexpensive cached-feature screening for normal cells,
then the Transformer only for anomalous cells.

The cost/latency claim must be demonstrated with measurements from the target
deployment; this repository does not claim that a particular hardware setup
beats an NWP system without benchmarking it.

## Physics-guided design

The neural model does not replace physical reasoning. The feature layer encodes
meteorological and hydrological signals that are meaningful to forecasters:
moisture, instability, lift, storm structure and terrain response.

CAPE/CIN and CTT-drop-rate are optional until the corresponding variables are
verified against the exact MOSDAC/NCMRWF product schemas. The code must not
silently invent them.

## Validation gates

Before real alerting:

- event-ledger labels must replace proxy labels where available;
- train/calibration/test splits must be time ordered;
- probability calibration must be evaluated on a held-out period;
- compare against the v0.1 gradient-boosting baseline;
- measure Brier score, Brier skill score, ROC-AUC, ECE, POD, FAR and CSI;
- measure CPU latency, memory, model size and throughput;
- run the full smoke test;
- keep `shadow_mode: true`.

## Synthetic demo

Run:

    python tools/demo_transformer.py

The output only proves that the MTL Transformer trains and infers end-to-end.
It is not a meteorological validation result.
