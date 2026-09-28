# MeghPrahari v1.0 — target architecture

Status legend: ✅ built and run on real data · 🟡 built, not yet run on real data / partial · ⬜ planned.
Current code state: see [ROADMAP.md](ROADMAP.md). This document is the design the roadmap builds towards.

## 1. Purpose and users

A catchment- and village-level early-warning **decision-support** system for Himalayan cloudbursts, severe
thunderstorms and flash floods, 0–6 h ahead. Users: district/state disaster management authorities (DDMA/SDMA) and IMD
forecasters. The system drafts; authorised humans approve (see `documentation/GOVERNANCE.md`).

```text
Probability  →  Impact  →  Decision  →  Governed alert  →  Verification
```

## 2. Design principles

1. **Features, not products.** Every data source enters through an adapter that produces the same feature definitions
   (`pipeline.FEATURE_SETS`), so a model trained on reanalysis/archives can run on live feeds. The accuracy change
   between the two must be measured, never assumed.
2. **Fail closed.** Missing or implausible inputs stop a step with a clear error or mark the output degraded; nothing is
   invented (CAPE/CIN/CTT/pressure/labels).
3. **One feature code path.** Live worker, replay and training all call `pipeline.features_at`.
4. **Honest verification.** Time-ordered splits, replay events excluded from training (`settings/region.yaml`), skill
   reported against persistence and extrapolation baselines.
5. **CPU-only, one server per state.** No GPU dependency in operations.

## 3. Training data vs live data

| Role | Training / replay (history) | Live operation |
|---|---|---|
| Rain rate (≤ 4 km) | NCMRWF **MERA** hourly ✅ (IMERG kept for comparison ✅) | **MOSDAC INSAT HEM** (30 min) ⬜, **IMD Doppler radar** ⬜; Phase A stand-in: **IMERG Early Run** (~4 h latency, 10 km) 🟡 `tools/live_imerg.py` |
| Cloud-top temperature | INSAT TIR1 archive ⬜; GridSat-B1 tried and rejected -- ~2% coverage over the pilot box, see `backend/meghprahari/ctt.py` | INSAT TIR1 live ⬜ |
| Atmospheric ingredients | **ERA5** (pressure + single levels) ✅, IMDAA (≤ 2020) ⬜ | **NCMRWF NCUM** analyses/forecasts ⬜ (ERA5 arrives ~5 days late); Phase A stand-in: **NOAA GFS 0.25deg analysis** (~4 h latency, same grid as ERA5) 🟡 `tools/live_gfs.py` |
| Truth / labels | MERA thresholds ✅ + event ledger ✅ | gauges ⬜ + citizen reports (`/api/observations`) 🟡 + ledger |

ERA5 and MERA are for training and replay. Live mode needs MOSDAC, NCUM and radar adapters producing the same
features; Phase A (`tools/live_imerg.py`, `tools/live_gfs.py`) builds and runs on real data with public,
no-MOSDAC substitutes (IMERG Early Run, GFS analysis) so live ingestion + inference can be exercised (still shadow
mode, roadmap Q1/Q2) before MOSDAC access is resolved; the accuracy gap against MOSDAC/NCUM has not been measured
yet (Q3).

## 4. Components

```text
 SOURCES ─► [1] INGEST ─► [2] DATA CUBE ─► [3] FEATURES ─► [4] PREDICTION ─► [5] IMPACT ─► [6] DECISION
                                                                                              │
             [8] VERIFICATION / MLOps ◄──────────────────── [7] GOVERNANCE & DISSEMINATION ◄──┘
```

| # | Component | Responsibility | Code today |
|---|---|---|---|
| 1 | **Ingest** | One adapter per source; checksum ledger; QC; cadence/gap checks; data-age monitor; degraded-mode flag | `ingest.py` ✅ (HEM/IMERG/MERA/ERA5 readers, `MetIndex`), data-age monitor ⬜ |
| 2 | **Data cube** | Align time (sat 30–60 min vs met 3 h: `select_met_times`) and space (nearest cell per catchment) | `pipeline.select_met_times`, `assemble` ✅ |
| 3 | **Features** | Storm dynamics (rain, trend, growth, motion, stall); ingredients (IWV, CAPE, CIN, CIN trend, convergence, upslope flux, shear); terrain (slope, HAND, Tc, area); CTT cooling ⬜; catchment accumulations ⬜ | `satellite.py`, `meteo.py`, `terrain.py`, `pipeline.FEATURES_V2` ✅ |
| 4a | **Rain nowcast** | 0–2 h extrapolation; 2–6 h blend with NWP (NCUM) | `baselines.py` (persistence, uniform-motion) ✅, NWP blend ⬜ |
| 4b | **Hazard models** | P(ts/cb/ff) per catchment × lead bin; gradient boosting baseline, Transformer only if it beats it on held-out data | `model.py` ✅ (trained on synthetic only), `mtl_transformer.py` 🟡 |
| 4c | **Hydrology** | Catchment rain accumulation vs flash-flood threshold; antecedent wetness; SCS-CN | Tc-window proxy labels ✅, thresholds ⬜ |
| 4d | **Calibration** | Isotonic (GBM) / temperature scaling (Transformer); reliability checks | isotonic ✅, temperature scaling ⬜ |
| 5 | **Impact** | Villages, population, schools, hospitals, roads per catchment; minutes to act; refuge direction | `alerts.actionable_margin_min`, refuge map ✅, exposure layer 🟡 |
| 6 | **Decision** | Cost-loss thresholds, hysteresis, escalation levels | `alerts.py` ✅ |
| 7 | **Governance & dissemination** | Draft → four-eyes approval → hash-locked CAP 1.2 (XSD-validated) → outbox → SACHET/SMS gateway | `governance.py`, `api.py`, CAP XSD test ✅, SACHET ⬜ |
| 8 | **Verification / MLOps** | POD/FAR/CSI, Brier/BSS, reliability, FSS, lead time vs baselines; drift monitoring; retraining | `model.py`, `verify.py`, `tools/evaluate.py` ✅, monitoring ⬜ |

Explanations: `explain.ingredient_card` (moisture, instability, lift, storm signal, terrain response; non-causal) ✅,
served at `/api/explain/{alert_id}`.

## 5. Model design

* **Targets:** 3 hazards × 3 lead bins (0–2, 2–4, 4–6 h). Labels are **neighbourhood** labels (`labels.neighbourhood_km`)
  so a storm displaced by a few km is not counted as both a miss and a false alarm.
* **Label definitions must match the rain product's resolution.** At 4 km / hourly (MERA), cloudburst-scale rates are
  smoothed (the 2025 Mandi cloudburst peaks at 54 mm/h in MERA), so the cloudburst label needs an agreed, justified
  definition (e.g. accumulation-based or percentile-based) before training (roadmap F2/F3).
* **Baseline first:** calibrated gradient boosting per target. The Transformer is adopted only if it improves held-out
  skill; the comparison is reported either way.
* **Physics constraints:** probabilities non-decreasing in cumulative windows; flash flood requires rain.
* **Splits:** train 2021–2023, calibrate 2024, test = replay events (2025) + other held-out events.

## 6. Deployment

Docker services on one server: PostgreSQL/PostGIS (state, audit), API + dashboard (FastAPI + MapLibre), ingest/model
worker. Data under `/data` (landing, atlas, models, outbox). Shadow mode on until the go-live gates in
`documentation/GOVERNANCE.md` pass.

## 7. Quality attributes

| Attribute | Mechanism |
|---|---|
| Reproducibility | data register with checksums; model SHA-256; config in git; replay gives identical output |
| Correctness | unit tests with synthetic fixtures in the real file layouts; real-data smoke runs |
| Safety | shadow mode, four-eyes, calibrated-only models, hash-locked CAP, append-only audit chain |
| Honesty | only measured results are claimed (`CLAUDE.md` hard rule 1) |
| Operability | data-age and processing-time monitoring ⬜; degraded mode with dashboard warning ⬜ |
