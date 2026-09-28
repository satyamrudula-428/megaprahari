# SIH26077 — MeghPrahari technical alignment

## What the current code actually implements

MeghPrahari v0.3 is a **catchment-native, probabilistic severe-weather decision-support prototype**. It combines INSAT-3DR HEM rain-rate dynamics, IMDAA/IMDAA-like atmospheric profiles, and SRTM-derived terrain features on a common analysis grid, then produces calibrated probabilities for severe thunderstorm (`ts`), cloudburst-type extreme rainfall (`cb`), and flash flood (`ff`) targets over 0–2 h, 2–4 h and 4–6 h lead windows. Real 2025 pilot data now loads (see "Data sources actually used" below), but no model has yet been trained or run on real data.

The model that produces forecasts in the worker and API is **one calibrated scikit-learn HistGradientBoosting model per hazard/lead target**.

The repository also contains a **physics-guided multi-task Transformer** (`backend/meghprahari/mtl_transformer.py`: modality projections, bidirectional cross-modal attention, temporal Transformer encoder, three hazard heads × three lead bins). It is unit-tested, and its operational wrapper (`backend/meghprahari/mtl_backend.py`: training-split normalisation, temperature calibration, SHA-256-checked save/load) and the `model.backend` switch in the worker exist and are tested. It **has not yet been trained on real data** (roadmap G5), so do not describe it as producing forecasts until it has been trained and evaluated on held-out events.

## Predictor matrix

| SIH concept | v0.3 implementation (`FEATURES_V1`) |
|---|---|
| Moisture | Integrated water vapour (IWV) |
| Rapid moisture change | 3-hour IWV difference |
| Low-level lift/convergence | Moisture-flux convergence |
| Orographic forcing | Orographic moisture-flux index |
| Instability | K-index and Total Totals (undefined where 850 hPa is below ground: ~90 % of the Mandi pilot); CAPE, CIN and 3-h CIN change in `FEATURES_V2` |
| Wind structure | Low-level to 500-hPa bulk shear |
| Satellite precipitation | INSAT-3DR HEM rain rate |
| Convective evolution | Rain-rate trend, acceleration, area growth |
| Storm motion | Phase-correlation motion estimate |
| Storm stalling | Residence/stall index |
| Terrain | Slope, HAND, upstream area, flow length, time of concentration |
| Hyper-local unit | Micro-catchment / village |

### Deliberately not claimed in v0.3

- CAPE/CIN as features of a *trained* model. They are in `FEATURES_V2` (from ERA5 single levels) and used in the
  explanation card, but no model has been trained on them yet.
- Cloud Top Temperature (CTT) drop rate. The current HEM reader is a precipitation product reader; it does not ingest an INSAT TIR brightness-temperature field.
- Transformer or cross-attention **in production**. The architecture, its operational wrapper (normalisation, temperature
  calibration, SHA-256-checked loading) and the `model.backend` switch exist and are tested, but no Transformer has been
  trained on real data yet.
- SHAP. Explanations are importance-weighted anomaly drivers plus an ingredient card (moisture, instability, lift,
  storm signal, terrain response) built from the alert's inputs; both are explicitly labelled as non-causal.
- Forecast skill on real events. Synthetic unit tests verify numerical/software behaviour; they do not prove operational forecast skill.

## Operational safety model

The system is designed as decision support, not autonomous public warning issuance:

```text
HEM + IMDAA + SRTM
       ↓
QC + spatial/temporal alignment
       ↓
meteorological + satellite + terrain features
       ↓
calibrated probability models
       ↓
catchment/village risk
       ↓
alert draft
       ↓
human approval / four-eyes for highest level
       ↓
CAP 1.2 exercise outbox
```

`shadow_mode: true` is the default. Real feeds and public dissemination require separate data-access, domain-validation, governance and operational gates.

## Proposal wording that is safe to use

> MeghPrahari is a hyper-local severe-weather nowcasting and decision-support system that fuses satellite precipitation dynamics, atmospheric thermodynamic and kinematic predictors, and terrain-derived catchment characteristics. It estimates calibrated hazard probabilities for severe thunderstorms, cloudburst-type extreme rainfall and flash floods across 0–2, 2–4 and 4–6 hour lead windows. The system converts these probabilities into catchment-level risk maps and governed alert drafts, with meteorological/terrain driver explanations, human approval and CAP 1.2 exercise output.

Do **not** replace the above with claims of a Transformer forecast engine, CAPE/CIN or CTT until those components are wired in, trained and evaluated on real data. It is accurate to say the repository contains a unit-tested multi-task Transformer architecture that is being integrated.

## Data sources actually used for the 2025 pilot

| Role in the design | Planned source | Used now | Why |
|---|---|---|---|
| Satellite rain rate (30 min) | INSAT-3DR/3DS HEM (MOSDAC) | NASA GPM IMERG Final V07, 0.1°, 30 min | MOSDAC access pending; IMERG smooths cloudburst peaks (max ~21–46 mm/h vs ~100 mm/h cloudburst rates) |
| Observed rain for verification | Event ledger + gauges | NCMRWF MERA hourly 4 km (satellite + radar) + news-based event ledger | MERA places the 30 Jun–1 Jul 2025 rain core over Pandoh–Siyanj–Thunag; units (mm/h) and time zone (UTC) inferred, not yet confirmed |
| Rain-rate model input (pilot) | INSAT HEM | **NCMRWF MERA hourly** (IMERG kept for comparison) | IMERG showed 0–1.5 mm/h at Siyanj during the cloudburst; MERA shows the build-up to 21 mm/h |
| Atmospheric profiles (3 h) | IMDAA / IMDAA-like (NCMRWF) | ERA5 pressure levels (CDS) + single levels incl. surface pressure, orography, CAPE, CIN (ARCO-ERA5), 0.25°, 3-hourly | NCMRWF RDS offers IMDAA only to 2020; IMDAA stays the planned training source for 2017–2020 |
| Terrain | SRTM 30 m | SRTM GL1 30 m (OpenTopography) | as planned |

## Pilot scope

The pilot region (Mandi, Himachal Pradesh, 31.5–32.0 °N, 76.8–77.3 °E) and the replay events are defined in `settings/region.yaml` and validated by `meghprahari.settings.load_region`. Every replay event's download window is returned by `meghprahari.settings.replay_exclusion_windows` so it can be excluded from training and calibration.
