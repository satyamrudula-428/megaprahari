# MeghPrahari architecture — v0.2

```text
INSAT-3DR HEM rain-rate files ─┐
                              ├─> QC / alignment ─> feature extraction ─┐
IMDAA / IMDAA-like profiles ─┤                                        │
                              │                                        ▼
SRTM DEM ─> atlas/catchments ─┘                             calibrated target models
                                                                       │
                                      ┌────────────────────────────────┼─────────────────────────┐
                                      │                                │                         │
                                      ▼                                ▼                         ▼
                             TS: 0–2/2–4/4–6                 CB: 0–2/2–4/4–6          FF: 0–2/2–4/4–6
                                      │                                │                         │
                                      └────────────────────────────────┼─────────────────────────┘
                                                                       ▼
                                                           catchment/village risk
                                                                       │
                                                                       ▼
                                                            explainable drivers
                                                                       │
                                                                       ▼
                                                                  alert draft
                                                                       │
                                                                       ▼
                                                               human approval
                                                                       │
                                                                       ▼
                                                                 CAP 1.2 outbox
```

## Current model architecture

Each hazard/lead target is trained independently with `HistGradientBoostingClassifier`, followed by isotonic calibration when enough positive calibration examples exist. The model is time-split into training, calibration and test periods.

This is a deliberate v0.2 engineering choice: it is lightweight, auditable and practical for a pilot. A future neural multi-task model can be added only after the real-data training table and event labels are established.

## Current feature groups

- HEM: current rain, recent maximum, acceleration, spatial growth, motion speed, stall duration.
- Meteorology: IWV, 3-hour IWV change, moisture-flux convergence, orographic moisture flux, 850/500-hPa bulk shear, K-index, Total Totals.
- Terrain: slope, HAND, time of concentration and upstream catchment area.

## Demo mode

`deploy_demo.sh` starts PostgreSQL/PostGIS and FastAPI, then seeds synthetic models, catchments, forecasts and exercise alerts. It does **not** start the worker and it does not contact MOSDAC/NCMRWF. This is intended only to demonstrate the UI and approval workflow.
