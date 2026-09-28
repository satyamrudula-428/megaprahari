# Roadmap — MeghPrahari v0.4

Principle: **get one real event through the whole pipeline first, then make each stage smarter.**
Priority: **[M]** must · **[S]** should · **[N]** nice. Drop [N] first, then [S], never [M].
Critical path: A3–A6 → C1–C7 → D1–D4 → F1–F4 → G1–G5 → H4 → L1 → P3

Status legend: ☐ todo · ◐ in progress · ☑ done

---

## Phase 1 — Walking skeleton on real data
**Done when:** the dashboard shows real Mandi catchments with probabilities computed from real 30 Jun 2025 data.

| ID | Pri | Task | Status |
|---|---|---|---|
| A1 | M | Fix pilot box (31.5–32.0 N, 76.8–77.3 E) in `settings/region.yaml` | ☑ |
| A2 | M | Fix replay events — Mandi 30 Jun–1 Jul 2025 and Padhar/Chauhar 31 Jul–1 Aug 2024 confirmed (held out of training and calibration); Dharali candidate (outside box) | ◐ |
| A3 | M | Download DEM for pilot box — SRTM GL1, `data/inputs/mandi_dem.tif` | ☑ |
| A4 | M | Download IMDAA-like for event dates — NCMRWF RDS has no 2025 pressure levels; ERA5 pressure levels (CDS) + single levels (ARCO-ERA5) instead | ☑ |
| A5 | M | Order MOSDAC HEM for event dates | ☐ |
| A6 | M | Order MOSDAC TIR1 for event dates | ☐ |
| A11 | M | Villages CSV for pilot region — `data/inputs/villages.csv`: 31 places from OSM (top 30 by stream size + Mandi); district/population blank, fill from Census later (optional, for exposure) | ◐ |
| A15 | M | Data register (source, dates, licence) — template in `documentation/DATA_REGISTER.md`, fill at download | ◐ |
| B1 | M | Rewrite README for v0.4, remove contradictions, fix demo commands | ☑ |
| B2 | M | Make Transformer test run under unittest; all tests pass | ☑ |
| B3 | M | Requirements: torch (CPU), metpy, xarray, pysteps, shap, captum — pysteps needs MSVC Build Tools/conda on Windows | ☑ |
| B4 | M | GitHub repo, branch rules, PR template — PR template added; repo + branch rules need a GitHub account | ◐ |
| B5 | S | GitHub Actions CI (tests + lint) — workflow added, not yet run on GitHub | ◐ |
| B6 | M | `data/` folder structure script | ☑ |
| B7 | S | `settings/region.yaml` | ☑ |
| C1 | M | Inspect real HEM files, fill `hem` settings | ☐ |
| C2 | M | Inspect real IMDAA-like files, fill `met` settings — ERA5 used for 2025 (no IMDAA-like 2025); `met` fully filled from inspected files | ☑ |
| C3 | M | GRIB → NetCDF conversion step — not needed so far: ERA5, MERA delivered as NetCDF, IMERG as HDF5 | ☑ |
| C4 | M | RH → specific humidity (`humidity.py`) — not needed: ERA5 provides specific humidity (kg/kg) | ☑ |
| C5 | M | Test `read_hem_raw` on real files | ☐ |
| C6 | M | Test `read_met_state` on real files — `read_met_states`/`MetIndex` build MetState for all 64 real ERA5 times (ps 534–972 hPa) | ☑ |
| C10 | M | Time alignment (30 min sat vs 3 h met) — `pipeline.select_met_times` (tested), used by worker | ☑ |
| C11 | M | Spatial alignment / regridding on real grids — nearest-cell sampling per catchment tested on IMERG/ERA5/MERA grid orientations | ☑ |
| E1 | M | Run `build_atlas.py` on real DEM — mandi_v1: 1320 catchments, void 0, 24 s | ☑ |
| E2 | M | Visual QA of catchments in QGIS — quick matplotlib check OK (Beas path matches); full QGIS check pending | ◐ |
| E3 | M | Calibrate `safe_hand_m` for hills | ☐ |
| E4 | M | Load villages → catchments | ☐ |
| M1 | M | Worker runs end-to-end on real replay data (GBM, proxy labels) — worker reads IMERG + ERA5; `tools/replay_event.py` produced 246,840 real feature rows (187 anchors × 1,320 catchments); needs trained models + DB for forecasts | ◐ |

## Phase 2 — Complete the science
**Done when:** results table compares GBM vs pysteps vs persistence on held-out real events.

| ID | Pri | Task | Status |
|---|---|---|---|
| A7 | S | Order WV channel + sounder precipitable water | ☐ |
| A8 | S | Download IMERG (events + training years) — event done: 192 half-hourly files 29 Jun–2 Jul 2025; training years pending. Also MERA hourly 4 km rain (NCMRWF) for the event in `data/landing/mera/` | ◐ |
| A9 | M | Download training data — MERA Jun–Sep 2021–2024 done (11,712 files); ERA5 2021–2024 via CDS API running (`fetch_era5_cds.py`, harvests finished jobs) | ◐ |
| A10 | S | ERA5 CAPE/CIN fallback — in `data/landing/era5_sl/` (ARCO-ERA5); not yet a model feature (D3) | ☑ |
| C7 | M | TIR1 reader → brightness temperature (`tir.py`) | ☐ |
| C8 | S | IMERG reader (`imerg.py`) — done via `read_hem_raw` (axis_order lon,lat + bbox crop) and `imerg:` settings; real files load | ☑ |
| C9 | S | WV / TPW readers | ☐ |
| C13 | S | IMERG fallback when HEM missing | ☐ |
| D1 | M | CTT drop rate — `ctt.ctt_drop_rate` + `pipeline.ctt_fields`/`FEATURES_V3`, unit-tested; real cadence is 3 h (GridSat-B1), not the 15–30 min originally envisaged for live INSAT TIR1 (needs C7) | ◐ |
| D2 | M | CTT min + cold-cloud area (< ~221 K) — `ctt.ctt_min`/`cold_cloud_area_km2`, unit-tested and reads real GridSat-B1 files correctly, BUT real coverage over the Mandi pilot box is ~2% (checked on 5 real files incl. the event's peak hour, which was 100% missing) — not usable for training until C7 (MOSDAC TIR1) or a better-coverage source; see `backend/meghprahari/ctt.py` docstring | ◐ |
| D3 | M | CAPE, CIN into `FEATURES_V2` — from ERA5 single levels via `met.surface_vars`; defined over high terrain (CAPE 0 % missing on the Mandi replay) | ☑ |
| D4 | M | CIN trend — `met_d_cin_3h` (negative = lid eroding) | ☑ |
| D5 | S | Satellite IWV, cross-check with reanalysis | ☐ |
| D6 | S | IWV anomaly vs monsoon climatology | ☐ |
| D7 | M | Verify existing features on real data — replay done: K-index/Total Totals missing for 90 % (850 hPa below ground over high terrain); IMERG misses the Siyanj cloudburst, MERA captures the 5-h build-up (now the rain input); stall index unrealistic when cells barely move | ◐ |
| D8 | M | Normalisation stats from train split only — `mtl_backend.standardize_fit` + missing-value flags | ☑ |
| D9 | S | pysteps extrapolated rain as input feature | ☐ |
| D10 | M | Unit tests for all new features | ☐ |
| F1 | M | Event ledger — 13 places: Mandi 2025 (news) + Padhar 2024 (HPSDMA via SANDRP); clock times not reported | ◐ |
| F2 | M | Labelling rules — ts ≥ 10 mm/h; cb = 3-h total ≥ 33 mm (P99.9 of 2021–22 MERA, the training years; proxy, 4-km hourly data cannot resolve 100 mm/h); 10-km neighbourhood | ☑ |
| F3 | M | `ff` threshold — per-catchment P99.9 of Tc-window totals from training-year MERA (`ff_thresholds.py`); rare-rain proxy, hydrological thresholds remain E6 | ☑ |
| F4 | M | Build real training table — builder rewritten (MERA + ERA5, same feature code as live); smoke run on non-event 2025 days OK; waits for 2021–24 data | ◐ |
| F5 | M | Class imbalance — negative subsampling with inverse-probability weights + isotonic calibration on natural rates | ☑ |
| F6 | M | Time + region splits, replay events excluded — date-based splits in `train.py`; builder drops anchors touching replay windows; `fetch_era5_cds.py` refuses replay months | ☑ |
| G1 | M | Retrain GBM on real data — `train.py` trains all 9 targets, skips targets without both classes; waits for training data | ◐ |
| G2 | M | Persistence baseline (`baselines.py`) — done; scored on the Mandi event in `evidence/` | ☑ |
| G3 | M | pysteps baseline — pysteps not installable on Windows without build tools; simple uniform-motion extrapolation stand-in in `baselines.py` | ◐ |
| H1 | M | POD, FAR, CSI, Brier, BSS, AUC, reliability/ECE — `model.py` (existing) used by `tools/evaluate.py` | ☑ |
| H2 | M | Fractions Skill Score — `verify.fss` | ☑ |
| H3 | M | Median lead time per event — `verify.warning_lead_min`, `storm_onset`, `lead_time_summary` | ☑ |
| H4 | M | Results table — `evaluate.py` (baselines on rain) + `evaluate_models.py` (models vs persistence on the same labels, lead time); model rows once trained | ◐ |

## Phase 3 — AI engine
**Done when:** worker runs the Transformer on real replay data; results table has a Transformer row.

| ID | Pri | Task | Status |
|---|---|---|---|
| G4 | M | `model.backend` switch — settings `model.backend: gbm | transformer`; worker loads an approved 'mtl' Transformer or per-target GBMs | ☑ |
| G5 | M | Train Transformer on real data — `train_transformer.py` (sequences from `build_training_table --seq-out`, early stopping) tested; waits for ERA5 training data | ◐ |
| G6 | M | Temperature-scaling calibration — per target on the calibration split (`mtl_backend.fit_temperature`) | ☑ |
| G7 | M | Transformer save/load with SHA-256 — `mtl_backend.save/load_transformer` (hash check + torch weights_only); uncalibrated refused | ☑ |
| G8 | S | Spatial encoder (patches / U-Net) | ☐ |
| G9 | S | Physics-guided loss constraints | ☐ |
| G10 | S | Uncertainty (ensemble / MC dropout) | ☐ |
| G11 | S | Risk-adaptive anomaly gate with real thresholds | ☐ |
| G12 | N | ONNX export | ☐ |
| H5 | S | Ablation study (CTT, CAPE/CIN, terrain) | ☐ |
| H6 | M | Latency, memory, model size | ☐ |
| H7 | S | Plots for slides | ☐ |

## Phase 4 — Impact and explainability
**Done when:** every alert shows ingredients, villages/people at risk, minutes to act, local-language text.

| ID | Pri | Task | Status |
|---|---|---|---|
| I1 | M | Three-ingredients card — `explain.py`, stored with each draft, `/api/explain/{alert_id}` | ☑ |
| I2 | S | SHAP for GBM | ☐ |
| I3 | S | Integrated gradients / attention for Transformer | ☐ |
| I4 | M | Plain-language explanation text — in the ingredient card | ☑ |
| A12 | S | OSM roads, schools, hospitals | ☐ |
| A13 | S | WorldPop population | ☐ |
| A14 | S | CWC gauge data | ☐ |
| E5 | S | SCS Curve Number runoff (`runoff.py`) | ☐ |
| E6 | S | Catchment flash-flood thresholds | ☐ |
| E7 | S | Validate vs CWC gauges | ☐ |
| E8 | S | Exposure layer (`exposure.py`) — schools, health facilities, villages, main-road km per catchment from OSM (`build_exposure.py`); OSM under-maps rural schools | ☑ |
| J1 | M | Tune cost-loss thresholds + hysteresis on real data | ☐ |
| J2 | M | CAP XSD validation — all 36 draft variants validate against the OASIS CAP 1.2 XSD (`tests/test_cap_schema.py`) | ☑ |
| J3 | S | SACHET-aligned CAP profile | ☐ |
| J4 | S | Hindi / Telugu / English templates — English + Hindi (draft, needs review) as CAP info blocks; Telugu pending decision | ◐ |
| J5 | S | Mock SMS / WhatsApp / IVR screens | ☐ |
| J6 | M | Actionable margin in every alert — CAP parameter `actionable_margin_min` + plain sentence; 'unknown' when no refuge | ☑ |
| K1 | N | ESP32 rain gauge prototype | ☐ |
| K2 | N | Gauge ingest endpoint | ☐ |
| K3 | S | Citizen report form (`/observe`) | ☐ |
| K4 | S | Gauge/citizen data into verification | ☐ |

## Phase 5 — Demo and polish
**Done when:** anyone on the team runs the 3-minute demo from a fresh laptop.

| ID | Pri | Task | Status |
|---|---|---|---|
| L1 | M | Replay mode with time slider — `frontend/replay.html` + `export_replay.py` (standalone, no DB); probabilities appear once models exist | ◐ |
| L2 | M | Hazard × lead selector, clear legend | ☐ |
| L3 | M | Ingredient panel — per village and hour in the replay page | ☑ |
| L4 | M | 'What actually happened' overlay — event-ledger places and observed MERA rain in the replay page | ☑ |
| L5 | M | Baseline comparison view | ☐ |
| L6 | S | Exposure panel — schools, health facilities, road km per selected village's catchment | ☑ |
| L7 | S | Action-deadline countdown | ☐ |
| L8 | S | Terrain/catchment layers | ☐ |
| L9 | S | Mobile layout — replay page stacks map and panel below 820 px; not tested on devices | ◐ |
| L10 | N | Low-bandwidth mode | ☐ |
| M2 | M | Docker image with CPU torch — `docker build --build-arg WITH_AI=1`; not yet built/tested here | ◐ |
| M3 | M | Render deployment with real replay | ☐ |
| M4 | S | Monitoring (data age, missing files) — `monitor.py` + `/api/status` from the ingest ledger | ☑ |
| M5 | S | Degraded mode + dashboard warning — red banner on the dashboard when an input is stale; no automatic fallback source yet | ◐ |
| M6 | S | DB backup/restore test | ☐ |
| Q1 | M | Live rain feed (Phase A, no MOSDAC) — `tools/live_imerg.py` downloads GPM IMERG Early Run (~4 h latency, no login beyond existing Earthdata), verified on a real file to have the same layout as the archival Final Run; `imerg_nrt:` config section; `rain_source: imerg_nrt` for live mode | ☑ |
| Q2 | M | Live atmospheric-ingredients feed (Phase A, no MOSDAC/NCUM) — `tools/live_gfs.py` downloads+converts NOAA GFS 0.25deg analysis (~4 h latency) to the same NetCDF layout as the archival ERA5 files; verified on a real file: identical grid to ERA5 (no regridding), units match except CIN sign (GFS negative, flipped to match); `met_nrt:` config section; `worker.met_section()`/`met_source: met_nrt` for live mode | ☑ |
| Q3 | S | Cross-check GFS vs ERA5 and IMERG Early Run vs Final Run on real overlapping data (architecture design principle 1: the accuracy gap must be measured, not assumed) — not done yet | ☐ |
| Q4 | M | MOSDAC standing order / API for live HEM + TIR1 (Phase B; task referenced as "C12"/`mdapi.py` in `documentation/plan/DATA_SETUP.md` but never added here) — blocked on the same `/download/` portal auth problem as the archival bulk download | ☐ |
| N1 | M | Update `SIH_ALIGNMENT.md` | ☐ |
| N2 | M | Data sources + licences doc | ☐ |
| N3 | M | Model card | ☐ |
| N4 | M | v0.4 architecture diagram — `documentation/plan/ARCHITECTURE.md` (v1.0 target design mapped to the code) | ☑ |
| N5 | S | Shadow mode + go-live gates doc | ☐ |
| N6 | S | Verification log | ☐ |
| O1 | M | Integration test: real files → alert — `tests/test_integration_real.py` (real MERA/ERA5/atlas → features → CAP draft, XSD-valid; runs where data/ exists) | ☑ |
| O2 | M | Replay reproducibility test — export_replay twice gives identical bundles (except timestamp) | ☑ |
| O3 | S | API load test | ☐ |
| O4 | M | Full smoke test before every demo | ☐ |
| P1 | M | SIH submission requirements | ☐ |
| P2 | M | Pitch deck | ☐ |
| P3 | M | 3-minute demo script — draft `documentation/pitch/DEMO_SCRIPT.md`; results slot to fill after training | ◐ |
| P4 | M | Backup demo video | ☐ |
| P5 | S | Business/impact slide | ☐ |
| P6 | M | Judge Q&A prep — draft `documentation/pitch/JUDGE_QA.md` | ◐ |
| P7 | S | One-page summary | ☐ |

---

## Risks and fallbacks
| Risk | Fallback |
|---|---|
| MOSDAC approval slow | Build with IMERG + ERA5 first |
| No INSAT data on event date | Switch to backup event |
| No specific humidity in IMDAA-like | RH → q conversion (C4) |
| Too few events | Pool Himalayan districts; rain-threshold labels |
| Transformer doesn't beat GBM | Report honestly; ensemble; focus on system |
| Training slow | Smaller model / fewer years / Colab GPU |
| Live demo fails | Backup video + local replay |
