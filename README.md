# MeghPrahari — hyper-local severe-weather nowcasting (SIH26077)

MeghPrahari is a severe-weather nowcasting and early-warning **decision-support prototype** for Smart India
Hackathon 2026, problem statement **SIH26077** (MoES / NCMRWF, Disaster Management). It is designed to estimate
calibrated probabilities of **severe thunderstorms (`ts`)**, **cloudbursts (`cb`)** and **flash floods (`ff`)** over
**0–2 h, 2–4 h and 4–6 h** lead windows for micro-catchments and villages, explain the drivers, and produce
**human-approved CAP 1.2 exercise alerts**.

> **Status: v0.3 code, v0.4 in progress.** Real pilot data (terrain, rain, ERA5 profiles) now loads, but **no model
> has been trained or run on real data yet**; all models so far are trained on synthetic data. The unit tests prove that
> the code behaves as designed; they say nothing about forecast skill. No skill numbers exist yet — they will come
> only from `tools/evaluate.py` on held-out real events (roadmap task H4).

The plan for v0.4 lives in [`documentation/plan/`](documentation/plan/): [roadmap and task IDs](documentation/plan/ROADMAP.md),
[real-data setup guide](documentation/plan/DATA_SETUP.md) and [Claude Code session prompts](documentation/plan/CLAUDE_CODE_PROMPTS.md).

## Pipeline

```text
INSAT-3DR HEM rain rate + IMDAA/IMDAA-like profiles + SRTM DEM
      ↓
QC + spatial/temporal alignment
      ↓
satellite + meteorological + terrain features  (FEATURES_V1, 17 features)
      ↓
calibrated probability model per hazard × lead window  (gradient boosting)
      ↓
catchment/village risk + driver explanation
      ↓
alert DRAFT → human approval (four-eyes at the top level) → CAP 1.2 exercise outbox
```

## Which model runs

| Component | State |
|---|---|
| **Gradient boosting** (`backend/meghprahari/model.py`): one calibrated scikit-learn `HistGradientBoostingClassifier` per hazard/lead target | The model the worker and API use today. Trained only on synthetic data so far. |
| **Physics-guided multi-task Transformer** (`backend/meghprahari/mtl_transformer.py`): modality projections, cross-modal attention, temporal Transformer, 3 hazard heads × 3 lead bins | **Implemented, unit-tested and wired in** behind `model.backend: transformer` (`mtl_backend.py`: normalisation, temperature calibration, SHA-256-checked loading; `train_transformer.py`). **Not yet trained on real data** (roadmap G5). |

Both statements are true at once: the repository contains a working Transformer path, but no forecast in the system is
produced by it until it is trained on real data and approved.

## What is implemented

Unit-tested on synthetic fixtures:
- HEM rain-rate QC, trend, acceleration, area growth, phase-correlation storm motion, stall index (`satellite.py`).
- IWV, 3-hour IWV change, moisture-flux convergence, orographic moisture-flux index, K-index, Total Totals,
  low-level–500 hPa bulk shear, regridding (`meteo.py`).
- Terrain: depression filling, D8 flow, accumulation, HAND, stream links, micro-catchments, Kirpich time of
  concentration, refuge map (`terrain.py`).
- Feature assembly, proxy rain-threshold labels, scoring, village evaluation (`pipeline.py`).
- Calibration, Brier / Brier skill, AUC, reliability / ECE, POD / FAR / CSI, SHA-256 model integrity (`model.py`).
- Importance-weighted anomaly drivers — **not SHAP and not causal**.
- "Why this alert?" card (`explain.py`): moisture, instability, lift, storm signal and terrain response rated from the
  alert's own inputs with plain-language text; missing inputs shown as unknown. Served at `/api/explain/{alert_id}`.
- Verification (`verify.py`): Fractions Skill Score, warning lead time, storm onset; baselines (`baselines.py`):
  persistence and a simple uniform-motion extrapolation (a stand-in for pysteps).
- Cost-loss thresholds, hysteresis, actionable lead-time margin (written into every CAP message), CAP 1.2 builder
  (`alerts.py`); drafts validate against the official OASIS CAP 1.2 XSD in tests.
- RBAC, JWT, four-eyes approval, hash-chained audit log (`governance.py`, `store.py`, `database/init.sql`).
- CAPE, CIN and 3-h CIN change (`FEATURES_V2`, from ERA5 single levels) for instability over high terrain.
- Training-table builder (same feature code as live, 10-km neighbourhood labels, replay events excluded) and
  `train.py` with date-based time-ordered splits; `fetch_era5_cds.py` for ERA5 training data (CDS API key needed).
- Exposure per catchment (`exposure.py`, `build_exposure.py`): schools, health facilities, villages, main roads (OSM).
- Alerts in English and Hindi (CAP `<info>` per language; the Hindi text is a draft for review).
- Labels from training-year climatology: cloudburst-type burst = 3-h total ≥ 33 mm within 10 km (P99.9 of 2021–22); flash flood = per-catchment
  rare rain over the catchment's response time (`ff_thresholds.py`). Both are proxies (4-km hourly rain), documented as such.
- Multi-task Transformer backend (`mtl_backend.py`, `train_transformer.py`): training-split normalisation, missing flags,
  temperature calibration, SHA-256-checked save/load; selected with `model.backend` (gradient boosting stays default).
- Replay page `frontend/replay.html` (standalone): hourly rain, villages, ingredient card, exposure and what actually happened;
  bundles from `tools/export_replay.py`. Model scorecard on held-out events: `tools/evaluate_models.py`.
- Settings that refuse to start while any value is `CHANGE_ME`; pilot region and replay events in
  `settings/region.yaml`, validated by `mp.settings.load_region`.

Run on **real data** so far (pilot: Mandi, 30 Jun – 1 Jul 2025 cloudbursts; see [documentation/DATA_REGISTER.md](documentation/DATA_REGISTER.md)):
- `tools/build_atlas.py` on the SRTM GL1 DEM: 1,320 micro-catchments; stream network matches the Beas valley.
- Readers (`ingest.py`, `tools/inspect_inputs.py`) load NASA **GPM IMERG** half-hourly rain (stand-in for MOSDAC HEM
  until MOSDAC access is approved), NCMRWF **MERA** hourly rain (units inferred, see below; now the rain input and the label source) and
  **ERA5** pressure levels (Copernicus CDS) and single levels (surface pressure, orography, CAPE, CIN; from Google's
  public ARCO-ERA5 copy via `tools/fetch_arco_era5.py`). NCMRWF has no 2025 IMDAA-like product.
- `tools/replay_event.py` runs the worker's feature code over the whole event offline: 187 time steps × 1,320
  catchments. Findings: K-index and Total Totals are undefined for ~90 % of rows (the 850 hPa level is below ground over
  high terrain), and IMERG misses the Siyanj cloudburst almost entirely (0–1.5 mm/h where MERA shows ~170 mm in 24 h), so
  IMERG is a weak rain input for cloudbursts. The rain input is therefore **MERA** (`rain_source: mera`); with it the
  Siyanj rain rises from 3 to 21 mm/h over the 5 hours before the 00:30 IST peak. MERA's units (mm/h) and time zone (UTC)
  are **inferred** (source path `APCP-sfc`, lag match with IMERG), not yet confirmed from NCMRWF documentation.
  Rain-feature windows are defined in minutes, so they mean the same at 30-min and 60-min cadence.
  No model has been applied to these features yet.
- `tools/evaluate.py` scores the **baselines** (persistence, extrapolation) on the Mandi event against IMERG and MERA
  → `evidence/evaluation_mandi_2025_06_30.md`. Against MERA, both baselines give **0 min** warning for the night storm
  (onset 30 Jun 16:00 UTC). These are baseline numbers only; MeghPrahari's models are not in the table yet.

Written but **never run on real files**: the MOSDAC HEM configuration, `tools/load_villages.py`,
`tools/build_training_table.py`, `tools/train.py`, the worker loop (`worker.py`). The API (`api.py`) and dashboard
(`frontend/index.html`) have only been exercised with the synthetic demo.

## Not built yet

Cloud-top-temperature features and the TIR1 reader, a trained model (ERA5 training data 2021–2024 downloading),
a pysteps baseline (a simpler extrapolation stands in), model rows in the results tables (the scripts exist; models
are not trained yet), live-mode adapters (MOSDAC HEM, NCUM, radar). See the roadmap. See [the roadmap](documentation/plan/ROADMAP.md) for task IDs and priorities.

## Quick start (development)

Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate                     # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the unit tests:

```bash
make test                                              # = PYTHONPATH=backend python -m unittest discover -s tests -v
```

Without `make` (e.g. Windows PowerShell):

```powershell
$env:PYTHONPATH = "src"; python -m unittest discover -s tests -v
```

The suite has 146 tests. Without PyTorch the 3 Transformer tests are **skipped**, not failed. To run them, install
the CPU build of PyTorch and the AI extras:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ai.txt
python tools/demo_transformer.py                     # synthetic training/inference smoke test; no skill claim
```

Requirements files:

| File | Contents |
|---|---|
| `requirements.txt` | API, worker, science core (incl. xarray, MetPy) — installed in the Docker image |
| `requirements-ai.txt` | PyTorch (CPU build recommended), Captum, SHAP |
| `requirements-eval.txt` | pysteps and matplotlib for baselines and verification plots (on Windows pysteps needs the Microsoft C++ Build Tools, or install it with `conda install -c conda-forge pysteps`) |
| `requirements-dev.txt` | pytest (optional; tests are plain `unittest`), xmlschema (CAP 1.2 schema test) |

## Synthetic demo (UI and approval workflow only)

Needs Docker and a Linux/macOS/WSL shell (the script uses `os.getuid`, which native Windows Python lacks).

```bash
bash tools/deploy_demo.sh
```

It creates `.env` with random secrets, creates the `data/` folders, starts **only** PostgreSQL and the API, and
seeds **synthetic** catchments, forecasts and exercise alerts. It does not start the worker and does not contact
MOSDAC or NCMRWF. Open `http://127.0.0.1:8000` and log in as `approver` / `MeghPrahariDemo123!`.

In the dashboard you can switch hazard/lead target, view synthetic catchment probabilities on the map, review draft
alerts and their drivers, approve an alert, and publish the approved **exercise** CAP XML to the outbox.

For a hosted demo see [RENDER_DEPLOY.md](RENDER_DEPLOY.md) (also synthetic data).

## Real data

Follow [documentation/plan/DATA_SETUP.md](documentation/plan/DATA_SETUP.md). In short:

1. `sh tools/make_data_dirs.sh` creates the git-ignored `data/` layout.
2. Download the DEM, IMDAA-like, HEM and TIR1 data for the pilot box and dates in `settings/region.yaml`, and record
   each dataset in [documentation/DATA_REGISTER.md](documentation/DATA_REGISTER.md).
3. Inspect real files before filling `settings/settings.yaml` — **never guess variable names**:
   `h5ls -r`, `ncdump -h`, or `python tools/describe_file.py <file>` where those tools are missing.
4. Build the atlas (`tools/build_atlas.py`), load villages (`tools/load_villages.py`), build the training table,
   train, and have a **different** scientist approve each calibrated model.
5. Keep `shadow_mode: true` until the go-live gates in [documentation/GOVERNANCE.md](documentation/GOVERNANCE.md) are met.

## Repository layout

```text
backend/meghprahari/        science (pure functions), pipeline, models, alerts, governance, API, worker
tools/       atlas, villages, training table, training, demo seeding, data dirs, file inspection
settings/        settings.yaml (fill every CHANGE_ME), region.yaml (pilot box + replay events), model_v03.yaml
tests/         unittest suite (synthetic fixtures)
frontend/           MapLibre dashboard
database/            PostgreSQL/PostGIS schema
documentation/          plan/, governance, SIH alignment, architecture notes, data register
```

## Documentation

- [documentation/plan/](documentation/plan/) — roadmap, data setup, session prompts
- [documentation/SIH_ALIGNMENT.md](documentation/SIH_ALIGNMENT.md) — what the code implements against the SIH26077 statement
- [documentation/GOVERNANCE.md](documentation/GOVERNANCE.md) — controls and go-live gates
- [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) — v0.2 pipeline (still the production path);
  [documentation/ARCHITECTURE_V03.md](documentation/ARCHITECTURE_V03.md) — Transformer design (not yet wired in)
- [documentation/DATA_REGISTER.md](documentation/DATA_REGISTER.md) — every dataset, its source and licence

## Governance

The model produces probabilities and alert drafts; authorised humans decide what is sent. `shadow_mode: true` makes
every CAP message `Exercise` / `Restricted`, and replay runs are always forced into shadow mode. Read
[documentation/GOVERNANCE.md](documentation/GOVERNANCE.md) before any operational use.
