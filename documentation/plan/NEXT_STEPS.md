# Next steps — handoff note

Kept up to date during work sessions so the team (or a new session) can continue at any time.
**Last updated:** 2026-09-24 (session with Claude Code).

## Where things stand
- All code for data → features → labels → training → evaluation → alerts → replay page is built and tested
  (`make test`, 165 tests). **No model has been trained on real data yet.**
- ERA5 training-data download (2021–2024, Jun–Sep) is at **46/48 files**, stuck: 2 CDS jobs have been "accepted"
  with no `started` timestamp for hours, and two duplicate `fetch_era5_cds.py` processes were found running at once
  (likely why — they collide on the account's queue limit; 62 requests were rejected). Not touched further without
  the user's OK (their call, standing instruction): kill the duplicate process, then re-run
  `python tools/fetch_era5_cds.py --years 2021-2024 --months 6-9 --workers 2` (resumes; also cancel the 2 stuck
  "accepted" jobs on the CDS website — "Your requests" — if they still haven't started).
- MOSDAC bulk download (HEM + TIR1, 4 orders placed and approved) is still blocked: the `/download/` file-browser app
  needs its own separate login (OAuth `client_id=mosdac`, distinct from the `uops` dashboard login) and several
  Playwright automation attempts did not produce a lasting authenticated session. Recommended next step: the user
  installs a browser extension (e.g. "Download Them All") in their own already-logged-in Chrome and bulk-downloads
  each order folder by hand — no further automation attempts planned unless asked.
- 2026-09-24: built and wired up cloud-top-temperature features (roadmap D1/D2: `backend/meghprahari/ctt.py`,
  `pipeline.ctt_fields`/`FEATURES_V3`, `tools/fetch_gridsat.py`, all unit-tested) using NOAA GridSat-B1 (public, no
  login). **Real-data finding: GridSat-B1 coverage over the Mandi pilot box is ~2% — not usable for training.**
  See `backend/meghprahari/ctt.py` docstring and the GridSat row in `documentation/DATA_REGISTER.md`. Do not train on `FEATURES_V3` with
  GridSat-B1; stick to `FEATURES_V2` (current default) until MOSDAC TIR1 (C7) is available.
- 2026-09-24: built live-mode Phase A (roadmap Q1/Q2 — no MOSDAC needed), both run and verified on real data:
  - `tools/live_imerg.py`: live rain feed via GPM IMERG Early Run (~4 h latency), `imerg_nrt:` config,
    `rain_source: imerg_nrt`. Run every ~30 min (Task Scheduler/cron).
  - `tools/live_gfs.py`: live atmospheric-ingredients feed via NOAA GFS 0.25° analysis (~4 h latency), converted to
    the exact ERA5 file layout (units matched, CIN sign-flipped, grid verified identical after a lat-order fix);
    `met_nrt:` config, `met_source: met_nrt` (new `worker.met_section()`, mirrors `rain_section()`). Run every ~3 h.
  - Both feeds ran through the real, unmodified feature pipeline end-to-end. **Not yet done:** actually running
    `worker.py --mode=live` continuously with these two feeds (needs a trained+approved model in the DB, still
    blocked on ERA5 training data — see above), and Q3 (measuring the GFS-vs-ERA5 / IMERG-Early-vs-Final accuracy
    gap on real overlapping data, per architecture design principle 1 — do not assume they're equivalent).
  - `requirements.txt`: added `cfgrib` (GRIB2 reader, prebuilt Windows wheel, no MSVC Build Tools needed) and `requests`.

## Do these in order once ERA5 is complete
1. ~~Recompute label thresholds from the training years only~~ **done 2026-09-23**: cloudburst 33 mm / 3 h
   (P99.9 of 2021–22), flash-flood per-catchment thresholds from 2021–22 (`data/atlas/ff_thresholds.csv`).
2. Build the training table + Transformer sequences:
   `python tools/build_training_table.py --out data/work/train_table.csv.gz --start 2021-06-01 --end 2024-09-30 --seq-out data/work/seq.npz`
3. Train gradient boosting (splits: train ≤ 2022-09-30, calibrate ≤ 2023-09-30, test 2024):
   `python tools/train.py --table data/work/train_table.csv.gz --train-end 2022-09-30 --cal-end 2023-09-30 --out-dir data/models`
4. Train the Transformer (same splits):
   `python tools/train_transformer.py --seq data/work/seq.npz --train-end 2022-09-30 --cal-end 2023-09-30 --out-dir data/models`
5. Results on the held-out events (never used in training):
   `python tools/evaluate_models.py --event mandi_2025_06_30 --models-dir data/models --transformer data/models/<mtl file>.pt`
   and the same for `--event padhar_2024_07_31` (needs ERA5 Jul–Aug 2024, part of the download).
6. Replay page with probabilities: `python tools/export_replay.py --event mandi_2025_06_30 --models-dir data/models`,
   then open `frontend/replay.html`.
7. Fill the results slot in `documentation/pitch/DEMO_SCRIPT.md` with numbers from `evidence/models_*.md` only.
8. Model card: `python tools/model_card.py --models-dir data/models` (writes `documentation/MODEL_CARD.md` from the metrics
   files), then tune alert thresholds (J1) from the results.

## Needs a person
- MOSDAC approval → HEM + TIR1 download (cloud-top-temperature features C7/D1/D2; live rain feed).
- Native Hindi speaker to review `backend/meghprahari/alert_text.py` (Hindi text is a draft).
- Optional: MERA Documentation tab screenshot (units inferred as mm/h, time zone as UTC).
- Optional: DEM extended ~1 km north (to 32.02 °N) so Rajban/Terang (2024 event) get village alerts; rebuild the atlas.

## Later (see documentation/plan/ROADMAP.md for all task IDs)
Live adapters (MOSDAC HEM/TIR, NCUM forecasts, radar), hydrological flash-flood thresholds (E6), pysteps baseline (G3),
Docker build test with `--build-arg WITH_AI=1` (M2), deployment with the real replay (M3), pitch deck (P2), backup
video (P4), DB backup/restore test (M6).

## Important facts learned (keep in the pitch)
- Persistence and extrapolation give **0 min** warning for the Mandi night burst (onset 30 Jun 2025 ~16–17 UTC).
- IMERG (10 km) misses the Siyanj cloudburst; MERA (4 km, radar) shows a 5-h build-up → MERA is the rain input.
- K-index/Total Totals undefined for ~90 % of the pilot area (850 hPa below ground) → CAPE/CIN used instead.
- MERA cannot resolve true cloudburst rates; the cloudburst label is a documented proxy.
