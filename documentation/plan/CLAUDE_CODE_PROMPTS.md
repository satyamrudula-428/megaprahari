# Claude Code session prompts — MeghPrahari

Paste one prompt per session. Always one task group at a time. Review the plan Claude proposes before approving.

---

## 0. First session — orientation (no code changes)
```
Read CLAUDE.md and every file in documentation/plan/. Then read README.md, documentation/ARCHITECTURE_V03.md,
documentation/SIH_ALIGNMENT.md and all modules in backend/meghprahari/.
Run `make test` and report the result.
Give me: (1) a short map of how data flows through the code, (2) any mismatch you find between the code
and documentation/plan/ARCHITECTURE.md, (3) risks you see for Phase 1. Do not change any files yet.
```

## 1. Phase 1 — foundations (B1–B3, B6, B7)
```
Work on tasks B2, B3, B6, B7 from documentation/plan/ROADMAP.md.
- B2: make tests/test_mtl_transformer.py run under unittest and skip cleanly when torch is not installed.
- B3: update requirements files (keep torch in requirements-ai.txt, CPU build note).
- B6: add tools/make_data_dirs.sh creating the data/ layout in documentation/plan/DATA_SETUP.md.
- B7: add settings/region.yaml with the pilot box and replay events; load it via settings.py with validation.
Propose a plan first. All tests must pass.
```
```
Work on task B1: rewrite README.md for v0.4 (current state = v0.3 code + plan). Remove the contradiction about
the Transformer, fix the demo commands, link documentation/plan/. Be strictly honest about what is and is not verified.
```

## 2. Phase 1 — real data readers (C1–C6, C10, C11)
Run first yourself and paste the output into the prompt: `h5ls -r <hem file>` and `ncdump -h <met file>`
(on Windows without HDF5/netCDF tools: `python tools/describe_file.py <file>`).
```
Here is the structure of our real files:
<paste h5ls output>
<paste ncdump -h output>
Tasks C1, C2, C4, C5, C6: fill settings/settings.yaml from these exact names, add backend/meghprahari/humidity.py
(RH→specific humidity with MetPy) if needed, and make read_hem_raw / read_met_state work on these files.
Add a script tools/inspect_inputs.py that loads one of each file and prints shapes, ranges and times.
Never guess names that are not in the output above.
```
```
Tasks C10, C11: implement and test time alignment (30-min satellite vs 3-hourly met, respecting max_age_h)
and spatial alignment on the real grids. Add unit tests with synthetic grids of the same shapes.
```

## 3. Phase 1 — terrain + end-to-end (E1–E4, M1)
```
Tasks E1–E4: run tools/build_atlas.py on data/inputs/mandi_dem.tif (atlas id mandi_v1), fix any real-DEM
problems, then make tools/load_villages.py work with data/inputs/villages.csv. Report catchment counts,
void fraction and any skipped villages.
```
```
Task M1: add tools/replay_event.py that runs the worker pipeline over the files in data/landing for a
given time window, using the GBM backend and proxy rain-threshold labels, and writes forecasts to the DB so
the dashboard shows real Mandi catchments. Keep shadow mode and hindcast flags forced on.
```

## 4. Phase 2 — new features (C7, D1–D4, D10)
```
Tasks C7, D1, D2: add backend/meghprahari/tir.py (TIR1 reader → brightness temperature using the calibration in the
file; show me the relevant file attributes first), CTT minimum, CTT drop rate over 30 min (K per 30 min),
and cold-cloud area below a configurable threshold (default 221 K). Unit tests required.
```
```
Tasks D3, D4, D8: add CAPE, CIN and CIN trend (3 h) to a new FEATURES_V2 in pipeline.py using
meteo.cape_cin_metpy, keep FEATURES_V1 for backward compatibility, and compute normalisation stats from the
training split only. Fail loudly if inputs are missing — never fill fake values.
```

## 5. Phase 2 — labels, baselines, evaluation (F1–F6, G1–G3, H1–H4)
```
Tasks F2–F6: update tools/build_training_table.py for FEATURES_V2, event-ledger labels
(data/ledger/events.csv) combined with rain-threshold labels, class weights, and time-ordered splits
(train 2017–2019, calibrate 2020, test 2021+) that exclude the replay events listed in settings/region.yaml.
```
```
Tasks G2, G3, H1–H4: add backend/meghprahari/baselines.py (persistence, pysteps extrapolation) and tools/evaluate.py
producing POD, FAR, CSI, Brier, BSS, AUC, ECE, Fractions Skill Score and median lead time per event for
every model, as a markdown table + plots in evidence/. Then retrain GBM (G1) and run the evaluation.
```

## 6. Phase 3 — Transformer (G4–G7, H6)
```
Tasks G4, G7: add a model.backend switch (transformer | gbm) used by worker.py and pipeline.py. Implement
Transformer save/load as a state_dict + config with SHA-256 integrity, and keep the existing model approval
and calibrated-only rules. GBM must keep working.
```
```
Tasks G5, G6, H6: add tools/train_transformer.py (sequence dataset [N, 12, F] from the training table,
masked multitask BCE, early stopping on the calibration split, temperature scaling), register the model,
run tools/evaluate.py and measure latency / memory / size.
```

## 7. Phase 4 — explain, impact, alerts (I1, I4, E8, J1, J2, J4, J6)
```
Tasks I1, I4: add backend/meghprahari/explain.py producing a three-ingredients card (moisture, instability, lift,
storm signal, terrain response) with plain-language text for each alert, and expose it at /api/explain.
```
```
Tasks E8, J6, J2, J4: add backend/meghprahari/exposure.py (villages, population, facilities per catchment), include
actionable minutes in each alert, validate CAP output against the OASIS CAP 1.2 XSD in tests, and add
Hindi/Telugu/English alert text templates.
```

## 8. Phase 5 — demo (L1–L5, M2, M3, O1, O2)
```
Tasks L1–L5: extend frontend/index.html with a replay time slider, hazard × lead selector, ingredient panel,
"what actually happened" overlay (from the event ledger / observed rain) and a baseline comparison view.
Keep it dependency-light and mobile-friendly.
```
```
Tasks M2, M3, O1, O2: CPU-torch Docker image, Render deployment that seeds the real replay instead of
synthetic data, an integration test from real files to CAP alert, and a replay reproducibility test.
```

---

## Useful one-liners
- "Before changing anything, list the files you will touch and why."
- "Run make test and fix failures without weakening any check."
- "Update documentation/SIH_ALIGNMENT.md to reflect what this change implemented."
- "Give me a commit message in the form `<TASK-ID>: description`."
