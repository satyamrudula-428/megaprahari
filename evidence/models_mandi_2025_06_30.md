# Model results — Mandi cloudbursts (mandi_2025_06_30), held-out event

Generated 2026-09-23 17:06 UTC by `tools/evaluate_models.py`. 88 hourly anchors × 1320 catchments. Decision threshold p ≥ 0.1. Labels: settings `labels` (cloudburst = 3-h total ≥ 40.0 mm within 10.0 km; flash flood = per-catchment rare-rain proxy).

One event is not a climatology; these numbers describe this case only.

| target | method | n | positives | Brier | BSS | AUC | POD | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|
| ts_0_2 | persistence (last 2 h) | 113520 | 5861 | 0.0621 | -0.27 | 0.69 | 0.42 | 0.60 | 0.26 |
| ts_2_4 | persistence (last 2 h) | 113520 | 6270 | 0.0930 | -0.78 | 0.55 | 0.15 | 0.85 | 0.08 |
| ts_4_6 | persistence (last 2 h) | 113520 | 6813 | 0.1104 | -0.96 | 0.49 | 0.03 | 0.97 | 0.02 |
| cb_0_2 | persistence (last 2 h) | 113520 | 2115 | 0.0172 | 0.06 | 0.77 | 0.54 | 0.46 | 0.37 |
| cb_2_4 | persistence (last 2 h) | 113520 | 2115 | 0.0316 | -0.73 | 0.57 | 0.15 | 0.85 | 0.08 |
| cb_4_6 | persistence (last 2 h) | 113520 | 2115 | 0.0373 | -1.04 | 0.49 | 0.00 | 1.00 | 0.00 |
| ff_0_2 | persistence (last 2 h) | 99330 | 839 | 0.0109 | -0.30 | 0.68 | 0.37 | 0.64 | 0.23 |
| ff_2_4 | persistence (last 2 h) | 99330 | 829 | 0.0167 | -1.02 | 0.51 | 0.02 | 0.98 | 0.01 |
| ff_4_6 | persistence (last 2 h) | 99330 | 829 | 0.0171 | -1.06 | 0.50 | 0.00 | 1.00 | 0.00 |

## Warning lead time for the cloudburst-type burst

Observed onset (first anchor whose next 2 h contain a burst anywhere in the atlas): **2025-06-30 17:00 UTC**.

| method | first warning (any cb lead bin, any catchment) | lead time |
|---|---|---|
| persistence (last 2 h) | – | miss |
