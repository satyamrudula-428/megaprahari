# MeghPrahari — 3-minute demo script (task P3)

**Setup before the demo:** open `frontend/replay.html?event=mandi_2025_06_30` (served by the API at `/replay.html`, or any
static server in `frontend/`), plus `evidence/models_mandi_2025_06_30.md` in a second tab. Have the backup video ready (P4).
Keep the SHADOW MODE banner visible: everything shown is an exercise / hindcast.

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:25 | Title / map of Mandi | "On the night of 30 June 2025, a series of cloudbursts hit Mandi in Himachal. In Siyanj, two houses were swept away and nine people went missing. Standard forecasting methods gave **zero minutes** of warning. MeghPrahari is built to change that, village by village." |
| 0:25–0:55 | Replay page, slider at 30 Jun 14:00 UTC | "This is a replay of that night using real data: NCMRWF's MERA rain, ERA5 weather and a 30 m terrain map of 1,320 micro-catchments. The dashed box is our pilot area; red circles are the places the news reported as hit." |
| 0:55–1:35 | Press Play from 14:00 to 20:00 UTC | "Watch the rain build over Pandoh–Siyanj–Thunag: 3, 5, 8, 12, then 21 mm per hour at 00:30 IST. The storm grows for five hours before the peak and barely moves. That build-up is exactly what a nowcast can catch." |
| 1:35–2:10 | Click **Pandoh** | "For every village we show *why*: moist air, low-level convergence, moist wind pushed up the slopes, and a storm that is growing and stalling. Plus what is exposed (schools, health centres, roads) and how fast water reaches the stream." |
| 2:10–2:35 | Results report tab | "Held-out results: *[fill from evidence/models_mandi_2025_06_30.md after training: model vs persistence, lead time]*. The event was never used for training." |
| 2:35–3:00 | Alert / governance | "Alerts are drafts. A district officer approves them (two people for the highest level), they are issued in the official CAP format in Hindi and English, with the minutes left to act, and every step goes into a tamper-evident audit log. Nothing goes public automatically." |

**Rules for the speaker**
- Only quote numbers that appear in `evidence/`. Never say "accurate" or "predicts cloudbursts" without the number.
- Say "extreme short-burst rain" when talking about the cloudburst label (4 km hourly data cannot see 100 mm/h at a point).
- If the model has not beaten persistence on the held-out event, say so, and show the system, explanation and governance.
