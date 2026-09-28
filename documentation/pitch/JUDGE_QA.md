# Likely judge questions and honest answers (task P6)

**How accurate is it?**
Only quote `evidence/models_*.md` (held-out events: Mandi 30 Jun 2025 and Padhar 31 Jul 2024, neither used in
training). Always compare with persistence ("what happened in the last 2 hours repeats"), which gives zero minutes of
warning for the Mandi burst. If numbers are not in the report yet, say the model is being trained and show the baseline.

**Can you really predict a cloudburst?**
A true cloudburst (~100 mm/h over a few km²) cannot be observed directly with 4 km hourly data. We predict
*extreme short-duration rain* (a 3-hour total of 33 mm or more within 10 km; the rarest 0.1 % in the 2021–22 training years) plus
flash-flood-level rain for each catchment's response time. These are proxies, and the documents say so.

**Why not just use IMD or NWP forecasts?**
NWP models are good at the atmosphere a few hours ahead but coarse (12–25 km). MeghPrahari combines their ingredients
(moisture, instability, lift) with storm tracking and each catchment's terrain, and turns the result into a village-level
decision with the time left to act. In live mode, NCMRWF's NCUM forecasts are an input, not a competitor.

**What about false alarms?**
Thresholds come from each village's cost/loss ratio and are set by the authority; escalation needs repeated signals
(hysteresis). Every alert is reviewed by a human, and we report the false-alarm ratio (FAR) with every result.

**Where does the data come from, and is it available live?**
Training and replay: NCMRWF MERA rain, ERA5 (Copernicus), SRTM terrain, OpenStreetMap. Live: MOSDAC INSAT HEM and TIR
(access requested), NCUM forecasts and IMD radar, all through the same feature adapters. ERA5 and MERA arrive too late
for live use; we say so in the architecture.

**Why a Transformer?**
It learns across hazards and time jointly. We keep calibrated gradient boosting as the baseline and fallback, and the
Transformer is used only if it beats it on held-out events. Both are calibrated so a 30 % means about 30 %.

**How does it scale beyond Mandi?**
The terrain atlas, features and models are per region; a new district needs its DEM and villages (hours, not months).
Everything runs on CPU on one state server.

**Is it safe to deploy?**
It runs in shadow mode: every CAP message is an Exercise. Going live needs an authority agreement, a full monsoon of
shadow verification, and the go-live gates in `documentation/GOVERNANCE.md`.

**What is not done yet?** (answer openly)
Cloud-top-temperature features (need MOSDAC TIR), live adapters, hydrological flood thresholds, and a pysteps baseline.
