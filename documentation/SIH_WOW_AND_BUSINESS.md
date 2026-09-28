# MeghPrahari v0.3 — SIH Differentiator

## The product idea

**Risk-Adaptive Weather Intelligence:** compute follows atmospheric risk.

Instead of applying the most expensive inference uniformly to every location,
the system performs a cheap screening pass and activates the multimodal
Transformer for anomalous cells/catchments.

### Normal state

weather signals stable
→ cached/low-cost features
→ no expensive Transformer pass

### Anomaly state

rapid IWV change
+ rainfall acceleration
+ convergence / instability signal
→ high-resolution Transformer
→ multi-hazard probabilities
→ catchment impact assessment
→ alert draft

## The user-facing "wow" panel

The dashboard should answer four questions:

1. **What is likely?**
   - thunderstorm / cloudburst / flash flood probability

2. **When?**
   - 0–2 / 2–4 / 4–6 hour window

3. **Why?**
   - IWV anomaly, instability, convergence, CTT trend, rainfall dynamics,
     terrain response

4. **Who/what is exposed?**
   - villages, population, infrastructure and estimated action window

The product therefore converts a weather probability into a decision-support
object instead of stopping at a weather map.

## Business model

MeghPrahari can be packaged as a weather-risk intelligence platform for:

- state/district disaster management;
- municipalities;
- infrastructure operators;
- utilities;
- transport corridors;
- insurance/risk analytics.

The first commercial/operational unit should be a geographic pilot (one
catchment region or district), with annual software/support plus deployment
and integration services. Pricing must be validated with customers rather than
invented in the SIH presentation.

## What must not be claimed

Do not claim:
- operational forecast skill without real event validation;
- universal India-wide coverage from the v0.3 prototype;
- a proven cost reduction versus NWP without benchmark data;
- live MOSDAC/NCMRWF ingestion until credentials, schemas and downloads are
  tested;
- CTT/CAPE/CIN support unless the exact product variables are configured and
  tested.
