"""'Why this alert?' card (tasks I1, I4): the three classic ingredients of deep convection (moisture, instability,
lift) plus the observed storm signal and the terrain response of the catchment, each rated from the alert's own
feature values and described in plain language.

This is a physical-reasoning summary of the INPUTS, not an attribution of the model's output (not SHAP, not causal).
A missing input is reported as 'unknown'; nothing is filled in. Threshold values are documented starting points from
the literature, to be tuned with forecasters for the Himalayan pilot.
Feature units: IWV kg/m2 (= mm of precipitable water), K-index and Total Totals in degC, rain mm/h, rain acceleration
mm/h per hour, stall hours, time of concentration minutes, slope m/m, upstream area km2."""
import math

STRONG, MODERATE, WEAK, UNKNOWN = "strong", "moderate", "weak", "unknown"

# Precipitable water: monsoon columns are typically 40-60 mm; >= 50 mm is very moist (rule of thumb, to tune).
IWV_STRONG, IWV_MODERATE = 50.0, 40.0
IWV_RISE_3H = 2.0                    # kg/m2 in 3 h counted as "moisture increasing"
# K-index (George 1960): 31-35 widespread thunderstorms, > 35 numerous. Total Totals (Miller 1972): >= 44 possible,
# >= 50 severe thunderstorms likely.
KI_STRONG, KI_MODERATE = 35.0, 30.0
TT_STRONG, TT_MODERATE = 50.0, 44.0
# CAPE (J/kg): commonly used bands, ~1000 moderate and ~2500 strong instability (starting values; tune for the monsoon
# Himalaya). CIN (J/kg, positive magnitude): above ~100 a strong lid; a 3-h drop counts as "lid weakening".
CAPE_STRONG, CAPE_MODERATE = 2500.0, 1000.0
CIN_STRONG_LID = 100.0
CIN_ERODING_3H = -20.0
# Observed storm: 10 mm/h is the thunderstorm proxy threshold used for labels (settings labels.ts_mm_h).
RAIN_STRONG, RAIN_MODERATE = 10.0, 2.0
RAIN_TRACE = 0.5                     # below this (mm/h) the text says 'little or no rain'
STALL_LONG_H = 1.0                   # a cell that lingers >= 1 h over the same place
STALL_QUOTE_MAX_H = 6.0              # above this the stall index (sqrt(area)/speed) is not a meaningful duration
# Terrain: time of concentration (Kirpich) - how fast rain on the catchment reaches its outlet.
TC_FAST_MIN, TC_MODERATE_MIN = 30.0, 90.0


def _v(row, key):
    """Finite float value of a feature, or None if absent/NaN."""
    try:
        x = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _strongest(*statuses):
    order = {STRONG: 3, MODERATE: 2, WEAK: 1}
    known = [s for s in statuses if s in order]
    return max(known, key=order.get) if known else UNKNOWN


def moisture(row):
    iwv, d = _v(row, "met_iwv"), _v(row, "met_d_iwv_3h")
    if iwv is None:
        return UNKNOWN, "Moisture data are missing for this place.", {}
    s = STRONG if iwv >= IWV_STRONG else MODERATE if iwv >= IWV_MODERATE else WEAK
    text = {STRONG: "The air is very moist", MODERATE: "The air is moist", WEAK: "The air is fairly dry"}[s]
    text += f" (about {iwv:.0f} mm of water vapour overhead)"
    if d is not None and d >= IWV_RISE_3H:
        text += f", and moisture rose by {d:.0f} mm in the last 3 hours"
    return s, text + ".", {"met_iwv": iwv, "met_d_iwv_3h": d}


def instability(row):
    ki, tt = _v(row, "met_kindex"), _v(row, "met_totals")
    cape, cin, dcin = _v(row, "met_cape"), _v(row, "met_cin"), _v(row, "met_d_cin_3h")
    if ki is None and tt is None and cape is None:
        return UNKNOWN, ("Instability indices are not available here (K-index and Total Totals need the 850 hPa level, "
                         "which can be below the ground over high terrain, and CAPE is not configured)."), {}
    s_ki = None if ki is None else STRONG if ki >= KI_STRONG else MODERATE if ki >= KI_MODERATE else WEAK
    s_tt = None if tt is None else STRONG if tt >= TT_STRONG else MODERATE if tt >= TT_MODERATE else WEAK
    s_cape = None if cape is None else STRONG if cape >= CAPE_STRONG else MODERATE if cape >= CAPE_MODERATE else WEAK
    s = _strongest(s_ki, s_tt, s_cape)
    text = {STRONG: "The atmosphere is very unstable: storms can grow quickly",
            MODERATE: "The atmosphere is unstable enough for thunderstorms",
            WEAK: "The atmosphere is fairly stable"}[s]
    parts = [f"CAPE {cape:.0f} J/kg" if cape is not None else None, f"K-index {ki:.0f}" if ki is not None else None,
             f"Total Totals {tt:.0f}" if tt is not None else None]
    text = f"{text} ({', '.join(p for p in parts if p)})"
    if cin is not None and cin >= CIN_STRONG_LID:
        text += "; a strong 'lid' of stable air is still holding storms back"
    if dcin is not None and dcin <= CIN_ERODING_3H:
        text += "; the lid is weakening"
    return s, text + ".", {"met_cape": cape, "met_cin": cin, "met_d_cin_3h": dcin, "met_kindex": ki, "met_totals": tt}


def lift(row):
    mfc, omfi = _v(row, "met_mfc_low"), _v(row, "met_omfi_low")
    if mfc is None and omfi is None:
        return UNKNOWN, "Lifting indicators are not available here.", {}
    conv, upslope = mfc is not None and mfc > 0, omfi is not None and omfi > 0
    s = STRONG if conv and upslope else MODERATE if conv or upslope else WEAK
    bits = []
    if conv:
        bits.append("low-level moist air is converging")
    if upslope:
        bits.append("moist wind is blowing up the mountain slopes")
    if not bits:
        bits.append("there is no clear low-level convergence or upslope flow")
    missing = [n for n, x in (("convergence", mfc), ("upslope flow", omfi)) if x is None]
    tail = f" ({' and '.join(missing)} data missing)" if missing else ""
    return s, "Lift: " + " and ".join(bits) + tail + ".", {"met_mfc_low": mfc, "met_omfi_low": omfi}


def storm_signal(row):
    now, acc, grow, stall = (_v(row, k) for k in ("hem_rain_now", "hem_rain_accel", "hem_area_growth", "hem_stall_h"))
    if now is None:
        return UNKNOWN, "Satellite rain data are missing for this place.", {}
    intensifying = (acc is not None and acc > 0) or (grow is not None and grow > 0)
    s = STRONG if now >= RAIN_STRONG and intensifying else MODERATE if now >= RAIN_MODERATE or intensifying else WEAK
    if now < RAIN_TRACE:
        text = "Satellite shows little or no rain here yet"
        if intensifying:
            text += ", but rain nearby is building"
    else:
        text = f"Satellite shows rain of about {now:.0f} mm/h here"
        if intensifying:
            text += " and it is intensifying or spreading"
    if now >= RAIN_TRACE and stall is not None and stall >= STALL_LONG_H:
        if stall > STALL_QUOTE_MAX_H:
            text += "; the rain cell is barely moving, so rain may keep falling on the same place"
        else:
            text += f"; the rain cell is moving slowly and may sit over the same place for about {stall:.1f} h"
    return s, text + ".", {"hem_rain_now": now, "hem_rain_accel": acc, "hem_area_growth": grow, "hem_stall_h": stall}


def terrain_response(row):
    tc, slope, area = _v(row, "ter_tc_min"), _v(row, "ter_slope"), _v(row, "ter_area_up_km2")
    if tc is None:
        return UNKNOWN, "Terrain data are missing for this catchment.", {}
    s = STRONG if tc <= TC_FAST_MIN else MODERATE if tc <= TC_MODERATE_MIN else WEAK
    text = f"Rain falling on this catchment reaches the stream outlet in about {tc:.0f} min"
    if area is not None:
        text += f" (it drains about {area:.0f} km2 upstream)"
    if slope is not None:
        text += f"; slopes are steep (average {math.degrees(math.atan(slope)):.0f} degrees)" if slope >= 0.3 else ""
    return s, text + ".", {"ter_tc_min": tc, "ter_slope": slope, "ter_area_up_km2": area}


INGREDIENTS = (("moisture", moisture), ("instability", instability), ("lift", lift),
               ("storm_signal", storm_signal), ("terrain_response", terrain_response))


def ingredient_card(row):
    """Explanation card for one catchment's feature row (Series or dict with FEATURES_V1 names).
    Returns {"ingredients": [{name, status, text, evidence}], "summary": str, "note": str}; JSON-serialisable."""
    items = []
    for name, fn in INGREDIENTS:
        status, text, ev = fn(row)
        items.append(dict(name=name, status=status, text=text,
                          evidence={k: (None if v is None else round(v, 6)) for k, v in ev.items()}))
    strong = [i["name"].replace("_", " ") for i in items if i["status"] == STRONG]
    unknown = [i["name"].replace("_", " ") for i in items if i["status"] == UNKNOWN]
    summary = (f"Strong signals: {', '.join(strong)}." if strong else "No ingredient is strong on its own.")
    if unknown:
        summary += f" Not available: {', '.join(unknown)}."
    return dict(ingredients=items, summary=summary,
                note="Describes the input conditions behind the forecast; it is not a causal attribution of the model.")
