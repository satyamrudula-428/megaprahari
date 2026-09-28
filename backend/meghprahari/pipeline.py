"""Pure (I/O-free) pipeline: features per catchment, labels, scoring, alert evaluation."""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import alerts as A, ctt as C, explain as E, meteo as M, model as MD, satellite as S

FEATURES_V1 = [
    "hem_rain_now", "hem_rain_max_1h", "hem_rain_accel", "hem_area_growth", "hem_speed_kmh", "hem_stall_h",
    "met_iwv", "met_d_iwv_3h", "met_mfc_low", "met_omfi_low", "met_shear_low_500", "met_kindex", "met_totals",
    "ter_slope", "ter_hand", "ter_tc_min", "ter_area_up_km2"]
# V2 adds instability that is defined over high terrain (unlike K-index/Total Totals, which need 850 hPa above ground):
# CAPE (J/kg), CIN (J/kg, positive magnitude; NaN where undefined) and its 3-h change (negative = the lid is eroding).
FEATURES_V2 = FEATURES_V1 + ["met_cape", "met_cin", "met_d_cin_3h"]
# V3 adds cloud-top temperature dynamics (see ctt.py: GridSat-B1, TRAINING/REPLAY only, not live-capable).
FEATURES_V3 = FEATURES_V2 + ["sat_ctt_min", "sat_ctt_drop_rate", "sat_cold_area_km2"]
FEATURE_SETS = {"v1": FEATURES_V1, "v2": FEATURES_V2, "v3": FEATURES_V3}
HAZARDS = ("ts", "cb", "ff")
BINS = ((0, 120), (120, 240), (240, 360))          # minutes ahead
# Rain-feature time windows in MINUTES, so features mean the same at any cadence (30-min HEM/IMERG, 60-min MERA).
HISTORY_MIN = 150        # trend and area growth use the frames from t-150 min to t (6 frames at 30 min, 3 at 60 min)
MAX_WINDOW_MIN = 60      # hem_rain_max_1h: maximum over t-60 min .. t
MOTION_LAG_MIN = 60      # storm motion from the frames 60 min apart


def frames_for(minutes, dt_min):
    """Number of frames within the last `minutes` before the anchor, anchor included (150 min: 6 at 30-min cadence,
    3 at 60-min cadence, i.e. t-120..t)."""
    if dt_min <= 0 or dt_min > minutes:
        raise ValueError(f"cadence {dt_min} min is not usable for a {minutes}-min feature window")
    return minutes // dt_min + 1


def target_name(hazard, b):
    return f"{hazard}_{b[0] // 60}_{b[1] // 60}"


@dataclass
class MetState:
    time: object
    lat: np.ndarray
    lon: np.ndarray
    p_hpa: np.ndarray            # strictly decreasing
    q: np.ndarray                # (K,y,x) kg/kg
    t_k: np.ndarray
    u: np.ndarray
    v: np.ndarray
    ps_hpa: np.ndarray           # (y,x) surface pressure; required so below-ground levels are never used
    terrain_m: np.ndarray        # (y,x) mean terrain on this grid
    extra: dict = None           # optional single-level fields (y,x), e.g. {"cape": J/kg, "cin": J/kg}; NaN = undefined

    def validate(self):
        K, ny, nx = self.q.shape
        for name in ("t_k", "u", "v"):
            if getattr(self, name).shape != (K, ny, nx):
                raise ValueError(f"{name} shape mismatch")
        if self.ps_hpa.shape != (ny, nx) or self.terrain_m.shape != (ny, nx):
            raise ValueError("ps/terrain shape mismatch")
        for name, a in (self.extra or {}).items():
            if np.shape(a) != (ny, nx):
                raise ValueError(f"extra field '{name}' shape mismatch")
        if len(self.p_hpa) != K or np.any(np.diff(self.p_hpa) >= 0):
            raise ValueError("levels must be strictly decreasing")
        for ax, n in ((self.lat, ny), (self.lon, nx)):
            d = np.diff(ax)
            if len(ax) != n or n < 3 or not np.allclose(d, d[0], rtol=1e-3, atol=1e-6):
                raise ValueError("lat/lon axes must be regular and match the grid")
        for name, lo, hi in (("q", 0.0, 0.05), ("t_k", 150.0, 350.0), ("ps_hpa", 300.0, 1100.0)):
            a = getattr(self, name)
            bad = np.isfinite(a) & ((a < lo) | (a > hi))
            if bad.any():
                raise ValueError(f"{name} has values outside [{lo}, {hi}] (check units/scale factors)")
        return self


def _level(p_hpa, hpa):
    i = np.flatnonzero(np.isclose(p_hpa, hpa))
    if i.size != 1:
        raise ValueError(f"level {hpa} hPa not present in met state")
    return int(i[0])


def met_fields(m):
    i850, i700, i500 = (_level(m.p_hpa, x) for x in (850, 700, 500))
    ps = m.ps_hpa
    q_low, u_low, v_low = (M.lowest_above_ground(a, m.p_hpa, ps) for a in (m.q, m.u, m.v))
    t850, t700, t500 = (m.t_k[i] - 273.15 for i in (i850, i700, i500))
    td850 = M.dewpoint_c(M.vapor_pressure_hpa(m.q[i850], 850.0))
    td700 = M.dewpoint_c(M.vapor_pressure_hpa(m.q[i700], 700.0))
    ok850 = ps >= 850.0 + M.LOW_MARGIN_HPA
    ok500 = ps >= 500.0 + M.LOW_MARGIN_HPA
    extra = {k: np.asarray(m.extra[k], float) for k in ("cape", "cin") if m.extra and k in m.extra}
    return dict(
        **extra,
        iwv=M.iwv(m.q, m.p_hpa, ps),
        mfc_low=M.mfc(q_low, u_low, v_low, m.lat, m.lon),
        omfi_low=M.omfi(q_low, u_low, v_low, m.terrain_m, m.lat, m.lon),
        shear_low_500=np.where(ok500, M.bulk_shear(m.u[i500], m.v[i500], u_low, v_low), np.nan),
        kindex=np.where(ok850, M.k_index(t850, td850, t700, td700, t500), np.nan),
        totals=np.where(ok850, M.total_totals(t850, td850, t500), np.nan))


def hem_fields(stack, dt_min=30, pixel_km=4.0, growth_thr=10.0, cell_thr=10.0, tile=32):
    """Rain-dynamics fields from a (T, y, x) stack of rain-rate frames (mm/h) ending at the anchor, dt_min apart.
    Windows are fixed in minutes (HISTORY_MIN, MAX_WINDOW_MIN, MOTION_LAG_MIN); at 30-min cadence this is the
    original behaviour (6 / 3 frames, motion over 2 frames)."""
    n_hist, n_max = frames_for(HISTORY_MIN, dt_min), frames_for(MAX_WINDOW_MIN, dt_min)
    if MOTION_LAG_MIN % dt_min:
        raise ValueError(f"cadence {dt_min} min does not divide the {MOTION_LAG_MIN}-min motion lag")
    lag = MOTION_LAG_MIN // dt_min
    if n_hist < 3:
        raise ValueError(f"cadence {dt_min} min gives fewer than 3 frames in {HISTORY_MIN} min; trend undefined")
    if stack.shape[0] < n_hist:
        raise ValueError(f"need at least {n_hist} frames ({HISTORY_MIN} min at {dt_min}-min cadence)")
    tile = min(tile, stack.shape[1], stack.shape[2])
    now = stack[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx1h = np.nanmax(stack[-n_max:], axis=0)
    vy, vx = S.motion_tiles(stack[-1 - lag], stack[-1], lag * dt_min, pixel_km, tile=tile)
    speed = S.speed_field(vy, vx, now.shape, tile)
    stall = S.stall_index(S.cell_area_map(now, cell_thr, pixel_km ** 2), speed)
    return dict(rain_now=now, rain_max_1h=mx1h, accel=S.linear_trend(stack[-n_hist:], dt_min / 60.0),
                growth=S.exceed_growth(stack[-n_hist:], growth_thr), speed=speed, stall=stall)


def ctt_fields(stack, dt_h, thr_k=221.0, pixel_km2=64.0):
    """Cloud-top-temperature fields from a (T, y, x) stack of brightness-temperature frames (K) ending at the anchor,
    dt_h apart (see ctt.py: GridSat-B1 is 3-hourly, TRAINING/REPLAY only). thr_k/pixel_km2: see ctt.cold_cloud_area_km2
    (pixel_km2 default 64.0 = GridSat-B1's ~8 km grid spacing squared)."""
    return dict(ctt_min=C.ctt_min(stack), ctt_drop_rate=C.ctt_drop_rate(stack, dt_h),
                cold_area_km2=C.cold_cloud_area_km2(stack[-1], thr_k, pixel_km2))


def nearest_idx(axis, vals, max_dist=None):
    """Index of the nearest grid coordinate (asc or desc axis); -1 if farther than max_dist."""
    axis = np.asarray(axis, float)
    vals = np.asarray(vals, float)
    if axis[0] > axis[-1]:
        i = axis.size - 1 - nearest_idx(axis[::-1], vals, max_dist)
        return np.where(i == axis.size, -1, i) if max_dist is not None else i
    i = np.clip(np.searchsorted(axis, vals), 1, axis.size - 1)
    idx = np.where(np.abs(vals - axis[i - 1]) <= np.abs(vals - axis[i]), i - 1, i)
    if max_dist is not None:
        idx = np.where(np.abs(vals - axis[idx]) <= max_dist, idx, -1)
    return idx


def _take(field, iy, ix):
    ok = (iy >= 0) & (ix >= 0)
    out = np.full(iy.shape, np.nan)
    out[ok] = field[iy[ok], ix[ok]]
    return out


def assemble(catch, hem, met_now, met_prev, hem_lat, hem_lon, met_lat, met_lon, features=FEATURES_V1,
             ctt=None, ctt_lat=None, ctt_lon=None):
    """catch: DataFrame indexed by catchment id with lat, lon, slope_mean, hand_mean, tc_min, area_up_km2.
    features: FEATURES_V1, FEATURES_V2 (needs 'cape'/'cin' in met_now) or FEATURES_V3 (also needs ctt/ctt_lat/ctt_lon,
    ValueError otherwise, never filled)."""
    lat, lon = catch["lat"].to_numpy(float), catch["lon"].to_numpy(float)
    hd = 0.75 * np.array([abs(hem_lat[1] - hem_lat[0]), abs(hem_lon[1] - hem_lon[0])])
    md = 0.75 * np.array([abs(met_lat[1] - met_lat[0]), abs(met_lon[1] - met_lon[0])])
    hy, hx = nearest_idx(hem_lat, lat, hd[0]), nearest_idx(hem_lon, lon, hd[1])
    my, mx = nearest_idx(met_lat, lat, md[0]), nearest_idx(met_lon, lon, md[1])
    f = pd.DataFrame(index=catch.index)
    f["hem_rain_now"] = _take(hem["rain_now"], hy, hx)
    f["hem_rain_max_1h"] = _take(hem["rain_max_1h"], hy, hx)
    f["hem_rain_accel"] = _take(hem["accel"], hy, hx)
    f["hem_area_growth"] = _take(hem["growth"], hy, hx)
    f["hem_speed_kmh"] = _take(hem["speed"], hy, hx)
    f["hem_stall_h"] = _take(hem["stall"], hy, hx)
    f["met_iwv"] = _take(met_now["iwv"], my, mx)
    f["met_d_iwv_3h"] = _take(met_now["iwv"] - met_prev["iwv"], my, mx) if met_prev is not None else np.nan
    f["met_mfc_low"] = _take(met_now["mfc_low"], my, mx)
    f["met_omfi_low"] = _take(met_now["omfi_low"], my, mx)
    f["met_shear_low_500"] = _take(met_now["shear_low_500"], my, mx)
    f["met_kindex"] = _take(met_now["kindex"], my, mx)
    f["met_totals"] = _take(met_now["totals"], my, mx)
    f["ter_slope"] = catch["slope_mean"].to_numpy(float)
    f["ter_hand"] = catch["hand_mean"].to_numpy(float)
    f["ter_tc_min"] = catch["tc_min"].to_numpy(float)
    f["ter_area_up_km2"] = catch["area_up_km2"].to_numpy(float)
    if "met_cape" in features:
        missing = [k for k in ("cape", "cin") if k not in met_now]
        if missing:
            raise ValueError(f"feature set needs {missing} but the met state has none (configure met.surface_vars)")
        f["met_cape"] = _take(met_now["cape"], my, mx)
        f["met_cin"] = _take(met_now["cin"], my, mx)
        f["met_d_cin_3h"] = (_take(met_now["cin"] - met_prev["cin"], my, mx)
                             if met_prev is not None and "cin" in met_prev else np.nan)
    if "sat_ctt_min" in features:
        if ctt is None or ctt_lat is None or ctt_lon is None:
            raise ValueError("feature set needs ctt/ctt_lat/ctt_lon (configure a ctt source, e.g. GridSat-B1)")
        cd = 0.75 * np.array([abs(ctt_lat[1] - ctt_lat[0]), abs(ctt_lon[1] - ctt_lon[0])])
        cy, cx = nearest_idx(ctt_lat, lat, cd[0]), nearest_idx(ctt_lon, lon, cd[1])
        f["sat_ctt_min"] = _take(ctt["ctt_min"], cy, cx)
        f["sat_ctt_drop_rate"] = _take(ctt["ctt_drop_rate"], cy, cx)
        f["sat_cold_area_km2"] = _take(ctt["cold_area_km2"], cy, cx)
    return f[list(features)]


def window_frames_from_tc(tc_min, dt_min=30, default=4, wmin=1, wmax=12):
    tc = np.asarray(tc_min, float)
    w = np.clip(np.ceil(np.nan_to_num(tc, nan=default * dt_min) / dt_min), wmin, wmax)
    return w.astype(int)


def neighbourhood_max(frames, radius_px):
    """Per-frame maximum within a disc of radius_px pixels (NaN ignored), so a label means 'within r of the catchment'
    rather than 'at this exact pixel' (avoids penalising a storm displaced by a few km twice). radius 0 = unchanged."""
    if radius_px <= 0:
        return frames
    from scipy import ndimage
    r = int(radius_px)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    fp = (yy ** 2 + xx ** 2) <= r * r
    out = np.empty_like(frames, dtype=float)
    for k, f in enumerate(frames):
        valid = np.isfinite(f)
        m = ndimage.maximum_filter(np.where(valid, f, -np.inf), footprint=fp, mode="constant", cval=-np.inf)
        out[k] = np.where(np.isfinite(m), m, np.nan)
    return out


def make_labels(future, iy, ix, thr, window_frames, dt_min=30, bins=BINS, radius_px=0, history=None):
    """Proxy labels from rain rate AFTER the anchor time (first frame = anchor + dt_min), rain in mm/h.
    ts (and cb when thr has no 'cb_accum_mm'): peak rate >= thr[h] (mm/h) within the lead bin, taken over a disc of
    radius_px pixels around the catchment point.
    cb with thr['cb_accum_mm'] and thr['cb_accum_min']: the largest cb_accum_min-long rain TOTAL (mm) that ENDS inside
    the lead bin reaches cb_accum_mm, totals computed per pixel first and then maximised over the disc. Windows may
    start before the anchor (the storm is still producing inside the bin), so `history` (H, y, x) = observed frames
    ending AT the anchor must hold at least cb_accum_min/dt_min - 1 frames.
    ff_proxy: maximum rolling accumulation over each catchment's Tc window >= thr['ff_mm'] at the catchment point
    (no neighbourhood). Missing frames count as 0 rain in totals.
    These are PROXIES: replace/augment with event-ledger labels before claiming skill on real events."""
    accum = "cb_accum_mm" in thr
    if accum:
        w_cb = int(thr["cb_accum_min"]) // dt_min
        if w_cb < 1 or int(thr["cb_accum_min"]) % dt_min:
            raise ValueError(f"cb_accum_min must be a positive multiple of the {dt_min}-min cadence")
        if history is None or history.shape[0] < w_cb - 1:
            raise ValueError(f"cloudburst accumulation labels need {w_cb - 1} observed frames before the anchor")
        full = np.concatenate([history[history.shape[0] - (w_cb - 1):] if w_cb > 1 else history[:0], future])
        cs_full = np.concatenate([np.zeros((1,) + full.shape[1:]), np.cumsum(np.nan_to_num(full), axis=0)])
    peak = neighbourhood_max(future, radius_px)
    pser = peak[:, iy, ix]
    ser = future[:, iy, ix]
    ser0 = np.nan_to_num(ser)
    Tf, n = ser.shape
    out = {}
    ffthr = np.broadcast_to(np.asarray(thr["ff_mm"], float), (n,))
    dt_h = dt_min / 60.0
    for a, b in bins:
        fa, fb = a // dt_min, min(b // dt_min, Tf)
        if fb <= fa:
            raise ValueError("not enough future frames for the requested lead bin")
        seg = pser[fa:fb]
        valid = np.isfinite(seg).any(axis=0)
        mx = np.max(np.where(np.isfinite(seg), seg, -np.inf), axis=0)
        for h in ("ts", "cb"):
            if h == "cb" and accum:
                ends = np.arange(fa, fb) + (w_cb - 1)                    # window end indices in `full`
                tot = (cs_full[ends + 1] - cs_full[ends + 1 - w_cb]) * dt_h
                field = neighbourhood_max(tot.max(axis=0)[None], radius_px)[0]
                lab = (field[iy, ix] >= thr["cb_accum_mm"]).astype(float)
            else:
                lab = (mx >= thr[h]).astype(float)
            lab[~valid] = np.nan
            out[f"y_{h}_{a // 60}_{b // 60}"] = lab
        ff = np.full(n, np.nan)
        for w in np.unique(window_frames):
            cols = window_frames == w
            lo = max(fa, int(w) - 1)
            if lo >= fb:
                continue
            cs = np.vstack([np.zeros((1, int(cols.sum()))), np.cumsum(ser0[:, cols], axis=0)])
            ends = np.arange(lo, fb)
            acc = (cs[ends + 1] - cs[ends + 1 - int(w)]) * dt_h
            ff[cols] = (acc.max(axis=0) >= ffthr[cols]).astype(float)
        ff[~valid] = np.nan
        out[f"y_ff_{a // 60}_{b // 60}"] = ff
    return pd.DataFrame(out)


def score_all(features, models):
    """models: {target_name: TargetModel}. Raises if any model is uncalibrated (fail closed)."""
    out = pd.DataFrame(index=features.index)
    for name, tm in models.items():
        out[name] = MD.predict(tm, features, require_calibrated=True)
    return out


def evaluate_villages(probs, features, villages, states, models, cfg, now, shadow, sender, sender_name, up_needed=None):
    """Update escalation state per (village, hazard); return alert DRAFTS for upward transitions only."""
    drafts = []
    for v in villages:
        if v.catchment_id not in probs.index:
            continue
        for h in HAZARDS:
            cols = [(target_name(h, b), b) for b in BINS if target_name(h, b) in probs.columns]
            if not cols:
                continue
            vals = [(probs.at[v.catchment_id, c], b) for c, b in cols]
            vals = [(p, b) for p, b in vals if np.isfinite(p)]
            if not vals:
                continue
            p, b = max(vals, key=lambda t: t[0])
            st = states.setdefault((v.id, h), A.VillageState())
            prev = st.level
            A.step(st, float(p), A.level_thresholds(v.cost, v.loss), up_needed, cfg.get("down_needed", 4), cfg.get("release", 0.8))
            if st.level > prev:
                tn = target_name(h, b)
                drivers = MD.top_drivers(models[tn], features.loc[v.catchment_id]) if tn in models else []
                d = A.draft_alert(v, st.level, float(p), h, b[0], b[1], drivers, cfg, now, shadow, sender, sender_name)
                d["explanation"] = E.ingredient_card(features.loc[v.catchment_id])
                drafts.append(d)
    return drafts


def select_met_times(met_times, t, max_age_h, prev_offset_h=3.0, prev_tol_h=1.0):
    """Time alignment of met analyses (e.g. 3-hourly) with a satellite anchor time t (e.g. 30-min cadence).
    Returns (t_now, t_prev): t_now = latest met time <= t that is at most max_age_h old (hold the last analysis;
    never use a future one), t_prev = met time closest to t_now - prev_offset_h within +-prev_tol_h (for 3-h
    tendencies such as d_iwv_3h). Either is None when unavailable."""
    cands = [k for k in met_times if k <= t and (t - k).total_seconds() <= max_age_h * 3600]
    if not cands:
        return None, None
    t_now = max(cands)
    target = prev_offset_h * 3600
    prev = [k for k in met_times if abs((t_now - k).total_seconds() - target) <= prev_tol_h * 3600]
    t_prev = min(prev, key=lambda k: abs((t_now - k).total_seconds() - target)) if prev else None
    return t_now, t_prev


def features_at(t, frames, hem_axes, met_times, get_met, catch, rain_cfg, max_age_h, features=FEATURES_V1):
    """Features for every catchment at anchor time t, exactly as the worker computes them. frames: {time: rain frame
    (mm/h)} at rain_cfg['cadence_min']; hem_axes: (lat, lon) of the frames; met_times: available met analysis times;
    get_met(time) -> MetState. Returns (DataFrame, None), or (None, reason) when inputs are incomplete."""
    stack = pick_window(frames, t, frames_for(HISTORY_MIN, rain_cfg["cadence_min"]), rain_cfg["cadence_min"])
    tm_now, tm_prev = select_met_times(met_times, t, max_age_h)
    if stack is None or tm_now is None:
        return None, "frame window incomplete or met state too old"
    m = get_met(tm_now)
    met_prev = met_fields(get_met(tm_prev)) if tm_prev is not None else None
    hem = hem_fields(stack, rain_cfg["cadence_min"], rain_cfg["pixel_km"], rain_cfg["growth_thr_mm_h"], rain_cfg["cell_thr_mm_h"])
    return assemble(catch, hem, met_fields(m), met_prev, hem_axes[0], hem_axes[1], m.lat, m.lon, features), None


def pick_window(frames, t, n=6, step_min=30, tol_min=5):
    """Stack the n frames ending at time t (one per step_min, +-tol_min) from {datetime: array}; None if any is missing."""
    picked = []
    for k in range(n - 1, -1, -1):
        target = t.timestamp() - k * step_min * 60
        best = min(frames, key=lambda ts: abs(ts.timestamp() - target), default=None)
        if best is None or abs(best.timestamp() - target) > tol_min * 60:
            return None
        picked.append(frames[best])
    return np.stack(picked)
