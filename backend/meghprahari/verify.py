"""Verification scores that complement model.py (POD/FAR/CSI, Brier, BSS, AUC, reliability/ECE):
Fractions Skill Score for neighbourhood (hyper-local) accuracy and warning lead time per event.
Pure functions: numpy arrays in, numbers out. Rain fields in mm/h, NaN = missing."""
from datetime import timedelta

import numpy as np
from scipy import ndimage


def _fractions(binary, valid, window):
    """Fraction of valid pixels exceeding the threshold inside each window x window neighbourhood."""
    num = ndimage.uniform_filter(np.where(valid, binary, 0.0), size=window, mode="constant")
    den = ndimage.uniform_filter(valid.astype(float), size=window, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def fss(fcst, obs, thr, window):
    """Fractions Skill Score (Roberts & Lean 2008) of forecast vs observed fields for exceedance of thr, with a square
    neighbourhood of `window` pixels (odd). fcst/obs: (y, x) or (t, y, x); scores are aggregated over all times by
    summing numerator and reference terms. Pixels missing in either field are ignored. Returns NaN when neither field
    exceeds thr anywhere (score undefined). 1 = perfect, 0 = no skill."""
    f, o = np.asarray(fcst, float), np.asarray(obs, float)
    if f.shape != o.shape:
        raise ValueError(f"shape mismatch {f.shape} vs {o.shape}")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number of pixels")
    if f.ndim == 2:
        f, o = f[None], o[None]
    mse = ref = 0.0
    for fk, ok in zip(f, o):
        valid = np.isfinite(fk) & np.isfinite(ok)
        if not valid.any():
            continue
        pf = _fractions(np.nan_to_num(fk) >= thr, valid, window)
        po = _fractions(np.nan_to_num(ok) >= thr, valid, window)
        m = valid & np.isfinite(pf) & np.isfinite(po)
        mse += float(np.sum((pf[m] - po[m]) ** 2))
        ref += float(np.sum(pf[m] ** 2) + np.sum(po[m] ** 2))
    return float("nan") if ref == 0 else 1.0 - mse / ref


def fss_useful(obs_base_rate):
    """FSS above which a forecast counts as 'useful' at a given scale: 0.5 + f0/2 (Roberts & Lean 2008),
    f0 = observed fraction of pixels exceeding the threshold."""
    return 0.5 + float(obs_base_rate) / 2.0


def warning_lead_min(issue_times, warned, onset, max_lead_min=360):
    """Lead time (minutes) of the FIRST warning issued in [onset - max_lead_min, onset] for one event, or None when
    no warning was issued in that window (a miss). issue_times: datetimes; warned: bools (same length)."""
    if len(issue_times) != len(warned):
        raise ValueError("issue_times and warned must have the same length")
    leads = [(onset - t).total_seconds() / 60.0 for t, w in zip(issue_times, warned) if w]
    leads = [x for x in leads if 0 <= x <= max_lead_min]
    return max(leads) if leads else None


def storm_onset(times, peak, thr, after, quiet_h=3.0, max_gap_h=1.0):
    """Onset of a NEW storm in an area: first time >= after whose peak rain (mm/h) reaches thr, preceded by at least
    quiet_h hours of data all below thr (so rain continuing from an earlier storm is not an onset). The quiet period
    must be covered by data starting within max_gap_h of its beginning. times: sorted datetimes; peak: floats."""
    for i, t in enumerate(times):
        if t < after or not peak[i] >= thr:
            continue
        start = t - timedelta(hours=quiet_h)
        before = [(k, p) for k, p in zip(times[:i], peak[:i]) if k >= start]
        if before and (before[0][0] - start).total_seconds() <= max_gap_h * 3600 and all(p < thr for _, p in before):
            return t
    return None


def lead_time_summary(leads):
    """Summary over events: number of events, hits (warned before onset), hit rate and median lead of the hits (min)."""
    hits = [x for x in leads if x is not None]
    return dict(events=len(leads), hits=len(hits), hit_rate=(len(hits) / len(leads)) if leads else float("nan"),
                median_lead_min=float(np.median(hits)) if hits else float("nan"))
