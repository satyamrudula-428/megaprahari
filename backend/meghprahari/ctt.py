"""Cloud-top temperature (CTT) dynamics from a stack of brightness-temperature frames (Kelvin, shape (T, H, W),
NaN = missing). Colder cloud tops indicate taller, more vigorous convection.

Source used so far: NOAA GridSat-B1 (variable irwin_cdr), 3-hourly, ~8 km, license "No constraints on data access or
use" (verified from the file's own metadata, no account needed). It is NOT INSAT: India is viewed at an angle by
GOES/Himawari/Meteosat, and the product has a multi-month processing lag, so it is usable for TRAINING and REPLAY
only, never for live nowcasting. See documentation/DATA_REGISTER.md and CLAUDE.md hard rule 1: do not claim this is a
substitute for MOSDAC INSAT TIR1 (roadmap C7), which would give 15-30 min cadence at ~1 km for live operation.

IMPORTANT (checked on real files, 2026-09-24, tools/fetch_gridsat.py): over the small Mandi pilot box specifically,
GridSat-B1's per-timestep coverage is very poor -- 5 real files spanning the Mandi 2025-06-29 event window had mean
coverage 2% of pixels (4 of 5 hours had ZERO valid pixels in the pilot box), including the file for the event's own
peak hour (2025-06-30 18 UTC), which was checked globally and found 100% missing over the whole pilot box (99.95%
missing over the entire globe at that timestep). The pilot box sits in a gap between the geostationary satellites
this composite blends (nothing near 77 degrees E). This module and its pipeline wiring (pipeline.ctt_fields,
FEATURES_V3) are built and unit-tested and read real GridSat-B1 files correctly, but GridSat-B1 is NOT currently a
usable real feature source for the Mandi pilot region -- do not train on FEATURES_V3 with GridSat-B1 data until
MOSDAC TIR1 (roadmap C7) is available, or a satellite composite with proper coverage of this longitude is found."""
import numpy as np
from scipy import ndimage


def ctt_min(frames):
    """Coldest brightness temperature (K) at each grid point across the stack; NaN where every frame is missing."""
    import warnings
    has_data = np.isfinite(frames).any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns: nanmin warns, result discarded below
        vals = np.nanmin(frames, axis=0)
    return np.where(has_data, vals, np.nan)


def ctt_drop_rate(frames, dt_h):
    """Per-pixel least-squares slope of brightness temperature (K per hour; negative = cooling) across the stack,
    ignoring NaNs. NaN where fewer than 3 valid frames. Same construction as satellite.linear_trend, over whatever
    cadence `frames` actually has (see module docstring: 3-hourly for GridSat-B1, not the 15-30 min originally
    envisaged for live INSAT TIR1)."""
    T = frames.shape[0]
    t = (np.arange(T) * dt_h).reshape(-1, 1, 1)
    m = np.isfinite(frames)
    n = m.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.where(m, t, 0.0).sum(0) / np.maximum(n, 1)
        fm = np.where(m, frames, 0.0).sum(0) / np.maximum(n, 1)
        num = np.where(m, (t - tm) * (frames - fm), 0.0).sum(0)
        den = np.where(m, (t - tm) ** 2, 0.0).sum(0)
        return np.where((n >= 3) & (den > 0), num / np.where(den > 0, den, 1.0), np.nan)


def cold_cloud_area_km2(frame, thr_k, pixel_km2):
    """Area (km2) of the contiguous patch of cloud colder than thr_k (8-connected) containing each pixel; 0 where
    the pixel itself is not that cold or is missing. thr_k: brightness-temperature threshold in Kelvin (a common
    convective-overshoot cut-off is 221 K, i.e. about -52 degC; tune per basin/season)."""
    cold = np.isfinite(frame) & (frame <= thr_k)
    lab, _ = ndimage.label(cold, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return sizes[lab] * pixel_km2
