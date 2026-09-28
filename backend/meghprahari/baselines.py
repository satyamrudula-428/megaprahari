"""Reference nowcasts that MeghPrahari must beat (task G2, and a simple stand-in for G3):
- persistence: the latest rain field stays exactly where it is (Eulerian persistence);
- extrapolation: the latest field moves with ONE domain-mean velocity estimated from the last two frames.
The extrapolation is a deliberately simple substitute for a pysteps nowcast (pysteps does not install on the team's
Windows machines without build tools); it must be labelled as such in results.
Rain fields (y, x) in mm/h, NaN = missing; rows increase southward, columns eastward (as in satellite.py)."""
import numpy as np
from scipy import ndimage

from . import satellite as S


def persistence(last_frame, n_steps):
    """(n_steps, y, x) stack repeating the last observed frame."""
    f = np.asarray(last_frame, float)
    return np.repeat(f[None], n_steps, axis=0)


def mean_motion(prev_frame, last_frame, dt_min, pixel_km, min_rain=1.0, min_frac=0.02):
    """Domain-mean storm motion (vy, vx) in km/h between two frames dt_min apart (+vy southward, +vx eastward), via
    phase correlation over the whole domain (one tile). (0, 0) when there is too little rain to estimate it."""
    a, b = np.asarray(prev_frame, float), np.asarray(last_frame, float)
    vy, vx = S.motion_tiles(a, b, dt_min, pixel_km, tile=min(a.shape), min_rain=min_rain, min_frac=min_frac)
    vy, vx = float(vy[0, 0]), float(vx[0, 0])
    return (0.0, 0.0) if not (np.isfinite(vy) and np.isfinite(vx)) else (vy, vx)


def extrapolate(last_frame, vy_kmh, vx_kmh, dt_min, n_steps, pixel_km):
    """(n_steps, y, x) stack: the last frame shifted by the mean motion for leads dt_min, 2*dt_min, ...
    Pixels advected in from outside the domain are NaN (unknown), never zero-filled."""
    f = np.asarray(last_frame, float)
    valid = np.isfinite(f)
    out = np.empty((n_steps,) + f.shape)
    for k in range(1, n_steps + 1):
        h = k * dt_min / 60.0
        shift = (vy_kmh * h / pixel_km, vx_kmh * h / pixel_km)
        moved = ndimage.shift(np.nan_to_num(f), shift, order=1, mode="constant", cval=0.0)
        inside = ndimage.shift(valid.astype(float), shift, order=1, mode="constant", cval=0.0)
        out[k - 1] = np.where(inside > 0.999, moved, np.nan)
    return out


def window_max(stack, dt_min, lead_bin_min):
    """Maximum over the frames whose lead (k * dt_min, k = 1..) falls in the bin (start, end] minutes: the forecast
    or observed peak rain rate (mm/h) in a lead window such as 0-2 h."""
    a, b = lead_bin_min
    leads = dt_min * np.arange(1, stack.shape[0] + 1)
    sel = (leads > a) & (leads <= b)
    if not sel.any():
        raise ValueError(f"no frames in lead bin {lead_bin_min}")
    part = stack[sel]
    with np.errstate(invalid="ignore"):
        out = np.nanmax(np.where(np.isfinite(part), part, -np.inf), axis=0)
    return np.where(np.isfinite(out), out, np.nan)
