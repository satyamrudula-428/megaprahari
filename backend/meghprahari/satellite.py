"""Rain dynamics from a stack of satellite rain-rate frames (mm/h, shape (T, H, W), NaN = missing)."""
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def linear_trend(frames, dt_h):
    """Per-pixel least-squares slope (mm/h per hour) ignoring NaNs; NaN if fewer than 3 valid frames."""
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


def exceed_growth(frames, thr, half_window_px=2):
    """Change in local fraction of pixels >= thr between the first and second half of the window."""
    h = frames.shape[0] // 2
    ex = np.where(np.isfinite(frames), (frames >= thr).astype(float), np.nan)
    with np.errstate(invalid="ignore"):
        a = np.nan_to_num(np.nanmean(ex[:h], 0))
        b = np.nan_to_num(np.nanmean(ex[h:], 0))
    size = 2 * half_window_px + 1
    return ndimage.uniform_filter(b, size) - ndimage.uniform_filter(a, size)


def motion_tiles(a, b, dt_min, pixel_km, tile=32, min_rain=1.0, min_frac=0.02):
    """Storm motion (km/h) per tile by cross-correlation between frame a and the later frame b.
    Returns (vy, vx): +vy = southward (rows increase), +vx = eastward (columns increase). NaN if too little rain."""
    H, W = a.shape
    ny, nx = H // tile, W // tile
    vy = np.full((ny, nx), np.nan)
    vx = np.full((ny, nx), np.nan)
    hours = dt_min / 60.0
    for i in range(ny):
        for j in range(nx):
            A = np.nan_to_num(a[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile])
            B = np.nan_to_num(b[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile])
            if (A >= min_rain).mean() < min_frac or (B >= min_rain).mean() < min_frac:
                continue
            shift = phase_cross_correlation(A, B, upsample_factor=4, normalization=None)[0]
            vy[i, j], vx[i, j] = -shift[0] * pixel_km / hours, -shift[1] * pixel_km / hours
    return vy, vx


def speed_field(vy, vx, shape, tile=32):
    """Expand per-tile speed (km/h) to a per-pixel map of the given shape."""
    sp = np.hypot(vy, vx)
    iy = np.minimum(np.arange(shape[0]) // tile, sp.shape[0] - 1)
    ix = np.minimum(np.arange(shape[1]) // tile, sp.shape[1] - 1)
    return sp[np.ix_(iy, ix)]


def cell_area_map(frame, thr, pixel_km2):
    """Area (km2) of the contiguous rain cell (>= thr, 8-connected) containing each pixel; 0 elsewhere."""
    lab, _ = ndimage.label(np.nan_to_num(frame) >= thr, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return sizes[lab] * pixel_km2


def stall_index(area_km2, speed_kmh, v_min=1.0):
    """Residence time (h) of a rain cell = sqrt(area) / speed. Hypothesis feature; NaN if no cell or no motion estimate."""
    area = np.asarray(area_km2, float)
    speed = np.asarray(speed_kmh, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(area > 0, np.sqrt(area) / np.maximum(speed, v_min), np.nan)
