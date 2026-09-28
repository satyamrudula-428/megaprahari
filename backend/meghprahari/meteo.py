"""Atmospheric precursors from pressure-level fields. Units: q kg/kg, pressure hPa, winds m/s, T degC/K as named."""
import numpy as np

G = 9.80665
R_EARTH = 6371000.0
LOW_MARGIN_HPA = 25.0


def iwv(q, p_hpa, ps_hpa=None):
    """Integrated water vapour (kg/m2) = (1/g) * integral q dp over the levels above ground.
    q: (K, ...) ordered surface->top like p_hpa (strictly decreasing). Levels below the surface
    (p > ps) are ignored; the layer between the surface and the lowest valid level uses that level's q."""
    q = np.asarray(q, float)
    p = np.asarray(p_hpa, float)
    if p.ndim != 1 or np.any(np.diff(p) >= 0):
        raise ValueError("p_hpa must be 1-D and strictly decreasing (surface -> top)")
    if q.shape[0] != p.size:
        raise ValueError("q first axis must match p_hpa")
    shape = q.shape[1:]
    ps = np.full(shape, p[0]) if ps_hpa is None else np.asarray(ps_hpa, float)
    valid = p.reshape((-1,) + (1,) * len(shape)) <= ps
    total = np.zeros(shape)
    seen = np.zeros(shape, bool)
    for k in range(p.size):
        vk = valid[k]
        total += np.where(vk & ~seen, q[k] * (ps - p[k]), 0.0)
        if k + 1 < p.size:
            total += np.where(vk & valid[k + 1], 0.5 * (q[k] + q[k + 1]) * (p[k] - p[k + 1]), 0.0)
        seen |= vk
    return total * 100.0 / G


def vapor_pressure_hpa(q, p_hpa):
    return q * p_hpa / (0.622 + 0.378 * q)


def dewpoint_c(e_hpa):
    """Inverse Bolton (1980)."""
    x = np.log(np.maximum(e_hpa, 1e-6) / 6.112)
    return 243.5 * x / (17.67 - x)


def k_index(t850, td850, t700, td700, t500):
    return (t850 - t500) + td850 - (t700 - td700)


def total_totals(t850, td850, t500):
    return (t850 - t500) + (td850 - t500)


def divergence(fx, fy, lat, lon):
    """Spherical divergence (1/s) of a horizontal vector field on a regular lat/lon grid."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    coslat = np.cos(np.deg2rad(lat))[:, None]
    dlam = np.deg2rad(lon[1] - lon[0])
    dphi = np.deg2rad(lat[1] - lat[0])
    return (np.gradient(fx, axis=1) / dlam + np.gradient(fy * coslat, axis=0) / dphi) / (R_EARTH * coslat)


def mfc(q, u, v, lat, lon):
    """Moisture-flux convergence = -div(qV) in 1/s * kg/kg. Positive = moisture converging."""
    return -divergence(q * u, q * v, lat, lon)


def gradient(h, lat, lon):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    coslat = np.cos(np.deg2rad(lat))[:, None]
    dhdx = np.gradient(h, axis=1) / (R_EARTH * coslat * np.deg2rad(lon[1] - lon[0]))
    dhdy = np.gradient(h, axis=0) / (R_EARTH * np.deg2rad(lat[1] - lat[0]))
    return dhdx, dhdy


def omfi(q, u, v, terrain_m, lat, lon):
    """Orographic moisture-flux index = q * (V . grad h). Positive = moist upslope flow."""
    dhdx, dhdy = gradient(terrain_m, lat, lon)
    return q * (u * dhdx + v * dhdy)


def bulk_shear(u_hi, v_hi, u_lo, v_lo):
    return np.hypot(u_hi - u_lo, v_hi - v_lo)


def lowest_above_ground(field, p_hpa, ps_hpa, margin=LOW_MARGIN_HPA):
    """Value of field (K,y,x) at the lowest level at least `margin` hPa above the surface; NaN if none."""
    p = np.asarray(p_hpa, float)
    out = np.full(field.shape[1:], np.nan)
    seen = np.zeros(out.shape, bool)
    for k in range(p.size):
        ok = (p[k] <= ps_hpa - margin) & ~seen
        out = np.where(ok, field[k], out)
        seen |= ok
    return out


def ps_from_terrain(h_m):
    """Standard-atmosphere surface pressure (hPa). Approximate fallback only."""
    return 1013.25 * (1.0 - 2.25577e-5 * np.asarray(h_m, float)) ** 5.25588


def _edges(c):
    c = np.asarray(c, float)
    d = np.diff(c)
    return np.concatenate([[c[0] - d[0] / 2], c[:-1] + d / 2, [c[-1] + d[-1] / 2]])


def _bin(src, dst):
    dst = np.asarray(dst, float)
    e = _edges(dst)
    s = np.asarray(src, float)
    idx = (np.searchsorted(-e, -s, side="right") if dst[0] > dst[-1] else np.searchsorted(e, s, side="right")) - 1
    return np.where((idx >= 0) & (idx < dst.size), idx, -1)


def regrid_mean(values, src_lat, src_lon, dst_lat, dst_lon):
    """Mean of fine-grid values inside each coarse cell (e.g. 30 m SRTM -> 12 km met grid). NaN where empty."""
    iy, ix = _bin(src_lat, dst_lat), _bin(src_lon, dst_lon)
    ny, nx = len(dst_lat), len(dst_lon)
    yy, xx = np.meshgrid(iy, ix, indexing="ij")
    ok = (yy >= 0) & (xx >= 0) & np.isfinite(values)
    flat = yy[ok] * nx + xx[ok]
    num = np.bincount(flat, weights=values[ok], minlength=ny * nx)
    den = np.bincount(flat, minlength=ny * nx)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan).reshape(ny, nx)


def cape_cin_metpy(p_hpa, t_c, td_c):
    """Surface-parcel CAPE/CIN (J/kg) via MetPy. Optional; not used in FEATURES_V1 (too slow per column
    for national grids and MetPy is not part of the tested core)."""
    from metpy.calc import cape_cin, parcel_profile
    from metpy.units import units
    p, t, td = p_hpa * units.hPa, t_c * units.degC, td_c * units.degC
    cape, cin = cape_cin(p, t, td, parcel_profile(p, t[0], td[0]).to("degC"))
    return float(cape.m), float(cin.m)
