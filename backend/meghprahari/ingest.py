"""Landing-zone ingestion helpers. Pure helpers (scan, name->time, QC, cadence) are unit-tested.
The file READERS at the bottom need h5py/netCDF4 and the product's variable names (settings/settings.yaml). read_hem_raw
and read_met_states are unit-tested on synthetic files with the real layouts and have loaded the real 2025 IMERG, MERA
and ERA5 files (tools/inspect_inputs.py); MOSDAC HEM and the one-time-per-file read_met_state are still UNTESTED on
real files. All readers validate shapes/regularity strictly so a wrong mapping fails loudly."""
import glob
import hashlib
import os
import re
import time
from datetime import datetime, timezone

import numpy as np


class IngestError(Exception):
    pass


def sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_new_files(directory, pattern, seen_sha, min_age_s=10, now=None, cache=None):
    """Stable (not modified for min_age_s) files whose SHA-256 is not in seen_sha. `cache` avoids re-hashing."""
    now = time.time() if now is None else now
    cache = {} if cache is None else cache
    out = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        st = os.stat(path)
        if now - st.st_mtime < min_age_s:
            continue
        key = (st.st_size, st.st_mtime)
        if path not in cache or cache[path][0] != key:
            cache[path] = (key, sha256_path(path))
        sha = cache[path][1]
        if sha not in seen_sha:
            out.append((path, sha))
    return out


def parse_time_from_name(name, regex, fmt):
    m = re.search(regex, name)
    if not m:
        raise IngestError(f"file name '{name}' does not match the configured time regex")
    try:
        return datetime.strptime(m.group(1), fmt).replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise IngestError(f"cannot parse time from '{m.group(1)}': {e}")


def qc_rain_frame(raw, fill_values, scale=1.0, offset=0.0, max_mm_h=500.0, max_missing_frac=0.5):
    """Raw product array -> mm/h float32 with NaN for fill/negative/implausible values."""
    a = np.asarray(raw, np.float32).copy()
    for fv in fill_values:
        a[a == fv] = np.nan
    a = a * np.float32(scale) + np.float32(offset)
    a[(a < 0) | (a > max_mm_h)] = np.nan
    frac = float(np.isnan(a).mean())
    if frac > max_missing_frac:
        raise IngestError(f"frame rejected: {frac:.0%} missing/invalid")
    return a, frac


def qc_ctt_frame(raw, valid_min=140.0, valid_max=375.0, max_missing_frac=0.5):
    """Brightness-temperature array (already in Kelvin: netCDF4 applies the file's own scale_factor/add_offset/
    _FillValue on read) -> float32 with NaN outside the physically valid range. Same fail-loud contract as
    qc_rain_frame. valid_min/max default to GridSat-B1's own 'valid_range' attribute (140-375 K)."""
    a = np.asarray(raw, np.float32).copy()
    a[(a < valid_min) | (a > valid_max)] = np.nan
    frac = float(np.isnan(a).mean())
    if frac > max_missing_frac:
        raise IngestError(f"frame rejected: {frac:.0%} missing/invalid")
    return a, frac


def check_cadence(times, expected_min=30, tol_min=5):
    """Return (t_prev, t_next, gap_min) for every gap longer than expected+tol."""
    ts = sorted(times)
    return [(a, b, (b - a).total_seconds() / 60) for a, b in zip(ts, ts[1:])
            if (b - a).total_seconds() / 60 > expected_min + tol_min]


def _axis(a, name):
    a = np.asarray(a, float)
    if a.ndim == 2:
        a2 = a
        a = a2[:, 0] if name == "lat" else a2[0, :]
        ref = a[:, None] if name == "lat" else a[None, :]
        if not np.allclose(a2, ref):
            raise IngestError(f"{name} is a curvilinear 2-D field; regrid to a regular lat/lon grid first")
    d = np.diff(a)
    if a.size < 3 or not np.allclose(d, d[0], rtol=1e-3, atol=1e-6):
        raise IngestError(f"{name} axis is not regular; regrid first")
    return a


def crop_to_bbox(arr, lat, lon, bbox):
    """Cut a (lat, lon) array to bbox {lat_min, lat_max, lon_min, lon_max} (degrees, inclusive); axes keep their
    direction. Raises IngestError if nothing is left."""
    iy = np.flatnonzero((lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]))
    ix = np.flatnonzero((lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"]))
    if iy.size < 3 or ix.size < 3:
        raise IngestError("crop bbox leaves fewer than 3 grid points on an axis; check the bbox and the file's grid")
    return arr[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], lat[iy[0]:iy[-1] + 1], lon[ix[0]:ix[-1] + 1]


def read_hem_raw(path, c):
    """Satellite rain-rate reader (HEM, IMERG, MERA...). Returns (raw 2-D array (lat, lon), lat, lon); raw values are
    NOT yet scaled or QC'd (use qc_rain_frame). c = one rain section of the settings:
    var/lat_var/lon_var (names as in the file), optional axis_order ("lat,lon" default, or "lon,lat" when the file
    stores longitude first, e.g. IMERG) and optional bbox (crop, degrees)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5", ".he5"):
        import h5py
        with h5py.File(path, "r") as f:
            arr, lat, lon = f[c["var"]][()], f[c["lat_var"]][()], f[c["lon_var"]][()]
    elif ext in (".nc", ".nc4"):
        import netCDF4
        with netCDF4.Dataset(path) as f:
            arr = np.ma.filled(f[c["var"]][:].astype(float), np.nan)
            lat, lon = f[c["lat_var"]][:], f[c["lon_var"]][:]
    else:
        raise IngestError(f"unsupported file type: {ext}")
    arr = np.squeeze(arr)
    lat, lon = _axis(lat, "lat"), _axis(lon, "lon")
    order = c.get("axis_order", "lat,lon")
    if order == "lon,lat":
        arr = arr.T
    elif order != "lat,lon":
        raise IngestError(f"axis_order must be 'lat,lon' or 'lon,lat', got '{order}'")
    if arr.ndim != 2 or arr.shape != (lat.size, lon.size):
        raise IngestError(f"array shape {arr.shape} does not match lat/lon ({lat.size}, {lon.size}); check axis_order")
    if c.get("bbox"):
        arr, lat, lon = crop_to_bbox(arr, lat, lon, c["bbox"])
    return arr, lat, lon


def _utc_times(tvar):
    """netCDF time variable -> list of timezone-aware UTC datetimes (the files' time units are UTC)."""
    import netCDF4
    out = []
    for t in netCDF4.num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"), only_use_cftime_datetimes=False):
        out.append(datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, tzinfo=timezone.utc))
    return out


def met_file_times(path, c):
    """Valid times (UTC) stored in a multi-time met file; c = settings['met'] with time_var set."""
    import netCDF4
    with netCDF4.Dataset(path) as f:
        return _utc_times(f[c["time_var"]])


def read_met_states(path, c, terrain_m, ps_path=None, times=None):
    """Reader for a NetCDF pressure-level file holding MANY times (e.g. ERA5 from the Copernicus CDS), dims
    (time, level, lat, lon). Returns {UTC datetime: MetState}. c = settings['met'] with time_var set.
    Surface pressure: variable vars.ps from this file, or, if absent here, from ps_path (a single-level file with the
    same grid and the same time variable name). Missing surface pressure raises IngestError (never invented).
    Optional c['surface_vars'] = {name: file variable} (e.g. cape, cin in J/kg) are read from the same file as surface
    pressure into MetState.extra; a configured field that is absent raises IngestError. Values the product leaves
    undefined (e.g. ERA5 CIN without a level of free convection) stay NaN.
    times: optional iterable of datetimes to read (default: all). Units via c['units'] (q kg/kg, ps and level hPa)."""
    import netCDF4
    from .pipeline import MetState
    v, u = c["vars"], c["units"]
    with netCDF4.Dataset(path) as f:
        all_t = _utc_times(f[c["time_var"]])
        want = all_t if times is None else [t for t in all_t if t in set(times)]
        if not want:
            return {}
        lat, lon = _axis(f[v["lat"]][:], "lat"), _axis(f[v["lon"]][:], "lon")
        lev_all = np.asarray(f[v["level"]][:], float) * u["level_scale"]
        keep = level_subset(lev_all, c.get("levels_hpa"))
        lev = lev_all[keep]
        fields = {}
        for name in ("q", "t", "u", "v"):
            a = f[v[name]]
            if a.ndim != 4 or a.shape[1:] != (lev_all.size, lat.size, lon.size):
                raise IngestError(f"variable '{v[name]}' must be (time, level, lat, lon); got {a.shape}")
            fields[name] = a
        ps_here = v["ps"] in f.variables
        idx = [all_t.index(t) for t in want]
        data = {n: np.ma.filled(fields[n][idx].astype(float), np.nan)[:, keep] for n in fields}
        surf_names = dict(c.get("surface_vars") or {})
        extra = {}
        if ps_here:
            ps = np.ma.filled(f[v["ps"]][idx].astype(float), np.nan)
            extra = {k: _surface(f, name, idx, path) for k, name in surf_names.items()}
    if not ps_here:
        if not ps_path:
            raise IngestError(f"surface pressure '{v['ps']}' is not in {os.path.basename(path)} and no ps file was given")
        with netCDF4.Dataset(ps_path) as g:
            if not (np.allclose(np.asarray(g[v["lat"]][:], float), lat) and np.allclose(np.asarray(g[v["lon"]][:], float), lon)):
                raise IngestError("surface-pressure file grid differs from the pressure-level grid")
            pt = _utc_times(g[c["time_var"]])
            missing = [t for t in want if t not in pt]
            if missing:
                raise IngestError(f"surface-pressure file lacks {len(missing)} time(s), first {missing[0].isoformat()}")
            gidx = [pt.index(t) for t in want]
            ps = np.ma.filled(g[v["ps"]][gidx].astype(float), np.nan)
            extra = {k: _surface(g, name, gidx, ps_path) for k, name in surf_names.items()}
    ps = ps * u["ps_scale"]
    order = np.argsort(-lev)
    out = {}
    for k, t in enumerate(want):
        out[t] = MetState(t, lat, lon, lev[order], data["q"][k][order] * u["q_scale"], data["t"][k][order],
                          data["u"][k][order], data["v"][k][order], ps[k], terrain_m,
                          {n: a[k] for n, a in extra.items()}).validate()
    return out


def level_subset(levels_hpa, wanted):
    """Indices of `wanted` pressure levels (hPa) within the file's levels; all levels when wanted is None. Missing
    levels raise IngestError, so training and test data always use the same vertical levels (met.levels_hpa)."""
    if not wanted:
        return np.arange(len(levels_hpa))
    idx = []
    for w in wanted:
        hit = np.flatnonzero(np.isclose(levels_hpa, float(w)))
        if hit.size != 1:
            raise IngestError(f"pressure level {w} hPa (met.levels_hpa) is not in the file")
        idx.append(int(hit[0]))
    return np.array(sorted(idx, key=lambda i: -levels_hpa[i]))


def _surface(ds, name, idx, path):
    """(len(idx), lat, lon) float array of a single-level variable; NaN where the product leaves it undefined."""
    if name not in ds.variables:
        raise IngestError(f"surface variable '{name}' configured in met.surface_vars is not in {os.path.basename(path)}")
    return np.ma.filled(ds[name][idx].astype(float), np.nan)


class MetIndex:
    """Met analyses spread over files that each hold MANY times (e.g. ERA5). add_file() records which file holds which
    time; get(t) reads that single time on demand and keeps the most recent `keep` states in memory. terrain_lat/lon
    (optional) must match the met grid (checked on every read)."""

    def __init__(self, c, terrain_m, ps_path=None, terrain_lat=None, terrain_lon=None, keep=12):
        self.c, self.terrain, self.ps_path, self.keep = c, terrain_m, ps_path, keep
        self.terrain_axes = (terrain_lat, terrain_lon)
        self.where, self.cache, self.ps_of = {}, {}, {}

    def add_file(self, path, ps_path=None):
        """Index the times in `path`; ps_path overrides the default surface-pressure file for this file."""
        ts = met_file_times(path, self.c)
        for t in ts:
            self.where[t] = path
        if ps_path:
            self.ps_of[path] = ps_path
        return ts

    def times(self):
        return sorted(self.where)

    def get(self, t):
        if t not in self.cache:
            if t not in self.where:
                raise IngestError(f"no met file holds {t.isoformat()}")
            path = self.where[t]
            m = read_met_states(path, self.c, self.terrain, self.ps_of.get(path, self.ps_path), times=[t])[t]
            la, lo = self.terrain_axes
            if la is not None and not (np.allclose(m.lat, la) and np.allclose(m.lon, lo)):
                raise IngestError("met grid does not match terrain_on_met_grid file")
            for k in sorted(self.cache)[:max(0, len(self.cache) - self.keep + 1)]:
                del self.cache[k]
            self.cache[t] = m
            return m
        return self.cache[t]


def read_met_state(path, c, when, terrain_m):
    """UNTESTED reader for a NetCDF pressure-level file (one time per file). c = settings['met']."""
    import netCDF4
    from .pipeline import MetState
    v, u = c["vars"], c["units"]
    with netCDF4.Dataset(path) as f:
        lat, lon = _axis(f[v["lat"]][:], "lat"), _axis(f[v["lon"]][:], "lon")
        lev = np.asarray(f[v["level"]][:], float) * u["level_scale"]

        def grab(name):
            a = np.ma.filled(f[v[name]][:].astype(float), np.nan)
            if a.ndim == 4 and a.shape[0] == 1:
                a = a[0]
            if a.ndim != 3 or a.shape[0] != lev.size:
                raise IngestError(f"variable '{v[name]}' must be (level, lat, lon) for a single time; got {a.shape}")
            return a

        q, t, uu, vv = grab("q") * u["q_scale"], grab("t"), grab("u"), grab("v")
        ps = np.squeeze(np.ma.filled(f[v["ps"]][:].astype(float), np.nan)) * u["ps_scale"]
    order = np.argsort(-lev)
    return MetState(when, lat, lon, lev[order], q[order], t[order], uu[order], vv[order], ps, terrain_m).validate()
