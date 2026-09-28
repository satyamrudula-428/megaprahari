"""Load real input files with the configured readers and print what they contain: file count, time range, gaps,
grid, and value ranges (tasks C5, C6, C8). It uses ONLY names from settings/settings.yaml; a CHANGE_ME value makes that
source report "not configured" instead of guessing. Paths starting with /data/ (Docker) map to --data-root.

Usage: python tools/inspect_inputs.py [--settings settings/settings.yaml] [--data-root data]
                                        [--write-met-axes data/work/met_axes.npz]"""
import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from meghprahari import ingest as I  # noqa: E402


def resolve(path, data_root):
    """Map a Docker path (/data/...) to the local data folder; other paths are returned unchanged."""
    if path and path.replace("\\", "/").startswith("/data/"):
        return os.path.join(data_root, path.replace("\\", "/")[len("/data/"):])
    return path


def unset(section, keys):
    """Names in `keys` that are missing or still CHANGE_ME in a settings section (possibly nested with '.')."""
    out = []
    for k in keys:
        node = section
        for part in k.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node in (None, "CHANGE_ME"):
            out.append(k)
    return out


def value_summary(a):
    """Dict with NaN fraction, min, mean, max of an array (NaN-aware); None values if all NaN."""
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if not ok.any():
        return dict(nan_frac=1.0, min=None, mean=None, max=None)
    return dict(nan_frac=float(1 - ok.mean()), min=float(a[ok].min()), mean=float(a[ok].mean()), max=float(a[ok].max()))


def fmt(s):
    if s["min"] is None:
        return "all missing"
    return f"min {s['min']:.3g}  mean {s['mean']:.3g}  max {s['max']:.3g}  missing {s['nan_frac']:.1%}"


def inspect_rain(name, c, data_root):
    print(f"\n=== {name}")
    missing = unset(c, ["dir", "pattern", "time_regex", "time_format", "var", "lat_var", "lon_var"])
    if missing:
        print(f"  not configured: {missing}")
        return
    d = resolve(c["dir"], data_root)
    files = sorted(glob.glob(os.path.join(d, c["pattern"])))
    if not files:
        print(f"  no files matching {c['pattern']} in {d}")
        return
    times = {}
    for p in files:
        try:
            times[I.parse_time_from_name(os.path.basename(p), c["time_regex"], c["time_format"])] = p
        except I.IngestError as e:
            print(f"  name error: {e}")
    ts = sorted(times)
    print(f"  files: {len(files)}  times: {ts[0]:%Y-%m-%d %H:%M} -> {ts[-1]:%Y-%m-%d %H:%M} UTC")
    gaps = I.check_cadence(ts, c["cadence_min"])
    print(f"  cadence {c['cadence_min']} min: {len(gaps)} gap(s)" + (f", largest {max(g[2] for g in gaps):.0f} min" if gaps else ""))
    peak_t, peak = None, -1.0
    for t in ts:
        raw, lat, lon = I.read_hem_raw(times[t], c)
        frame, _ = I.qc_rain_frame(raw, c["fill_values"], c.get("scale", 1.0), c.get("offset", 0.0), c["max_mm_h"], 1.0)
        m = np.nanmax(frame) if np.isfinite(frame).any() else -1.0
        if m > peak:
            peak_t, peak = t, m
    print(f"  grid: {lat.size} x {lon.size}  lat {lat[0]:.3f}..{lat[-1]:.3f}  lon {lon[0]:.3f}..{lon[-1]:.3f}  "
          f"step {abs(lat[1] - lat[0]):.4f} deg")
    raw, _, _ = I.read_hem_raw(times[peak_t], c)
    frame, _ = I.qc_rain_frame(raw, c["fill_values"], c.get("scale", 1.0), c.get("offset", 0.0), c["max_mm_h"], 1.0)
    print(f"  strongest frame {peak_t:%Y-%m-%d %H:%M} UTC: {fmt(value_summary(frame))}")
    for flag in ("units_confirmed", "time_zone_confirmed"):
        if c.get(flag) is False:
            print(f"  WARNING: {flag} is false in settings")


def inspect_met(c, data_root, axes_out=None):
    print("\n=== met")
    missing = unset(c, ["dir", "pattern", "time_var", "vars.q", "vars.t", "vars.u", "vars.v", "vars.level",
                        "vars.lat", "vars.lon"])
    if missing:
        print(f"  not configured: {missing}")
        return
    files = sorted(glob.glob(os.path.join(resolve(c["dir"], data_root), c["pattern"])))
    if not files:
        print("  no files")
        return
    import netCDF4
    for p in files:
        ts = I.met_file_times(p, c)
        print(f"  {os.path.basename(p)}: {len(ts)} times {ts[0]:%Y-%m-%d %H:%M} -> {ts[-1]:%Y-%m-%d %H:%M} UTC")
        v = c["vars"]
        with netCDF4.Dataset(p) as f:
            lat, lon, lev = f[v["lat"]][:], f[v["lon"]][:], f[v["level"]][:]
            print(f"  grid {lat.size} x {lon.size}  lat {lat[0]:.2f}..{lat[-1]:.2f}  lon {lon[0]:.2f}..{lon[-1]:.2f}  "
                  f"levels {lev.size}: {lev.max():.0f}..{lev.min():.0f} hPa")
            for name in ("q", "t", "u", "v"):
                print(f"  {name} ({v[name]}, {getattr(f[v[name]], 'units', '?')}): {fmt(value_summary(np.ma.filled(f[v[name]][:].astype(float), np.nan)))}")
        if axes_out:
            os.makedirs(os.path.dirname(os.path.abspath(axes_out)), exist_ok=True)
            np.savez(axes_out, lat=np.asarray(lat, float), lon=np.asarray(lon, float))
            print(f"  wrote met grid axes to {axes_out} (for build_atlas.py --met-axes)")
    ps_missing = unset(c, ["vars.ps", "ps_file"])
    if ps_missing:
        print(f"  surface pressure not configured ({ps_missing}): MetState cannot be built yet (no fallback is invented)")
        return
    terr = resolve(c["terrain_file"], data_root)
    if not os.path.exists(terr):
        print(f"  terrain file {terr} missing: run build_atlas.py with --met-axes")
        return
    t = np.load(terr)
    states = I.read_met_states(files[0], c, t["h"], ps_path=resolve(c["ps_file"], data_root))
    m = states[min(states)]
    print(f"  MetState OK for {len(states)} times; surface pressure {fmt(value_summary(m.ps_hpa))} hPa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="settings/settings.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--write-met-axes")
    a = ap.parse_args()
    with open(a.settings, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for name in ("hem", "imerg", "mera"):
        if name in cfg:
            c = dict(cfg[name])
            if name == "hem":
                c.setdefault("dir", cfg["landing"]["hem_dir"])
            inspect_rain(name, c, a.data_root)
    inspect_met(cfg["met"], a.data_root, a.write_met_axes)


if __name__ == "__main__":
    main()
