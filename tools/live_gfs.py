"""Live atmospheric-ingredients feed (Phase A of live mode, see documentation/plan/ARCHITECTURE.md sec. 3): download the
latest NOAA GFS 0.25-degree analysis (f000, refreshed every 6 h, ~4 h publication delay) for the pilot region and
write it in the same NetCDF layout as the archival ERA5 files, so the EXISTING reader (ingest.read_met_states) and
feature code run unchanged -- only `met_source: met_nrt` in settings/settings.yaml points worker.py at it instead of
ERA5 (see worker.met_section). ERA5 itself cannot be used live: it is a reanalysis that arrives ~5 days late.

GFS is a different product from ERA5 (forecast-model analysis, not reanalysis) with its own biases; this script does
not claim it is equivalent -- that gap must be measured (architecture design principle 1), not assumed.

Verified on a real file (2026-09-24, tools/fetch_era5_cds.py's own AREA box): GFS 0.25-degree analysis shares
ERA5's exact grid alignment (same 0.25-degree lattice, e.g. 76.00, 76.25, ... degrees E), so no regridding is
needed and the existing terrain_on_met_grid.npz (built on the ERA5 grid) applies unchanged. Units match ERA5's
convention (T: K, q: kg/kg, u/v: m/s, sp: Pa, cape: J/kg) with ONE exception: GFS reports CIN as a NEGATIVE quantity
(observed range -98 to -0.4 J/kg) where this codebase's convention (pipeline.py, ERA5-derived) is POSITIVE magnitude
(observed ERA5 range 0 to 996 J/kg) -- this script takes abs(cin) to match. Downloaded via NOAA NOMADS' public GRIB
filter service (no login) and decoded with cfgrib/eccodes (pip-installable prebuilt wheel on Windows, no MSVC Build
Tools needed -- unlike pysteps, see documentation/plan/ROADMAP.md B3).

Run this every ~3 h (Windows Task Scheduler / cron); it finds the latest run+forecast-hour that is actually
published, skips it if already converted, and exits.

Usage: python tools/live_gfs.py [--out-dir data/landing/gfs_nrt] [--region settings/region.yaml]
"""
import argparse
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from meghprahari.settings import load_region  # noqa: E402

NOMADS_DIR = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
AREA = [33, 76, 30, 78]                # N, W, S, E -- same box as tools/fetch_era5_cds.py (shared grid)
LEVELS_HPA = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 500, 400, 300, 200]
RUN_HOURS = ("18", "12", "06", "00")   # most recent first


def latest_run(session, now=None, max_age_days=2):
    """(date, hour) of the most recent GFS run whose f000 analysis file is published, checked newest first."""
    now = now or datetime.now(timezone.utc)
    for days_back in range(max_age_days + 1):
        d = now - timedelta(days=days_back)
        for hh in RUN_HOURS:
            run_dt = d.replace(hour=int(hh), minute=0, second=0, microsecond=0)
            if run_dt > now:
                continue
            url = f"{NOMADS_DIR}/gfs.{d:%Y%m%d}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f000"
            r = session.head(url, timeout=30)
            if r.status_code == 200:
                return d.date(), hh
    raise RuntimeError(f"no published GFS run found in the last {max_age_days} days")


def filter_query(run_date, run_hour, area):
    top, left, bottom, right = area[0], area[1], area[2], area[3]
    levels = "".join(f"&lev_{lv}_mb=on" for lv in LEVELS_HPA)
    variables = "".join(f"&var_{v}=on" for v in ("TMP", "SPFH", "UGRD", "VGRD", "PRES", "CAPE", "CIN"))
    return (f"{NOMADS_FILTER}?dir=%2Fgfs.{run_date:%Y%m%d}%2F{run_hour}%2Fatmos"
            f"&file=gfs.t{run_hour}z.pgrb2.0p25.f000{variables}&lev_surface=on{levels}"
            f"&subregion=&leftlon={left}&rightlon={right}&toplat={top}&bottomlat={bottom}")


def download_grib(session, url, dst, timeout=180):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    if not r.content.startswith(b"GRIB"):
        raise RuntimeError("response is not a GRIB2 file (NOMADS filter query likely malformed)")
    with open(dst, "wb") as f:
        f.write(r.content)


def grib_to_netcdf(grib_path, out_path, when):
    """GFS GRIB2 (pressure-level + surface groups) -> one NetCDF file laid out like the archival ERA5 files
    (dims valid_time/pressure_level/latitude/longitude for q,t,u,v; valid_time/latitude/longitude for sp,cape,cin),
    so ingest.read_met_states reads it unchanged. `when`: the run's valid time (UTC)."""
    import netCDF4
    import numpy as np
    import xarray as xr

    pl = xr.open_dataset(grib_path, engine="cfgrib",
                         backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
    sfc = xr.open_dataset(grib_path, engine="cfgrib",
                          backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": "surface"}})
    # cfgrib returns latitude ascending (south to north); ERA5/CDS files (and terrain_on_met_grid.npz) are descending
    # (north to south) -- flip so this file matches that convention exactly (checked on a real file, 2026-09-24).
    pl, sfc = pl.isel(latitude=slice(None, None, -1)), sfc.isel(latitude=slice(None, None, -1))
    lat, lon = pl.latitude.values.astype(float), pl.longitude.values.astype(float)
    lev = pl.isobaricInhPa.values.astype(float)
    epoch = int((when - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds())

    with netCDF4.Dataset(out_path + ".part", "w") as f:
        f.createDimension("valid_time", 1)
        f.createDimension("pressure_level", lev.size)
        f.createDimension("latitude", lat.size)
        f.createDimension("longitude", lon.size)
        tv = f.createVariable("valid_time", "i8", ("valid_time",))
        tv.units, tv.calendar = "seconds since 1970-01-01", "standard"
        tv[:] = [epoch]
        lv = f.createVariable("pressure_level", "f4", ("pressure_level",))
        lv.units = "hPa"
        lv[:] = lev
        la = f.createVariable("latitude", "f4", ("latitude",))
        la.units = "degrees_north"
        la[:] = lat
        lo = f.createVariable("longitude", "f4", ("longitude",))
        lo.units = "degrees_east"
        lo[:] = lon
        for name, da, units in (("q", pl["q"], "kg kg**-1"), ("t", pl["t"], "K"),
                                ("u", pl["u"], "m s**-1"), ("v", pl["v"], "m s**-1")):
            v = f.createVariable(name, "f4", ("valid_time", "pressure_level", "latitude", "longitude"), zlib=True)
            v.units = units
            v[0] = np.asarray(da.values, np.float32)
        sp = f.createVariable("sp", "f4", ("valid_time", "latitude", "longitude"), zlib=True)
        sp.units = "Pa"
        sp[0] = np.asarray(sfc["sp"].values, np.float32)
        cape = f.createVariable("cape", "f4", ("valid_time", "latitude", "longitude"), zlib=True)
        cape.units, cape.long_name = "J kg**-1", "Convective available potential energy"
        cape[0] = np.asarray(sfc["cape"].values, np.float32)
        cin = f.createVariable("cin", "f4", ("valid_time", "latitude", "longitude"), zlib=True)
        cin.units = "J kg**-1"
        cin.long_name = "Convective inhibition (sign-flipped from GFS's raw negative convention to match ERA5)"
        cin[0] = np.abs(np.asarray(sfc["cin"].values, np.float32))
    os.replace(out_path + ".part", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/landing/gfs_nrt")
    ap.add_argument("--region", default="settings/region.yaml")
    args = ap.parse_args()

    load_region(args.region)             # validated but AREA (above) matches ERA5's training grid, not the bbox+margin
    os.makedirs(args.out_dir, exist_ok=True)
    session = requests.Session()
    run_date, run_hour = latest_run(session)
    when = datetime(run_date.year, run_date.month, run_date.day, int(run_hour), tzinfo=timezone.utc)
    out_path = os.path.join(args.out_dir, f"gfs_{when:%Y%m%d%H}.nc")
    if os.path.exists(out_path):
        print(f"already have {out_path}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        grib_path = os.path.join(tmp, "gfs.grib2")
        download_grib(session, filter_query(run_date, run_hour, AREA), grib_path)
        grib_to_netcdf(grib_path, out_path, when)
    print(f"wrote {out_path} (GFS run {when:%Y-%m-%d %HZ})")


if __name__ == "__main__":
    main()
