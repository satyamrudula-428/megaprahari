"""Live rain feed (Phase A of live mode, see documentation/plan/ARCHITECTURE.md sec. 3): download new GPM IMERG *Early Run*
half-hourly granules (near-real-time, ~4 h latency) as they become available, for `worker.py --mode=live` to pick up
via the `imerg_nrt` config section (`rain_source: imerg_nrt`).

Early Run (GPM_3IMERGHHE V07) is NOT the same product as the archival GPM_3IMERGHH (Final Run) already in
data/landing/GPM_3IMERGHH_07/: Final Run is gauge-corrected and arrives ~3.5 months late, so it can only ever be used
for training/replay. Early Run has no gauge correction (satellite-only) and is what a live system actually gets;
verified on a real file (2026-09-24) to have the identical internal layout (Grid/precipitation, mm/hr, fill
-9999.9) as Final Run, so the existing imerg reader code and time_regex work unchanged -- only `imerg_nrt.dir` in
settings/settings.yaml points somewhere else. See documentation/DATA_REGISTER.md for the verification note.

Granule URLs are looked up via NASA's CMR (Common Metadata Repository) search API rather than hardcoded, because the
product-version suffix in the filename (currently V07C) changes with algorithm updates -- CMR always returns the
current one. Downloads need a NASA Earthdata login in ~/.netrc (same one used for the archival IMERG download; see
documentation/plan/DATA_SETUP.md).

Run this every ~30 min (Windows Task Scheduler / cron); each run catches up anything new in the last --hours-back
hours and skips files already on disk, so missing a run is harmless.

Usage: python tools/live_imerg.py [--hours-back 6] [--out-dir data/landing/GPM_3IMERGHHE_07_nrt]
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME = "GPM_3IMERGHHE"
VERSION = "07"


def parse_granules(cmr_json):
    """CMR granules.json response -> [(filename, download_url)], skipping entries with no direct .HDF5 data link."""
    out = []
    for e in cmr_json["feed"]["entry"]:
        url = next((l["href"] for l in e.get("links", [])
                    if l.get("href", "").startswith("https://data.gesdisc.earthdata.nasa.gov") and l["href"].endswith(".HDF5")), None)
        if url:
            out.append((os.path.basename(url), url))
    return out


def list_granules(start, end, page_size=200):
    """[(filename, download_url)] for Early-Run granules whose time range overlaps [start, end) (UTC)."""
    params = {"short_name": SHORT_NAME, "version": VERSION, "page_size": page_size,
              "temporal": f"{start:%Y-%m-%dT%H:%M:%SZ},{end:%Y-%m-%dT%H:%M:%SZ}"}
    r = requests.get(CMR_URL, params=params, timeout=60)
    r.raise_for_status()
    return parse_granules(r.json())


def download(url, dst, timeout=120):
    tmp = dst + ".part"
    with requests.get(url, stream=True, timeout=timeout) as r:      # requests reads ~/.netrc automatically
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours-back", type=float, default=6.0, help="how far back to check for new/updated granules")
    ap.add_argument("--out-dir", default="data/landing/GPM_3IMERGHHE_07_nrt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=args.hours_back)
    granules = list_granules(start, now)
    if not granules:
        print(f"no granules found for {start:%Y-%m-%d %H:%M} - {now:%Y-%m-%d %H:%M} UTC "
              "(Early Run usually lags ~4 h behind real time -- this is expected near the start of a run)")
        return

    n_ok, n_skip, n_fail = 0, 0, 0
    for name, url in sorted(granules):
        dst = os.path.join(args.out_dir, name)
        if os.path.exists(dst):
            n_skip += 1
            continue
        try:
            download(url, dst)
            print(f"downloaded {name}")
            n_ok += 1
        except requests.RequestException as e:
            print(f"FAILED {name}: {e}")
            n_fail += 1

    print(f"\n{n_ok} downloaded, {n_skip} already present, {n_fail} failed "
          f"(checked {len(granules)} granules from the last {args.hours_back:g} h).")
    if n_fail and n_ok == 0 and n_skip == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
