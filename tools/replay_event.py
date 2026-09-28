"""Offline replay of a real event through the SAME feature code the worker uses (pipeline.features_at), without a
database (step towards task M1). Writes one row per catchment per anchor time with all FEATURES_V1 columns.
No model is applied: no model has been trained on real data yet, and synthetic-data models must not be presented as
forecasts. Surface pressure is required; the script stops with a clear message if it is not configured.

Usage: python tools/replay_event.py [--event mandi_2025_06_30] [--settings settings/settings.yaml]
                                      [--region settings/region.yaml] [--atlas-dir data/atlas] [--data-root data]
                                      [--out data/work/replay_<event>_features.csv.gz]"""
import argparse
import glob
import os
import sys
from datetime import datetime, time, timezone

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from meghprahari import ingest as I, pipeline as P  # noqa: E402
from meghprahari.settings import load_region  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_frames  # noqa: E402
from inspect_inputs import resolve, unset  # noqa: E402


def replay(anchors, frames, axes, met_times, get_met, catch, rain_cfg, max_age_h, features=P.FEATURES_V1):
    """Features for every anchor with complete inputs: (DataFrame with 'anchor' + catchment index + features,
    {reason: count} for skipped anchors)."""
    parts, skipped = [], {}
    for t in anchors:
        f, why = P.features_at(t, frames, axes, met_times, get_met, catch, rain_cfg, max_age_h, features)
        if f is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        parts.append(f.assign(anchor=t))
    if not parts:
        return pd.DataFrame(columns=["anchor"] + list(features)), skipped
    return pd.concat(parts), skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="mandi_2025_06_30")
    ap.add_argument("--settings", default="settings/settings.yaml")
    ap.add_argument("--region", default="settings/region.yaml")
    ap.add_argument("--atlas-dir", default="data/atlas")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out")
    a = ap.parse_args()
    with open(a.settings, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    Mc = cfg["met"]
    missing = unset(Mc, ["time_var", "vars.ps", "ps_file"])
    if missing:
        sys.exit(f"cannot replay yet: met settings {missing} are not configured (ERA5 surface pressure is required; "
                 "download the single-level file and fill them in). Nothing is invented.")
    ev = next(e for e in load_region(a.region)["replay_events"] if e["id"] == a.event)
    start = datetime.combine(ev["download_start"], time.min, timezone.utc)
    end = datetime.combine(ev["download_end"], time.max, timezone.utc)
    src = cfg.get("rain_source", "hem")
    rain = cfg[src]
    frames, axes = load_frames(rain, a.data_root, start, end)
    if not frames:
        sys.exit(f"no {src} files for {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    terr = resolve(Mc["terrain_file"], a.data_root)
    import numpy as np
    t = np.load(terr)
    idx = I.MetIndex(Mc, t["h"], resolve(Mc["ps_file"], a.data_root), t["lat"], t["lon"])
    for p in sorted(glob.glob(os.path.join(resolve(Mc["dir"], a.data_root), Mc["pattern"]))):
        idx.add_file(p)
    catch = pd.read_csv(os.path.join(a.atlas_dir, "catchments.csv")).set_index("link_id")
    feats = P.FEATURE_SETS[cfg.get("features", "v1")]
    df, skipped = replay(sorted(frames), frames, axes, idx.times(), idx.get, catch, rain, Mc["max_age_h"], feats)
    out = a.out or os.path.join(a.data_root, "work", f"replay_{a.event}_features.csv.gz")
    df.to_csv(out, index_label="link_id")
    print(f"{src} frames: {len(frames)}; anchors with features: {df['anchor'].nunique()}; skipped: {skipped}")
    print(f"rows: {len(df)} -> {out}")
    print("missing fraction per feature:")
    print(df[feats].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
