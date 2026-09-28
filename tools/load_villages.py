"""Load villages (CSV: name,state,district,lat,lon,population,action_cost,loss) and link each to its catchment and
refuge using the atlas arrays. UNTESTED on real data. action_cost/loss are policy inputs agreed with the authority.
population, state and district may be blank (stored as NULL); population is only used for impact/exposure display."""
import argparse
import json
import os

import numpy as np
import psycopg
from affine import Affine
import pandas as pd


def opt_int(v):
    """CSV cell -> int, or None if blank (e.g. population not yet filled from the Census). Rejects negatives."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    n = int(float(str(v).replace(",", "")))
    if n < 0:
        raise ValueError(f"negative value: {v}")
    return n


def opt_str(v):
    """CSV cell -> stripped string, or None if blank."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    return str(v).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--atlas-dir", required=True)
    ap.add_argument("--atlas-id", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.atlas_dir, "meta.json")))
    inv = ~Affine(*meta["transform"])
    labels = np.load(os.path.join(a.atlas_dir, "labels.npy"))
    ld = lambda n: np.load(os.path.join(a.atlas_dir, n)) if os.path.exists(os.path.join(a.atlas_dir, n)) else None
    rd, rb = ld("refuge_dist.npy"), ld("refuge_bearing.npy")
    df = pd.read_csv(a.csv)
    n_ok = 0
    with psycopg.connect(os.environ["MP_DATABASE_URL"]) as c:
        for v in df.itertuples():
            col, row = inv * (v.lon, v.lat)
            row, col = int(np.floor(row)), int(np.floor(col))
            if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]) or labels[row, col] < 0:
                print(f"skip {v.name}: outside the atlas tile or not draining to a mapped stream")
                continue
            cid = c.execute("SELECT id FROM catchment WHERE atlas_id=%s AND link_id=%s", (a.atlas_id, int(labels[row, col]))).fetchone()
            if not cid:
                print(f"skip {v.name}: catchment not in DB")
                continue
            dist = None if rd is None or not np.isfinite(rd[row, col]) else float(rd[row, col])
            brg = None if rb is None or dist is None else float(rb[row, col])
            c.execute("INSERT INTO village (name,state,district,geom,population,action_cost,loss,catchment_id,refuge_dist_m,refuge_bearing_deg) "
                      "VALUES (%s,%s,%s,ST_SetSRID(ST_MakePoint(%s,%s),4326),%s,%s,%s,%s,%s,%s)",
                      (v.name, opt_str(v.state), opt_str(v.district), v.lon, v.lat, opt_int(v.population),
                       float(v.action_cost), float(v.loss), cid[0], dist, brg))
            n_ok += 1
    print(f"loaded {n_ok}/{len(df)} villages")


if __name__ == "__main__":
    main()
