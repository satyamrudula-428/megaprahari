"""Exposure layer (task E8): what is at risk in each micro-catchment. Pure functions over the atlas grid:
points (schools, health facilities, villages) and lines (roads) are assigned to the catchment (stream-link id) whose
cell they fall in. Coordinates in degrees (EPSG:4326); the atlas transform is the 6-element affine from meta.json
(a = lon step, c = west edge, e = -lat step, f = north edge). Road lengths in km."""
import math

import numpy as np
import pandas as pd

R_EARTH_M = 6371000.0


def cells(lat, lon, transform, shape):
    """(row, col) of each point on the atlas grid; -1 for points outside it."""
    a, _, c, _, e, f = transform
    row = np.floor((np.asarray(lat, float) - f) / e).astype(int)
    col = np.floor((np.asarray(lon, float) - c) / a).astype(int)
    ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    return np.where(ok, row, -1), np.where(ok, col, -1)


def link_of_points(lat, lon, labels, transform):
    """Catchment (link id) containing each point; -1 when outside the atlas or not draining to a mapped stream."""
    r, c = cells(lat, lon, transform, labels.shape)
    ok = r >= 0
    out = np.full(r.shape, -1)
    out[ok] = labels[r[ok], c[ok]]
    return out


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_M * math.asin(math.sqrt(h))


def road_km_per_link(lines, labels, transform, step_m=30.0):
    """{link id: km of line inside that catchment}. lines: iterable of [(lat, lon), ...] polylines. Each segment is
    sampled every ~step_m metres and each sample carries its share of the segment length."""
    acc = {}
    for line in lines:
        for (la1, lo1), (la2, lo2) in zip(line, line[1:]):
            d = haversine_m(la1, lo1, la2, lo2)
            if d == 0:
                continue
            n = max(1, int(math.ceil(d / step_m)))
            fr = (np.arange(n) + 0.5) / n
            links = link_of_points(la1 + fr * (la2 - la1), lo1 + fr * (lo2 - lo1), labels, transform)
            for lk in links[links >= 0]:
                acc[int(lk)] = acc.get(int(lk), 0.0) + d / n / 1000.0
    return acc


def exposure_table(link_ids, points_by_kind, roads_km):
    """DataFrame indexed by link_id with one count column per point kind (e.g. schools, health, villages) and road_km.
    points_by_kind: {kind: array of link ids (-1 ignored)}; roads_km: {link id: km}."""
    t = pd.DataFrame(index=pd.Index(sorted(int(x) for x in link_ids), name="link_id"))
    for kind, links in points_by_kind.items():
        s = pd.Series(np.asarray(links)[np.asarray(links) >= 0]).value_counts()
        t[kind] = s.reindex(t.index).fillna(0).astype(int)
    t["road_km"] = pd.Series(roads_km, dtype=float).reindex(t.index).fillna(0.0).round(3)
    return t
