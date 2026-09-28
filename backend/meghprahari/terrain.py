"""Terrain intelligence: void/depression filling, D8 flow, accumulation, HAND, stream links,
micro-catchments, Kirpich time of concentration, refuge distance.

Pure numpy/scipy. Sized for tiles of ~1e6 cells (Python loops in the O(N) passes); for national runs
tile by basin, or swap in WhiteboxTools/pysheds and load the same tables."""
import heapq

import numpy as np
import pandas as pd
from scipy import ndimage

OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def cell_size_m(lat_center_deg, res_deg):
    """Approximate (dy, dx) in metres of a geographic cell; adequate at tile scale (< ~1 degree)."""
    return 110540.0 * res_deg, 111320.0 * np.cos(np.deg2rad(lat_center_deg)) * res_deg


def fill_voids_nearest(dem, void_mask):
    if void_mask.all():
        raise ValueError("DEM is entirely void")
    if not void_mask.any():
        return dem
    idx = ndimage.distance_transform_edt(void_mask, return_distances=False, return_indices=True)
    return dem[tuple(idx)]


def fill_depressions(dem, eps=1e-4):
    """Priority-Flood with epsilon (Barnes et al. 2014). Tile-border cells are outlets."""
    if not np.isfinite(dem).all():
        raise ValueError("DEM has NaN/inf; fill voids first")
    z = dem.astype(np.float64).copy()
    h, w = z.shape
    closed = np.zeros((h, w), bool)
    border = {(y, x) for y in (0, h - 1) for x in range(w)} | {(y, x) for x in (0, w - 1) for y in range(h)}
    heap = []
    for y, x in border:
        closed[y, x] = True
        heap.append((z[y, x], y, x))
    heapq.heapify(heap)
    while heap:
        zc, y, x = heapq.heappop(heap)
        for dy, dx in OFFS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not closed[ny, nx]:
                closed[ny, nx] = True
                if z[ny, nx] <= zc:
                    z[ny, nx] = zc + eps
                heapq.heappush(heap, (z[ny, nx], ny, nx))
    return z


def d8_down(z, cell):
    """Flat index of the steepest strictly-lower neighbour (-1 = outlet). Border cells are outlets."""
    h, w = z.shape
    best = np.zeros((h, w))
    down = -np.ones((h, w), np.int64)
    pz = np.pad(z, 1, constant_values=np.inf)
    pidx = np.pad(np.arange(h * w).reshape(h, w), 1, constant_values=-1)
    for dy, dx in OFFS:
        zn = pz[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        s = (z - zn) / np.hypot(dy * cell[0], dx * cell[1])
        better = s > best
        best = np.where(better, s, best)
        down = np.where(better, pidx[1 + dy:1 + dy + h, 1 + dx:1 + dx + w], down)
    down[0, :] = down[-1, :] = -1
    down[:, 0] = down[:, -1] = -1
    return down.ravel()


def flow_accumulation(down, order_high_to_low):
    dl = down.tolist()
    acc = [1.0] * len(dl)
    for i in order_high_to_low.tolist():
        d = dl[i]
        if d >= 0:
            acc[d] += acc[i]
    return np.array(acc)


def hand(z, down, stream, order_low_to_high):
    """Height Above Nearest Drainage: elevation minus elevation of the stream cell the flow path reaches.
    NaN where the path leaves the tile without meeting a stream (unknown, never guessed)."""
    zf = z.ravel()
    dl = down.tolist()
    sm = stream.ravel().tolist()
    drain = [np.nan] * zf.size
    for i in order_low_to_high.tolist():
        if sm[i]:
            drain[i] = zf[i]
        elif dl[i] >= 0:
            drain[i] = drain[dl[i]]
    return (zf - np.array(drain)).reshape(z.shape)


def label_links(down, stream, order_high_to_low):
    """Stream links: a new link starts at every source and every junction."""
    n = down.size
    up_cnt = np.zeros(n, np.int64)
    only_up = -np.ones(n, np.int64)
    for i in np.flatnonzero(stream):
        d = down[i]
        if d >= 0 and stream[d]:
            up_cnt[d] += 1
            only_up[d] = i
    link = -np.ones(n, np.int64)
    nl = 0
    for i in order_high_to_low.tolist():
        if not stream[i]:
            continue
        if up_cnt[i] == 1:
            link[i] = link[only_up[i]]
        else:
            link[i] = nl
            nl += 1
    return link, nl


def label_catchments(down, link, order_low_to_high):
    """Every cell gets the id of the stream link it drains into (-1 if it leaves the tile first)."""
    catch = link.copy()
    for i in order_low_to_high.tolist():
        if link[i] >= 0:
            continue
        d = down[i]
        catch[i] = catch[d] if d >= 0 else -1
    return catch


def flow_length_and_top(z, down, cell, order_high_to_low):
    """Longest upstream flow path (m) into each cell, and the elevation where that path starts."""
    h, w = z.shape
    n = z.size
    idx = np.arange(n)
    ok = down >= 0
    step = np.zeros(n)
    dd = np.where(ok, down, 0)
    step[ok] = np.hypot(np.abs(idx // w - dd // w)[ok] * cell[0], np.abs(idx % w - dd % w)[ok] * cell[1])
    zf = z.ravel()
    length = [0.0] * n
    top = zf.tolist()
    dl = down.tolist()
    st = step.tolist()
    for i in order_high_to_low.tolist():
        d = dl[i]
        if d >= 0 and length[i] + st[i] > length[d]:
            length[d] = length[i] + st[i]
            top[d] = top[i]
    return np.array(length), np.array(top)


def slope_m_per_m(z, cell):
    gy, gx = np.gradient(z, cell[0], cell[1])
    return np.hypot(gy, gx)


def kirpich_tc_min(L_m, S):
    """Kirpich (1940) time of concentration in minutes; L in metres, S in m/m. First-order estimate."""
    L = np.asarray(L_m, float)
    S = np.maximum(np.asarray(S, float), 1e-4)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(L > 0, 0.0195 * L ** 0.77 * S ** -0.385, np.nan)


def analyze(dem, cell, stream_cells):
    """Full terrain pass on one void-free tile. Returns arrays plus one row per stream link."""
    z = fill_depressions(dem)
    down = d8_down(z, cell)
    hl = np.argsort(-z.ravel(), kind="stable")
    lh = hl[::-1]
    acc = flow_accumulation(down, hl)
    stream = acc >= stream_cells
    hnd = hand(z, down, stream, lh)
    link, _ = label_links(down, stream, hl)
    catch = label_catchments(down, link, lh)
    length, top = flow_length_and_top(z, down, cell, hl)
    slope = slope_m_per_m(z, cell)
    shape = z.shape
    table = link_table(z, acc.reshape(shape), link, catch, hnd, slope, length, top, cell)
    return dict(z=z, down=down, acc=acc.reshape(shape), stream=stream.reshape(shape), hand=hnd,
                link=link.reshape(shape), catch=catch.reshape(shape), slope=slope, table=table)


def link_table(z, acc, link, catch, hand_arr, slope_arr, length, top, cell):
    cols = ["link_id", "row", "col", "area_local_km2", "area_up_km2", "slope_mean", "hand_mean", "flow_len_m", "tc_min"]
    w = z.shape[1]
    zf = z.ravel()
    s = np.flatnonzero(link >= 0)
    if s.size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({"link": link[s], "z": zf[s], "cell": s})
    outlet = df.loc[df.groupby("link")["z"].idxmin()].set_index("link")["cell"]
    links, oc = outlet.index.to_numpy(), outlet.to_numpy()
    n_l = int(link.max()) + 1
    cv = catch.ravel()
    inside = cv >= 0
    cell_km2 = cell[0] * cell[1] / 1e6

    def bmean(values):
        v = values.ravel()
        ok = inside & np.isfinite(v)
        den = np.bincount(cv[ok], minlength=n_l)
        num = np.bincount(cv[ok], weights=v[ok], minlength=n_l)
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    lo = length[oc]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(lo > 0, (top[oc] - zf[oc]) / lo, np.nan)
    return pd.DataFrame({
        "link_id": links, "row": oc // w, "col": oc % w,
        "area_local_km2": np.bincount(cv[inside], minlength=n_l)[links] * cell_km2,
        "area_up_km2": acc.ravel()[oc] * cell_km2,
        "slope_mean": bmean(slope_arr)[links], "hand_mean": bmean(hand_arr)[links],
        "flow_len_m": lo, "tc_min": kirpich_tc_min(lo, S)})


def refuge_map(hand_arr, safe_m, cell):
    """Distance (m) and compass bearing (deg, clockwise from north) from every cell to the nearest cell whose
    HAND >= safe_m. Straight-line lower bound; real paths are longer (apply a detour factor downstream)."""
    safe = np.nan_to_num(hand_arr, nan=-1.0) >= safe_m
    if not safe.any():
        return None, None
    dist, (iy, ix) = ndimage.distance_transform_edt(~safe, sampling=cell, return_indices=True)
    yy, xx = np.indices(hand_arr.shape)
    east = (ix - xx) * cell[1]
    north = -(iy - yy) * cell[0]
    bearing = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    return dist, bearing
