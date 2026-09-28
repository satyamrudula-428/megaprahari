"""Continuous worker: watch landing zone -> QC -> features -> calibrated scoring -> alert DRAFTS.
UNTESTED here (needs PostgreSQL, h5py/netCDF4 and your real files). The pure steps it calls are unit-tested.
Never publishes anything; it writes drafts that a human must approve through the API."""
import hashlib
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from . import alerts as A, model as MD, pipeline as P
from .ingest import (IngestError, MetIndex, parse_time_from_name, qc_rain_frame, read_hem_raw, read_met_state,
                     scan_new_files)
from .settings import load_settings
from .store import audit_append

log = logging.getLogger("meghprahari.worker")
KEEP_FRAMES = 12


class Ctx:
    def __init__(self):
        self.seen, self.cache = set(), {}
        self.frames, self.met = {}, {}
        self.met_index = None                            # MetIndex when met files hold many times (ERA5)
        self.hem_axes = None
        self.states = {}
        self.last_anchor = None
        self.catch = self.villages = self.terrain = None
        self.models = {}
        self.model_ids = {}
        self.mtl = None                                  # (TransformerModel, network) when model.backend = transformer
        self.feat_hist = deque(maxlen=12)                # recent feature tables for the Transformer's sequence input


def ledger(conn, source, path, sha, when, status, detail=None):
    conn.execute("INSERT INTO ingest_ledger (source, path, sha256, size_bytes, obs_time, status, detail) "
                 "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sha256) DO NOTHING",
                 (source, path, sha, os.path.getsize(path), when, status, detail))


def load_reference(conn, cfg, ctx):
    rows = conn.execute("SELECT id, ST_Y(geom) AS lat, ST_X(geom) AS lon, slope_mean, hand_mean, tc_min, area_up_km2 "
                        "FROM catchment WHERE atlas_id=%s ORDER BY id", (cfg["atlas_id"],)).fetchall()
    ctx.catch = pd.DataFrame(rows).set_index("id")
    vs = conn.execute("SELECT v.id, v.name, ST_Y(v.geom) AS lat, ST_X(v.geom) AS lon, v.action_cost, v.loss, "
                      "v.refuge_dist_m, v.refuge_bearing_deg, c.tc_min, v.catchment_id FROM village v "
                      "JOIN catchment c ON c.id=v.catchment_id WHERE c.atlas_id=%s", (cfg["atlas_id"],)).fetchall()
    ctx.villages = [A.VillageCtx(r["id"], r["name"], r["lat"], r["lon"], r["action_cost"], r["loss"], r["refuge_dist_m"],
                                 r["refuge_bearing_deg"], r["tc_min"] or 0.0, r["catchment_id"]) for r in vs]
    for r in conn.execute("SELECT village_id, hazard, level, up_count, down_count FROM village_state").fetchall():
        ctx.states[(r["village_id"], r["hazard"])] = A.VillageState(r["level"], r["up_count"], r["down_count"])
    Mc = met_section(cfg)[1]
    t = np.load(Mc["terrain_file"])
    ctx.terrain = (t["lat"], t["lon"], t["h"])
    if Mc.get("time_var"):
        ctx.met_index = MetIndex(Mc, t["h"], Mc.get("ps_file"), t["lat"], t["lon"], keep=KEEP_FRAMES)


def rain_section(cfg):
    """Name and settings of the rain-rate source feeding the features: settings 'rain_source' (default 'hem')."""
    src = cfg.get("rain_source", "hem")
    return src, cfg[src]


def met_section(cfg):
    """Name and settings of the atmospheric-ingredients source feeding the features: settings 'met_source'
    (default 'met', ERA5 -- training/replay). A live-mode alternative (e.g. 'met_nrt', GFS analysis) is selected the
    same way as rain_source; terrain_file/max_age_h/levels_hpa/surface_vars are read from that section too, so a
    live source needs its own complete met-shaped config section (see settings/settings.yaml met_nrt: for the pattern)."""
    src = cfg.get("met_source", "met")
    return src, cfg[src]


MTL_TARGET = "mtl"          # model_version.target of a multi-task Transformer (one artifact for all 9 targets)


def backend(cfg):
    """'gbm' (default, per-target gradient boosting) or 'transformer' (settings model.backend)."""
    b = (cfg.get("model") or {}).get("backend", "gbm")
    if b not in ("gbm", "transformer"):
        raise ValueError(f"model.backend must be 'gbm' or 'transformer', got '{b}'")
    return b


def load_transformer_model(conn, ctx):
    from . import mtl_backend as T
    r = conn.execute("SELECT id, artifact_path, sha256 FROM model_version WHERE status='approved' AND target=%s "
                     "ORDER BY approved_at DESC LIMIT 1", (MTL_TARGET,)).fetchone()
    if r is None:
        ctx.mtl, ctx.model_ids = None, {}
        return
    if ctx.model_ids.get(T.TARGETS[0]) != r["id"]:
        tm, net = T.load_transformer(r["artifact_path"], r["sha256"])      # refuses a file whose hash differs
        ctx.mtl, ctx.model_ids = (tm, net), {t: r["id"] for t in T.TARGETS}
        ctx.feat_hist = deque(maxlen=tm.seq_len)


def load_models(conn, ctx, cfg=None):
    if cfg is not None and backend(cfg) == "transformer":
        load_transformer_model(conn, ctx)
        return
    rows = conn.execute("SELECT id, target, artifact_path, sha256 FROM model_version WHERE status='approved' "
                        "AND target <> %s", (MTL_TARGET,)).fetchall()
    for r in rows:
        if ctx.model_ids.get(r["target"]) != r["id"]:
            ctx.models[r["target"]] = MD.load_model(r["artifact_path"], r["sha256"])
            ctx.model_ids[r["target"]] = r["id"]
    for gone in set(ctx.models) - {r["target"] for r in rows}:
        ctx.models.pop(gone), ctx.model_ids.pop(gone)


def ingest_files(conn, cfg, ctx):
    replay = cfg["mode"] == "replay"
    L, Mc = cfg["landing"], met_section(cfg)[1]
    src, H = rain_section(cfg)
    new = []
    for path, sha in scan_new_files(H.get("dir") or L["hem_dir"], H["pattern"], ctx.seen, L["min_age_s"], cache=ctx.cache):
        try:
            new.append((parse_time_from_name(os.path.basename(path), H["time_regex"], H["time_format"]), path, sha))
        except IngestError as e:
            ledger(conn, src, path, sha, None, "rejected", str(e)); ctx.seen.add(sha)
    new.sort()
    if replay:
        new = new[:1]                                    # one file per cycle, in time order
    for when, path, sha in new:
        try:
            raw, lat, lon = read_hem_raw(path, H)
            frame, miss = qc_rain_frame(raw, H["fill_values"], H["scale"], H["offset"], H["max_mm_h"])
            if ctx.hem_axes is None:
                ctx.hem_axes = (lat, lon)
            elif frame.shape != (len(ctx.hem_axes[0]), len(ctx.hem_axes[1])) or not (
                    np.allclose(lat, ctx.hem_axes[0]) and np.allclose(lon, ctx.hem_axes[1])):
                raise IngestError(f"{src} grid changed between files")
            ctx.frames[when] = frame
            ledger(conn, src, path, sha, when, "ok", f"missing={miss:.3f}")
        except IngestError as e:
            ledger(conn, src, path, sha, when, "rejected", str(e))
        ctx.seen.add(sha)
    horizon = max(ctx.frames) if replay and ctx.frames else None
    for path, sha in scan_new_files(Mc.get("dir") or L["met_dir"], Mc["pattern"], ctx.seen, L["min_age_s"], cache=ctx.cache):
        if ctx.met_index is not None:                    # many times per file: index now, read each time on demand
            try:                                         # (replay safety: select_met_times never picks a future time)
                ts = ctx.met_index.add_file(path)
                ledger(conn, "met", path, sha, min(ts) if ts else None, "ok", f"{len(ts)} times indexed")
            except (IngestError, OSError, KeyError) as e:
                ledger(conn, "met", path, sha, None, "rejected", str(e))
            ctx.seen.add(sha)
            continue
        try:
            when = parse_time_from_name(os.path.basename(path), Mc["time_regex"], Mc["time_format"])
            if replay and horizon is not None and when > horizon:
                continue                                 # not yet "released" by the virtual clock
            m = read_met_state(path, Mc, when, ctx.terrain[2])
            if not (np.allclose(m.lat, ctx.terrain[0]) and np.allclose(m.lon, ctx.terrain[1])):
                raise IngestError("met grid does not match terrain_on_met_grid file")
            ctx.met[when] = m
            ledger(conn, "met", path, sha, when, "ok")
        except IngestError as e:
            ledger(conn, "met", path, sha, None, "rejected", str(e))
        ctx.seen.add(sha)
    for d in (ctx.frames, ctx.met):
        for k in sorted(d)[:-KEEP_FRAMES]:
            del d[k]


def run_cycle(conn, cfg, ctx):
    met_times = ctx.met_index.times() if ctx.met_index is not None else list(ctx.met)
    if not ctx.frames or not met_times or not (ctx.models or ctx.mtl):
        return "waiting for data or an approved model", None, None
    t = max(ctx.frames)
    if t == ctx.last_anchor:
        return "no new frame", None, None
    get_met = ctx.met_index.get if ctx.met_index is not None else ctx.met.__getitem__
    feats, why = P.features_at(t, ctx.frames, ctx.hem_axes, met_times, get_met, ctx.catch, rain_section(cfg)[1],
                               met_section(cfg)[1]["max_age_h"], P.FEATURE_SETS[cfg.get("features", "v1")])
    if feats is None:
        return why, None, None
    if ctx.mtl is not None:
        from . import mtl_backend as T
        ctx.feat_hist.append(feats)
        probs = T.score_transformer(ctx.mtl[0], ctx.mtl[1], list(ctx.feat_hist))
    else:
        probs = P.score_all(feats, ctx.models)
    W = cfg["worker"]
    with conn.cursor() as cur:
        rows = [(int(cid), tgt, t, float(p), ctx.model_ids[tgt]) for tgt in probs.columns
                for cid, p in probs[tgt].items() if np.isfinite(p)]
        cur.executemany("INSERT INTO forecast_latest (catchment_id,target,issue_time,prob,model_id) VALUES (%s,%s,%s,%s,%s) "
                        "ON CONFLICT (catchment_id,target) DO UPDATE SET issue_time=EXCLUDED.issue_time, "
                        "prob=EXCLUDED.prob, model_id=EXCLUDED.model_id", rows)
        cur.executemany("INSERT INTO forecast_hist (issue_time,catchment_id,target,prob,model_id) VALUES (%s,%s,%s,%s,%s) "
                        "ON CONFLICT DO NOTHING", [r for r in rows if r[3] >= W["store_min_prob"]])
    replay = cfg["mode"] == "replay"
    shadow = bool(cfg["shadow_mode"]) or replay          # replay can never produce a non-exercise alert
    now = t if replay else datetime.now(timezone.utc)
    states = {k: A.VillageState(s.level, s.up, s.down) for k, s in ctx.states.items()}   # commit to memory only after the DB commit
    drafts = P.evaluate_villages(probs, feats, ctx.villages, states, ctx.models, cfg["alerts"], now, shadow,
                                 cfg["sender"], cfg["sender_name"], cfg["alerts"].get("up_needed"))
    for d in drafts:
        conn.execute("INSERT INTO alert (village_id,hazard,level,prob,lead_start_min,lead_end_min,margin_min,drivers,model_ids,"
                     "explanation,cap_xml,cap_sha256,shadow,hindcast,required_approvals) "
                     "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                     (d["village_id"], d["hazard"], d["level"], d["prob"], d["lead_start_min"], d["lead_end_min"],
                      d["margin_min"], Jsonb(d["drivers"]), Jsonb(ctx.model_ids), Jsonb(d.get("explanation")),
                      d["cap_xml"], hashlib.sha256(d["cap_xml"].encode()).hexdigest(), d["shadow"], replay, d["required_approvals"]))
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO village_state (village_id,hazard,level,up_count,down_count) VALUES (%s,%s,%s,%s,%s) "
                        "ON CONFLICT (village_id,hazard) DO UPDATE SET level=EXCLUDED.level, up_count=EXCLUDED.up_count, "
                        "down_count=EXCLUDED.down_count, updated_at=now()",
                        [(v, h, s.level, s.up, s.down) for (v, h), s in states.items()])
    conn.execute("DELETE FROM forecast_hist WHERE issue_time < now() - make_interval(days => %s)", (W["retention_days"],))
    audit_append(conn, "worker", "cycle", {"anchor": t.isoformat(), "catchments": len(feats), "drafts": len(drafts),
                                           "model_ids": ctx.model_ids, "shadow": shadow})
    return f"cycle ok: {len(drafts)} draft(s)", states, t


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_settings(os.environ.get("MP_CONFIG", "settings/settings.yaml"))
    pool = ConnectionPool(os.environ["MP_DATABASE_URL"], min_size=1, max_size=3, kwargs={"row_factory": dict_row})
    ctx = Ctx()
    with pool.connection() as c:
        load_reference(c, cfg, ctx)
        ctx.seen = {r["sha256"] for r in c.execute("SELECT sha256 FROM ingest_ledger").fetchall()}
    log.info("worker started: mode=%s shadow=%s catchments=%d villages=%d", cfg["mode"], cfg["shadow_mode"],
             len(ctx.catch), len(ctx.villages))
    while True:
        try:
            with pool.connection() as c:
                load_models(c, ctx, cfg)
                ingest_files(c, cfg, ctx)
            with pool.connection() as c:
                msg, states, anchor = run_cycle(c, cfg, ctx)
            log.info(msg)
            if states is not None:
                ctx.states, ctx.last_anchor = states, anchor
        except Exception:                                # noqa: BLE001 - keep the loop alive, never fabricate output
            log.exception("cycle failed")
        time.sleep(cfg["worker"]["poll_s"])


if __name__ == "__main__":
    main()
