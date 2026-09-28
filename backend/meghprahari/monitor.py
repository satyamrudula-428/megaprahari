"""Data-age monitoring and degraded-mode detection (tasks M4, M5). Pure functions: the API feeds them the latest
observation time per source from the ingest ledger; the dashboard shows a warning when any input is stale.
Ages in minutes."""
from datetime import datetime

STALE_FACTOR = 2.0        # a satellite/radar feed is stale when its newest frame is older than 2 x its cadence


def expected_max_age(cfg):
    """{source: maximum acceptable age in minutes}: the rain source at STALE_FACTOR x cadence, met at met.max_age_h."""
    src = cfg.get("rain_source", "hem")
    return {src: STALE_FACTOR * float(cfg[src]["cadence_min"]), "met": 60.0 * float(cfg["met"]["max_age_h"])}


def data_status(latest, max_age_min, now):
    """Rows {source, last, age_min, max_age_min, stale} and an overall 'degraded' flag.
    latest: {source: datetime or None}; a source never received counts as stale."""
    rows = []
    for src, limit in max_age_min.items():
        t = latest.get(src)
        age = None if t is None else (now - t).total_seconds() / 60.0
        rows.append(dict(source=src, last=None if t is None else t.isoformat(), age_min=None if age is None else round(age, 1),
                         max_age_min=limit, stale=age is None or age > limit))
    return dict(degraded=any(r["stale"] for r in rows), sources=rows,
                message=("inputs stale: " + ", ".join(r["source"] for r in rows if r["stale"]) +
                         " — forecasts may be missing or based on old data") if any(r["stale"] for r in rows) else "all inputs current")


def as_utc(t):
    """Keep only timezone-aware datetimes (the ledger stores timestamptz)."""
    return t if isinstance(t, datetime) and t.tzinfo is not None else None
