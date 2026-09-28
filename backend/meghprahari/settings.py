"""Settings loader. Fails closed: any value left as CHANGE_ME stops start-up."""
from datetime import date, datetime, time, timezone

import yaml

HAZARD_IDS = ("ts", "cb", "ff")
EVENT_STATUSES = ("confirmed", "candidate")


class ConfigError(Exception):
    pass


def load_settings(path, only=None):
    """only: top-level keys to validate (the API needs just a few; the worker validates everything)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if only:
        missing = [k for k in only if k not in cfg]
        if missing:
            raise ConfigError(f"missing settings: {missing}")
        _check({k: cfg[k] for k in only}, "")
    else:
        _check(cfg, "")
    return cfg


def _check(node, prefix):
    if isinstance(node, dict):
        for k, v in node.items():
            _check(v, f"{prefix}{k}.")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check(v, f"{prefix}{i}.")
    elif node == "CHANGE_ME":
        raise ConfigError(f"Set '{prefix[:-1]}' in the settings file before starting")


def _date(v, where):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        raise ConfigError(f"{where}: '{v}' is not an ISO date (YYYY-MM-DD)")


def _bbox(b, where):
    try:
        la0, la1, lo0, lo1 = (float(b[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
    except (KeyError, TypeError, ValueError):
        raise ConfigError(f"{where}: bbox needs numeric lat_min, lat_max, lon_min, lon_max (degrees)")
    if not (-90 <= la0 < la1 <= 90 and -180 <= lo0 < lo1 <= 180):
        raise ConfigError(f"{where}: bbox must satisfy lat_min < lat_max and lon_min < lon_max within valid degrees")
    return dict(lat_min=la0, lat_max=la1, lon_min=lo0, lon_max=lo1)


def load_region(path):
    """Load settings/region.yaml (pilot box, DEM grid, replay events) and validate it; raises ConfigError.
    Dates become datetime.date. Bbox values are degrees (EPSG:4326)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _check(cfg, "")
    for k in ("region", "grid", "replay_events"):
        if not cfg or k not in cfg:
            raise ConfigError(f"region file missing '{k}'")
    r = cfg["region"]
    r["bbox"] = _bbox(r.get("bbox"), "region")
    if float(r.get("met_margin_deg", 0)) < 0:
        raise ConfigError("region.met_margin_deg must be >= 0")
    g = cfg["grid"]
    b = r["bbox"]
    res_deg = float(g["dem_arcsec"]) / 3600.0
    cells = round((b["lat_max"] - b["lat_min"]) / res_deg) * round((b["lon_max"] - b["lon_min"]) / res_deg)
    if cells > int(g["max_dem_cells"]):
        raise ConfigError(f"region bbox is {cells} DEM cells at {g['dem_arcsec']} arcsec; build_atlas.py allows "
                          f"{g['max_dem_cells']}. Shrink the box or coarsen the DEM")
    ids = set()
    for i, e in enumerate(cfg["replay_events"] or []):
        where = f"replay_events[{i}]"
        for k in ("id", "name", "status", "hazards", "event_start", "event_end", "download_start", "download_end"):
            if k not in e:
                raise ConfigError(f"{where}: missing '{k}'")
        if e["id"] in ids:
            raise ConfigError(f"{where}: duplicate id '{e['id']}'")
        ids.add(e["id"])
        if e["status"] not in EVENT_STATUSES:
            raise ConfigError(f"{where}: status must be one of {EVENT_STATUSES}")
        bad = [h for h in e["hazards"] if h not in HAZARD_IDS]
        if bad:
            raise ConfigError(f"{where}: unknown hazards {bad}; use {HAZARD_IDS}")
        for k in ("event_start", "event_end", "download_start", "download_end"):
            e[k] = _date(e[k], f"{where}.{k}")
        if not (e["download_start"] <= e["event_start"] <= e["event_end"] <= e["download_end"]):
            raise ConfigError(f"{where}: need download_start <= event_start <= event_end <= download_end")
        if "bbox" in e:
            e["bbox"] = _bbox(e["bbox"], where)
    return cfg


def replay_exclusion_windows(region_cfg):
    """(start, end) UTC datetimes covering every replay event's download window, whole days inclusive.
    Candidate events are excluded too: data from a possible demo event must never reach training/calibration."""
    out = []
    for e in region_cfg["replay_events"] or []:
        start = datetime.combine(e["download_start"], time.min, timezone.utc)
        end = datetime.combine(e["download_end"], time.max, timezone.utc)
        out.append((start, end))
    return out
