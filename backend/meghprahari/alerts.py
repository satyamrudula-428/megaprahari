"""Alert engine: cost-loss thresholds, escalation state machine, actionable margin, refuge walking time,
and CAP 1.2 drafting. Alerts are DRAFTS; publication is a governed, human-approved step (see governance.py)."""
import math
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import timedelta

from . import alert_text as T
from .governance import approvals_needed

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
LEVELS = {0: "NONE", 1: "WATCH", 2: "WARNING", 3: "ACT_NOW"}
# Design choice: align with the issuing authority's CAP profile before any operational use.
CAP_MAP = {1: ("Future", "Moderate", "Possible"), 2: ("Expected", "Severe", "Likely"), 3: ("Immediate", "Extreme", "Likely")}
EVENTS = {"ts": "Severe thunderstorm", "cb": "Cloudburst-type extreme rainfall", "ff": "Flash flood"}
CAP_STATUS = {"Actual", "Exercise", "System", "Test", "Draft"}
CAP_SCOPE = {"Public", "Restricted", "Private"}


def cost_loss_threshold(cost, loss):
    """Static cost-loss decision rule: act when P >= C/L (Murphy 1977)."""
    if not (0 < cost < loss):
        raise ValueError("need 0 < cost < loss")
    return cost / loss


def level_thresholds(cost, loss, watch_factor=0.5, act_factor=3.0, act_floor=0.5, cap=0.95):
    """WARNING at C/L; WATCH and ACT_NOW multipliers are design parameters to tune with the authority."""
    t2 = cost_loss_threshold(cost, loss)
    return {1: watch_factor * t2, 2: t2, 3: max(t2, min(cap, max(act_factor * t2, act_floor)))}


@dataclass
class VillageState:
    level: int = 0
    up: int = 0
    down: int = 0


def step(state, p, thr, up_needed=None, down_needed=4, release=0.8):
    """Escalate after N consecutive cycles proposing a higher level; de-escalate one level after M consecutive
    cycles with p < release * current threshold (hysteresis)."""
    up_needed = up_needed or {1: 1, 2: 2, 3: 2}
    target = 0
    for lv in (1, 2, 3):
        if p >= thr[lv]:
            target = lv
    if target > state.level:
        state.up += 1
        state.down = 0
        if state.up >= up_needed[target]:
            state.level, state.up = target, 0
    else:
        state.up = 0
        if state.level > 0 and p < release * thr[state.level]:
            state.down += 1
            if state.down >= down_needed:
                state.level, state.down = state.level - 1, 0
        else:
            state.down = 0
    return state


def walk_time_min(dist_m, speed_kmh=3.0, detour=1.5):
    return dist_m * detour / (speed_kmh * 1000.0 / 60.0)


def actionable_margin_min(lead_min, dissemination_min, walk_min, tc_min=0.0, lag_fraction=0.0):
    """Minutes to spare: (forecast lead + optional rain-to-flood lag) - dissemination delay - time to reach refuge.
    lag_fraction=0 is the conservative default (flood arrives as rain starts); Tc only ever adds time."""
    return lead_min + lag_fraction * tc_min - dissemination_min - walk_min


def compass(bearing_deg):
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][int((bearing_deg + 22.5) % 360 // 45)]


@dataclass
class VillageCtx:
    id: int
    name: str
    lat: float
    lon: float
    cost: float
    loss: float
    refuge_dist_m: float = None
    refuge_bearing_deg: float = None
    tc_min: float = 0.0
    catchment_id: int = None


def _sub(parent, tag, text=None):
    el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
    if text is not None:
        el.text = str(text)
    return el


def build_cap(*, identifier, sender, sent, status, scope, event, urgency, severity, certainty, expires, sender_name,
              headline, description, instruction, area_desc, lat, lon, radius_km, language="en-IN",
              msg_type="Alert", note=None, restriction=None, parameters=None, extra_infos=None):
    """CAP 1.2 alert XML. extra_infos: further <info> blocks for other languages, each a dict with language, event,
    headline, description, instruction (same urgency/severity/certainty/expires/area; parameters only in the first)."""
    if sent.tzinfo is None or expires.tzinfo is None:
        raise ValueError("CAP timestamps must be timezone-aware")
    if status not in CAP_STATUS or scope not in CAP_SCOPE:
        raise ValueError("invalid CAP status/scope")
    ET.register_namespace("", CAP_NS)
    a = ET.Element(f"{{{CAP_NS}}}alert")
    _sub(a, "identifier", identifier)
    _sub(a, "sender", sender)
    _sub(a, "sent", sent.isoformat(timespec="seconds"))
    _sub(a, "status", status)
    _sub(a, "msgType", msg_type)
    _sub(a, "scope", scope)
    if restriction:
        _sub(a, "restriction", restriction)
    if note:
        _sub(a, "note", note)
    infos = [dict(language=language, event=event, headline=headline, description=description, instruction=instruction,
                  parameters=parameters or {})] + [dict(x, parameters={}) for x in (extra_infos or [])]
    for info in infos:
        i = _sub(a, "info")
        for tag, val in (("language", info["language"]), ("category", "Met"), ("event", info["event"]),
                         ("urgency", urgency), ("severity", severity), ("certainty", certainty),
                         ("expires", expires.isoformat(timespec="seconds")), ("senderName", sender_name),
                         ("headline", info["headline"]), ("description", info["description"]),
                         ("instruction", info["instruction"])):
            _sub(i, tag, val)
        for k, v in info["parameters"].items():
            p = _sub(i, "parameter")
            _sub(p, "valueName", k)
            _sub(p, "value", v)
        ar = _sub(i, "area")
        _sub(ar, "areaDesc", area_desc)
        _sub(ar, "circle", f"{lat:.5f},{lon:.5f} {radius_km:g}")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(a, encoding="unicode")


def draft_alert(v, level, prob, hazard, lead_start_min, lead_end_min, drivers, cfg, now, shadow, sender, sender_name):
    """Build one alert draft (dict ready for the database). Never marks anything as published."""
    if level not in CAP_MAP:
        raise ValueError("level must be 1..3")
    walk = None if v.refuge_dist_m is None else walk_time_min(v.refuge_dist_m, cfg["walk_speed_kmh"], cfg["detour_factor"])
    margin = None if walk is None else actionable_margin_min(
        lead_start_min, cfg["dissemination_delay_min"], walk, v.tc_min, cfg["lag_fraction"])
    urgency, severity, certainty = CAP_MAP[level]
    langs = list(cfg.get("languages", ["en"]))
    if not langs:
        raise ValueError("alerts.languages must list at least one language")
    texts = [T.render(lg, level=level, hazard=hazard, village=v.name, prob=prob, lead_start_min=lead_start_min,
                      lead_end_min=lead_end_min, drivers=drivers, margin=margin, walk=walk,
                      dissemination_min=cfg["dissemination_delay_min"], refuge_dist_m=v.refuge_dist_m,
                      refuge_bearing_deg=v.refuge_bearing_deg) for lg in langs]
    main = texts[0]
    params = {"probability": f"{prob:.3f}", "hazard": hazard}
    if margin is not None:
        params["actionable_margin_min"] = f"{margin:.0f}"
    xml = build_cap(
        identifier=f"{sender}-{uuid.uuid4()}", sender=sender, sent=now, expires=now + timedelta(minutes=cfg["horizon_min"]),
        status="Exercise" if shadow else "Actual", scope="Restricted" if shadow else "Public",
        restriction="Designated exercise participants only" if shadow else None,
        note="MeghPrahari shadow-mode decision-support output; not an official warning." if shadow else None,
        event=main["event"], urgency=urgency, severity=severity, certainty=certainty, sender_name=sender_name,
        headline=main["headline"], description=main["description"], instruction=main["instruction"],
        language=main["language"], area_desc=v.name, lat=v.lat, lon=v.lon, radius_km=cfg.get("radius_km", 3.0),
        parameters=params, extra_infos=texts[1:])
    return dict(village_id=v.id, hazard=hazard, level=level, prob=float(prob), lead_start_min=lead_start_min,
                lead_end_min=lead_end_min, margin_min=margin, drivers=drivers, cap_xml=xml, shadow=shadow,
                required_approvals=approvals_needed(level))
