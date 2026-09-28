"""REST API + static dashboard. Every state change is RBAC-checked and written to the audit chain.
UNTESTED here (no FastAPI/PostgreSQL in the build sandbox): run tools/smoke_test.sh after `docker compose up`."""
import hashlib
import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from . import __version__
from .governance import allowed, decode_token, hash_password, is_approved, issue_token, verify_password
from .monitor import as_utc, data_status, expected_max_age
from .pipeline import BINS, HAZARDS, target_name
from .settings import load_settings
from .store import audit_append, audit_verify

CFG = load_settings(os.environ.get("MP_CONFIG", "settings/settings.yaml"), only=("mode", "shadow_mode", "outbox_dir"))
SECRET = os.environ["MP_JWT_SECRET"]
TARGETS = {target_name(h, b) for h in HAZARDS for b in BINS}
pool = ConnectionPool(os.environ["MP_DATABASE_URL"], min_size=1, max_size=8, open=False, kwargs={"row_factory": dict_row})
DUMMY_HASH = hash_password("dummy-password-for-timing")


@asynccontextmanager
async def lifespan(_):
    pool.open()
    yield
    pool.close()


app = FastAPI(title="MeghPrahari", version=__version__, lifespan=lifespan)
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")


def need(perm):
    def dep(token: str = Depends(oauth2)):
        try:
            claims = decode_token(SECRET, token)
        except jwt.InvalidTokenError:
            raise HTTPException(401, "invalid or expired token")
        if not allowed(claims["role"], perm):
            raise HTTPException(403, f"role '{claims['role']}' lacks permission '{perm}'")
        return claims
    return dep


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "shadow_mode": CFG["shadow_mode"], "mode": CFG["mode"]}


@app.get("/ready")
def ready():
    try:
        with pool.connection() as c:
            models = c.execute("SELECT count(*) AS n FROM model_version WHERE status='approved'").fetchone()["n"]
            last = c.execute("SELECT max(ingested_at) AS t FROM ingest_ledger WHERE status='ok'").fetchone()["t"]
    except Exception as e:                                             # noqa: BLE001
        return JSONResponse({"ready": False, "detail": f"database: {e}"}, status_code=503)
    ok = models > 0
    return JSONResponse({"ready": ok, "approved_models": models, "last_ingest": last.isoformat() if last else None},
                        status_code=200 if ok else 503)


@app.get("/api/status")
def status(user=Depends(need("forecast:read"))):
    """Data age per input source and a degraded flag (tasks M4, M5)."""
    from datetime import datetime, timezone
    cfg = load_settings(os.environ.get("MP_CONFIG", "settings/settings.yaml"), only=("rain_source", "met"))
    limits = expected_max_age(cfg)
    with pool.connection() as c:
        rows = c.execute("SELECT source, max(obs_time) AS t FROM ingest_ledger WHERE status='ok' GROUP BY source").fetchall()
    latest = {r["source"]: as_utc(r["t"]) for r in rows}
    return data_status(latest, limits, datetime.now(timezone.utc))


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    with pool.connection() as c:
        u = c.execute("SELECT username, pw_hash, role::text AS role, active FROM app_user WHERE username=%s",
                      (form.username,)).fetchone()
        ok = verify_password(form.password, u["pw_hash"] if u else DUMMY_HASH) and bool(u) and u["active"]
        audit_append(c, form.username, "login_ok" if ok else "login_fail", {})
    if not ok:
        raise HTTPException(401, "invalid credentials")
    return {"access_token": issue_token(SECRET, u["username"], u["role"]), "token_type": "bearer"}


@app.get("/api/risk.geojson")
def risk(target: str = "ff_0_2", min_prob: float = 0.0, limit: int = 20000, user=Depends(need("forecast:read"))):
    if target not in TARGETS:
        raise HTTPException(422, f"target must be one of {sorted(TARGETS)}")
    with pool.connection() as c:
        rows = c.execute(
            "SELECT c.id, f.prob, f.issue_time, ST_AsGeoJSON(c.geom)::json AS geom FROM forecast_latest f "
            "JOIN catchment c ON c.id=f.catchment_id WHERE f.target=%s AND f.prob>=%s ORDER BY f.prob DESC LIMIT %s",
            (target, min_prob, min(limit, 50000))).fetchall()
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": r["geom"],
         "properties": {"id": r["id"], "prob": r["prob"], "issue_time": r["issue_time"].isoformat()}} for r in rows]}


@app.get("/api/alerts")
def alerts(status: Literal["draft", "approved", "published", "cancelled"] = "draft", user=Depends(need("alert:read"))):
    with pool.connection() as c:
        return c.execute(
            "SELECT a.id::text AS id, v.name AS village, a.hazard, a.level, a.prob, a.lead_start_min, a.lead_end_min, "
            "a.margin_min, a.drivers, a.status, a.shadow, a.hindcast, a.required_approvals, a.created_at, "
            "(SELECT count(*) FROM alert_approval p WHERE p.alert_id=a.id AND p.decision='approve') AS approvals "
            "FROM alert a JOIN village v ON v.id=a.village_id WHERE a.status=%s ORDER BY a.created_at DESC LIMIT 200",
            (status,)).fetchall()


@app.get("/api/explain/{alert_id}")
def explain(alert_id: uuid.UUID, user=Depends(need("alert:read"))):
    """'Why this alert?' card stored with the alert (ingredients + plain-language text; see mp.explain)."""
    with pool.connection() as c:
        a = c.execute("SELECT a.explanation, a.hazard, a.level, a.prob, v.name AS village FROM alert a "
                      "JOIN village v ON v.id=a.village_id WHERE a.id=%s", (str(alert_id),)).fetchone()
    if not a:
        raise HTTPException(404, "alert not found")
    if a["explanation"] is None:
        raise HTTPException(404, "no explanation stored for this alert (drafted before explanations existed)")
    return {"alert": str(alert_id), "village": a["village"], "hazard": a["hazard"], "level": a["level"],
            "prob": a["prob"], **a["explanation"]}


def _user_id(c, username):
    return c.execute("SELECT id FROM app_user WHERE username=%s AND active", (username,)).fetchone()["id"]


@app.post("/api/alerts/{alert_id}/approve")
def approve(alert_id: uuid.UUID, user=Depends(need("alert:approve"))):
    with pool.connection() as c:
        a = c.execute("SELECT level, status FROM alert WHERE id=%s FOR UPDATE", (str(alert_id),)).fetchone()
        if not a:
            raise HTTPException(404, "alert not found")
        if a["status"] not in ("draft", "approved"):
            raise HTTPException(409, f"alert is {a['status']}")
        c.execute("INSERT INTO alert_approval (alert_id, user_id, decision) VALUES (%s,%s,'approve') "
                  "ON CONFLICT (alert_id, user_id) DO NOTHING", (str(alert_id), _user_id(c, user["sub"])))
        rows = c.execute("SELECT u.id, u.role::text AS role FROM alert_approval p JOIN app_user u ON u.id=p.user_id "
                         "WHERE p.alert_id=%s AND p.decision='approve'", (str(alert_id),)).fetchall()
        ok = is_approved(a["level"], [(r["id"], r["role"]) for r in rows])
        if ok:
            c.execute("UPDATE alert SET status='approved' WHERE id=%s", (str(alert_id),))
        audit_append(c, user["sub"], "alert_approve", {"alert": str(alert_id), "fully_approved": ok})
    return {"fully_approved": ok}


@app.post("/api/alerts/{alert_id}/publish")
def publish(alert_id: uuid.UUID, user=Depends(need("alert:publish"))):
    """Writes the approved CAP file to the outbox. Delivery to the public is done by the authority's own gateway
    picking up outbox/actual; shadow-mode alerts land in outbox/exercise and are never marked Actual."""
    with pool.connection() as c:
        a = c.execute("SELECT status, shadow, cap_xml, cap_sha256 FROM alert WHERE id=%s FOR UPDATE", (str(alert_id),)).fetchone()
        if not a:
            raise HTTPException(404, "alert not found")
        if a["status"] != "approved":
            raise HTTPException(409, "alert must be fully approved first")
        if hashlib.sha256(a["cap_xml"].encode()).hexdigest() != a["cap_sha256"]:
            audit_append(c, user["sub"], "publish_blocked_hash_mismatch", {"alert": str(alert_id)})
            raise HTTPException(500, "CAP content does not match its approved hash")
        folder = os.path.join(CFG["outbox_dir"], "exercise" if a["shadow"] else "actual")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{alert_id}.xml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(a["cap_xml"])
        c.execute("UPDATE alert SET status='published' WHERE id=%s", (str(alert_id),))
        audit_append(c, user["sub"], "alert_publish", {"alert": str(alert_id), "path": path, "cap_sha256": a["cap_sha256"]})
    return {"published": path}


@app.post("/api/alerts/{alert_id}/cancel")
def cancel(alert_id: uuid.UUID, user=Depends(need("alert:cancel"))):
    with pool.connection() as c:
        n = c.execute("UPDATE alert SET status='cancelled' WHERE id=%s AND status IN ('draft','approved','published')",
                      (str(alert_id),)).rowcount
        if not n:
            raise HTTPException(409, "alert not found or already cancelled")
        audit_append(c, user["sub"], "alert_cancel", {"alert": str(alert_id)})
    return {"cancelled": True}


class Observation(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    kind: Literal["rain_mm_h", "water_level_cm", "flood_seen"]
    value: float | None = Field(default=None, ge=0, le=5000)


@app.post("/api/observations")
def observe(o: Observation, user=Depends(need("observation:create"))):
    with pool.connection() as c:
        c.execute("INSERT INTO observation (geom, kind, value, user_id) VALUES (ST_SetSRID(ST_MakePoint(%s,%s),4326),%s,%s,%s)",
                  (o.lon, o.lat, o.kind, o.value, _user_id(c, user["sub"])))
        audit_append(c, user["sub"], "observation_create", {"kind": o.kind})
    return {"stored": True}


@app.post("/api/models/{model_id}/approve")
def approve_model(model_id: int, user=Depends(need("model:approve"))):
    with pool.connection() as c:
        m = c.execute("SELECT target, status, calibrated, trained_by FROM model_version WHERE id=%s FOR UPDATE", (model_id,)).fetchone()
        if not m:
            raise HTTPException(404, "model not found")
        uid = _user_id(c, user["sub"])
        if m["status"] != "candidate" or not m["calibrated"]:
            raise HTTPException(409, "only calibrated candidate models can be approved")
        if uid == m["trained_by"]:
            raise HTTPException(403, "separation of duties: the trainer cannot approve their own model")
        c.execute("UPDATE model_version SET status='retired' WHERE target=%s AND status='approved'", (m["target"],))
        c.execute("UPDATE model_version SET status='approved', approved_by=%s, approved_at=now() WHERE id=%s", (uid, model_id))
        audit_append(c, user["sub"], "model_approve", {"model": model_id, "target": m["target"]})
    return {"approved": model_id}


@app.get("/api/audit/verify")
def verify(user=Depends(need("audit:verify"))):
    with pool.connection() as c:
        ok, bad = audit_verify(c)
    return {"chain_ok": ok, "first_bad_index": bad}


app.mount("/", StaticFiles(directory=os.environ.get("MP_WEB_DIR", "frontend"), html=True), name="web")
