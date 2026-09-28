"""Governance primitives: tamper-evident audit chain, RBAC, JWT, password hashing, approval policy."""
import base64
import hashlib
import hmac
import json
import os
import time

import jwt

GENESIS = "0" * 64

# ---- audit chain -------------------------------------------------------------------------------------------

def canonical_body(ts_iso, actor, action, payload):
    return json.dumps({"ts": ts_iso, "actor": actor, "action": action, "payload": payload},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def chain_hash(prev_hash, body):
    return hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()


def verify_chain(rows):
    """rows: dicts ordered by seq with prev_hash, body, hash (and optionally actor, action).
    Returns (True, None) or (False, index_of_first_bad_row)."""
    prev = GENESIS
    for i, r in enumerate(rows):
        if r["prev_hash"] != prev or chain_hash(prev, r["body"]) != r["hash"]:
            return False, i
        b = json.loads(r["body"])
        if ("actor" in r and r["actor"] != b["actor"]) or ("action" in r and r["action"] != b["action"]):
            return False, i
        prev = r["hash"]
    return True, None

# ---- RBAC --------------------------------------------------------------------------------------------------

PERMISSIONS = {
    "admin": {"user:manage", "audit:read", "audit:verify", "alert:read", "forecast:read"},
    "scientist": {"model:train", "model:approve", "model:read", "forecast:read", "alert:read"},
    "approver": {"alert:read", "alert:approve", "alert:publish", "alert:cancel", "forecast:read"},
    "operator": {"forecast:read", "alert:read"},
    "volunteer": {"observation:create"},
    "auditor": {"audit:read", "audit:verify", "alert:read", "model:read"},
}


def allowed(role, perm):
    return perm in PERMISSIONS.get(role, set())

# ---- approvals (four-eyes for the highest level) -----------------------------------------------------------

def approvals_needed(level):
    return 2 if level >= 3 else 1


def is_approved(level, approvals):
    """approvals: iterable of (user_id, role). Distinct users holding the approver role."""
    return len({u for u, r in approvals if r == "approver"}) >= approvals_needed(level)

# ---- JWT ---------------------------------------------------------------------------------------------------

def _check_secret(secret):
    if not isinstance(secret, str) or len(secret) < 32:
        raise ValueError("JWT secret must be at least 32 characters")


def issue_token(secret, sub, role, ttl_s=3600, now=None):
    _check_secret(secret)
    t = int(now if now is not None else time.time())
    return jwt.encode({"sub": sub, "role": role, "iat": t, "exp": t + ttl_s}, secret, algorithm="HS256")


def decode_token(secret, token):
    _check_secret(secret)
    return jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub", "role"]})

# ---- passwords ---------------------------------------------------------------------------------------------

def hash_password(password, n=2 ** 14, r=8, p=1):
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return "scrypt${}${}${}${}${}".format(n, r, p, base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_password(password, stored):
    try:
        _, n, r, p, salt, dk = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(dk))
    except Exception:
        return False
