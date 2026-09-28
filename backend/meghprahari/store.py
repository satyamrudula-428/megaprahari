"""DB helpers shared by API and worker (psycopg 3, dict rows). UNTESTED here: no PostgreSQL in the build sandbox."""
from datetime import datetime, timezone

from .governance import GENESIS, canonical_body, chain_hash, verify_chain


def audit_append(conn, actor, action, payload):
    conn.execute("SELECT pg_advisory_xact_lock(727001)")            # serialise the chain
    row = conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
    prev = row["hash"] if row else GENESIS
    ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    body = canonical_body(ts, actor, action, payload)
    conn.execute("INSERT INTO audit_log (ts, actor, action, prev_hash, body, hash) VALUES (%s::timestamptz,%s,%s,%s,%s,%s)",
                 (ts, actor, action, prev, body, chain_hash(prev, body)))


def audit_verify(conn):
    rows = conn.execute("SELECT actor, action, prev_hash, body, hash FROM audit_log ORDER BY seq").fetchall()
    return verify_chain(rows)
