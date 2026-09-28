"""Render entrypoint for the MeghPrahari synthetic demonstration.
Initialises the database schema once, seeds the synthetic demo state once,
and then starts FastAPI on Render's PORT.
"""
import os
import time
import urllib.request

import psycopg

DB = os.environ["MP_DATABASE_URL"]
PORT = os.environ.get("PORT", "10000")


def db_ready():
    try:
        with psycopg.connect(DB, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
            return True
    except Exception as exc:
        print(f"Waiting for database: {exc}", flush=True)
        return False


def ensure_schema():
    with psycopg.connect(DB) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        with open("database/init.sql", encoding="utf-8") as f:
            conn.execute(f.read())
        conn.commit()


def seed_if_empty():
    with psycopg.connect(DB) as conn:
        row = conn.execute("SELECT count(*) FROM app_user").fetchone()
        empty = row[0] == 0
    if empty:
        os.environ.setdefault("MP_DEMO_PASSWORD", "MeghPrahariDemo123!")
        import tools.seed_demo as seed_demo
        seed_demo.main()
        print("Synthetic demo seeded.", flush=True)
    else:
        print("Existing database detected; keeping existing data.", flush=True)


for _ in range(60):
    if db_ready():
        break
    time.sleep(2)
else:
    raise SystemExit("Database was not reachable within 120 seconds")

ensure_schema()
seed_if_empty()

os.execvp("uvicorn", ["uvicorn", "meghprahari.api:app", "--host", "0.0.0.0", "--port", PORT])
