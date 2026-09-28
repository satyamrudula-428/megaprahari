"""Render entrypoint for the MeghPrahari demonstration."""

import os
import time

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
        existing = conn.execute(
            "SELECT to_regclass('public.app_user')"
        ).fetchone()

        if existing[0] is not None:
            print("Existing database schema detected; skipping initialization.", flush=True)
            return

        conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

        with open("database/init.sql", encoding="utf-8") as f:
            conn.execute(f.read())

        conn.commit()
        print("Database schema initialized.", flush=True)


def seed_if_empty():
    with psycopg.connect(DB) as conn:
        row = conn.execute("SELECT count(*) FROM app_user").fetchone()

    if row[0] == 0:
        print(
            "Database is empty. Synthetic demo seeding is not configured; continuing.",
            flush=True,
        )
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

os.execvp(
    "uvicorn",
    [
        "uvicorn",
        "meghprahari.api:app",
        "--host",
        "0.0.0.0",
        "--port",
        PORT,
    ],
)
