-- MeghPrahari schema v0.1 (PostgreSQL 16 + PostGIS). Applied once by the postgis container on first start.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TYPE user_role AS ENUM ('admin', 'scientist', 'approver', 'operator', 'volunteer', 'auditor');

CREATE TABLE app_user (
  id BIGSERIAL PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  pw_hash TEXT NOT NULL,
  role user_role NOT NULL,
  active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE atlas_version (
  id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  dem_source TEXT NOT NULL,
  params JSONB NOT NULL
);

CREATE TABLE catchment (
  id BIGSERIAL PRIMARY KEY,
  atlas_id TEXT NOT NULL REFERENCES atlas_version(id),
  link_id INTEGER NOT NULL,
  geom geometry(Point, 4326) NOT NULL,          -- outlet cell of the stream link
  area_local_km2 REAL, area_up_km2 REAL, slope_mean REAL, hand_mean REAL, flow_len_m REAL, tc_min REAL,
  UNIQUE (atlas_id, link_id)
);
CREATE INDEX catchment_geom_gix ON catchment USING GIST (geom);

CREATE TABLE village (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL, state TEXT, district TEXT,
  geom geometry(Point, 4326) NOT NULL,
  population INTEGER CHECK (population >= 0),
  action_cost REAL NOT NULL CHECK (action_cost > 0),      -- cost of taking protective action (same unit as loss)
  loss REAL NOT NULL CHECK (loss > 0),                    -- loss if the event happens unprotected
  catchment_id BIGINT REFERENCES catchment(id),
  refuge_dist_m REAL, refuge_bearing_deg REAL,
  CHECK (action_cost < loss)
);
-- No phone numbers or personal contacts are stored here by design: dissemination goes through the
-- authority's own gateway (data minimisation, DPDP Act 2023).

CREATE TABLE ingest_ledger (
  id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL CHECK (source IN ('hem', 'met')),
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  size_bytes BIGINT,
  obs_time TIMESTAMPTZ,
  status TEXT NOT NULL CHECK (status IN ('ok', 'rejected')),
  detail TEXT,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_version (
  id BIGSERIAL PRIMARY KEY,
  target TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'approved', 'retired')),
  calibrated BOOLEAN NOT NULL,
  features JSONB NOT NULL,
  metrics JSONB NOT NULL,
  artifact_path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  trained_by BIGINT REFERENCES app_user(id),
  approved_by BIGINT REFERENCES app_user(id),
  approved_at TIMESTAMPTZ,
  CHECK (status <> 'approved' OR (calibrated AND approved_by IS NOT NULL))
);

CREATE TABLE forecast_latest (
  catchment_id BIGINT NOT NULL REFERENCES catchment(id),
  target TEXT NOT NULL,
  issue_time TIMESTAMPTZ NOT NULL,
  prob REAL NOT NULL CHECK (prob BETWEEN 0 AND 1),
  model_id BIGINT NOT NULL REFERENCES model_version(id),
  PRIMARY KEY (catchment_id, target)
);

CREATE TABLE forecast_hist (             -- only rows above worker.store_min_prob; pruned by retention_days
  issue_time TIMESTAMPTZ NOT NULL,
  catchment_id BIGINT NOT NULL,
  target TEXT NOT NULL,
  prob REAL NOT NULL,
  model_id BIGINT NOT NULL,
  PRIMARY KEY (issue_time, catchment_id, target)
);

CREATE TABLE village_state (
  village_id BIGINT NOT NULL REFERENCES village(id),
  hazard TEXT NOT NULL,
  level SMALLINT NOT NULL, up_count SMALLINT NOT NULL, down_count SMALLINT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (village_id, hazard)
);

CREATE TABLE alert (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  village_id BIGINT NOT NULL REFERENCES village(id),
  hazard TEXT NOT NULL CHECK (hazard IN ('ts', 'cb', 'ff')),
  level SMALLINT NOT NULL CHECK (level BETWEEN 1 AND 3),
  prob REAL NOT NULL, lead_start_min INTEGER, lead_end_min INTEGER, margin_min REAL,
  drivers JSONB, model_ids JSONB,
  explanation JSONB,                                    -- 'why this alert' ingredient card (mp.explain)
  cap_xml TEXT NOT NULL,
  cap_sha256 TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'published', 'cancelled')),
  shadow BOOLEAN NOT NULL,
  hindcast BOOLEAN NOT NULL DEFAULT FALSE,
  required_approvals SMALLINT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX alert_status_idx ON alert (status, created_at DESC);

CREATE TABLE alert_approval (
  alert_id UUID NOT NULL REFERENCES alert(id),
  user_id BIGINT NOT NULL REFERENCES app_user(id),
  decision TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (alert_id, user_id)
);

CREATE TABLE observation (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  geom geometry(Point, 4326) NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('rain_mm_h', 'water_level_cm', 'flood_seen')),
  value REAL,
  user_id BIGINT REFERENCES app_user(id),
  verified BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE audit_log (                  -- hash-chained (see mp.governance) and append-only (triggers below)
  seq BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  body TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE
);
CREATE FUNCTION audit_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'audit_log is append-only'; END $$;
CREATE TRIGGER audit_no_mod BEFORE UPDATE OR DELETE ON audit_log
  FOR EACH ROW EXECUTE FUNCTION audit_append_only();
CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_log
  FOR EACH STATEMENT EXECUTE FUNCTION audit_append_only();
