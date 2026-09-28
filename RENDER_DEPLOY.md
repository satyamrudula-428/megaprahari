# Render deployment — MeghPrahari demo

This version is prepared for a first public demo on Render. It uses the existing synthetic exercise data, not live MOSDAC/NCMRWF weather data.

## 1. Create a Render PostgreSQL database
Create a PostgreSQL database in the same Render region as the web service. Enable PostGIS if the database dashboard exposes the extension. Copy its **internal database URL**.

## 2. Create a Render Web Service
Use the GitHub repository containing this project.

- Runtime: Docker
- Branch: `main`
- Root Directory: blank
- Plan: Free for the first test

The Dockerfile automatically uses Render's `PORT` environment variable.

## 3. Add these environment variables

`MP_DATABASE_URL` = the internal URL of the Render PostgreSQL database

`MP_JWT_SECRET` = a random secret of at least 32 characters

Optional:

`MP_DEMO_PASSWORD` = your demo password (minimum 12 characters). If omitted, the demo password is `MeghPrahariDemo123!`.

## 4. Deploy
The container will:

1. wait for PostgreSQL;
2. enable PostGIS;
3. create the MeghPrahari schema;
4. seed synthetic exercise data if the database is empty;
5. start FastAPI.

Open `/health` after deployment.

Demo login when the database was freshly seeded:

- username: `approver`
- password: `MeghPrahariDemo123!` unless `MP_DEMO_PASSWORD` was set.

## Important
This is a demonstration deployment. The seeded forecasts and alerts are synthetic exercise data. It does **not** claim live weather forecasting skill or live MOSDAC/NCMRWF ingestion.

## One-click Blueprint option

This repository includes `render.yaml`. In Render, create a new Blueprint from the repository and let Render create the web service and PostgreSQL database from that file. Set `MP_JWT_SECRET` and `MP_DEMO_PASSWORD` when prompted.
