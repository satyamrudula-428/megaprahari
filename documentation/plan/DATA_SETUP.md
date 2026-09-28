# Data setup — MeghPrahari pilot (Mandi, Himachal Pradesh)

Step-by-step guide for getting real data into MeghPrahari. Do **Part A** (registrations) first, then
**Parts B–H** as approvals arrive. Portal menus change occasionally; if a button name differs slightly, look for
the closest match.

Record every dataset you download in [`documentation/DATA_REGISTER.md`](../DATA_REGISTER.md) (task A15).

## Part A: Register everywhere (about 30 minutes)

Use one shared team email, so approvals and download links reach everyone.

| # | Site | What to do |
|---|---|---|
| 1 | https://mosdac.gov.in | Click **SignUp**, fill in the form (institution: your college; purpose: "Research – SIH 2026 severe weather nowcasting"), click **I Agree**, then wait for the approval email |
| 2 | https://rds.ncmrwf.gov.in | Register / sign up, then wait for confirmation |
| 3 | https://bhuvan.nrsc.gov.in | Browsing works without an account; register if a CartoDEM download asks you to log in |
| 4 | https://urs.earthdata.nasa.gov | Create a NASA Earthdata login (for IMERG and SRTM) |
| 5 | https://opentopography.org | Create a free account (easiest SRTM download) |

MOSDAC approval can take a few days. Do Parts C and D while you wait.

Keep passwords in a `.env` file (git-ignored). Never commit them or paste them into chats.

## Part B: Pilot area and dates

These are fixed in [`settings/region.yaml`](../../settings/region.yaml) (tasks A1, A2, B7). Everyone downloads for the
**same box and dates**.

- **Region:** Mandi area, Himachal Pradesh
- **Box:** latitude 31.5–32.0 °N, longitude 76.8–77.3 °E
  - About 55 × 55 km: 1800 × 1800 cells at 1 arc-second (~30 m), inside `build_atlas.py`'s limit of 4 M cells.
- **Event:** Mandi cloudbursts, 30 Jun – 1 Jul 2025
- **Download window:** 29 Jun – 2 Jul 2025, so the model sees the build-up and the aftermath
- **Met download box:** the pilot box plus 0.5° margin (31.0–32.5 °N, 76.3–77.8 °E), because the met grid is ~12 km

Check the file loads: `PYTHONPATH=backend python -c "from meghprahari.settings import load_region; print(load_region('settings/region.yaml')['region'])"`

## Part C: Terrain data (DEM) and the atlas

**1. Download the DEM from OpenTopography** (easiest option)

1. Log in, then go to **Find Data** → **SRTM GL1 (30 m)**.
2. Enter the box from Part B.
3. Choose output **GeoTIFF**, geographic coordinates (WGS84 / EPSG:4326). `build_atlas.py` rejects any other CRS.
4. Download and rename the file to `mandi_dem.tif`.

(CartoDEM from Bhuvan also works; pick one and stick with it.)

**2. Set up the project folders**

```bash
sh tools/make_data_dirs.sh            # creates the data/ layout below (safe to re-run)
cp ~/Downloads/mandi_dem.tif data/inputs/
```

```text
data/
  inputs/          DEM, villages.csv
  inputs/tir/      MOSDAC TIR1 (Level-1 imager) files
  landing/hem/     MOSDAC HEM rain-rate files (the worker watches this)
  landing/met/     NCMRWF IMDAA / IMDAA-like NetCDF files (the worker watches this)
  ledger/          event ledger (events.csv)
  atlas/           build_atlas.py output
  work/            scratch / training tables
  models/          trained model artefacts
  outbox/          CAP exercise alerts
```

**3. Build the atlas**

```bash
pip install -r requirements.txt
PYTHONPATH=backend python tools/build_atlas.py \
  --dem data/inputs/mandi_dem.tif \
  --atlas-id mandi_v1 \
  --out-dir data/atlas
```

**4. Check the output**
- `data/atlas/catchments.csv` should list hundreds of catchments.
- `meta.json` should show a small `void_fraction`, close to 0.
- Open the DEM and `catchments.csv` in QGIS (free) and confirm the streams follow the real valleys.
- Set `atlas_id: mandi_v1` in `settings/settings.yaml`.

## Part D: Villages CSV

Shortcut: `python tools/osm_village_candidates.py --osm-json data/work/osm_places.json` lists every named
OpenStreetMap village in the box with coordinates, linked to the atlas and sorted by stream size. Keep the rows you
want. `population`, `state` and `district` may be left blank (stored as NULL); population is only needed later for
the exposure panel.

1. Get village names and populations for Mandi district from https://censusindia.gov.in or https://lgdirectory.gov.in.
2. Get coordinates for each village from OpenStreetMap or Bhuvan.
3. Save the file as `data/inputs/villages.csv` with at least these columns (extra columns are ignored):

```csv
name,state,district,lat,lon,population,action_cost,loss
Village A,Himachal Pradesh,Mandi,31.7081,76.9318,1250,1,10
```

`action_cost` and `loss` are the cost of acting versus the loss if no one acts. Use `1` and `10` for now and state
in the slides that the disaster authority would set them.

About 20–50 villages is enough for the demo. Load them later with `tools/load_villages.py`, once the database is
running.

## Part E: Atmospheric data from NCMRWF RDS

1. Log in to https://rds.ncmrwf.gov.in and open **Datasets**.
2. For 2025 dates, choose the **IMDAA-like** pressure-level dataset. (Choose IMDAA for 2017–2020 when building
   training data later.)
3. **Select these variables** (the code needs them):
   - Temperature
   - U and V wind components
   - Specific humidity. If only relative humidity is offered, download it; it will be converted with MetPy (task C4).
   - Surface pressure from the single-level dataset.
4. **Area:** the met download box from Part B.
5. **Dates:** 29 Jun – 2 Jul 2025, all available times (3-hourly).
6. **Format:** NetCDF if offered, otherwise GRIB.
7. Submit. You will get an email with a wget script; run it:

```bash
cd data/landing/met
bash downloaded_script.sh
```

8. If the files are GRIB, convert them (task C3): `cdo -f nc copy input.grib output.nc`

## Part F: Satellite data from MOSDAC (once approved)

**1. Rain-rate (HEM) files**
1. Log in and go to **Data Access** → **Order Data**.
2. In the catalog choose **Satellite** → INSAT-3DR (or INSAT-3DS if 3DR has no data for the dates) → the **HEM**
   (hydro-estimator rain rate) product.
3. Choose **Archival** order, dates 29 Jun – 2 Jul 2025.
4. Set the region of interest to the pilot box; format **HDF5**.
5. Place the order and wait for the download link.
6. Put all files in `data/landing/hem/`.

**2. Thermal-infrared (TIR1) the same way**
- Choose the Level-1 imager product with the **TIR1** band, same dates and area.
- Put the files in `data/inputs/tir/`. They feed the cloud-top temperature features (C7, D1, D2).

**3. Later, for live mode**
- Set up a **standing order** for HEM and TIR1, or use the MOSDAC API (https://mosdac.gov.in/node/2070) with
  `mdapi.py` (task C12).

## Part G: Inspect the real files and fill `settings.yaml`

This is the most important step: **never guess variable names.**

```bash
# Satellite HDF5: list everything inside
h5ls -r data/landing/hem/<one_file>.h5

# Weather NetCDF: show variables, dimensions, units
ncdump -h data/landing/met/<one_file>.nc

# Windows / no HDF5 or netCDF command-line tools: same information from Python
python tools/describe_file.py data/landing/hem/<one_file>.h5 data/landing/met/<one_file>.nc
```

Then copy the exact names into `settings/settings.yaml`:

| Setting | Where the value comes from |
|---|---|
| `hem.var`, `hem.lat_var`, `hem.lon_var` | Names in the HDF5 listing (rain-rate array and its latitude/longitude) |
| `hem.fill_values`, `scale`, `offset` | Attributes of the rain-rate variable |
| `hem.time_regex`, `time_format` | The file-name pattern (see below) |
| `met.vars` (q, t, u, v, ps, level, lat, lon) | Names in the NetCDF header |
| `met.units` | Scale factors so humidity is kg/kg and pressure is hPa |
| `sender`, `sender_name` | `meghprahari.demo` / `MeghPrahari Exercise` for now |

**File-name example.** INSAT files are usually named like `3RIMG_30JUN2025_0015_L2B_HEM_V01R00.h5`. Check the
actual files; only if they match, use:

```yaml
time_regex: "3[RS]IMG_(\\d{2}[A-Z]{3}\\d{4}_\\d{4})"
time_format: "%d%b%Y_%H%M"
```

## Part H: Check that everything loads

```bash
PYTHONPATH=backend python -c "
from meghprahari.settings import load_settings
cfg = load_settings('settings/settings.yaml')
print('Settings OK')
"
```

This fails with `ConfigError` until every `CHANGE_ME` is filled — that is intended (fail closed).
Then open one HEM file with `read_hem_raw` and plot it. Rain over Himachal on 30 June means the first real data is
in MeghPrahari.

## Team checklist

- [ ] Registered on all 5 portals
- [ ] Region and dates agreed (`settings/region.yaml`)
- [ ] DEM downloaded and atlas built
- [ ] Villages CSV made
- [ ] IMDAA-like files downloaded
- [ ] MOSDAC HEM + TIR1 ordered and downloaded
- [ ] Every dataset recorded in `documentation/DATA_REGISTER.md`
- [ ] All `CHANGE_ME` values filled from real file inspection
- [ ] Real data loads without errors

## Part I: Training data (past monsoons, not the replay event)

The model must learn from **other** storms: the 2025 replay window in `settings/region.yaml` is never used for training
(`tools/fetch_era5_cds.py` refuses it).

**1. ERA5 via the CDS API (one-time key setup, ~3 minutes)**
1. Log in at https://cds.climate.copernicus.eu → click your name → **Your profile** → copy the **API Token**.
2. Make sure the licence is accepted on both pages: `reanalysis-era5-pressure-levels` and
   `reanalysis-era5-single-levels` (Download tab → Terms of use → Accept).
3. In a separate PowerShell window (never paste the token into a chat):

```powershell
$k = Read-Host "CDS API token"
"url: https://cds.climate.copernicus.eu/api`nkey: $k" | Out-File -Encoding ascii "$HOME\.cdsapirc"
```

4. Download Jun–Sep 2021–2024 (112 small requests; runs unattended, resumes if interrupted):

```bash
python tools/fetch_era5_cds.py --years 2021-2024 --months 6-9 --dry-run   # check the plan
python tools/fetch_era5_cds.py --years 2021-2024 --months 6-9
```

**2. MERA rain for the same months (NCMRWF RDS, your login)** — one request per year (2021, 2022, 2023, 2024):
Year = that year · Month = Jun, Jul, Aug, Sep · Day = Select All · Time = all 24 · Area N 33, S 30, E 78, W 76 ·
NetCDF4 · Zip · accept licence. Put each zip in `data/landing/`.
