"""FEATURES_V2: CAPE, CIN and 3-h CIN change from single-level fields (ERA5 layout), failing loudly when absent."""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import netCDF4
import numpy as np
import pandas as pd

from meghprahari import ingest as I, pipeline as P
from test_readers import MET_CFG, write_era5, write_ps

UTC = timezone.utc
T0 = datetime(2025, 6, 30, tzinfo=UTC)
LAT, LON = np.array([32.5, 32.25, 32.0, 31.75, 31.5]), np.array([76.5, 76.75, 77.0, 77.25])
LEV = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])


def add_cape_cin(path, cape, cin):
    with netCDF4.Dataset(path, "a") as f:
        f.createVariable("cape", "f4", ("valid_time", "latitude", "longitude"))[:] = cape
        v = f.createVariable("cin", "f4", ("valid_time", "latitude", "longitude"), fill_value=np.float32(-9999.0))
        v[:] = np.ma.masked_invalid(cin)


class TestFeaturesV2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.pl, self.sl = os.path.join(d, "pl.nc"), os.path.join(d, "sl.nc")
        write_era5(self.pl, [0, 3], LAT, LON, LEV)
        write_ps(self.sl, [0, 3], LAT, LON, pa=95000.0)
        cin = np.full((2, 5, 4), 80.0)
        cin[1] = 30.0                                                 # the lid erodes from 80 to 30 J/kg
        cin[:, 0, 0] = np.nan                                         # undefined CIN (no level of free convection)
        add_cape_cin(self.sl, np.full((2, 5, 4), 1500.0), cin)
        self.cfg = dict(MET_CFG, surface_vars={"cape": "cape", "cin": "cin"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_reader_attaches_surface_fields(self):
        st = I.read_met_states(self.pl, self.cfg, np.full((5, 4), 500.0), ps_path=self.sl)
        m = st[T0]
        self.assertAlmostEqual(float(m.extra["cape"][2, 2]), 1500.0)
        self.assertTrue(np.isnan(m.extra["cin"][0, 0]))               # undefined stays NaN, never 0
        with self.assertRaises(I.IngestError):
            I.read_met_states(self.pl, dict(self.cfg, surface_vars={"cape": "no_such_var"}), np.full((5, 4), 500.0),
                              ps_path=self.sl)

    def test_assemble_v2(self):
        st = I.read_met_states(self.pl, self.cfg, np.full((5, 4), 500.0), ps_path=self.sl)
        now, prev = P.met_fields(st[T0 + timedelta(hours=3)]), P.met_fields(st[T0])
        hem = {k: np.zeros((5, 4)) for k in ("rain_now", "rain_max_1h", "accel", "growth", "speed", "stall")}
        catch = pd.DataFrame({"lat": [32.0, 32.5], "lon": [77.0, 76.5], "slope_mean": 0.5, "hand_mean": 50.0,
                              "tc_min": 30.0, "area_up_km2": 5.0}, index=[1, 2])
        f = P.assemble(catch, hem, now, prev, LAT, LON, LAT, LON, P.FEATURES_V2)
        self.assertEqual(list(f.columns), P.FEATURES_V2)
        self.assertAlmostEqual(f.at[1, "met_cape"], 1500.0)
        self.assertAlmostEqual(f.at[1, "met_cin"], 30.0)
        self.assertAlmostEqual(f.at[1, "met_d_cin_3h"], -50.0)        # lid eroding
        self.assertTrue(np.isnan(f.at[2, "met_cin"]) and np.isnan(f.at[2, "met_d_cin_3h"]))
        self.assertTrue(np.isnan(P.assemble(catch, hem, now, None, LAT, LON, LAT, LON, P.FEATURES_V2)["met_d_cin_3h"]).all())
        self.assertEqual(list(P.assemble(catch, hem, now, prev, LAT, LON, LAT, LON).columns), P.FEATURES_V1)

    def test_assemble_v3(self):
        st = I.read_met_states(self.pl, self.cfg, np.full((5, 4), 500.0), ps_path=self.sl)
        now, prev = P.met_fields(st[T0 + timedelta(hours=3)]), P.met_fields(st[T0])
        hem = {k: np.zeros((5, 4)) for k in ("rain_now", "rain_max_1h", "accel", "growth", "speed", "stall")}
        catch = pd.DataFrame({"lat": [32.0, 32.5], "lon": [77.0, 76.5], "slope_mean": 0.5, "hand_mean": 50.0,
                              "tc_min": 30.0, "area_up_km2": 5.0}, index=[1, 2])
        ctt_lat, ctt_lon = np.array([32.5, 32.0, 31.5]), np.array([76.5, 77.0])
        ctt_stack = np.full((4, 3, 2), 260.0, np.float32)
        ctt_stack[:, 1, 1] = [250.0, 240.0, 230.0, 220.0]                  # cooling into the anchor at (32.0, 77.0)
        ctt = P.ctt_fields(ctt_stack, dt_h=3.0)
        with self.assertRaises(ValueError):
            P.assemble(catch, hem, now, prev, LAT, LON, LAT, LON, P.FEATURES_V3)
        f = P.assemble(catch, hem, now, prev, LAT, LON, LAT, LON, P.FEATURES_V3, ctt=ctt, ctt_lat=ctt_lat,
                       ctt_lon=ctt_lon)
        self.assertEqual(list(f.columns), P.FEATURES_V3)
        self.assertAlmostEqual(f.at[1, "sat_ctt_min"], 220.0)
        self.assertLess(f.at[1, "sat_ctt_drop_rate"], 0)

    def test_v2_without_cape_fails_loudly(self):
        st = I.read_met_states(self.pl, MET_CFG, np.full((5, 4), 500.0), ps_path=self.sl)   # no surface_vars
        hem = {k: np.zeros((5, 4)) for k in ("rain_now", "rain_max_1h", "accel", "growth", "speed", "stall")}
        catch = pd.DataFrame({"lat": [32.0], "lon": [77.0], "slope_mean": 0.5, "hand_mean": 50.0, "tc_min": 30.0,
                              "area_up_km2": 5.0})
        with self.assertRaises(ValueError):
            P.assemble(catch, hem, P.met_fields(st[T0]), None, LAT, LON, LAT, LON, P.FEATURES_V2)


if __name__ == "__main__":
    unittest.main()
