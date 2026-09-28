"""Readers on small synthetic files that copy the layouts of the real 2025 files (inspected with
tools/describe_file.py): IMERG HDF5 (time, lon, lat), MERA NetCDF (time, lat desc, lon), ERA5 pressure levels
(valid_time, pressure_level, lat desc, lon) with surface pressure in a separate single-level file."""
import os
import tempfile
import unittest
from datetime import datetime, timezone

import numpy as np

from meghprahari import ingest as I

UTC = timezone.utc
MET_CFG = dict(time_var="valid_time",
               vars=dict(q="q", t="t", u="u", v="v", ps="sp", level="pressure_level", lat="latitude", lon="longitude"),
               units=dict(q_scale=1.0, ps_scale=0.01, level_scale=1.0))


def write_imerg(path, lat, lon, value_at):
    import h5py
    a = np.zeros((1, lon.size, lat.size), np.float32)
    a[0, 0, :] = -9999.9                                          # one fill column (first longitude)
    (iy, ix), val = value_at
    a[0, ix, iy] = val
    with h5py.File(path, "w") as f:
        f["Grid/precipitation"] = a
        f["Grid/lat"], f["Grid/lon"] = lat.astype(np.float32), lon.astype(np.float32)


def write_era5(path, times_h, lat, lon, lev, with_ps=False):
    import netCDF4
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("valid_time", len(times_h))
        f.createDimension("pressure_level", lev.size)
        f.createDimension("latitude", lat.size)
        f.createDimension("longitude", lon.size)
        tv = f.createVariable("valid_time", "i8", ("valid_time",))
        tv.units, tv.calendar = "seconds since 1970-01-01", "proleptic_gregorian"
        t0 = int(datetime(2025, 6, 30, tzinfo=UTC).timestamp())
        tv[:] = [t0 + 3600 * h for h in times_h]
        f.createVariable("pressure_level", "f8", ("pressure_level",))[:] = lev
        f.createVariable("latitude", "f8", ("latitude",))[:] = lat
        f.createVariable("longitude", "f8", ("longitude",))[:] = lon
        shape = (len(times_h), lev.size, lat.size, lon.size)
        prof = lambda vals: np.broadcast_to(np.asarray(vals, float)[None, :, None, None], shape)
        f.createVariable("q", "f4", ("valid_time", "pressure_level", "latitude", "longitude"))[:] = prof(
            [0.016, 0.012, 0.008, 0.003, 0.0005][:lev.size])
        f.createVariable("t", "f4", ("valid_time", "pressure_level", "latitude", "longitude"))[:] = prof(
            [302, 293, 283, 266, 240][:lev.size])
        for n, val in (("u", 5.0), ("v", 2.0)):
            f.createVariable(n, "f4", ("valid_time", "pressure_level", "latitude", "longitude"))[:] = np.full(shape, val)
        if with_ps:
            f.createVariable("sp", "f4", ("valid_time", "latitude", "longitude"))[:] = np.full(shape[:1] + shape[2:], 92000.0)


def write_ps(path, times_h, lat, lon, pa=92000.0):
    import netCDF4
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("valid_time", len(times_h))
        f.createDimension("latitude", lat.size)
        f.createDimension("longitude", lon.size)
        tv = f.createVariable("valid_time", "i8", ("valid_time",))
        tv.units = "seconds since 1970-01-01"
        t0 = int(datetime(2025, 6, 30, tzinfo=UTC).timestamp())
        tv[:] = [t0 + 3600 * h for h in times_h]
        f.createVariable("latitude", "f8", ("latitude",))[:] = lat
        f.createVariable("longitude", "f8", ("longitude",))[:] = lon
        f.createVariable("sp", "f4", ("valid_time", "latitude", "longitude"))[:] = np.full((len(times_h), lat.size, lon.size), pa)


class TestRainReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_imerg_lon_lat_order_and_crop(self):
        lat, lon = np.round(np.arange(29.05, 34.0, 0.1), 2), np.round(np.arange(75.05, 79.0, 0.1), 2)
        iy, ix = int(np.argmin(abs(lat - 31.75))), int(np.argmin(abs(lon - 77.05)))
        p = os.path.join(self.d, "3B-HHR.MS.MRG.3IMERG.20250630-S193000-E195959.1170.V07B.HDF5")
        write_imerg(p, lat, lon, ((iy, ix), 21.4))
        c = dict(var="Grid/precipitation", lat_var="Grid/lat", lon_var="Grid/lon", axis_order="lon,lat",
                 bbox=dict(lat_min=30.0, lat_max=33.0, lon_min=76.0, lon_max=78.0))
        raw, la, lo = I.read_hem_raw(p, c)
        self.assertEqual(raw.shape, (la.size, lo.size))
        self.assertTrue(30.0 <= la.min() and la.max() <= 33.0 and 76.0 <= lo.min() and lo.max() <= 78.0)
        self.assertAlmostEqual(float(raw[np.argmin(abs(la - 31.75)), np.argmin(abs(lo - 77.05))]), 21.4, places=4)
        frame, _ = I.qc_rain_frame(raw, [-9999.9])
        self.assertFalse(np.isnan(frame).any())                   # the fill column was cropped away
        with self.assertRaises(I.IngestError):                    # wrong axis order is caught, not silently accepted
            I.read_hem_raw(p, dict(c, axis_order="lat,lon"))
        t = I.parse_time_from_name(os.path.basename(p), r"3IMERG\.(\d{8}-S\d{6})", "%Y%m%d-S%H%M%S")
        self.assertEqual(t, datetime(2025, 6, 30, 19, 30, tzinfo=UTC))

    def test_mera_descending_lat(self):
        import netCDF4
        lat, lon = 32.99 - 0.0375 * np.arange(80), 76.02 + 0.0375 * np.arange(53)
        p = os.path.join(self.d, "mera_2025070100.nc")
        with netCDF4.Dataset(p, "w") as f:
            f.createDimension("time", None)
            f.createDimension("latitude", lat.size)
            f.createDimension("longitude", lon.size)
            f.createVariable("latitude", "f8", ("latitude",))[:] = lat
            f.createVariable("longitude", "f8", ("longitude",))[:] = lon
            r = f.createVariable("Rainfall", "f4", ("time", "latitude", "longitude"), fill_value=-999.0)
            a = np.zeros((1, lat.size, lon.size), np.float32)
            a[0, 10, 20] = 25.5
            a[0, 0, 0] = -999.0
            r[:] = a
        raw, la, lo = I.read_hem_raw(p, dict(var="Rainfall", lat_var="latitude", lon_var="longitude"))
        self.assertEqual(raw.shape, (80, 53))
        self.assertGreater(la[0], la[-1])                          # descending latitude kept as in the file
        self.assertTrue(np.isnan(raw[0, 0]))                       # masked fill value -> NaN
        self.assertAlmostEqual(float(raw[10, 20]), 25.5, places=4)
        t = I.parse_time_from_name("mera_2025070100.nc", r"mera_(\d{10})", "%Y%m%d%H")
        self.assertEqual(t, datetime(2025, 7, 1, 0, tzinfo=UTC))

    def test_crop_rejects_empty(self):
        with self.assertRaises(I.IngestError):
            I.crop_to_bbox(np.zeros((5, 5)), np.arange(5.0), np.arange(5.0), dict(lat_min=10, lat_max=11, lon_min=0, lon_max=4))


class TestEra5Reader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.lat, self.lon = np.array([32.5, 32.25, 32.0, 31.75, 31.5]), np.array([76.5, 76.75, 77.0, 77.25])
        self.lev = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])
        self.pl, self.ps = os.path.join(d, "era5_pl.nc"), os.path.join(d, "era5_sl.nc")
        write_era5(self.pl, [0, 3, 6, 9], self.lat, self.lon, self.lev)
        write_ps(self.ps, [0, 3, 6, 9], self.lat, self.lon)
        self.terrain = np.full((5, 4), 900.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_times_with_separate_ps(self):
        self.assertEqual(len(I.met_file_times(self.pl, MET_CFG)), 4)
        states = I.read_met_states(self.pl, MET_CFG, self.terrain, ps_path=self.ps)
        self.assertEqual(sorted(states), [datetime(2025, 6, 30, h, tzinfo=UTC) for h in (0, 3, 6, 9)])
        m = states[datetime(2025, 6, 30, 3, tzinfo=UTC)]
        self.assertEqual(list(m.p_hpa), [1000.0, 850.0, 700.0, 500.0, 300.0])
        self.assertAlmostEqual(float(m.ps_hpa[0, 0]), 920.0, places=3)   # Pa -> hPa
        self.assertAlmostEqual(float(m.q[0, 0, 0]), 0.016, places=6)
        self.assertEqual(m.q.shape, (5, 5, 4))
        sub = I.read_met_states(self.pl, MET_CFG, self.terrain, ps_path=self.ps,
                                times=[datetime(2025, 6, 30, 6, tzinfo=UTC), datetime(2025, 6, 30, 7, tzinfo=UTC)])
        self.assertEqual(list(sub), [datetime(2025, 6, 30, 6, tzinfo=UTC)])

    def test_level_subset(self):
        sub = dict(MET_CFG, levels_hpa=[850, 500, 1000])
        m = I.read_met_states(self.pl, sub, self.terrain, ps_path=self.ps)[datetime(2025, 6, 30, tzinfo=UTC)]
        self.assertEqual(list(m.p_hpa), [1000.0, 850.0, 500.0])
        self.assertAlmostEqual(float(m.q[1, 0, 0]), 0.012, places=6)                       # the 850 hPa value
        with self.assertRaises(I.IngestError):
            I.read_met_states(self.pl, dict(MET_CFG, levels_hpa=[925]), self.terrain, ps_path=self.ps)

    def test_ps_in_same_file(self):
        p = os.path.join(self.tmp.name, "with_ps.nc")
        write_era5(p, [0, 3], self.lat, self.lon, self.lev, with_ps=True)
        self.assertEqual(len(I.read_met_states(p, MET_CFG, self.terrain)), 2)

    def test_missing_surface_pressure_fails_loudly(self):
        with self.assertRaises(I.IngestError):
            I.read_met_states(self.pl, MET_CFG, self.terrain)                     # no ps anywhere
        short = os.path.join(self.tmp.name, "sl_short.nc")
        write_ps(short, [0, 3], self.lat, self.lon)
        with self.assertRaises(I.IngestError):
            I.read_met_states(self.pl, MET_CFG, self.terrain, ps_path=short)      # ps missing for 06 and 09 UTC
        other = os.path.join(self.tmp.name, "sl_grid.nc")
        write_ps(other, [0, 3, 6, 9], self.lat + 0.1, self.lon)
        with self.assertRaises(I.IngestError):
            I.read_met_states(self.pl, MET_CFG, self.terrain, ps_path=other)      # different grid

    def test_wrong_units_are_caught(self):
        with self.assertRaises(ValueError):                                       # ps left in Pa -> out of range
            I.read_met_states(self.pl, dict(MET_CFG, units=dict(MET_CFG["units"], ps_scale=1.0)), self.terrain,
                              ps_path=self.ps)


class TestMetIndex(unittest.TestCase):
    def test_lazy_many_time_files(self):
        with tempfile.TemporaryDirectory() as d:
            lat, lon = np.array([32.5, 32.25, 32.0, 31.75, 31.5]), np.array([76.5, 76.75, 77.0, 77.25])
            lev = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])
            a, b, ps = (os.path.join(d, n) for n in ("era5_pl_a.nc", "era5_pl_b.nc", "sl.nc"))
            write_era5(a, [0, 3], lat, lon, lev)
            write_era5(b, [6, 9], lat, lon, lev)
            write_ps(ps, [0, 3, 6, 9], lat, lon)
            idx = I.MetIndex(MET_CFG, np.full((5, 4), 900.0), ps, lat, lon, keep=2)
            self.assertEqual(len(idx.add_file(a)) + len(idx.add_file(b)), 4)
            ts = idx.times()
            self.assertEqual(ts[-1], datetime(2025, 6, 30, 9, tzinfo=UTC))
            for t in ts[::-1]:                                            # newest first, then older ones
                self.assertEqual(idx.get(t).time, t)
                self.assertLessEqual(len(idx.cache), 2)
            with self.assertRaises(I.IngestError):
                idx.get(datetime(2025, 7, 1, tzinfo=UTC))
            bad = I.MetIndex(MET_CFG, np.full((5, 4), 900.0), ps, lat + 1.0, lon)
            bad.add_file(a)
            with self.assertRaises(I.IngestError):                        # terrain grid mismatch is caught
                bad.get(ts[0])


if __name__ == "__main__":
    unittest.main()
