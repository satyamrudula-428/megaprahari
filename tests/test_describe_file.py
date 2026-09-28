import os
import tempfile
import unittest

import numpy as np

from scripts import describe_file as D


class TestDescribeFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_hdf5(self):
        import h5py
        p = os.path.join(self.d, "x.h5")
        with h5py.File(p, "w") as f:
            f.attrs["title"] = "synthetic"
            ds = f.create_dataset("grp/rain", data=np.zeros((4, 5), np.int16))
            ds.attrs["_FillValue"] = np.int16(-999)
            ds.attrs["scale_factor"] = 0.1
            f.create_dataset("lat", data=np.linspace(31.5, 32.0, 4))
        out = "\n".join(D.describe(p))
        self.assertIn("grp/rain  Dataset (4, 5) int16", out)
        self.assertIn("@_FillValue = -999", out)
        self.assertIn("@scale_factor = 0.1", out)
        self.assertIn("lat  Dataset (4,) float64", out)
        self.assertIn("@title = 'synthetic'", out)

    def test_netcdf(self):
        import netCDF4
        p = os.path.join(self.d, "x.nc")
        with netCDF4.Dataset(p, "w") as f:
            f.createDimension("time", None)
            f.createDimension("level", 3)
            v = f.createVariable("t", "f4", ("time", "level"))
            v.units = "K"
            f.source = "synthetic"
        out = "\n".join(D.describe(p))
        self.assertIn("time = UNLIMITED 0", out)
        self.assertIn("level = 3", out)
        self.assertIn("float32 t(time, level)", out)
        self.assertIn("@units = 'K'", out)
        self.assertIn("@source = 'synthetic'", out)

    def test_geotiff(self):
        import rasterio
        from rasterio.transform import from_origin
        p = os.path.join(self.d, "dem.tif")
        with rasterio.open(p, "w", driver="GTiff", height=3, width=4, count=1, dtype="float32", crs="EPSG:4326",
                           transform=from_origin(76.8, 32.0, 1 / 3600, 1 / 3600), nodata=-32768) as ds:
            ds.write(np.ones((1, 3, 4), np.float32))
        out = "\n".join(D.describe(p))
        self.assertIn("EPSG:4326", out)
        self.assertIn("3 rows x 4 cols", out)
        self.assertIn("nodata = -32768", out)

    def test_long_attributes_are_truncated_and_unknown_types_rejected(self):
        self.assertTrue(D._fmt("x" * 1000).endswith("..."))
        self.assertIn("array(100,)", D._fmt(np.arange(100)))
        with self.assertRaises(ValueError):
            D.describe(os.path.join(self.d, "x.grib"))


if __name__ == "__main__":
    unittest.main()
