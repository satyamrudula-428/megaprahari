import unittest
import numpy as np
from meghprahari import terrain as T, meteo as M, satellite as S


class TestTerrain(unittest.TestCase):
    def test_kirpich_example(self):
        self.assertAlmostEqual(float(T.kirpich_tc_min(3000, 0.15)), 19.3, delta=0.2)
        self.assertTrue(np.isnan(T.kirpich_tc_min(0, 0.1)))

    def test_fill_pit_and_conservation(self):
        h = w = 30
        y, _ = np.mgrid[0:h, 0:w]
        dem = 100.0 - y
        dem[15, 15] -= 20
        z = T.fill_depressions(dem)
        self.assertGreater(z[15, 15], 84.0)
        self.assertLess(z[15, 15], 84.01)
        down = T.d8_down(z, (1.0, 1.0))
        interior = np.zeros((h, w), bool)
        interior[1:-1, 1:-1] = True
        self.assertTrue((down.reshape(h, w)[interior] >= 0).all())
        acc = T.flow_accumulation(down, np.argsort(-z.ravel(), kind="stable"))
        self.assertEqual(acc[down == -1].sum(), h * w)

    def test_plane_accumulation(self):
        h = w = 30
        y, _ = np.mgrid[0:h, 0:w]
        z = T.fill_depressions(100.0 - y)
        down = T.d8_down(z, (1.0, 1.0))
        acc = T.flow_accumulation(down, np.argsort(-z.ravel(), kind="stable")).reshape(h, w)
        self.assertTrue((acc[h - 1, 1:w - 1] == h - 1).all())

    def test_v_valley_full_chain(self):
        h = w = 41
        cx = 20
        y, x = np.mgrid[0:h, 0:w]
        dem = 1.0 * np.abs(x - cx) + 0.05 * (h - 1 - y)
        r = T.analyze(dem, (1.0, 1.0), stream_cells=78)
        self.assertEqual(len(r["table"]), 1)
        row = r["table"].iloc[0]
        self.assertAlmostEqual(r["hand"][30, cx + 5], 5.0, places=6)
        self.assertAlmostEqual(r["hand"][30, cx], 0.0, places=6)
        self.assertEqual(round(row.area_up_km2 * 1e6), (h - 2) * (w - 2) + 1)
        self.assertEqual(round(row.area_local_km2 * 1e6), (h - 2) * (w - 2) + 1)
        self.assertAlmostEqual(row.flow_len_m, 58.0, places=6)
        top = 19 * 1.0 + 0.05 * (h - 2)
        self.assertAlmostEqual(row.tc_min, float(T.kirpich_tc_min(58.0, top / 58.0)), places=6)

    def test_label_links_junction(self):
        down = np.array([2, 2, 3, 4, -1])
        stream = np.ones(5, bool)
        link, n = T.label_links(down, stream, np.array([0, 1, 2, 3, 4]))
        self.assertEqual(n, 3)
        self.assertEqual(list(link), [0, 1, 2, 2, 2])

    def test_refuge(self):
        hnd = np.zeros((5, 21))
        hnd[2, 15] = 20.0
        dist, bearing = T.refuge_map(hnd, 10.0, (30.0, 30.0))
        self.assertAlmostEqual(dist[2, 5], 300.0)
        self.assertAlmostEqual(bearing[2, 5], 90.0)
        self.assertEqual(dist[2, 15], 0.0)
        self.assertIsNone(T.refuge_map(np.zeros((3, 3)), 10.0, (1, 1))[0])

    def test_fill_voids(self):
        dem = np.array([[1.0, 2.0, 3.0], [4.0, -9999.0, 6.0], [7.0, 8.0, 9.0]])
        out = T.fill_voids_nearest(dem, dem < -1000)
        self.assertIn(out[1, 1], (2.0, 4.0, 6.0, 8.0))
        with self.assertRaises(ValueError):
            T.fill_depressions(np.array([[1.0, np.nan], [1.0, 1.0]]))


class TestMeteo(unittest.TestCase):
    def test_iwv_constant_q(self):
        p = [1000, 850, 700, 500]
        q = np.full((4, 2, 2), 0.01)
        self.assertAlmostEqual(float(M.iwv(q, p)[0, 0]), 0.01 * 500 * 100 / M.G, places=6)
        self.assertAlmostEqual(float(M.iwv(q, p, np.full((2, 2), 1013.0))[0, 0]), 0.01 * 513 * 100 / M.G, places=6)
        self.assertAlmostEqual(float(M.iwv(q, p, np.full((2, 2), 900.0))[0, 0]), 0.01 * 400 * 100 / M.G, places=6)
        with self.assertRaises(ValueError):
            M.iwv(q, [500, 700, 850, 1000])

    def test_dewpoint(self):
        self.assertAlmostEqual(float(M.dewpoint_c(6.112)), 0.0, places=6)
        es = 6.112 * np.exp(17.67 * 25 / (25 + 243.5))
        self.assertAlmostEqual(float(M.dewpoint_c(0.5 * es)), 13.87, delta=0.02)

    def test_indices(self):
        self.assertEqual(M.k_index(20, 16, 8, 2, -10), 40)
        self.assertEqual(M.total_totals(20, 16, -10), 56)

    def test_divergence_and_mfc(self):
        lat = np.linspace(20, 30, 21)
        lon = np.linspace(70, 80, 21)
        a = 1e-5
        lam = np.deg2rad(lon - 75.0)[None, :]
        fx = a * M.R_EARTH * np.cos(np.deg2rad(lat))[:, None] * lam
        div = M.divergence(fx, np.zeros_like(fx), lat, lon)
        self.assertTrue(np.allclose(div, a, rtol=1e-9))
        q = np.full_like(fx, 0.01)
        self.assertTrue(np.allclose(M.mfc(q, fx, np.zeros_like(fx), lat, lon), -0.01 * a, rtol=1e-9))

    def test_omfi_upslope(self):
        lat = np.linspace(-1, 1, 11)
        lon = np.linspace(0, 1, 11)
        s = 0.1
        h = s * M.R_EARTH * np.deg2rad(lon)[None, :] * np.ones((11, 1))
        q = np.full((11, 11), 0.01)
        up = M.omfi(q, np.full((11, 11), 10.0), np.zeros((11, 11)), h, lat, lon)
        down = M.omfi(q, np.full((11, 11), -10.0), np.zeros((11, 11)), h, lat, lon)
        self.assertTrue(np.allclose(up[5], 0.01 * 10 * s, rtol=2e-3))
        self.assertTrue(np.allclose(down[5], -0.01 * 10 * s, rtol=2e-3))

    def test_lowest_above_ground_and_regrid(self):
        p = [1000, 850, 700]
        f = np.stack([np.full((1, 2), 1.0), np.full((1, 2), 2.0), np.full((1, 2), 3.0)])
        ps = np.array([[1010.0, 800.0]])
        out = M.lowest_above_ground(f, p, ps)
        self.assertEqual(list(out[0]), [2.0, 3.0])  # 1000 hPa is inside the 25 hPa margin at ps=1010
        self.assertTrue(np.isnan(M.lowest_above_ground(f, p, np.array([[600.0, 600.0]]))).all())
        src = np.arange(16.0).reshape(4, 4)
        r = M.regrid_mean(src, [3.5, 2.5, 1.5, 0.5], [0.5, 1.5, 2.5, 3.5], [3.0, 1.0], [1.0, 3.0])
        self.assertEqual(r.shape, (2, 2))
        self.assertAlmostEqual(r[0, 0], np.mean([0, 1, 4, 5]))
        self.assertAlmostEqual(r[1, 1], np.mean([10, 11, 14, 15]))


class TestSatellite(unittest.TestCase):
    def test_trend(self):
        t = np.arange(6) * 0.5
        frames = (2.0 + 3.0 * t).reshape(6, 1, 1) * np.ones((6, 3, 3))
        self.assertTrue(np.allclose(S.linear_trend(frames, 0.5), 3.0))
        frames[:4] = np.nan
        self.assertTrue(np.isnan(S.linear_trend(frames, 0.5)).all())

    def test_growth(self):
        fr = np.zeros((6, 20, 20))
        fr[3:, 8:12, 8:12] = 15.0
        g = S.exceed_growth(fr, 10.0, half_window_px=3)
        self.assertAlmostEqual(g[10, 10], 16 / 49, places=9)
        self.assertEqual(g[0, 0], 0.0)

    def test_motion_direction_and_speed(self):
        yy, xx = np.mgrid[0:64, 0:64]
        blob = lambda cy, cx: 20 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
        vy, vx = S.motion_tiles(blob(20, 20), blob(22, 23), 30, 4.0, tile=64)
        self.assertAlmostEqual(vy[0, 0], 16.0, delta=2.0)
        self.assertAlmostEqual(vx[0, 0], 24.0, delta=2.0)

    def test_cells_and_stall(self):
        fr = np.zeros((10, 10))
        fr[1:4, 1:4] = 20.0
        fr[7, 7] = 20.0
        a = S.cell_area_map(fr, 10.0, 16.0)
        self.assertEqual(a[2, 2], 9 * 16.0)
        self.assertEqual(a[7, 7], 16.0)
        self.assertEqual(a[5, 5], 0.0)
        self.assertAlmostEqual(float(S.stall_index(100, 10)), 1.0)
        self.assertAlmostEqual(float(S.stall_index(100, 2)), 5.0)
        self.assertAlmostEqual(float(S.stall_index(100, 0.2)), 10.0)
        self.assertTrue(np.isnan(S.stall_index(0, 5)))


if __name__ == "__main__":
    unittest.main()
