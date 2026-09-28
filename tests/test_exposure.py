import unittest

import numpy as np

from meghprahari import exposure as X
from scripts import build_exposure as S

TR = (0.1, 0.0, 76.8, 0.0, -0.1, 32.0)                         # 0.1-deg cells, north-west corner 32.0 N, 76.8 E
LABELS = np.array([[0, 0, 1, 1]] * 4)                           # west half link 0, east half link 1
LABELS[3, 3] = -1


class TestExposure(unittest.TestCase):
    def test_points_to_links(self):
        links = X.link_of_points([31.95, 31.95, 31.65, 30.0], [76.85, 77.15, 77.15, 76.85], LABELS, TR)
        self.assertEqual(links.tolist(), [0, 1, -1, -1])

    def test_road_length_split_between_catchments(self):
        line = [(31.95, 76.85), (31.95, 77.15)]                 # west-east across both catchments
        km = X.road_km_per_link([line], LABELS, TR, step_m=50)
        total = X.haversine_m(31.95, 76.85, 31.95, 77.15) / 1000
        self.assertAlmostEqual(sum(km.values()), total, places=6)
        self.assertAlmostEqual(km[0], km[1], delta=0.1 * total)  # the boundary is at the middle of the line

    def test_table(self):
        t = X.exposure_table([0, 1, 2], {"schools": np.array([0, 0, 1, -1]), "health": np.array([], int)}, {1: 2.5})
        self.assertEqual(t["schools"].tolist(), [2, 1, 0])
        self.assertEqual(t["health"].tolist(), [0, 0, 0])
        self.assertEqual(t["road_km"].tolist(), [0.0, 2.5, 0.0])

    def test_split_elements(self):
        els = [dict(type="node", lat=31.9, lon=76.9, tags={"amenity": "school"}),
               dict(type="way", center={"lat": 31.8, "lon": 77.0}, tags={"amenity": "hospital"}),
               dict(type="node", lat=31.7, lon=77.1, tags={"healthcare": "clinic"}),
               dict(type="way", tags={"highway": "primary"}, geometry=[{"lat": 31.9, "lon": 76.9}, {"lat": 31.9, "lon": 77.0}]),
               dict(type="way", tags={"highway": "footway"}, geometry=[{"lat": 31.9, "lon": 76.9}])]
        schools, health, roads = S.split_elements(els)
        self.assertEqual(schools, [(31.9, 76.9)])
        self.assertEqual(len(health), 2)
        self.assertEqual(len(roads), 1)


if __name__ == "__main__":
    unittest.main()
