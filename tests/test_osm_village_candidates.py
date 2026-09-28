import unittest

import numpy as np
import pandas as pd

from scripts import osm_village_candidates as V

BBOX = dict(lat_min=31.5, lat_max=32.0, lon_min=76.8, lon_max=77.3)


class TestVillageCandidates(unittest.TestCase):
    def test_places_frame_filters_bbox_and_unnamed(self):
        els = [dict(id=1, lat=31.7, lon=76.9, tags=dict(name="A", place="village")),
               dict(id=2, lat=33.0, lon=76.9, tags=dict(name="Outside", place="village")),
               dict(id=3, lat=31.6, lon=77.0, tags=dict(place="village")),
               dict(id=4, tags=dict(name="NoCoords"))]
        df = V.places_frame(els, BBOX)
        self.assertEqual(df.name.tolist(), ["A"])

    def test_link_to_atlas(self):
        # 4x4 grid of 0.1 deg cells starting at (lon 76.8, lat 32.0); left half link 0, right half link 1, one void
        labels = np.array([[0, 0, 1, 1]] * 4)
        labels[3, 3] = -1
        transform = (0.1, 0.0, 76.8, 0.0, -0.1, 32.0)
        catch = pd.DataFrame(dict(link_id=[0, 1], area_up_km2=[5.0, 50.0], tc_min=[10.0, 60.0]))
        df = pd.DataFrame(dict(name=["W", "E", "Void", "Far"], lat=[31.95, 31.95, 31.65, 31.0],
                               lon=[76.85, 77.15, 77.15, 76.85]))
        out = V.link_to_atlas(df, labels, transform, catch)
        self.assertEqual(out.link_id.tolist(), [0, 1, -1, -1])
        self.assertEqual(out.area_up_km2.tolist()[:2], [5.0, 50.0])
        self.assertTrue(np.isnan(out.area_up_km2.iloc[2]))

    def test_query_uses_bbox(self):
        q = V.overpass_query(BBOX)
        self.assertIn("(31.5,76.8,32.0,77.3)", q)


if __name__ == "__main__":
    unittest.main()
