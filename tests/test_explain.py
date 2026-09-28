import json
import unittest

import numpy as np
import pandas as pd

from meghprahari import explain as E
from meghprahari.pipeline import FEATURES_V1

LOADED = dict(met_iwv=55.0, met_d_iwv_3h=3.0, met_kindex=38.0, met_totals=46.0, met_mfc_low=2e-7, met_omfi_low=1e-5,
              hem_rain_now=25.0, hem_rain_accel=8.0, hem_area_growth=0.1, hem_speed_kmh=5.0, hem_stall_h=1.5,
              hem_rain_max_1h=30.0, met_shear_low_500=8.0, ter_slope=0.6, ter_hand=40.0, ter_tc_min=20.0,
              ter_area_up_km2=45.0)


class TestIngredientCard(unittest.TestCase):
    def test_loaded_atmosphere(self):
        card = E.ingredient_card(pd.Series(LOADED)[FEATURES_V1])
        st = {i["name"]: i["status"] for i in card["ingredients"]}
        self.assertEqual(st, dict(moisture="strong", instability="strong", lift="strong", storm_signal="strong",
                                  terrain_response="strong"))
        text = " ".join(i["text"] for i in card["ingredients"])
        self.assertIn("55 mm of water vapour", text)
        self.assertIn("rose by 3 mm", text)
        self.assertIn("25 mm/h", text)
        self.assertIn("about 1.5 h", text)
        slow = {i["name"]: i["text"] for i in E.ingredient_card(dict(LOADED, hem_stall_h=61.4))["ingredients"]}
        self.assertIn("barely moving", slow["storm_signal"])
        self.assertNotIn("61", slow["storm_signal"])
        self.assertIn("20 min", text)
        self.assertIn("Strong signals", card["summary"])
        json.dumps(card)                                                  # must be storable as JSONB

    def test_weak_and_moderate(self):
        row = dict(LOADED, met_iwv=35.0, met_d_iwv_3h=0.0, met_kindex=31.0, met_totals=40.0, met_mfc_low=-1e-7,
                   met_omfi_low=-1e-6, hem_rain_now=0.5, hem_rain_accel=-1.0, hem_area_growth=0.0, ter_tc_min=120.0)
        st = {i["name"]: i["status"] for i in E.ingredient_card(row)["ingredients"]}
        self.assertEqual(st, dict(moisture="weak", instability="moderate", lift="weak", storm_signal="weak",
                                  terrain_response="weak"))
        self.assertEqual(E.ingredient_card(row)["summary"], "No ingredient is strong on its own.")
        items = {i["name"]: i for i in E.ingredient_card(dict(row, hem_rain_now=0.0, hem_rain_accel=0.5))["ingredients"]}
        self.assertEqual(items["storm_signal"]["text"], "Satellite shows little or no rain here yet, but rain nearby is building.")

    def test_cape_cin_over_high_terrain(self):
        row = dict(LOADED, met_kindex=np.nan, met_totals=np.nan, met_cape=1800.0, met_cin=120.0, met_d_cin_3h=-40.0)
        it = {i["name"]: i for i in E.ingredient_card(row)["ingredients"]}["instability"]
        self.assertEqual(it["status"], "moderate")
        self.assertIn("CAPE 1800 J/kg", it["text"])
        self.assertIn("lid", it["text"])
        self.assertIn("the lid is weakening", it["text"])

    def test_missing_inputs_are_unknown_not_invented(self):
        row = dict(LOADED, met_kindex=np.nan, met_totals=np.nan, met_omfi_low=np.nan)
        card = E.ingredient_card(row)
        items = {i["name"]: i for i in card["ingredients"]}
        self.assertEqual(items["instability"]["status"], "unknown")
        self.assertIn("850 hPa", items["instability"]["text"])
        self.assertEqual(items["lift"]["status"], "moderate")                # convergence only
        self.assertIn("upslope flow data missing", items["lift"]["text"])
        self.assertIn("Not available: instability", card["summary"])
        self.assertEqual(E.ingredient_card({})["summary"].count("Not available"), 1)
        self.assertTrue(all(i["status"] == "unknown" for i in E.ingredient_card({})["ingredients"]))


if __name__ == "__main__":
    unittest.main()
