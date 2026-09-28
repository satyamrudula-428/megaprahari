"""Task J2: every CAP message the alert engine drafts must validate against the official OASIS CAP 1.2 schema
(tests/fixtures/CAP-v1.2.xsd, downloaded from docs.oasis-open.org). Task J6: the actionable margin is in the message."""
import importlib.util
import os
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from meghprahari import alerts as A

HAS_XMLSCHEMA = importlib.util.find_spec("xmlschema") is not None
XSD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "CAP-v1.2.xsd")
IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2025, 7, 1, 0, 30, tzinfo=IST)
CFG = dict(walk_speed_kmh=3.0, detour_factor=1.5, dissemination_delay_min=15, lag_fraction=0.0, horizon_min=180,
           radius_km=3.0)
NS = {"c": A.CAP_NS}


def params(xml):
    root = ET.fromstring(xml.split("\n", 1)[1])
    return {p.find("c:valueName", NS).text: p.find("c:value", NS).text for p in root.findall("c:info/c:parameter", NS)}


def drafts():
    with_refuge = A.VillageCtx(1, "Siyanj & <Pandoh>", 31.6124, 77.0693, 1, 10, refuge_dist_m=400, refuge_bearing_deg=200,
                               tc_min=30)
    no_refuge = A.VillageCtx(2, "Thunag", 31.5586, 77.1666, 1, 10)
    for v in (with_refuge, no_refuge):
        for level in (1, 2, 3):
            for hazard in ("ts", "cb", "ff"):
                for shadow in (True, False):
                    yield A.draft_alert(v, level, 0.5, hazard, 120, 240, ["met_kindex high (90th percentile)"], CFG, NOW,
                                        shadow, "meghprahari.demo", "MeghPrahari Exercise")


@unittest.skipUnless(HAS_XMLSCHEMA, "xmlschema not installed (pip install -r requirements-dev.txt)")
class TestCapSchema(unittest.TestCase):
    def test_all_drafts_validate(self):
        import xmlschema
        schema = xmlschema.XMLSchema(XSD)
        n = 0
        for d in drafts():
            schema.validate(d["cap_xml"])                         # raises with the exact violation if invalid
            n += 1
        self.assertEqual(n, 36)

    def test_schema_rejects_bad_messages(self):
        import xmlschema
        schema = xmlschema.XMLSchema(XSD)
        good = next(drafts())["cap_xml"]
        self.assertTrue(schema.is_valid(good))
        self.assertFalse(schema.is_valid(good.replace("<status>Exercise</status>", "<status>Practice</status>")))
        self.assertFalse(schema.is_valid(good.replace("+05:30</sent>", "Z</sent>")))   # CAP forbids 'Z'


class TestActionableMargin(unittest.TestCase):
    def test_margin_in_message(self):
        d = next(drafts())                                        # lead 120 min, 400 m refuge -> walk 12 min
        self.assertAlmostEqual(d["margin_min"], 120 - 15 - 12)
        self.assertEqual(params(d["cap_xml"])["actionable_margin_min"], "93")
        self.assertIn("time to spare: about 93 min", d["cap_xml"])

    def test_unknown_and_negative_margin(self):
        v = A.VillageCtx(2, "Thunag", 31.5586, 77.1666, 1, 10)
        d = A.draft_alert(v, 2, 0.5, "ff", 0, 120, [], CFG, NOW, True, "s", "S")
        self.assertNotIn("actionable_margin_min", params(d["cap_xml"]))
        self.assertIn("unknown (no refuge mapped", d["cap_xml"])
        far = A.VillageCtx(3, "Bada", 31.6, 77.0, 1, 10, refuge_dist_m=1000, refuge_bearing_deg=0)
        d = A.draft_alert(far, 2, 0.5, "ff", 0, 120, [], CFG, NOW, True, "s", "S")   # 0 - 15 - 30 = -45
        self.assertEqual(params(d["cap_xml"])["actionable_margin_min"], "-45")
        self.assertIn("short by about 45 min", d["cap_xml"])


if __name__ == "__main__":
    unittest.main()
