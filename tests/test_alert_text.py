import importlib.util
import unittest
import xml.etree.ElementTree as ET

from meghprahari import alert_text as T, alerts as A
from test_cap_schema import CFG, NOW, XSD

NS = {"c": A.CAP_NS}
V = A.VillageCtx(1, "Siyanj", 31.6124, 77.0693, 1, 10, refuge_dist_m=400, refuge_bearing_deg=200, tc_min=30)


class TestAlertText(unittest.TestCase):
    def test_bilingual_cap(self):
        d = A.draft_alert(V, 3, 0.7, "cb", 120, 240, ["met_cape high"], dict(CFG, languages=["en", "hi"]), NOW, True,
                          "meghprahari.demo", "MeghPrahari Exercise")
        root = ET.fromstring(d["cap_xml"].split("\n", 1)[1])
        infos = root.findall("c:info", NS)
        self.assertEqual([i.find("c:language", NS).text for i in infos], ["en-IN", "hi-IN"])
        hi = infos[1]
        self.assertEqual(hi.find("c:event", NS).text, "बादल फटने जैसी अत्यधिक वर्षा")
        self.assertIn("तुरंत कार्रवाई करें", hi.find("c:headline", NS).text)
        self.assertIn("दक्षिण दिशा", hi.find("c:instruction", NS).text)          # 200 degrees -> south sector
        self.assertEqual(T.direction("hi", 225), "दक्षिण-पश्चिम")
        self.assertIn("93 मिनट", hi.find("c:description", NS).text)             # same margin as the English text
        self.assertEqual(len(hi.findall("c:parameter", NS)), 0)                  # parameters only in the first info
        self.assertEqual(hi.find("c:area/c:circle", NS).text, infos[0].find("c:area/c:circle", NS).text)
        if importlib.util.find_spec("xmlschema"):
            import xmlschema
            xmlschema.XMLSchema(XSD).validate(d["cap_xml"])

    def test_english_default_and_unknown_language(self):
        d = A.draft_alert(V, 1, 0.2, "ts", 0, 120, [], CFG, NOW, True, "s", "S")
        self.assertEqual(d["cap_xml"].count("<info>"), 1)
        with self.assertRaises(ValueError):
            A.draft_alert(V, 1, 0.2, "ts", 0, 120, [], dict(CFG, languages=["te"]), NOW, True, "s", "S")
        with self.assertRaises(ValueError):
            A.draft_alert(V, 1, 0.2, "ts", 0, 120, [], dict(CFG, languages=[]), NOW, True, "s", "S")

    def test_hindi_without_refuge(self):
        t = T.render("hi", level=2, hazard="ff", village="Thunag", prob=0.4, lead_start_min=0, lead_end_min=120,
                     drivers=[], margin=None, walk=None, dissemination_min=15, refuge_dist_m=None, refuge_bearing_deg=None)
        self.assertIn("ज्ञात नहीं", t["description"])
        self.assertEqual(t["instruction"], "स्थानीय प्रशासन के निर्देशों का पालन करें।")


if __name__ == "__main__":
    unittest.main()
