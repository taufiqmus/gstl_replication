import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import siapkan_demo_presentasi as sdp  # noqa: E402


class TestDemoPresentasi(unittest.TestCase):
    def test_ringkas_dan_index(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "scores.csv")
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["window", "sample", "src_frame", "time_s", "score"])
                for i, s in enumerate([0.1, 0.2, 0.7, 0.9]):
                    w.writerow([0, i, i, i * 0.1, s])
            r = sdp.ringkas(p, 0.5)
            self.assertEqual(r["frame"], 4)
            self.assertAlmostEqual(r["maks"], 0.9)
            self.assertAlmostEqual(r["peringatan"], 0.2)
            self.assertIsNone(sdp.ringkas(p, 0.95)["peringatan"])

            class A:
                det_model, threshold = "cascade_rcnn", 0.5

            sdp.tulis_index(d, [{"nama": "a<b", "label": "1", "ringkas": r}, {"nama": "c", "label": "", "galat": "SystemExit: x"}], A, "abc1234")
            s = open(os.path.join(d, "index.html"), encoding="utf-8").read()
            self.assertIn("a&lt;b", s)
            self.assertIn("GAGAL", s)
            self.assertTrue(os.path.exists(os.path.join(d, "RINGKASAN.csv")))


if __name__ == "__main__":
    unittest.main()
