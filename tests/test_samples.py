"""Uji klip contoh (samples/): hash sesuai manifest, berkas terbaca, tiap klip 50 frame. Dilewati bila samples/ tidak ada atau opencv tidak terpasang."""
import csv
import hashlib
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(os.path.dirname(HERE), "samples")
MANIFEST = os.path.join(SAMPLES, "MANIFEST.csv")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@unittest.skipUnless(os.path.exists(MANIFEST), "samples/ tidak ada")
class TestSamples(unittest.TestCase):
    def setUp(self):
        self.rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))

    def test_manifest_complete_and_hashes_match(self):
        self.assertGreaterEqual(len(self.rows), 6)
        for r in self.rows:
            p = os.path.join(SAMPLES, r["file"])
            self.assertTrue(os.path.exists(p), r["file"])
            self.assertEqual(hashlib.sha256(open(p, "rb").read()).hexdigest(), r["sha256"], r["file"])
        self.assertEqual({r["label"] for r in self.rows}, {"0", "1"})   # ada positif dan negatif

    def test_expected_scores_cover_every_sample(self):
        exp = os.path.join(SAMPLES, "expected_scores.csv")
        if not os.path.exists(exp):
            self.skipTest("expected_scores.csv belum dibuat (jalankan tools/run_samples.py --update-expected)")
        roles = {}
        for r in csv.DictReader(open(exp)):
            roles.setdefault(r["role"], []).append(float(r["score"]))
        for r in self.rows:
            self.assertIn(r["role"], roles)
            self.assertEqual(len(roles[r["role"]]), 50)
            self.assertTrue(all(0.0 <= s <= 1.0 for s in roles[r["role"]]))

    @unittest.skipUnless(HAS_CV2, "opencv tidak terpasang")
    def test_videos_readable_50_frames_720p(self):
        for r in self.rows:
            cap = cv2.VideoCapture(os.path.join(SAMPLES, r["file"]))
            n, ok = 0, True
            while ok:
                ok, fr = cap.read()
                if ok:
                    n += 1
                    shape = fr.shape
            cap.release()
            self.assertEqual(n, 50, r["file"])
            self.assertEqual(shape[:2], (720, 1280), r["file"])


if __name__ == "__main__":
    unittest.main()
