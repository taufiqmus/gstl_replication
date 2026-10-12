"""Uji fungsi visualisasi demo.py (tanpa detektor/GPU). Dilewati bila opencv tidak terpasang."""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import demo  # noqa: E402


class TestParseViz(unittest.TestCase):
    def test_presets_and_lists(self):
        self.assertEqual(demo.parse_viz("minimal"), ("boxes", "bar"))
        self.assertEqual(demo.parse_viz("full"), demo.VIZ_ALL)
        self.assertEqual(demo.parse_viz("clean"), ())
        self.assertIsNone(demo.parse_viz("none"))
        self.assertEqual(demo.parse_viz("curve, glow"), ("curve", "glow"))

    def test_unknown_element_rejected(self):
        with self.assertRaises(ValueError):
            demo.parse_viz("curve,pelangi")


class TestRiskColor(unittest.TestCase):
    def test_endpoints_and_monotone_red(self):
        self.assertEqual(demo.risk_color(0.0, 0.5), (0, 200, 0))       # hijau
        self.assertEqual(demo.risk_color(0.5, 0.5), (0, 140, 255))     # oranye pada ambang
        self.assertEqual(demo.risk_color(5.0, 0.5), (0, 0, 255))       # merah (dijepit)
        reds = [demo.risk_color(s, 0.5)[2] for s in np.linspace(0, 1, 21)]
        self.assertEqual(reds, sorted(reds))                            # kanal merah tidak turun saat skor naik
        greens = [demo.risk_color(s, 0.5)[1] for s in np.linspace(0, 1, 21)]
        self.assertEqual(greens, sorted(greens, reverse=True))          # kanal hijau tidak naik


@unittest.skipUnless(HAS_CV2, "opencv tidak terpasang")
class TestCompose(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), np.uint8)
        self.boxes = [(100, 100, 300, 250, 0.9, 0), (500, 200, 560, 330, 0.8, 1)]
        self.hist = [0.01, 0.2, 0.6, 0.9]

    def test_shapes_for_each_element(self):
        for items, extra in (((), 0), (("boxes",), 0), (("bar",), 0), (("glow",), 0), (("alert",), 0), (("curve",), 150), (demo.VIZ_ALL, 150)):
            im = demo.compose_frame(self.frame, self.boxes, self.hist, 0.3, 0.5, items, 10.0, width=640, first_alert=0.2)
            self.assertEqual(im.shape, (360 + extra, 640, 3), items)
            self.assertEqual(im.dtype, np.uint8)

    def test_alert_only_above_threshold(self):
        below = demo.compose_frame(self.frame, [], [0.1], 0.0, 0.5, ("alert",), 10.0, width=640)
        above = demo.compose_frame(self.frame, [], [0.9], 0.0, 0.5, ("alert",), 10.0, width=640)
        self.assertEqual(int(below.sum()), 0)       # frame hitam tanpa banner tetap hitam
        self.assertGreater(int(above.sum()), 0)     # banner merah muncul

    def test_curve_panel_window_and_empty(self):
        p = demo.draw_curve_panel(640, 150, [0.1] * 400 + [0.9], 10.0, 0.5, curve_seconds=30.0)
        self.assertEqual(p.shape, (150, 640, 3))
        p1 = demo.draw_curve_panel(640, 150, [0.3], 10.0, 0.5, 30.0)   # satu titik tidak boleh error
        self.assertEqual(p1.shape, (150, 640, 3))


@unittest.skipUnless(HAS_CV2, "opencv tidak terpasang")
class TestFinalizeVideo(unittest.TestCase):
    def _make(self, d):
        import cv2
        p = os.path.join(d, "a.tmp.mp4")
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(5):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()
        return p

    def test_without_ffmpeg_keeps_file(self):
        import tempfile
        from unittest import mock
        d = tempfile.mkdtemp()
        tmp, final = self._make(d), os.path.join(d, "a.mp4")
        with mock.patch.object(demo.shutil, "which", return_value=None):
            self.assertFalse(demo.finalize_video(tmp, final))
        self.assertTrue(os.path.exists(final))
        self.assertFalse(os.path.exists(tmp))

    @unittest.skipUnless(__import__("shutil").which("ffmpeg"), "ffmpeg tidak ada")
    def test_with_ffmpeg_makes_h264(self):
        import subprocess
        import tempfile
        d = tempfile.mkdtemp()
        tmp, final = self._make(d), os.path.join(d, "a.mp4")
        self.assertTrue(demo.finalize_video(tmp, final))
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", final],
                               capture_output=True, text=True).stdout.strip()
        self.assertEqual(probe, "h264")


class TestHtml(unittest.TestCase):
    def test_html_contains_data_and_is_self_contained(self):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "r.html")
        demo.write_html(p, "uji.mp4", np.array([0.1, 0.8]), np.array([[0.1, 0.0], [0.2, 0.0]]), 10.0, 0.5, "overlay.mp4")
        s = open(p, encoding="utf-8").read()
        self.assertIn('src="overlay.mp4"', s)
        start = s.index("const D=") + len("const D=")
        data = json.loads(s[start:s.index(";const cv=")])
        self.assertEqual(data["s"], [0.1, 0.8])
        self.assertEqual(data["thr"], 0.5)
        self.assertTrue(data["video"])
        self.assertNotIn("http://", s.replace("http://www.w3.org", ""))   # tanpa sumber eksternal


if __name__ == "__main__":
    unittest.main()
