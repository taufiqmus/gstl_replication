"""Uji registri detektor (gstl/detectors.py) tanpa mengunduh bobot (pretrained=False) dan tanpa GPU."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from gstl import detectors as D  # noqa: E402
from gstl.detector import bbox_sampling  # noqa: E402


class TestMapping(unittest.TestCase):
    def test_coco_to_legacy_classes_and_order(self):
        boxes = [[0, 0, 10, 10], [5, 5, 50, 50], [1, 1, 9, 9], [0, 0, 5, 5], [2, 2, 8, 8]]
        scores = [0.4, 0.9, 0.5, 0.7, 0.3]
        labels = [3, 8, 1, 4, 99]            # car, truck, person, motorcycle, di luar pemetaan
        out = D.coco_to_legacy(boxes, scores, labels)
        self.assertEqual(len(out), 3)
        self.assertEqual([len(o) for o in out], [2, 1, 1])            # Car: car+truck; Pedestrian: person; Cyclist: motorcycle
        self.assertEqual(out[0].shape[1], 6)
        self.assertEqual(list(out[0][:, 4]), [0.9, 0.4])               # terurut skor menurun
        self.assertEqual(int(out[2][0, 5]), 4)                         # label COCO asli dipertahankan

    def test_empty(self):
        out = D.coco_to_legacy(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
        self.assertEqual([o.shape for o in out], [(0, 6)] * 3)

    def test_box_name(self):
        self.assertEqual(D.box_name(np.array([0, 0, 1, 1, .9, 4]), 2), "motorcycle")
        self.assertEqual(D.box_name(np.array([0, 0, 1, 1, .9]), 1), "Pedestrian")   # larik lama (5 kolom)


class TestCompatibility(unittest.TestCase):
    def test_bbox_sampling_accepts_six_columns(self):
        res = D.coco_to_legacy([[10, 10, 100, 100], [200, 50, 400, 300]], [0.9, 0.6], [3, 1])
        bb = bbox_sampling(res, nbox=19, imsize=(720, 1280), rng=np.random.RandomState(0))
        self.assertEqual(bb.shape, (19, 6))
        self.assertTrue((bb[:, 2] > bb[:, 0]).all() and (bb[:, 3] > bb[:, 1]).all())

    def test_bbox_sampling_with_no_detection_uses_full_frame(self):
        res = D.coco_to_legacy(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
        bb = bbox_sampling(res, nbox=19, imsize=(720, 1280), rng=np.random.RandomState(0))
        self.assertEqual(bb.shape, (19, 6))
        self.assertEqual(len(np.unique(bb[:, :4], axis=0)), 1)           # satu kotak cadangan seluas frame


class _Fake:
    name = "fake"

    def detect(self, img):
        return [np.array([[0, 0, 10, 10, 0.9], [0, 0, 5, 5, 0.01]]), np.zeros((0, 5)), np.array([[1, 1, 2, 2, 0.2]])]


class TestConfFilter(unittest.TestCase):
    def test_filter_removes_low_scores(self):
        out = D.ConfFilter(_Fake(), 0.5).detect(None)
        self.assertEqual([len(o) for o in out], [1, 0, 0])
        self.assertIn("conf0.50", D.ConfFilter(_Fake(), 0.5).name)

    def test_unknown_detector_and_missing_ckpt(self):
        with self.assertRaises(ValueError):
            D.build_detector("yolo_ajaib")
        with self.assertRaises(ValueError):
            D.build_detector("cascade_rcnn", "cpu", ckpt=None)


class TestTorchvision(unittest.TestCase):
    def test_contract_on_random_frame_cpu(self):
        try:
            import torchvision  # noqa: F401
        except ImportError:
            self.skipTest("torchvision tidak terpasang")
        for name in ("tv_ssdlite", "tv_fasterrcnn"):
            det = D.build_detector(name, "cpu", conf=0.0, pretrained=False)
            frame = (np.random.RandomState(0).rand(240, 320, 3) * 255).astype(np.uint8)
            out = det.detect(frame)
            self.assertEqual(len(out), 3, name)
            for o in out:
                self.assertEqual(o.ndim, 2, name)
                self.assertEqual(o.shape[1], 6, name)


if __name__ == "__main__":
    unittest.main()
