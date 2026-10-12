"""Uji varian GSTL untuk eksperimen geometri: mode kotak +cls/+score dan no_frame (geometri saja)."""
import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from helpers import N_OBJ, T, make_batch, random_detections  # noqa: E402
from gstl.gstl import BOX_MODES, box_dim, box_geometry, build_gstl, n_params_registered  # noqa: E402


def det_tensor(B=2, with_class=True, seed=0):
    rng = np.random.default_rng(seed)
    d = np.stack([random_detections(rng, T=T, n_empty=4) for _ in range(B)])
    d[..., 5] = rng.integers(0, 3, size=d[..., 5].shape) * (d[..., 2] > d[..., 0])
    d[..., 4] = rng.uniform(0, 1, size=d[..., 4].shape) * (d[..., 2] > d[..., 0])
    return torch.from_numpy(d.astype(np.float32))


class TestBoxModes(unittest.TestCase):
    def test_dims_and_old_modes_unchanged(self):
        det = det_tensor()
        for m in BOX_MODES:
            self.assertEqual(box_geometry(det, m).shape[-1], box_dim(m), m)
        a = box_geometry(det, "xyxy")
        b = box_geometry(det, "xyxywh")
        torch.testing.assert_close(a, b[..., :4])                                   # prefiks sama
        torch.testing.assert_close(b[..., 4], b[..., 2] - b[..., 0])                # lebar

    def test_cls_onehot_and_padding_zero(self):
        det = det_tensor()
        g = box_geometry(det, "xyxywh+cls+score")
        pad = (det[..., 2] - det[..., 0]) <= 0
        self.assertTrue(pad.any())
        self.assertEqual(float(g[pad].abs().sum()), 0.0)                            # kotak kosong -> semua fitur 0
        onehot = g[..., 6:9]
        self.assertTrue(torch.all(onehot[~pad].sum(-1) == 1))                       # tepat satu kelas
        torch.testing.assert_close(g[..., 9][~pad], det[..., 4][~pad])              # skor dibawa apa adanya


class TestVariants(unittest.TestCase):
    def _fwd(self, m, x, det, g, w, y, toa):
        m.eval()
        for lin in (m.predictor.l1, m.predictor.l2):
            with torch.no_grad():
                lin.weight_rho.fill_(-40.0)
                lin.bias_rho.fill_(-40.0)
        with torch.no_grad():
            _, o, _ = m(x, y, toa, g, edge_weights=w, det=det)
        return torch.stack([d["pred_mean"] for d in o])

    def test_no_frame_ignores_frame_feature_and_uses_boxes(self):
        rng = np.random.default_rng(0)
        x, y, toa, g, w = make_batch(rng, B=3, dim=64)
        det = det_tensor(3)
        m = build_gstl(T, 10.0, seed=0, x_dim=64, h_dim=16, z_dim=8, no_frame=True, box_mode="xyxywh+cls", box_embed=12)
        self.assertIsNone(m.phi_f)
        self.assertEqual(m.node_dim, 12)
        p1 = self._fwd(m, x, det, g, w, y, toa)
        p2 = self._fwd(m, torch.randn_like(x), det, g, w, y, toa)
        torch.testing.assert_close(p1, p2)                                           # fitur frame TIDAK dipakai
        det2 = det.clone()
        det2[..., :4] = det2[..., :4] * 0.5
        self.assertFalse(torch.allclose(p1, self._fwd(m, x, det2, g, w, y, toa)))    # geometri dipakai

    def test_default_variant_matches_previous_node_dim(self):
        m = build_gstl(50, 10.0, seed=0)
        self.assertEqual(m.node_dim, 4 + 256)
        self.assertAlmostEqual(n_params_registered(m) / 1e6, 1.443, delta=0.002)     # perilaku bawaan tidak berubah

    def test_m2_style_and_gradients(self):
        rng = np.random.default_rng(1)
        x, y, toa, g, w = make_batch(rng, B=3, dim=64)
        det = det_tensor(3)
        m = build_gstl(T, 10.0, seed=0, x_dim=64, h_dim=16, z_dim=8, frame_embed=8, box_mode="xyxywh+cls+score", box_embed=16)
        self.assertEqual(m.node_dim, 16 + 8)
        m.train()
        losses, _, _ = m(x, y, toa, g, edge_weights=w, det=det, nbatch=5)
        (losses["cross_entropy"] + losses["auxloss"]).backward()
        self.assertIsNotNone(m.psi_b[0].weight.grad)
        self.assertIsNotNone(m.phi_f[0].weight.grad)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            build_gstl(T, 10.0, box_mode="bola")


if __name__ == "__main__":
    unittest.main()
