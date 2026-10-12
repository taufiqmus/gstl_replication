"""Uji GSTL (Mahmood 2023) di atas port UString."""
import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from helpers import N_OBJ, T, make_batch, random_detections  # noqa: E402
from gstl.gstl import box_geometry, build_gstl, n_params_registered  # noqa: E402


def det_batch(rng, B):
    return torch.from_numpy(np.stack([random_detections(rng, T=T, n_empty=0) for _ in range(B)]))


class TestGSTL(unittest.TestCase):
    def test_param_count_close_to_paper(self):
        """Konfigurasi bawaan (h=z=128, FC frame 256, Ψ 4-d mentah, SAA) ≈ 1,42 M (Mahmood Tabel 8)."""
        m = build_gstl(50, 10.0, seed=0)
        n = n_params_registered(m)
        self.assertAlmostEqual(n / 1e6, 1.443, delta=0.002)
        m2 = build_gstl(50, 10.0, seed=0, with_saa=False)
        self.assertAlmostEqual(n_params_registered(m2) / 1e6, 1.427, delta=0.002)

    def test_forward_backward_and_inputs_used(self):
        rng = np.random.default_rng(0)
        x, y, toa, g, w = make_batch(rng, B=3, dim=64)
        det = det_batch(rng, 3)
        m = build_gstl(T, 10.0, seed=0, x_dim=64, h_dim=16, z_dim=8, frame_embed=12)
        m.eval()
        for lin in (m.predictor.l1, m.predictor.l2):  # sigma ~0 → keluaran deterministik
            with torch.no_grad():
                lin.weight_rho.fill_(-40.0)
                lin.bias_rho.fill_(-40.0)
        with torch.no_grad():
            _, o1, _ = m(x, y, toa, g, edge_weights=w, det=det)
            x2 = x.clone()
            x2[:, :, 1:] = torch.randn_like(x2[:, :, 1:])          # fitur visual objek diubah
            _, o2, _ = m(x2, y, toa, g, edge_weights=w, det=det)
            det2 = det.clone()
            det2[..., :4] = det2[..., :4] * 0.5                      # geometri diubah
            _, o3, _ = m(x, y, toa, g, edge_weights=w, det=det2)
        p = lambda o: torch.stack([d["pred_mean"] for d in o])     # noqa: E731
        torch.testing.assert_close(p(o1), p(o2))                     # objek visual TIDAK dipakai
        self.assertFalse(torch.allclose(p(o1), p(o3)))               # geometri dipakai
        # hanya fitur frame juga cukup (dataset frame_only)
        with torch.no_grad():
            _, o4, _ = m(x[:, :, :1], y, toa, g, edge_weights=w, det=det)
        torch.testing.assert_close(p(o1), p(o4))
        m.train()
        losses, _, _ = m(x, y, toa, g, edge_weights=w, det=det, nbatch=5, eval_uncertain=True)
        self.assertEqual(float(losses["ranking"]), 0.0)              # pers. 4: tanpa ranking loss
        (losses["cross_entropy"] + losses["auxloss"]).backward()
        self.assertIsNotNone(m.phi_f[0].weight.grad)
        self.assertIsNone(m.rnn.weight_xz[0].weight.grad)           # GRU-GCN beku (setia ke dasar UString)

    def test_box_geometry(self):
        det = torch.tensor([[[[640.0, 360.0, 1280.0, 720.0, 0, 0]]]])
        torch.testing.assert_close(box_geometry(det), torch.tensor([[[[0.5, 0.5, 1.0, 1.0]]]]))
        self.assertEqual(box_geometry(det, "xyxywh").shape[-1], 6)


if __name__ == "__main__":
    unittest.main()
