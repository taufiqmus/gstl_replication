import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, os.path.dirname(HERE))

from dosis_respons_kotak import NEW_MODES, perturb_new  # noqa: E402


def _det():
    d = np.zeros((4, 19, 6))
    d[:, :3] = [[100, 100, 200, 220, 0.9, 1], [300, 50, 340, 150, 0.6, 2], [900, 400, 1100, 600, 0.8, 0]]
    return d


class TestPerturb(unittest.TestCase):
    def test_padding_stays_zero_and_shape(self):
        for mode in ("jitter", "acak_kelas", "acak_skor", "acak_posisi"):
            out = perturb_new(_det(), mode, 20.0, np.random.RandomState(0))
            self.assertEqual(out.shape, (4, 19, 6))
            self.assertTrue(np.all(out[:, 3:] == 0), mode)

    def test_jitter_zero_is_identity_and_bounds(self):
        d = _det()
        self.assertTrue(np.array_equal(perturb_new(d, "jitter", 0.0, np.random.RandomState(0)), d))
        out = perturb_new(d, "jitter", 100.0, np.random.RandomState(1))
        self.assertTrue(out[..., [0, 2]].min() >= 0 and out[..., [0, 2]].max() <= 1279)
        self.assertTrue(out[..., [1, 3]].min() >= 0 and out[..., [1, 3]].max() <= 719)

    def test_posisi_keeps_size_class_score(self):
        d = _det()
        out = perturb_new(d, "acak_posisi", 0.0, np.random.RandomState(2))
        self.assertTrue(np.allclose(out[:, :3, 2] - out[:, :3, 0], d[:, :3, 2] - d[:, :3, 0]))
        self.assertTrue(np.array_equal(out[..., 4:], d[..., 4:]))

    def test_slot_permutation_preserves_rows(self):
        d = _det()
        for mode in ("perm_slot_frame", "perm_slot_klip"):
            out = perturb_new(d, mode, 0.0, np.random.RandomState(3))
            for t in range(4):
                self.assertEqual(sorted(map(tuple, out[t])), sorted(map(tuple, d[t])), mode)
        same = perturb_new(d, "perm_slot_klip", 0.0, np.random.RandomState(3))
        self.assertTrue(np.array_equal(same[0], same[3]))

    def test_modes_registered(self):
        self.assertEqual(len(NEW_MODES), 6)


if __name__ == "__main__":
    unittest.main()
