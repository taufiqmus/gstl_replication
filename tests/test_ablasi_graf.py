import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, os.path.dirname(HERE))

from ablasi_graf import weights_norm, weights_shuffled, weights_uniform  # noqa: E402
from gstl.graph import edge_list  # noqa: E402


def _det(n_valid):
    d = np.zeros((3, 19, 6))
    for k in range(n_valid):
        d[:, k] = [100 + 200 * k, 100, 180 + 200 * k, 220, 1, 0]
    return d


class TestBobotGraf(unittest.TestCase):
    def setUp(self):
        self.e = edge_list(19)

    def test_uniform_sums_to_one(self):
        w = weights_uniform(3, len(self.e))
        self.assertEqual(w.shape, (3, 171))
        self.assertTrue(np.allclose(w.sum(1), 1.0, atol=1e-6))

    def test_norm_masks_padding_and_sums_to_one(self):
        w = weights_norm(_det(4), self.e)
        valid_pair = (self.e[:, 0] < 4) & (self.e[:, 1] < 4)
        self.assertTrue(np.allclose(w.sum(1), 1.0, atol=1e-5))
        self.assertTrue(np.all(w[:, ~valid_pair] == 0))
        self.assertTrue(np.all(w[:, valid_pair] > 0))

    def test_norm_falls_back_to_uniform_without_pairs(self):
        w = weights_norm(_det(1), self.e)
        self.assertTrue(np.allclose(w, 1.0 / len(self.e), atol=1e-7))

    def test_closer_pairs_weigh_more(self):
        w = weights_norm(_det(3), self.e)[0]
        idx = {tuple(p): k for k, p in enumerate(self.e)}
        self.assertGreater(w[idx[(0, 1)]], w[idx[(0, 2)]])

    def test_shuffle_preserves_multiset(self):
        w = weights_norm(_det(5), self.e)
        s = weights_shuffled(w, np.random.RandomState(0))
        self.assertTrue(np.allclose(np.sort(w, 1), np.sort(s, 1)))


if __name__ == "__main__":
    unittest.main()
