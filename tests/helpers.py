"""Fungsi bantu uji (data acak kecil); disalin dari tests/test_equivalence.py repo asal."""
import numpy as np
import torch

from gstl import graph as G

D, N_OBJ, T = 64, 19, 6  # dimensi kecil agar cepat di CPU


def random_detections(rng, T=T, n=N_OBJ, scale=1280.0, n_empty=5):
    """bbox piksel acak; n_empty objek terakhir = bbox kosong (0) seperti padding deteksi."""
    x1 = rng.uniform(0, scale * 0.9, size=(T, n))
    y1 = rng.uniform(0, scale * 0.5, size=(T, n))
    w = rng.uniform(10, 200, size=(T, n))
    h = rng.uniform(10, 200, size=(T, n))
    det = np.stack([x1, y1, x1 + w, y1 + h, rng.uniform(0, 1, (T, n)), np.ones((T, n))], -1).astype(np.float32)
    if n_empty:
        det[:, -n_empty:, :] = 0
    return det


def make_batch(rng, B=3, T=T, dim=D, n_pos=2, detections_fn=random_detections):
    x = torch.from_numpy(rng.normal(size=(B, T, N_OBJ + 1, dim)).astype(np.float32))
    y = torch.zeros(B, 2)
    y[:n_pos, 1] = 1
    y[n_pos:, 0] = 1
    toa = torch.tensor([[float(rng.integers(2, T))] if i < n_pos else [T + 1.0] for i in range(B)])
    graphs, weights = [], []
    for _ in range(B):
        g, w = G.generate_st_graph(detections_fn(rng, T=T))
        graphs.append(torch.from_numpy(g))
        weights.append(torch.from_numpy(w))
    return x, y, toa, torch.stack(graphs), torch.stack(weights)
