"""Ablasi masukan GSTL pada tingkat dataset (tanpa pelatihan ulang): membungkus dataset yang punya metode raw(i) dan _pack(...).

Mode: asli | tukar_kotak | tukar_fitur_frame | acak_waktu | balik_waktu | acak_paruh | buang (peluang p) | kotak_acak | tanpa_geometri.
Ablasi tingkat model (psi_nol): context manager `psi_nol(model)` mengganti keluaran box_geometry dengan nol.
"""
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset

W, H = 1280, 720
FULL = np.array([0, 0, W - 1, H - 1, 1, 0], dtype=np.float32)


def unique_rows(frame):
    _, idx = np.unique(frame[:, :4], axis=0, return_index=True)
    return frame[np.sort(idx)]


def fill19(rows, rng):
    rows = np.asarray(rows)
    if len(rows) == 0:
        rows = FULL[None]
    if len(rows) >= 19:
        return rows[:19]
    return np.vstack([rows, rows[rng.randint(0, len(rows), 19 - len(rows))]])


def perturb_det(det, mode, p, rng):
    out = np.zeros(det.shape, dtype=np.float32)
    for t in range(det.shape[0]):
        if mode == "buang":
            u = unique_rows(det[t])
            valid = u[:, 2] > u[:, 0]
            u = u[valid] if valid.any() else u                      # abaikan kotak padding saat memilih
            out[t] = fill19(u[rng.rand(len(u)) >= p], rng)
        elif mode == "tanpa_geometri":
            out[t] = np.tile(FULL, (19, 1))
        elif mode == "kotak_acak":
            x1, y1 = rng.uniform(0, W * 0.9, 19), rng.uniform(0, H * 0.9, 19)
            w, h = rng.uniform(20, 300, 19), rng.uniform(20, 300, 19)
            out[t] = np.stack([x1, y1, np.minimum(x1 + w, W - 1), np.minimum(y1 + h, H - 1), np.zeros(19), np.zeros(19)], -1)
    return out


class AblatedDataset(Dataset):
    def __init__(self, base, mode="asli", p=0.0, seed=0):
        self.base, self.mode, self.p, self.seed = base, mode, p, seed
        n = len(base)
        perm = np.random.RandomState(seed).permutation(n)
        for i in np.where(perm == np.arange(n))[0]:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
        self.perm = perm

    def __len__(self):
        return len(self.base)

    def __getattr__(self, k):                     # fps, n_frames, dim_feature, n_obj, ... dari dataset dasar
        if k in ("base", "mode", "p", "seed", "perm"):
            raise AttributeError(k)
        return getattr(self.base, k)

    def __getitem__(self, index):
        feats, labels, det, vid, toa = self.base.raw(index)
        if self.mode == "tukar_kotak":
            det = self.base.raw(int(self.perm[index]))[2]
        elif self.mode == "tukar_fitur_frame":
            feats = self.base.raw(int(self.perm[index]))[0]
        elif self.mode == "acak_waktu":                  # urutan frame kotak diacak DALAM klip yang sama (isi per frame tak berubah)
            det = det[np.random.RandomState(self.seed * 100003 + index).permutation(det.shape[0])]
        elif self.mode == "balik_waktu":                 # urutan frame kotak dibalik (isi per frame tak berubah)
            det = det[::-1].copy()
        elif self.mode == "acak_paruh":                  # acak HANYA di dalam paruh awal dan paruh akhir (fase kasar klip dipertahankan)
            rng = np.random.RandomState(self.seed * 100003 + index)
            h = det.shape[0] // 2
            order = np.concatenate([rng.permutation(h), h + rng.permutation(det.shape[0] - h)])
            det = det[order]
        elif self.mode in ("buang", "tanpa_geometri", "kotak_acak"):
            det = perturb_det(det, self.mode, self.p, np.random.RandomState(self.seed * 100003 + index))
        return self.base._pack(feats, labels, det, toa, vid)


@contextlib.contextmanager
def psi_nol(model):
    """Ganti keluaran box_geometry di gstl.gstl dengan nol selama konteks (geometri dihapus di dalam model; graf tak berubah)."""
    from . import gstl as G
    orig = G.box_geometry
    G.box_geometry = lambda det, mode="xyxy": torch.zeros_like(orig(det, mode))
    try:
        yield
    finally:
        G.box_geometry = orig
