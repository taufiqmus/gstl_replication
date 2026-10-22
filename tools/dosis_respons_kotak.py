"""Kurva dosis-respons masukan kotak GSTL pada set uji CCD TANPA melatih ulang: seberapa halus kotak boleh diganggu sebelum skor berubah?

  python tools/dosis_respons_kotak.py --data-root <CCD> [--seeds 3] [--only asli,jitter_20]

Bobot dan protokol evaluasi sama dengan tools/ablasi_kotak.py (yang dipakai kembali); hanya kotak valid (lebar > 0) yang diubah,
kotak padding (nol) tetap. Fitur frame, label, toa tidak berubah.
  asli               acuan
  jitter_5..100      noise Gaussian (sigma piksel) pada x1,y1,x2,y2 tiap kotak valid, dipotong ke bingkai 1280x720
  acak_kelas         kelas tiap kotak valid diundi seragam (0..2)
  acak_skor          skor deteksi tiap kotak valid diundi seragam (0..1)
  acak_posisi        ukuran, kelas, dan skor tetap; posisi diundi seragam di dalam bingkai
  perm_slot_frame    19 slot diacak per frame (independen antar frame)
  perm_slot_klip     19 slot diacak sekali per klip (urutan sama di semua frame)
  kotak_acak         pembanding dari ablasi_kotak (19 kotak acak per frame)
Mode acak dijalankan --seeds kali. Keluaran: results/dosis_respons_kotak.csv, docs/dosis_respons_kotak.md.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ablasi_kotak import AblatedCrash, run  # noqa: E402
from gstl.loader import load_gstl_model  # noqa: E402

W, H = 1280, 720
NEW_MODES = ("jitter", "acak_kelas", "acak_skor", "acak_posisi", "perm_slot_frame", "perm_slot_klip")


def perturb_new(det, mode, p, rng):
    """det: (T, N, 6) [x1,y1,x2,y2,skor,kelas]; kembalikan salinan float64 dengan gangguan hanya pada kotak valid."""
    out = np.array(det, dtype=np.float64)
    valid = (out[..., 2] - out[..., 0]) > 0                         # (T, N)
    if mode == "jitter":
        noise = rng.normal(0.0, p, out[..., :4].shape) * valid[..., None]
        xy = out[..., :4] + noise
        xy[..., [0, 2]] = np.clip(xy[..., [0, 2]], 0, W - 1)
        xy[..., [1, 3]] = np.clip(xy[..., [1, 3]], 0, H - 1)
        out[..., :4] = xy
    elif mode == "acak_kelas":
        out[..., 5] = np.where(valid, rng.randint(0, 3, valid.shape), out[..., 5])
    elif mode == "acak_skor":
        out[..., 4] = np.where(valid, rng.uniform(0, 1, valid.shape), out[..., 4])
    elif mode == "acak_posisi":
        w, h = out[..., 2] - out[..., 0], out[..., 3] - out[..., 1]
        x1 = rng.uniform(0, 1, valid.shape) * np.maximum(W - 1 - w, 0)
        y1 = rng.uniform(0, 1, valid.shape) * np.maximum(H - 1 - h, 0)
        new = np.stack([x1, y1, x1 + w, y1 + h], -1)
        out[..., :4] = np.where(valid[..., None], new, out[..., :4])
    elif mode == "perm_slot_frame":
        for t in range(out.shape[0]):
            out[t] = out[t][rng.permutation(out.shape[1])]
    elif mode == "perm_slot_klip":
        out = out[:, rng.permutation(out.shape[1])]
    else:
        raise ValueError(mode)
    return out


class DoseCrash(AblatedCrash):
    def __getitem__(self, index):
        if self.mode not in NEW_MODES:
            return super().__getitem__(index)
        with np.load(os.path.join(self.feat_dir, self.files_list[index])) as d:
            feats, labels, det, vid = d["data"], d["labels"], d["det"], str(d["ID"])
        toa = self.toa_dict[vid] if labels[1] > 0 else self.n_frames + 1
        det = perturb_new(det, self.mode, self.p, np.random.RandomState(self.seed * 100003 + index))
        return self._pack(feats, labels, det, toa, vid)


def main():
    import ablasi_kotak
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "gstl_ccd_seed123_epoch10.pth"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default=None, help="daftar nama mode dipisah koma")
    a = ap.parse_args()
    if not a.data_root:
        raise SystemExit("beri --data-root atau set CCD_ROOT")
    ablasi_kotak.AblatedCrash = DoseCrash                             # run() membangun dataset lewat nama ini
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_gstl_model(a.weights, device)
    s = a.seeds
    plan = [("asli", "asli", 0.0, 1)] + [("jitter_%d" % sg, "jitter", float(sg), s) for sg in (5, 10, 20, 50, 100)] + [
        ("acak_kelas", "acak_kelas", 0.0, s), ("acak_skor", "acak_skor", 0.0, s), ("acak_posisi", "acak_posisi", 0.0, s),
        ("perm_slot_frame", "perm_slot_frame", 0.0, s), ("perm_slot_klip", "perm_slot_klip", 0.0, s), ("kotak_acak", "kotak_acak", 0.0, s)]
    if a.only:
        keep = set(a.only.split(","))
        plan = [x for x in plan if x[0] in keep]
    rows, t0 = [], time.time()
    for name, mode, p, n in plan:
        for sd in range(n):
            m = run(model, a.data_root, mode, p, sd, device, a.workers)
            rows.append(dict(mode=name, seed=sd, **{k: m[k] for k in ("AP_orig", "mTTA_orig", "AP_video", "mTTA_sec", "AUC_skor_maks")}))
            print("%-16s seed %d | AP_orig %.4f AP_video %.4f AUC %.4f mTTA_sec %.2f (%.0f s)" % (
                name, sd, m["AP_orig"], m["AP_video"], m["AUC_skor_maks"], m["mTTA_sec"], time.time() - t0), flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "dosis_respons_kotak.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = list(dict.fromkeys(r["mode"] for r in rows))
    md = ["# Dosis-respons masukan kotak GSTL pada set uji CCD (tanpa pelatihan ulang)\n",
          "Bobot `gstl_ccd_seed123_epoch10.pth`, protokol = `tools/ablasi_kotak.py`. Rerata (± simpangan baku antar seed acakan). "
          "AUC memakai skor maks per video.\n",
          "| Mode | seed | AP asli (%) | AP video (%) | AUC | mTTA detik |", "|---|---|---|---|---|---|"]
    for n in names:
        g = [r for r in rows if r["mode"] == n]

        def f(k, sc=1.0, d=2):
            v = np.array([r[k] for r in g]) * sc
            return ("%.*f" % (d, v.mean())) + (" ± %.*f" % (d, v.std(ddof=1)) if len(v) > 1 else "")
        md.append("| %s | %d | %s | %s | %s | %s |" % (n, len(g), f("AP_orig", 100), f("AP_video", 100), f("AUC_skor_maks", 1, 3), f("mTTA_sec")))
    open(os.path.join(ROOT, "docs", "dosis_respons_kotak.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
