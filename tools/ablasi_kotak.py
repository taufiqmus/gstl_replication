"""Ablasi masukan GSTL pada set uji CCD TANPA melatih ulang: seberapa bergantung bobot yang ada pada kotak (geometri) dan pada fitur frame?

  python tools/ablasi_kotak.py --data-root <CCD> [--seeds 3]

Bobot dan protokol evaluasi sama dengan evaluate.py (batch 10, drop_last, seed sampling BNN 123); hanya masukan yang diubah:
  asli               tanpa perubahan (acuan)
  tukar_kotak        kotak (det) tiap klip diganti dengan kotak klip lain (permutasi acak); fitur frame, label, toa tetap
  buang_25/50/75     tiap kotak unik dibuang dengan peluang p; bila habis -> satu kotak seluas frame; kekurangan diisi duplikat dari yang tersisa
  kotak_acak         19 kotak acak (posisi/ukuran seragam) per frame
  tanpa_geometri     19 kotak seluas frame (graf dan geometri tak informatif)
  tukar_fitur_frame  fitur frame diganti milik klip lain; kotak, label, toa tetap (pembanding: ketergantungan pada fitur frame)
Mode acak dijalankan --seeds kali (seed berbeda untuk acakan). Keluaran: results/ablasi_kotak.csv, docs/ablasi_kotak.md.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from collision_eval.metrics import average_precision_video, video_scores  # noqa: E402
from gstl.data import CrashDataset  # noqa: E402
from gstl.loader import load_gstl_model  # noqa: E402
from train import compute_metrics, configure_for_model, run_inference, set_seed  # noqa: E402

W, H = 1280, 720
FULL = np.array([0, 0, W - 1, H - 1, 1, 0], dtype=np.float64)


class _A:
    loss_alpha, loss_beta, loss_yita = 1e-3, 10.0, 10.0


def unique_rows(frame):
    _, idx = np.unique(frame[:, :4], axis=0, return_index=True)
    return frame[np.sort(idx)]


def fill19(rows, rng):
    rows = np.asarray(rows)
    if len(rows) == 0:
        rows = FULL[None]
    if len(rows) >= 19:
        return rows[:19]
    extra = rows[rng.randint(0, len(rows), 19 - len(rows))]
    return np.vstack([rows, extra])


def perturb_det(det, mode, p, rng):
    out = np.zeros_like(det, dtype=np.float64)
    for t in range(det.shape[0]):
        if mode == "buang":
            u = unique_rows(det[t])
            keep = u[rng.rand(len(u)) >= p]
            out[t] = fill19(keep, rng)
        elif mode == "tanpa_geometri":
            out[t] = np.tile(FULL, (19, 1))
        elif mode == "kotak_acak":
            x1 = rng.uniform(0, W * 0.9, 19)
            y1 = rng.uniform(0, H * 0.9, 19)
            w = rng.uniform(20, 300, 19)
            h = rng.uniform(20, 300, 19)
            out[t] = np.stack([x1, y1, np.minimum(x1 + w, W - 1), np.minimum(y1 + h, H - 1), np.zeros(19), np.zeros(19)], -1)
    return out


class AblatedCrash(CrashDataset):
    def __init__(self, root, mode="asli", p=0.0, seed=0):
        super().__init__(root, "test")
        self.mode, self.p, self.seed = mode, p, seed
        n = len(self.files_list)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        for i in np.where(perm == np.arange(n))[0]:   # tanpa titik tetap
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
        self.perm = perm

    def __getitem__(self, index):
        def load(i):
            with np.load(os.path.join(self.feat_dir, self.files_list[i])) as d:
                return d["data"], d["labels"], d["det"], str(d["ID"])
        feats, labels, det, vid = load(index)
        toa = self.toa_dict[vid] if labels[1] > 0 else self.n_frames + 1
        if self.mode == "tukar_kotak":
            det = load(self.perm[index])[2]
        elif self.mode == "tukar_fitur_frame":
            feats = load(self.perm[index])[0]
        elif self.mode in ("buang", "tanpa_geometri", "kotak_acak"):
            det = perturb_det(det, self.mode, self.p, np.random.RandomState(self.seed * 100003 + index))
        return self._pack(feats, labels, det, toa, vid)


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    allv = np.concatenate([pos, neg])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(model, root, mode, p, seed, device, workers):
    ds = AblatedCrash(root, mode, p, seed)
    configure_for_model("gstl", ds)
    loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=True, num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0)
    set_seed(123)
    res, _ = run_inference(model, loader, device, _A, eval_uncertain=False)
    m = compute_metrics(res, ds.fps)
    vs = video_scores(res["pred"], res["label"], res["toa"])
    m["AUC_skor_maks"] = auc(vs, res["label"])
    m["AP_skor_maks"] = float(average_precision_video(vs, res["label"]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "gstl_ccd_seed123_epoch10.pth"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default=None, help="daftar mode dipisah koma")
    a = ap.parse_args()
    if not a.data_root:
        raise SystemExit("beri --data-root atau set CCD_ROOT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_gstl_model(a.weights, device)
    plan = [("asli", "asli", 0.0, 1), ("tanpa_geometri", "tanpa_geometri", 0.0, 1), ("tukar_kotak", "tukar_kotak", 0.0, a.seeds),
            ("buang_25", "buang", 0.25, a.seeds), ("buang_50", "buang", 0.50, a.seeds), ("buang_75", "buang", 0.75, a.seeds),
            ("kotak_acak", "kotak_acak", 0.0, a.seeds), ("tukar_fitur_frame", "tukar_fitur_frame", 0.0, a.seeds)]
    if a.only:
        keep = set(a.only.split(","))
        plan = [x for x in plan if x[0] in keep]
    rows, t0 = [], time.time()
    for name, mode, p, n in plan:
        for s in range(n):
            m = run(model, a.data_root, mode, p, s, device, a.workers)
            rows.append(dict(mode=name, seed=s, **{k: m[k] for k in ("AP_orig", "mTTA_orig", "AP_video", "mTTA_sec", "AUC_skor_maks")}))
            print("%-18s seed %d | AP_orig %.4f AP_video %.4f AUC %.4f mTTA_sec %.2f (%.0f s)" % (
                name, s, m["AP_orig"], m["AP_video"], m["AUC_skor_maks"], m["mTTA_sec"], time.time() - t0), flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "ablasi_kotak.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = []
    for r in rows:
        if r["mode"] not in names:
            names.append(r["mode"])
    md = ["# Ablasi masukan GSTL pada set uji CCD (tanpa pelatihan ulang)\n",
          "Bobot `gstl_ccd_seed123_epoch10.pth`, protokol = `evaluate.py`. Rerata (± simpangan baku antar seed acakan; '-' bila satu seed). AUC memakai skor maks per video (positif: sebelum toa).\n",
          "| Mode | seed | AP asli (%) | AP video (%) | AUC | mTTA detik |", "|---|---|---|---|---|---|"]
    for n in names:
        g = [r for r in rows if r["mode"] == n]

        def f(k, sc=1.0, d=2):
            v = np.array([r[k] for r in g]) * sc
            return ("%.*f" % (d, v.mean())) + (" ± %.*f" % (d, v.std(ddof=1)) if len(v) > 1 else "")
        md.append("| %s | %d | %s | %s | %s | %s |" % (n, len(g), f("AP_orig", 100), f("AP_video", 100), f("AUC_skor_maks", 1, 3), f("mTTA_sec")))
    open(os.path.join(ROOT, "docs", "ablasi_kotak.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
