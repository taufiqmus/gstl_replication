"""Ablasi graf GSTL pada set uji CCD TANPA melatih ulang: seberapa besar bobot sisi (dari kotak) memengaruhi skor?

  python tools/ablasi_graf.py --data-root <CCD> [--seeds 3] [--only asli,graf_seragam]

Bobot dan protokol sama dengan tools/ablasi_kotak.py. Topologi graf (171 sisi dari 19 slot) tidak pernah diubah; hanya bobot sisi
(dan, pada psi_tukar, geometri Psi) yang diganti. Fitur frame, label, toa tetap.
  asli             acuan (bobot exp(-jarak^2 piksel), runtuh)
  graf_seragam     semua sisi bobot sama (1/E)
  graf_norm        bobot exp(-d/s^2), d = jarak^2 antar pusat kotak dalam koordinat ternormalisasi (s = 0,25); pasangan yang memuat
                   kotak padding dibuang; bila < 2 kotak valid dalam frame -> seragam
  graf_bobot_acak  bobot asli diacak antar sisi dalam tiap frame (sebaran sama, pasangan salah)
  graf_tukar       bobot sisi dihitung dari kotak klip lain; Psi memakai kotak sendiri
  psi_tukar        Psi memakai kotak klip lain; bobot sisi dari kotak sendiri
  tukar_semua      Psi dan bobot sisi dari kotak klip lain (setara tukar_kotak; sanity check)
  psi_nol          (hanya lewat --only) Psi diganti nol di dalam model; bobot sisi asli
  psi_nol+graf_seragam  (hanya lewat --only) Psi nol DAN bobot sisi seragam: semua simpul identik, graf tanpa informasi
Keluaran: results/ablasi_graf.csv, docs/ablasi_graf.md.
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
from gstl.ablation import psi_nol  # noqa: E402
from gstl.graph import edge_list, generate_st_graph  # noqa: E402
from gstl.loader import load_gstl_model  # noqa: E402

W, H = 1280.0, 720.0
SIGMA2 = 0.25 ** 2
GRAF_MODES = ("graf_seragam", "graf_norm", "graf_bobot_acak", "graf_tukar", "psi_tukar", "tukar_semua")


def weights_uniform(n_frames, n_edges):
    return np.full((n_frames, n_edges), 1.0 / n_edges, dtype=np.float32)


def weights_norm(det, edges):
    """det: (T, N, >=4). Bobot exp(-d/SIGMA2) pada koordinat ternormalisasi, hanya pasangan kotak valid; (T, E) float32, jumlah 1."""
    box = np.asarray(det, dtype=np.float64)[..., :4]
    valid = (box[..., 2] - box[..., 0]) > 0
    cx = 0.5 * (box[..., 0] + box[..., 2]) / W
    cy = 0.5 * (box[..., 1] + box[..., 3]) / H
    i, j = edges[:, 0], edges[:, 1]
    d = (cx[:, i] - cx[:, j]) ** 2 + (cy[:, i] - cy[:, j]) ** 2
    w = np.exp(-d / SIGMA2) * (valid[:, i] & valid[:, j])
    s = w.sum(axis=1, keepdims=True)
    out = np.where(s > 0, w / np.where(s > 0, s, 1.0), 1.0 / len(edges))
    return out.astype(np.float32)


def weights_shuffled(w, rng):
    out = w.copy()
    for t in range(out.shape[0]):
        out[t] = out[t][rng.permutation(out.shape[1])]
    return out


class GrafCrash(AblatedCrash):
    def __getitem__(self, index):
        if self.mode not in GRAF_MODES:
            return super().__getitem__(index)

        def load(i):
            with np.load(os.path.join(self.feat_dir, self.files_list[i])) as d:
                return d["data"], d["labels"], d["det"], str(d["ID"])
        feats, labels, det, vid = load(index)
        toa = self.toa_dict[vid] if labels[1] > 0 else self.n_frames + 1
        other = load(int(self.perm[index]))[2]
        det_psi = other if self.mode in ("psi_tukar", "tukar_semua") else det
        out = list(self._pack(feats, labels, det_psi, toa, vid))
        edges = edge_list(det.shape[1])
        if self.mode == "graf_seragam":
            w = weights_uniform(det.shape[0], len(edges))
        elif self.mode == "graf_norm":
            w = weights_norm(det, edges)
        elif self.mode == "graf_bobot_acak":
            w = weights_shuffled(generate_st_graph(det)[1], np.random.RandomState(self.seed * 100003 + index))
        elif self.mode in ("graf_tukar", "tukar_semua"):
            w = generate_st_graph(other)[1]
        else:                                                           # psi_tukar: bobot dari kotak sendiri
            w = generate_st_graph(det)[1]
        out[3] = torch.from_numpy(w)
        return tuple(out)


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
    ablasi_kotak.AblatedCrash = GrafCrash
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_gstl_model(a.weights, device)
    s = a.seeds
    plan = [("asli", "asli", 0.0, 1), ("graf_seragam", "graf_seragam", 0.0, 1), ("graf_norm", "graf_norm", 0.0, 1),
            ("graf_bobot_acak", "graf_bobot_acak", 0.0, s), ("graf_tukar", "graf_tukar", 0.0, s),
            ("psi_tukar", "psi_tukar", 0.0, s), ("tukar_semua", "tukar_semua", 0.0, s),
            ("psi_nol", "asli", 0.0, 1), ("psi_nol+graf_seragam", "graf_seragam", 0.0, 1)]
    if a.only:
        keep = set(a.only.split(","))
        plan = [x for x in plan if x[0] in keep]
    else:
        plan = [x for x in plan if not x[0].startswith("psi_nol")]
    suffix = "_parsial" if a.only else ""   # --only tidak menimpa hasil lengkap
    rows, t0 = [], time.time()
    for name, mode, p, n in plan:
        for sd in range(n):
            if name.startswith("psi_nol"):
                with psi_nol(model):
                    m = run(model, a.data_root, mode, p, sd, device, a.workers)
            else:
                m = run(model, a.data_root, mode, p, sd, device, a.workers)
            rows.append(dict(mode=name, seed=sd, **{k: m[k] for k in ("AP_orig", "mTTA_orig", "AP_video", "mTTA_sec", "AUC_skor_maks")}))
            print("%-16s seed %d | AP_orig %.4f AP_video %.4f AUC %.4f mTTA_sec %.2f (%.0f s)" % (
                name, sd, m["AP_orig"], m["AP_video"], m["AUC_skor_maks"], m["mTTA_sec"], time.time() - t0), flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "ablasi_graf%s.csv" % suffix), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = list(dict.fromkeys(r["mode"] for r in rows))
    md = ["# Ablasi graf GSTL pada set uji CCD (tanpa pelatihan ulang)\n",
          "Bobot `gstl_ccd_seed123_epoch10.pth`, protokol = `tools/ablasi_kotak.py`. Topologi graf tidak diubah; hanya bobot sisi "
          "(dan Psi pada `psi_tukar`). Rerata (± simpangan baku antar seed acakan). AUC memakai skor maks per video.\n",
          "| Mode | seed | AP asli (%) | AP video (%) | AUC | mTTA detik |", "|---|---|---|---|---|---|"]
    for n in names:
        g = [r for r in rows if r["mode"] == n]

        def f(k, sc=1.0, d=2):
            v = np.array([r[k] for r in g]) * sc
            return ("%.*f" % (d, v.mean())) + (" ± %.*f" % (d, v.std(ddof=1)) if len(v) > 1 else "")
        md.append("| %s | %d | %s | %s | %s | %s |" % (n, len(g), f("AP_orig", 100), f("AP_video", 100), f("AUC_skor_maks", 1, 3), f("mTTA_sec")))
    open(os.path.join(ROOT, "docs", "ablasi_graf%s.md" % suffix), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
