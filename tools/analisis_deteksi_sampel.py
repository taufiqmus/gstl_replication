"""Berapa banyak deteksi per frame pada klip contoh, menurut ambang skor? Menjawab: apa yang DILIHAT pengguna di overlay (kotak skor >= 0,5)
berbeda dari apa yang DITERIMA model (semua deteksi, ambang skor 1e-6, maks 100 per frame, lalu dipilih 19 kotak oleh bbox_sampling).

  python tools/analisis_deteksi_sampel.py [--detector <latest.pth>]

Per klip dan per frame dihitung jumlah deteksi dengan skor > 1e-6 (semua), >= 0,1, >= 0,3, >= 0,5, >= 0,8, dan jumlah yang dikirim ke model
(kotak unik setelah bbox_sampling). Keluaran: results/analisis_deteksi_sampel.json dan docs/analisis_deteksi_sampel.md.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
THR = (0.1, 0.3, 0.5, 0.8)


def main():
    ap = argparse.ArgumentParser()
    _loc = os.path.join(ROOT, "weights", "detector", "latest.pth")
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT") or (_loc if os.path.exists(_loc) else None))
    a = ap.parse_args()
    from gstl.detector import bbox_sampling, load_detector
    det = load_detector(a.detector, device="cuda")
    man = list(csv.DictReader(open(os.path.join(ROOT, "samples", "MANIFEST.csv"), encoding="utf-8")))
    out = []
    for r in man:
        cap = cv2.VideoCapture(os.path.join(ROOT, "samples", r["file"]))
        rng = np.random.RandomState(123)
        n_all, n_thr, n_model, zero05 = [], {t: [] for t in THR}, [], 0
        nf = 0
        while nf < 50:
            ok, fr = cap.read()
            if not ok:
                break
            res = det.detect(fr)
            sc = np.concatenate([c[:, 4] for c in res]) if len(res) else np.zeros(0)
            n_all.append(int((sc > 1e-6).sum()))
            for t in THR:
                n_thr[t].append(int((sc >= t).sum()))
            zero05 += int((sc >= 0.5).sum() == 0)
            bb = bbox_sampling(res, nbox=19, imsize=fr.shape[:2], rng=rng)
            n_model.append(len(np.unique(bb[:, :4], axis=0)))
            nf += 1
        cap.release()
        row = dict(role=r["role"], label=int(r["label"]), n_frames=nf, deteksi_semua=float(np.mean(n_all)),
                   **{"skor>=%.1f" % t: float(np.mean(n_thr[t])) for t in THR}, kotak_unik_ke_model=float(np.mean(n_model)),
                   frame_tanpa_kotak_skor05_persen=100.0 * zero05 / max(nf, 1))
        out.append(row)
        print("%-24s semua %5.1f | >=0,5 %4.1f | >=0,8 %4.1f | ke model %4.1f | frame tanpa kotak >=0,5: %3.0f%%" % (
            r["role"], row["deteksi_semua"], row["skor>=0.5"], row["skor>=0.8"], row["kotak_unik_ke_model"], row["frame_tanpa_kotak_skor05_persen"]), flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "analisis_deteksi_sampel.json"), "w"), indent=1)

    def avg(rows, k):
        return float(np.mean([x[k] for x in rows])) if rows else float("nan")
    groups = {"semua klip": out, "DoTA": [x for x in out if x["role"].startswith("dota")], "CCD": [x for x in out if x["role"].startswith("ccd")],
              "positif": [x for x in out if x["label"] == 1], "negatif": [x for x in out if x["label"] == 0]}
    md = ["# Deteksi per frame pada klip contoh (apa yang terlihat vs apa yang diterima model)\n",
          "Rerata per klip (50 frame). 'Semua' = deteksi dengan skor > 1e-6 (ambang yang dipakai pipeline); 'ke model' = kotak unik yang dikirim ke GSTL setelah `bbox_sampling` (maks 19).\n",
          "| Kelompok | deteksi semua | skor >= 0,1 | skor >= 0,3 | skor >= 0,5 | skor >= 0,8 | kotak unik ke model | frame tanpa kotak (skor >= 0,5) |", "|---|---|---|---|---|---|---|---|"]
    for g, rows in groups.items():
        md.append("| %s (n=%d) | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.0f%% |" % (g, len(rows), avg(rows, "deteksi_semua"), avg(rows, "skor>=0.1"),
                  avg(rows, "skor>=0.3"), avg(rows, "skor>=0.5"), avg(rows, "skor>=0.8"), avg(rows, "kotak_unik_ke_model"), avg(rows, "frame_tanpa_kotak_skor05_persen")))
    open(os.path.join(ROOT, "docs", "analisis_deteksi_sampel.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
