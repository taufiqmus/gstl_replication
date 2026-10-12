"""Seberapa baik/buruk kotak deteksi pada fitur CCD rilis (set uji)? Hanya membaca det di .npz (tanpa model/GPU).

  python tools/analisis_kotak.py --data-root <CCD>

Definisi (dari perilaku bbox_sampling UString): bila detektor tidak menemukan apa pun pada sebuah frame, diisi SATU kotak seluas frame
(0,0,W-1,H-1) lalu diduplikasi ke 19 slot. Kekurangan kotak lain diisi duplikat acak. Maka per frame:
  kotak unik  = jumlah baris det yang berbeda;
  tanpa deteksi = hanya satu kotak unik dan luasnya >= 90% frame (kotak cadangan seluas frame).
Dilaporkan per kelompok: positif/negatif, dan untuk positif: siang/malam, cuaca. Keluaran: results/analisis_kotak.json dan docs/analisis_kotak.md.
"""
import argparse
import collections
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1280.0, 720.0


def frame_stats(det):
    """det (T,19,6) -> (kotak unik per frame (T,), tanpa deteksi per frame (T,) bool)."""
    T = det.shape[0]
    uniq = np.zeros(T, int)
    nodet = np.zeros(T, bool)
    for t in range(T):
        u = np.unique(det[t, :, :4], axis=0)
        uniq[t] = len(u)
        if len(u) == 1:
            b = u[0]
            nodet[t] = (b[2] - b[0]) * (b[3] - b[1]) >= 0.9 * W * H
    return uniq, nodet


def load_meta(root):
    meta = {}
    for line in open(os.path.join(root, "videos", "Crash-1500.txt")):
        line = line.strip()
        if line:
            vid = line.split(",[")[0]
            rest = line.split("],")[1].split(",")
            meta[vid] = dict(light=rest[2], weather=rest[3])
    return meta


def summarize(rows):
    if not rows:
        return None
    u = np.array([r["uniq_mean"] for r in rows])
    nd = np.array([r["nodet_frac"] for r in rows])
    return dict(n_klip=len(rows), kotak_unik_per_frame_rerata=float(u.mean()), kotak_unik_per_frame_median_klip=float(np.median(u)),
                frame_tanpa_deteksi_rerata=float(nd.mean()), klip_semua_frame_tanpa_deteksi=int((nd >= 0.999).sum()),
                klip_lebih_dari_separuh_frame_tanpa_deteksi=int((nd > 0.5).sum()), klip_tanpa_deteksi_di_frame_mana_pun=int((nd == 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    a = ap.parse_args()
    if not a.data_root:
        raise SystemExit("beri --data-root atau set CCD_ROOT")
    meta = load_meta(a.data_root)
    lines = [x.split() for x in open(os.path.join(a.data_root, "vgg16_features", "test.txt")) if x.strip()]
    rows = []
    for rel, lab in lines:
        det = np.load(os.path.join(a.data_root, "vgg16_features", rel))["det"]
        u, nd = frame_stats(det)
        vid = os.path.basename(rel)[:-4]
        m = meta.get(vid, {}) if lab == "1" else {}
        rows.append(dict(id=vid, label=int(lab), uniq_mean=float(u.mean()), nodet_frac=float(nd.mean()), light=m.get("light"), weather=m.get("weather")))
    groups = {"positif": [r for r in rows if r["label"] == 1], "negatif": [r for r in rows if r["label"] == 0]}
    for lt in ("Day", "Night"):
        groups["positif_" + lt] = [r for r in rows if r["label"] == 1 and r["light"] == lt]
    for wt in ("Normal", "Snowy", "Rainy"):
        groups["positif_" + wt] = [r for r in rows if r["label"] == 1 and r["weather"] == wt]
    res = {k: summarize(v) for k, v in groups.items()}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"ringkas": res, "klip": rows}, open(os.path.join(ROOT, "results", "analisis_kotak.json"), "w"), indent=1)
    md = ["# Kualitas kotak deteksi pada fitur CCD rilis (set uji, 900 klip)\n",
          "Dibuat oleh `tools/analisis_kotak.py`. 'Tanpa deteksi' = frame yang hanya berisi kotak cadangan seluas frame (detektor tidak menemukan objek).\n",
          "| Kelompok | klip | kotak unik/frame (rerata) | frame tanpa deteksi (rerata) | klip: semua frame tanpa deteksi | klip: >50% frame tanpa deteksi |", "|---|---|---|---|---|---|"]
    for k, v in res.items():
        if v:
            md.append("| %s | %d | %.2f | %.1f%% | %d | %d |" % (k, v["n_klip"], v["kotak_unik_per_frame_rerata"], 100 * v["frame_tanpa_deteksi_rerata"],
                                                                   v["klip_semua_frame_tanpa_deteksi"], v["klip_lebih_dari_separuh_frame_tanpa_deteksi"]))
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "analisis_kotak.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
