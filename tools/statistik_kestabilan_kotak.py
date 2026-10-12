"""Seberapa 'stabil' kotak deteksi dari frame ke frame, dan apakah itu membedakan positif dari negatif?

  python tools/statistik_kestabilan_kotak.py --data-root <folder compact> --tag rilis

Per klip: untuk tiap pasangan frame berurutan, tiap kotak valid di frame t dicocokkan ke kotak terbaik di frame t+1 (IoU maks). Statistik:
  persist_iou      rerata IoU maks (kestabilan/persistensi kotak)
  tak_cocok_05     porsi kotak dengan IoU maks < 0,5 (kotak 'berkedip'/hilang)
  perubahan_n      rerata |jumlah kotak t+1 - jumlah kotak t|
  persist_iou_dibalik: sama tetapi untuk urutan dibalik (harus identik secara simetris; sanity)
Dilaporkan AUC univariat pada set uji (0,5 = tak membedakan; jauh dari 0,5 = isyarat). Keluaran: results/kestabilan_kotak_<tag>.json.
"""
import argparse
import json
import os

import numpy as np


def iou_matrix(A, B):
    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)


def valid_boxes(frame):
    u = np.unique(frame[:, :4], axis=0)
    return u[u[:, 2] > u[:, 0]]


def clip_stats(det):
    seq = [valid_boxes(det[t]) for t in range(det.shape[0])]
    per, miss, dn = [], [], []
    for t in range(len(seq) - 1):
        a, b = seq[t], seq[t + 1]
        dn.append(abs(len(b) - len(a)))
        if len(a) == 0 or len(b) == 0:
            continue
        m = iou_matrix(a, b).max(1)
        per.append(m.mean())
        miss.append((m < 0.5).mean())
    f = lambda v: float(np.mean(v)) if len(v) else 0.0  # noqa: E731
    return {"persist_iou": f(per), "tak_cocok_05": f(miss), "perubahan_n": f(dn)}


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    p, n = s[y == 1], s[y == 0]
    allv = np.concatenate([p, n])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tag", default="rilis")
    a = ap.parse_args()
    rows, y = [], []
    for line in open(os.path.join(a.data_root, "test.txt")):
        if line.strip():
            rel, lab = line.split()
            rows.append(clip_stats(np.load(os.path.join(a.data_root, rel))["det"]))
            y.append(int(lab))
    y = np.array(y)
    res = {"n": int(len(y)), "positif": int(y.sum()), "AUC_univariat": {k: auc([r[k] for r in rows], y) for k in rows[0]},
           "rerata_positif": {k: float(np.mean([r[k] for r, l in zip(rows, y) if l == 1])) for k in rows[0]},
           "rerata_negatif": {k: float(np.mean([r[k] for r, l in zip(rows, y) if l == 0])) for k in rows[0]}}
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "kestabilan_kotak_%s.json" % a.tag)
    json.dump(res, open(out, "w"), indent=1)
    print("kestabilan kotak (%s), AUC univariat:" % a.tag, {k: round(v, 3) for k, v in res["AUC_univariat"].items()})
    print("  rerata positif:", {k: round(v, 3) for k, v in res["rerata_positif"].items()})
    print("  rerata negatif:", {k: round(v, 3) for k, v in res["rerata_negatif"].items()})


if __name__ == "__main__":
    main()
