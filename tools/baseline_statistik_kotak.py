"""Baseline 'jalan pintas kotak': klasifikasi klip hanya dari statistik ringkas kotak deteksi (tanpa urutan waktu/hubungan antar objek, tanpa fitur frame).

  python tools/baseline_statistik_kotak.py --data-root <folder compact> [--out results/baseline_statistik_kotak_<tag>.json] [--tag rilis]

Statistik per klip (dari kotak valid = lebar > 0): jumlah kotak unik, luas/tinggi rerata dan maks, posisi y bawah, sebaran x, porsi tiap kelas lama,
simpangan baku jumlah kotak antar frame, dan rerata skor deteksi (bila bermakna). Model: regresi logistik (numpy; L2) dilatih di train.txt dan
dievaluasi di test.txt (AP, AUC). Dipakai sebagai pembanding: varian GSTL berbasis geometri harus mengalahkan ini agar klaim 'geometri membawa sinyal
di luar statistik sederhana' sah.
"""
import argparse
import json
import os

import numpy as np


def stats(det):
    nu, area, h, yb, cx, hmax = [], [], [], [], [], []
    for t in range(det.shape[0]):
        u = np.unique(det[t][:, :4], axis=0)
        u = u[u[:, 2] > u[:, 0]]
        nu.append(len(u))
        if len(u):
            a = (u[:, 2] - u[:, 0]) * (u[:, 3] - u[:, 1]) / (1280 * 720)
            area += list(a)
            h += list((u[:, 3] - u[:, 1]) / 720)
            yb += list(u[:, 3] / 720)
            cx += list((u[:, 0] + u[:, 2]) / 2 / 1280)
            hmax.append(((u[:, 3] - u[:, 1]) / 720).max())
    valid = det[..., 2] > det[..., 0]
    cl = det[..., 5][valid]
    sc = det[..., 4][valid]
    f = lambda v: float(np.mean(v)) if len(v) else 0.0  # noqa: E731
    return [f(nu), float(np.std(nu)), f(area), f(h), f(hmax), f(yb), float(np.std(cx)) if len(cx) else 0.0,
            float((cl == 0).mean()) if len(cl) else 0.0, float((cl == 1).mean()) if len(cl) else 0.0, float((cl == 2).mean()) if len(cl) else 0.0, f(sc)]


NAMES = ["n_unik", "std_n", "luas", "tinggi", "tinggi_maks", "y_bawah", "std_x", "frac_car", "frac_ped", "frac_cyc", "skor"]


def load(root, phase):
    X, y = [], []
    for line in open(os.path.join(root, phase + ".txt")):
        if line.strip():
            rel, lab = line.split()
            X.append(stats(np.load(os.path.join(root, rel))["det"]))
            y.append(int(lab))
    return np.array(X), np.array(y)


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    allv = np.concatenate([p, n])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def ap(s, y):
    o = np.argsort(-s)
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / (np.arange(len(yy)) + 1)
    return float((prec * yy).sum() / yy.sum())


def logreg(X, y, l2=1e-2, it=3000, lr=0.1):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(it):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data-root", required=True)
    ap_.add_argument("--tag", default="rilis")
    a = ap_.parse_args()
    Xtr, ytr = load(a.data_root, "train")
    Xte, yte = load(a.data_root, "test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w, b = logreg((Xtr - mu) / sd, ytr)
    s = (Xte - mu) / sd @ w + b
    res = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "AP": ap(s, yte), "AUC": auc(s, yte), "AP_acak": float(yte.mean()),
           "bobot": {n: float(v) for n, v in zip(NAMES, w)}, "AUC_univariat": {n: auc(Xte[:, i], yte) for i, n in enumerate(NAMES)}}
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "baseline_statistik_kotak_%s.json" % a.tag)
    json.dump(res, open(out, "w"), indent=1)
    print("baseline statistik kotak (%s): AP %.3f | AUC %.3f | AP acak %.3f" % (a.tag, res["AP"], res["AUC"], res["AP_acak"]))
    print("AUC univariat:", {k: round(v, 3) for k, v in res["AUC_univariat"].items()})


if __name__ == "__main__":
    main()
