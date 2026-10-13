"""Diagnosis struktural: mengapa geometri kotak tidak berpengaruh pada reimplementasi GSTL (tanpa pelatihan).

  python tools/diagnosis_struktur_gstl.py --ccd <compact CCD> --dota <compact DoTA> [--out results/diagnosis_struktur_gstl.json]

A. Graf: porsi massa bobot sisi pada pasangan kotak kosong (padding) vs pasangan objek nyata; frame dengan fallback seragam.
B. Identitas slot: apakah slot ke-k pada frame t dan t+1 adalah objek yang sama (IoU), dibanding pencocokan terbaik.
C. Probe sinyal geometri (regresi logistik, tanpa jaringan): statistik klip, fitur dinamika tanpa pelacakan, dan rerata fitur frame; dilatih di latih, diuji di uji, hiperparameter tetap.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from baseline_statistik_kotak import ap, auc, logreg, stats  # noqa: E402
from gstl.graph import edge_list, generate_st_graph  # noqa: E402

W, H = 1280.0, 720.0


def listing(root, name):
    if name == "dota":
        rows = list(csv.DictReader(open(os.path.join(root, "manifest.csv"), encoding="utf-8")))
        tr = [(r["rel"], int(r["label"])) for r in rows if r["split"] == "training"]
        te = [(r["rel"], int(r["label"])) for r in rows if r["split"] == "testing"]
    else:
        def rd(f):
            return [(l.split()[0], int(l.split()[1])) for l in open(os.path.join(root, f)) if l.strip()]
        tr, te = rd("train.txt"), rd("test.txt")
    return tr, te


def valid_unique(d):
    u = np.unique(d[:, :4], axis=0)
    return u[u[:, 2] > u[:, 0]]


def iou(a, b):
    ix = np.maximum(0, np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]))
    iy = np.maximum(0, np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def graf_diag(dets):
    idx = edge_list(19)
    out = dict(frac_fallback=[], massa_pad_pad=[], massa_campur=[], massa_real_real=[], n_real_real_bobot_pos=[], frac_frame_tanpa_pasangan_real=[])
    for det in dets:
        _, w = generate_st_graph(det)                      # T x E
        valid = det[..., 2] > det[..., 0]                  # T x N
        vi, vj = valid[:, idx[:, 0]], valid[:, idx[:, 1]]
        pad_pad, real_real = (~vi) & (~vj), vi & vj
        mix = ~(pad_pad | real_real)
        tot = w.sum(1) + 1e-30
        out["frac_fallback"].append(float(np.mean(np.all(np.isclose(w, 1.0), axis=1))))
        out["massa_pad_pad"].append(float(((w * pad_pad).sum(1) / tot).mean()))
        out["massa_campur"].append(float(((w * mix).sum(1) / tot).mean()))
        out["massa_real_real"].append(float(((w * real_real).sum(1) / tot).mean()))
        out["n_real_real_bobot_pos"].append(float(((w > 1e-6) & real_real).sum(1).mean()))
        out["frac_frame_tanpa_pasangan_real"].append(float((real_real.sum(1) == 0).mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def slot_diag(dets):
    same, best, nv = [], [], 0
    for det in dets:
        for t in range(det.shape[0] - 1):
            a, b = det[t], det[t + 1]
            va, vb = a[:, 2] > a[:, 0], b[:, 2] > b[:, 0]
            for k in np.where(va & vb)[0]:
                same.append(iou(a[k], b[k]))
            for k in np.where(va)[0]:
                cand = [iou(a[k], b[m]) for m in np.where(vb)[0]]
                best.append(max(cand) if cand else 0.0)
            nv += int(va.sum())
    same, best = np.array(same), np.array(best)
    return {"IoU_slot_sama_median": float(np.median(same)) if len(same) else None,
            "frac_slot_sama_IoU>0.5": float((same > 0.5).mean()) if len(same) else None,
            "frac_objek_ada_pencocokan_terbaik_IoU>0.5": float((best > 0.5).mean()),
            "frac_objek_slot_sama_tanpa_pelacakan": float(len(same) / max(nv, 1))}


def series(det):
    amax, hmax, ybot, ncen, acen, n = [], [], [], [], [], []
    for t in range(det.shape[0]):
        u = valid_unique(det[t])
        n.append(len(u))
        if len(u) == 0:
            amax.append(0); hmax.append(0); ybot.append(0); ncen.append(0); acen.append(0)
            continue
        a = (u[:, 2] - u[:, 0]) * (u[:, 3] - u[:, 1]) / (W * H)
        cx = (u[:, 0] + u[:, 2]) / 2 / W
        c = np.abs(cx - 0.5) < 0.2
        amax.append(a.max()); hmax.append(((u[:, 3] - u[:, 1]) / H).max()); ybot.append((u[:, 3] / H).max())
        ncen.append(int(c.sum())); acen.append(a[c].max() if c.any() else 0)
    return np.array([amax, hmax, ybot, ncen, acen, n], dtype=np.float64)


def dyn_feats(det):
    s = series(det)                                         # 6 x T
    t = np.arange(s.shape[1])
    f = []
    for v in s:
        sl = np.polyfit(t, v, 1)[0] if v.std() > 0 else 0.0
        f += [v.mean(), sl, v[-15:].mean() - v[:15].mean(), v.max()]
    return np.array(f + list(stats(det)))


def fit_eval(Xtr, ytr, Xte, yte, l2=1e-2, it=3000, lr=0.1):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w, b = logreg((Xtr - mu) / sd, ytr, l2=l2, it=it, lr=lr)
    s = ((Xte - mu) / sd) @ w + b
    return {"AP": ap(s, yte), "AUC": auc(s, yte)}


def probe(root, name, workers_note=""):
    tr, te = listing(root, name)
    def load(items, want_frame):
        D, F, Y = [], [], []
        for rel, lab in items:
            with np.load(os.path.join(root, rel)) as z:
                D.append(z["det"].astype(np.float32)); Y.append(lab)
                if want_frame:
                    F.append(z["frame"].astype(np.float32).mean(0))
        return D, (np.array(F) if want_frame else None), np.array(Y)
    Dtr, Ftr, ytr = load(tr, True)
    Dte, Fte, yte = load(te, True)
    s11tr, s11te = np.array([stats(d) for d in Dtr]), np.array([stats(d) for d in Dte])
    dtr, dte = np.array([dyn_feats(d) for d in Dtr]), np.array([dyn_feats(d) for d in Dte])
    res = {"n_train": len(ytr), "n_test": len(yte), "AP_acak": float(yte.mean()),
           "statistik_11": fit_eval(s11tr, ytr, s11te, yte),
           "geometri_dinamika_tanpa_pelacakan": fit_eval(dtr, ytr, dte, yte),
           "rerata_fitur_frame_VGG16": fit_eval(Ftr, ytr, Fte, yte, l2=1e-1, it=1500, lr=0.05)}
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Dte), size=min(300, len(Dte)), replace=False)
    res["graf"] = graf_diag([Dte[i] for i in sel])
    res["identitas_slot"] = slot_diag([Dte[i] for i in sel[:100]])
    return res


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ccd", required=True)
    ap_.add_argument("--dota", required=True)
    ap_.add_argument("--out", default=os.path.join(ROOT, "results", "diagnosis_struktur_gstl.json"))
    a = ap_.parse_args()
    out = {}
    for name, root in (("dota", a.dota), ("ccd", a.ccd)):
        out[name] = probe(root, name)
        print(name, json.dumps(out[name], indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
