"""Validasi pipeline demo.py terhadap fitur CCD yang dirilis: untuk k video uji (k negatif + k positif, dipilih acak seed tetap),
bandingkan skor GSTL dari (a) fitur npz rilis dan (b) fitur yang diekstrak ulang dari mp4 oleh pipeline demo.

  python tools/validate_demo.py --data-root <CCD> --detector <latest.pth> [--k 6]

Catatan: untuk video POSITIF, fitur rilis diekstrak dari frame sumber yang berbeda dari mp4 rilis (temuan kami di repo kerja),
sehingga perbandingan per-frame pada positif tidak adil; angka positif hanya menunjukkan apakah demo tetap menaikkan skor.
Keluaran: results/demo_validasi.json dan docs/demo_validasi.md.
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from demo import DEFAULT_WEIGHTS, N_FRAMES, score_window  # noqa: E402
from gstl.data import crash_toa_dict  # noqa: E402
from gstl.detector import load_detector  # noqa: E402
from gstl.features import VGG16FC7, extract_frames_features  # noqa: E402
from gstl.loader import load_gstl_model  # noqa: E402


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(a.detector, device=str(dev))
    vgg = VGG16FC7().to(dev).eval()
    model, _ = load_gstl_model(DEFAULT_WEIGHTS, dev)
    toa = crash_toa_dict(os.path.join(a.data_root, "videos", "Crash-1500.txt"))
    lines = [x.split() for x in open(os.path.join(a.data_root, "vgg16_features", "test.txt")) if x.strip()]
    rng = random.Random(a.seed)
    neg = rng.sample([l for l in lines if l[1] == "0"], a.k)
    pos = rng.sample([l for l in lines if l[1] == "1"], a.k)
    rows = []
    for rel, lab in neg + pos:
        d = np.load(os.path.join(a.data_root, "vgg16_features", rel))
        vid = str(d["ID"])
        s_npz = score_window(model, d["data"], d["det"], dev, 123, (1280, 720))
        folder = "Crash-1500" if lab == "1" else "Normal"
        cap = cv2.VideoCapture(os.path.join(a.data_root, "videos", folder, vid + ".mp4"))
        frames = []
        while len(frames) < N_FRAMES:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        feat = extract_frames_features(frames, detector, vgg, n_frames=N_FRAMES, seed=123, device=str(dev))
        s_demo = score_window(model, feat["data"], feat["det"], dev, 123, (frames[0].shape[1], frames[0].shape[0]))
        t = toa[vid] if lab == "1" else N_FRAMES
        r = {"id": vid, "label": int(lab), "toa": int(t), "n_frames_mp4": len(frames),
             "max_npz": float(s_npz[:t].max()), "max_demo": float(s_demo[:t].max()),
             "corr_frame": float(np.corrcoef(s_npz, s_demo)[0, 1]) if s_npz.std() > 0 and s_demo.std() > 0 else None,
             "mean_abs_diff": float(np.abs(s_npz - s_demo).mean())}
        rows.append(r)
        print(r, flush=True)
    y = [r["label"] for r in rows]
    summ = {"k_per_kelas": a.k, "auc_npz": auc([r["max_npz"] for r in rows], y), "auc_demo": auc([r["max_demo"] for r in rows], y)}
    for name, lab in (("negatif", 0), ("positif", 1)):
        g = [r for r in rows if r["label"] == lab]
        summ[name] = {"mean_abs_diff": float(np.mean([r["mean_abs_diff"] for r in g])),
                      "corr_frame_median": float(np.median([r["corr_frame"] for r in g if r["corr_frame"] is not None])),
                      "max_npz_mean": float(np.mean([r["max_npz"] for r in g])), "max_demo_mean": float(np.mean([r["max_demo"] for r in g]))}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"ringkas": summ, "baris": rows}, open(os.path.join(ROOT, "results", "demo_validasi.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
