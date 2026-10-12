"""Buat dataset SINTETIS berformat CCD (sesuai asumsi DataLoader asli) untuk uji asap pipeline.

Bukan data sungguhan. Positif diberi sinyal yang menguat menjelang toa agar model bisa belajar
(menguji bahwa loss turun dan metrik bergerak). Ukuran per video: 50x20x4096 float32 ≈ 16 MB.

  python tools/make_synthetic_ccd.py --out <folder> --n-pos 20 --n-neg 40
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pos", type=int, default=20)
    ap.add_argument("--n-neg", type=int, default=40)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T, NB = 50, 19
    feat_dir = os.path.join(a.out, "vgg16_features")
    os.makedirs(os.path.join(feat_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, "negative"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "videos"), exist_ok=True)
    direction = rng.normal(size=a.dim).astype(np.float32)
    entries, anno = [], []
    for k in range(a.n_pos + a.n_neg):
        pos = k < a.n_pos
        vid = "%06d" % (k + 1)
        feats = np.abs(rng.normal(size=(T, NB + 1, a.dim))).astype(np.float32)
        det = np.zeros((T, NB, 6), dtype=np.float32)
        n_real = int(rng.integers(3, NB))
        x1 = rng.uniform(0, 1100, size=(T, n_real))
        y1 = rng.uniform(0, 600, size=(T, n_real))
        det[:, :n_real, 0], det[:, :n_real, 1] = x1, y1
        det[:, :n_real, 2], det[:, :n_real, 3] = x1 + 80, y1 + 60
        det[:, :n_real, 4], det[:, :n_real, 5] = 0.9, 1
        if pos:
            toa = int(rng.integers(30, 50))
            ramp = np.clip((np.arange(T) - (toa - 20)) / 20.0, 0, None)[:, None, None]
            feats += 0.5 * ramp * np.abs(direction)[None, None, :]
            lab = [0] * toa + [1] * (T - toa)
            anno.append("%s,[%s],%d,ytb_%s,day,normal,0" % (vid, ",".join(map(str, lab)), 0, vid))
            rel = "positive/%s.npz" % vid
            labels = np.array([0, 1])
        else:
            rel = "negative/%s.npz" % vid
            labels = np.array([1, 0])
        np.savez(os.path.join(feat_dir, rel), data=feats, labels=labels, det=det, ID=vid)
        entries.append((rel, int(pos)))
    idx = rng.permutation(len(entries))
    n_test = int(round(a.test_frac * len(entries)))
    with open(os.path.join(feat_dir, "test.txt"), "w") as f:
        for i in sorted(idx[:n_test]):
            f.write("%s %d\n" % entries[i])
    with open(os.path.join(feat_dir, "train.txt"), "w") as f:
        for i in sorted(idx[n_test:]):
            f.write("%s %d\n" % entries[i])
    with open(os.path.join(a.out, "videos", "Crash-1500.txt"), "w") as f:
        f.write("\n".join(anno) + "\n")
    print("selesai:", a.out, "train", len(entries) - n_test, "test", n_test)


if __name__ == "__main__":
    main()
