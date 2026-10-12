"""Gambar kotak pada klip CCD untuk menilai kualitas deteksi secara visual.

  python tools/plot_kotak_ccd.py --data-root <CCD> [--out outputs/plot_kotak]

Dua lembar (JPG):
  kotak_ccd_negatif.jpg : tiap baris satu klip normal CCD uji; kolom 1-2 = deteksi detektor kita (frame 10 dan 30),
                          kolom 3 = kotak yang DITERIMA MODEL dari .npz rilis (frame 30; untuk klip normal frame mp4 selaras dengan npz).
  kotak_ccd_positif.jpg : tiap baris satu klip kecelakaan; kolom = deteksi detektor kita pada frame 10, 30, 45.
                          (Fitur/kotak .npz rilis klip positif berasal dari frame sumber berbeda dari mp4 sehingga TIDAK digambar.)
Warna deteksi kita menurut skor: hijau tebal >= 0,5; kuning 0,1-0,5; merah tipis < 0,1 (ini yang membuat kotak "sampah" terlihat).
Kotak .npz rilis: satu warna per kelas (Car hijau, Pedestrian biru, Cyclist oranye); skor di .npz terpotong jadi bilangan bulat, jadi tak ada pita skor.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TW = 560
CLS_COL = ((0, 200, 0), (255, 150, 0), (0, 150, 255))
NEG_ROLES = ["ccd_neg_01", "ccd_neg_02", "ccd_neg_03", "ccd_neg_04"]
POS_ROLES = ["ccd_pos_malam_1", "ccd_pos_siang_salju_2", "ccd_pos_siang_ego", "ccd_pos_siang_nonego"]


def frames_of(path, need):
    cap = cv2.VideoCapture(path)
    out, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in need:
            out[i] = f
        i += 1
    cap.release()
    return out


def draw_det(frame, res):
    im = frame.copy()
    n = {"hi": 0, "mid": 0, "lo": 0}
    items = [(c[k], ci) for ci, c in enumerate(res) for k in range(len(c))]
    for b, ci in sorted(items, key=lambda t: t[0][4]):          # skor rendah dulu agar yang yakin di atas
        s = b[4]
        if s >= 0.5:
            col, th, key = (0, 220, 0), 5, "hi"
        elif s >= 0.1:
            col, th, key = (0, 220, 255), 3, "mid"
        else:
            col, th, key = (60, 60, 255), 1, "lo"
        n[key] += 1
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)
    return im, n


def tile(im, text):
    im = cv2.resize(im, (TW, int(im.shape[0] * TW / im.shape[1])))
    cv2.rectangle(im, (0, 0), (TW, 24), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "plot_kotak"))
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT") or os.path.join(ROOT, "weights", "detector", "latest.pth"))
    a = ap.parse_args()
    from gstl.detector import load_detector
    det = load_detector(a.detector, device="cuda")
    os.makedirs(a.out, exist_ok=True)
    man = {r["role"]: r for r in csv.DictReader(open(os.path.join(ROOT, "samples", "MANIFEST.csv"), encoding="utf-8"))}

    def sheet(roles, frames_idx, with_released, name):
        rows = []
        for role in roles:
            r = man[role]
            fr = frames_of(os.path.join(ROOT, "samples", r["file"]), set(frames_idx))
            tiles = []
            for t in frames_idx[:2] if with_released else frames_idx:
                im, n = draw_det(fr[t], det.detect(fr[t]))
                tiles.append(tile(im, "%s  f=%d | skor>=0,5: %d  0,1-0,5: %d  <0,1: %d" % (role, t, n["hi"], n["mid"], n["lo"])))
            if with_released:
                d = np.load(os.path.join(a.data_root, "vgg16_features", ("negative" if r["label"] == "0" else "positive"), r["nama_asli"][:-4] + ".npz"))["det"]
                t = frames_idx[1]
                im = fr[t].copy()
                u = np.unique(d[t], axis=0)
                for b in u:
                    cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), CLS_COL[int(b[5]) % 3], 3)
                tiles.append(tile(im, "kotak .npz rilis (diterima model) f=%d | kotak unik: %d" % (t, len(u))))
            rows.append(np.hstack(tiles))
        img = np.vstack(rows)
        cv2.imwrite(os.path.join(a.out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(name, img.shape, flush=True)

    sheet(NEG_ROLES, [10, 30], True, "kotak_ccd_negatif.jpg")
    sheet(POS_ROLES, [10, 30, 45], False, "kotak_ccd_positif.jpg")


if __name__ == "__main__":
    main()
