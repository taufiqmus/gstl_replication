"""Bandingkan dua detektor secara visual pada frame tertentu dari klip contoh.

  python tools/bandingkan_detektor.py [--a cascade_rcnn --a-conf 0.5] [--b tv_fasterrcnn --b-conf 0.3] [--out outputs/banding_detektor.jpg]

Kasus bawaan = kegagalan yang kita temukan: motor (DoTA dan CCD), malam, objek sangat dekat. Tiap baris = satu kasus; kiri detektor A, kanan detektor B.
Nama kelas + skor ditulis di atas kotak (nama COCO asli bila tersedia). Hanya deteksi dengan skor >= ambang masing-masing yang digambar.
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

CASES = [("dota_pos_naik", 36, "motor + pengendara"), ("ccd_pos_siang_nonego", 30, "motor sport"), ("ccd_neg_03", 10, "malam, mobil berlampu"),
         ("ccd_pos_siang_ego", 30, "mobil sangat dekat"), ("dota_pos_gerobak", 31, "gerobak motor"), ("ccd_pos_malam_hujan", 10, "malam + hujan")]
COLORS = ((0, 200, 0), (255, 150, 0), (0, 150, 255))


def frame_at(path, idx):
    cap = cv2.VideoCapture(path)
    for _ in range(idx + 1):
        ok, f = cap.read()
    cap.release()
    return f


def draw(frame, res, conf, title):
    from gstl.detectors import box_name
    im = frame.copy()
    n = 0
    for ci, cls in enumerate(res):
        for b in cls:
            if b[4] < conf:
                continue
            n += 1
            c = COLORS[ci % 3]
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 4)
            cv2.putText(im, "%s %.2f" % (box_name(b, ci), b[4]), (int(b[0]), max(int(b[1]) - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3, cv2.LINE_AA)
    im = cv2.resize(im, (800, 450))
    cv2.rectangle(im, (0, 0), (800, 30), (0, 0, 0), -1)
    cv2.putText(im, "%s | skor>=%.2f: %d kotak" % (title, conf, n), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="cascade_rcnn")
    ap.add_argument("--b", default="tv_fasterrcnn")
    ap.add_argument("--a-conf", type=float, default=0.5, help="ambang gambar untuk A")
    ap.add_argument("--b-conf", type=float, default=0.3, help="ambang gambar untuk B (juga ambang sumber)")
    ap.add_argument("--detector", default=os.path.join(ROOT, "weights", "detector", "latest.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "banding_detektor.jpg"))
    a = ap.parse_args()
    from gstl.detectors import build_detector
    A = build_detector(a.a, "cuda", a.detector)
    B = build_detector(a.b, "cuda", a.detector, a.b_conf if a.b != "cascade_rcnn" else None)
    man = {r["role"]: r for r in csv.DictReader(open(os.path.join(ROOT, "samples", "MANIFEST.csv"), encoding="utf-8"))}
    rows = []
    for role, idx, note in CASES:
        f = frame_at(os.path.join(ROOT, "samples", man[role]["file"]), idx)
        ia = draw(f, A.detect(f), a.a_conf, "%s | %s f=%d (%s)" % (a.a, role, idx, note))
        ib = draw(f, B.detect(f), a.b_conf, "%s" % a.b)
        rows.append(np.hstack([ia, ib]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("gambar:", a.out)


if __name__ == "__main__":
    main()
