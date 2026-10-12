"""Ubah fitur CCD rilis (.npz penuh, ±16 MB/klip) menjadi format RINGKAS untuk GSTL (±0,45 MB/klip): hanya fitur frame utuh (float16) dan kotak det.

  python tools/make_compact_ccd.py --data-root <CCD> --out <folder keluaran> [--workers 4]

Isi per klip (.npz): frame (50,4096) float16, det (50,19,6) float32, labels (2,), ID. Disalin pula: train.txt, test.txt, Crash-1500.txt.
Aman diulang (berkas yang sudah ada dilewati); tidak mengubah data sumber. float16 membulatkan fitur (selisih relatif ~1e-3).
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def convert(src, dst):
    if os.path.exists(dst):
        return 0
    with np.load(src) as d:
        frame = d["data"][:, 0, :].astype(np.float16)
        np.savez(dst + ".tmp.npz", frame=frame, det=d["det"].astype(np.float32), labels=d["labels"], ID=d["ID"])
    os.replace(dst + ".tmp.npz", dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    feat = os.path.join(a.data_root, "vgg16_features")
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for phase in ("train", "test"):
        shutil.copy2(os.path.join(feat, phase + ".txt"), os.path.join(a.out, phase + ".txt"))
        for line in open(os.path.join(feat, phase + ".txt")):
            if line.strip():
                rel = line.split()[0]
                os.makedirs(os.path.dirname(os.path.join(a.out, rel)), exist_ok=True)
                jobs.append((os.path.join(feat, rel), os.path.join(a.out, rel)))
    shutil.copy2(os.path.join(a.data_root, "videos", "Crash-1500.txt"), os.path.join(a.out, "Crash-1500.txt"))
    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(lambda j: convert(*j), jobs)):
            done += r
            if (k + 1) % 300 == 0 or k + 1 == len(jobs):
                print("[%d/%d] baru dikonversi %d (%.0f s)" % (k + 1, len(jobs), done, time.time() - t0), flush=True)
    print("selesai ->", a.out)


if __name__ == "__main__":
    main()
