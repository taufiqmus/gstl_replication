"""Jalankan demo.py pada klip contoh di samples/ dan (opsional) bandingkan skornya dengan nilai acuan (uji regresi visualisasi/pipeline).

  python tools/run_samples.py --detector <latest.pth>                      # jalankan semua, bandingkan dengan samples/expected_scores.csv
  python tools/run_samples.py --detector <latest.pth> --viz full --html    # pilih visualisasi (diteruskan ke demo.py)
  python tools/run_samples.py --detector <latest.pth> --only dota_pos_naik,ccd_neg
  python tools/run_samples.py --detector <latest.pth> --update-expected    # tulis ulang nilai acuan (setelah bobot/pipeline sengaja diubah)

Langkah: (1) verifikasi SHA-256 tiap klip terhadap samples/SHA256SUMS.txt; (2) jalankan demo.main per klip ke outputs/samples/<role>;
(3) bandingkan skor per frame dengan acuan (toleransi --tol, bawaan 5e-3; GPU/driver berbeda dapat memberi selisih kecil).
Kode keluar 1 bila ada klip yang hash-nya salah atau skornya menyimpang.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SAMPLES = os.path.join(ROOT, "samples")
EXPECTED = os.path.join(SAMPLES, "expected_scores.csv")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest():
    return list(csv.DictReader(open(os.path.join(SAMPLES, "MANIFEST.csv"), encoding="utf-8")))


def read_scores(path):
    return np.array([float(r["score"]) for r in csv.DictReader(open(path))])


def summary(sc, thr=0.5, fps=10.0):
    above = np.where(sc >= thr)[0]
    return "maks %.3f | rerata %.3f | ambang %.1f %s" % (sc.max(), sc.mean(), thr, ("terlampaui t=%.1fs" % (above[0] / fps)) if len(above) else "tidak terlampaui")


def main():
    ap = argparse.ArgumentParser()
    _loc = os.path.join(ROOT, "weights", "detector", "latest.pth")
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT") or (_loc if os.path.exists(_loc) else None))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "samples"))
    ap.add_argument("--only", default=None, help="daftar role dipisah koma")
    ap.add_argument("--viz", default="full")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--uncertainty", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-3, help="selisih mutlak maks skor per frame yang dianggap sama")
    ap.add_argument("--update-expected", action="store_true")
    a = ap.parse_args()
    if not a.detector or not os.path.exists(a.detector):
        raise SystemExit("bobot detektor tidak ditemukan: taruh di weights/detector/latest.pth, atau beri --detector / set DETECTOR_CKPT")
    rows = read_manifest()
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["role"] in keep]
        if not rows:
            raise SystemExit("tidak ada role yang cocok: %s" % a.only)
    bad = []
    for r in rows:
        p = os.path.join(SAMPLES, r["file"])
        if not os.path.exists(p) or sha256(p) != r["sha256"]:
            bad.append(r["file"])
    if bad:
        raise SystemExit("hash klip tidak cocok / berkas hilang: %s" % ", ".join(bad))
    print("hash %d klip cocok." % len(rows))

    import demo
    expected = {}
    if os.path.exists(EXPECTED) and not a.update_expected:
        for r in csv.DictReader(open(EXPECTED)):
            expected.setdefault(r["role"], []).append(float(r["score"]))
    new_expected, fails = [], []
    for r in rows:
        out = os.path.join(a.out, r["role"])
        argv = ["--video", os.path.join(SAMPLES, r["file"]), "--detector", a.detector, "--out", out, "--viz", a.viz, "--no-plot"]
        if a.html:
            argv.append("--html")
        if a.uncertainty:
            argv.append("--uncertainty")
        with contextlib.redirect_stdout(io.StringIO()):
            demo.main(argv)
        sc = read_scores(os.path.join(out, "scores.csv"))
        status = ""
        if a.update_expected:
            new_expected += [(r["role"], i, float(s)) for i, s in enumerate(sc)]
            status = "acuan ditulis"
        elif r["role"] in expected:
            ex = np.array(expected[r["role"]])
            if len(ex) != len(sc):
                status = "GAGAL (jumlah frame %d != acuan %d)" % (len(sc), len(ex))
                fails.append(r["role"])
            else:
                d = float(np.abs(sc - ex).max())
                ok = d <= a.tol
                status = "%s (selisih maks %.1e)" % ("LULUS" if ok else "GAGAL", d)
                if not ok:
                    fails.append(r["role"])
        else:
            status = "tanpa acuan"
        print("%-16s label %s | %s | %s" % (r["role"], r["label"], summary(sc), status), flush=True)
    if a.update_expected:
        # gabungkan dengan acuan lama: hanya role yang dijalankan yang diganti (aman bersama --only)
        merged = {}
        if os.path.exists(EXPECTED):
            for r in csv.DictReader(open(EXPECTED)):
                merged.setdefault(r["role"], []).append(float(r["score"]))
        fresh = {}
        for role, _, v in new_expected:
            fresh.setdefault(role, []).append(v)
        merged.update(fresh)
        man = read_manifest()
        order = [r["role"] for r in man if r["role"] in merged]
        lab = {r["role"]: r["label"] for r in man}
        with open(EXPECTED, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["role", "sample", "score"])
            for role in order:
                w.writerows((role, i, "%.6f" % v) for i, v in enumerate(merged[role]))
        with open(os.path.join(SAMPLES, "expected_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["role", "label", "skor_maks", "skor_rerata", "ambang_0.5_pertama_s"])
            for role in order:
                v = np.array(merged[role])
                ab = np.where(v >= 0.5)[0]
                w.writerow([role, lab[role], "%.3f" % v.max(), "%.3f" % v.mean(), ("%.1f" % (ab[0] / 10.0)) if len(ab) else "-"])
        print("acuan ditulis (%d role, %d diperbarui): %s" % (len(order), len(fresh), EXPECTED))
    print("\nkeluaran visual per klip di:", a.out)
    if fails:
        print("GAGAL pada:", ", ".join(fails))
        sys.exit(1)


if __name__ == "__main__":
    main()
