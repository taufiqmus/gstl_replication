"""Buat tabel hasil (paper vs repo ini) dari results/ccd_test_*.csv dan tulis ke README.md di antara penanda TABEL.

  python tools/make_results_table.py
Setiap results/ccd_test_<nama>.csv menjadi satu baris (rerata, dan +-sd bila baris > 1). Hasil dihitung dari berkas, bukan diketik tangan.
"""
import csv
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = json.load(open(os.path.join(ROOT, "results", "paper_reference.json"), encoding="utf-8"))


def fmt(v, d=2):
    return ("%." + str(d) + "f") % np.mean(v) + ((" ± %.*f" % (d, np.std(v, ddof=1))) if len(v) > 1 else "")


def main():
    lines = ["| hasil | n evaluasi | AP asli (%) | mTTA asli (s) | AP video eksak (%) | mTTA detik sungguhan |", "|---|---|---|---|---|---|",
             "| **Paper (Tabel 9, CCD)** | | %.1f | %.2f | | |" % (REF["ccd"]["AP"], REF["ccd"]["mTTA"])]
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "ccd_test_*.csv"))):
        name = os.path.basename(p)[len("ccd_test_"):-4]
        if name.startswith("all_epochs"):
            continue
        rows = list(csv.DictReader(open(p)))
        g = lambda k: [float(r[k]) for r in rows]  # noqa: E731
        lines.append("| %s | %d | %s | %s | %s | %s |" % (name, len(rows), fmt([100 * x for x in g("AP_orig")]), fmt(g("mTTA_orig")),
                                                          fmt([100 * x for x in g("AP_video")]), fmt(g("mTTA_sec"))))
    table = "\n".join(lines)
    open(os.path.join(ROOT, "results", "TABEL.md"), "w", encoding="utf-8").write(table + "\n")
    rp = os.path.join(ROOT, "README.md")
    s = open(rp, encoding="utf-8").read()
    a, b = "<!-- TABEL:BEGIN -->", "<!-- TABEL:END -->"
    if a in s and b in s:
        s = s[:s.index(a) + len(a)] + "\n" + table + "\n" + s[s.index(b):]
        open(rp, "w", encoding="utf-8").write(s)
    print(table)


if __name__ == "__main__":
    main()
