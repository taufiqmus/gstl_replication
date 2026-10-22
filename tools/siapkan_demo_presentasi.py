"""Hitung demo GSTL untuk banyak klip sekali jalan dan kumpulkan di satu folder siap-bawa untuk presentasi.

Dijalankan SEBELUM presentasi di mesin ber-GPU (atau Colab). Hasilnya (index.html + satu subfolder per klip) cukup
dibuka di peramban pada laptop mana pun, tanpa GPU, tanpa Python, tanpa internet.

  python tools/siapkan_demo_presentasi.py --detector weights/detector/latest.pth
  python tools/siapkan_demo_presentasi.py --videos a.mp4 b.mp4 --detector <latest.pth> --out outputs/demo_presentasi
  python tools/siapkan_demo_presentasi.py --only ccd_pos,dota_pos --max-windows 2 --detector <latest.pth>

Klip diambil dari --videos, atau semua video di --samples-dir (bawaan: samples/). Klip yang sudah punya scores.csv
dilewati (pakai --ulang untuk menghitung ulang). Folder keluaran bawaan outputs/demo_presentasi/ ada di .gitignore;
jangan di-commit: klip contoh berasal dari video YouTube berhak cipta.
"""
import argparse
import contextlib
import csv
import html
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

EKSTENSI = (".mp4", ".avi", ".mov", ".mkv")


def daftar_klip(a):
    if a.videos:
        paths = list(a.videos)
    else:
        paths = [os.path.join(a.samples_dir, f) for f in sorted(os.listdir(a.samples_dir)) if f.lower().endswith(EKSTENSI)]
    if a.only:
        kunci = [k.strip().lower() for k in a.only.split(",") if k.strip()]
        paths = [p for p in paths if any(k in os.path.basename(p).lower() for k in kunci)]
    if a.maks_klip:
        paths = paths[:a.maks_klip]
    if not paths:
        raise SystemExit("tidak ada klip yang dipilih")
    return paths


def label_manifest(samples_dir):
    """file -> (role, label, catatan) dari samples/MANIFEST.csv bila ada."""
    p = os.path.join(samples_dir, "MANIFEST.csv")
    if not os.path.exists(p):
        return {}
    return {r["file"]: (r.get("role", ""), r.get("label", ""), r.get("catatan", "")) for r in csv.DictReader(open(p, encoding="utf-8"))}


def ringkas(scores_csv, thr):
    rows = list(csv.DictReader(open(scores_csv)))
    sc = [float(r["score"]) for r in rows]
    t = [float(r["time_s"]) for r in rows]
    lewat = [ti for ti, s in zip(t, sc) if s >= thr]
    return {"frame": len(sc), "maks": max(sc), "rerata": sum(sc) / len(sc), "peringatan": lewat[0] if lewat else None}


def tulis_index(out, hasil, a, commit):
    baris = []
    for h in hasil:
        r = h.get("ringkas")
        if r is None:
            sel = '<td colspan="4" class="gagal">GAGAL: %s</td>' % html.escape(h["galat"])
        else:
            w = "t = %.1f s" % r["peringatan"] if r["peringatan"] is not None else "tidak terlampaui"
            sel = "<td>%d</td><td>%.3f</td><td>%.3f</td><td>%s</td>" % (r["frame"], r["maks"], r["rerata"], w)
        tautan = '<a href="%s/laporan.html">buka laporan</a>' % h["nama"] if h.get("ringkas") else ""
        label = {"1": "kecelakaan", "0": "normal"}.get(h["label"], h["label"] or "-")
        baris.append("<tr><td>%s</td><td>%s</td>%s<td>%s</td></tr>" % (html.escape(h["nama"]), html.escape(label), sel, tautan))
    isi = """<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demo GSTL</title><style>
body{font-family:system-ui,sans-serif;margin:24px auto;max-width:980px;padding:0 16px;color:#1b1b1f;background:#fff}
@media (prefers-color-scheme:dark){body{color:#e6e6ea;background:#14141a}a{color:#8ab4ff}th{background:#23232b}td,th{border-color:#34343e}}
table{border-collapse:collapse;width:100%%}td,th{border:1px solid #d6d6dd;padding:6px 10px;text-align:left}th{background:#f1f1f5}
.gagal{color:#c0392b}.cat{font-size:.9em;opacity:.8}</style></head><body>
<h1>Demo GSTL: risiko tabrakan dari video dashcam</h1>
<p class="cat">Detektor: %s | ambang peringatan %.2f | kode %s | dibuat %s. Skor bukan risiko terkalibrasi; model dilatih hanya pada CCD 1280x720,
jadi klip lain di luar distribusi latih. Skor tiap jendela 50 frame dihitung terpisah (GRU di-reset), sehingga skor rendah pada ~1 detik pertama tiap jendela adalah artefak.</p>
<table><tr><th>Klip</th><th>Label sebenarnya</th><th>Frame</th><th>Skor maks</th><th>Rerata</th><th>Peringatan pertama</th><th></th></tr>%s</table>
</body></html>""" % (html.escape(a.det_model), a.threshold, html.escape(commit), time.strftime("%Y-%m-%d %H:%M"), "\n".join(baris))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(isi)
    with open(os.path.join(out, "RINGKASAN.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["klip", "label", "frame", "skor_maks", "skor_rerata", "peringatan_pertama_s"])
        for h in hasil:
            r = h.get("ringkas")
            if r:
                w.writerow([h["nama"], h["label"], r["frame"], "%.4f" % r["maks"], "%.4f" % r["rerata"],
                            "" if r["peringatan"] is None else "%.1f" % r["peringatan"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    _loc = os.path.join(ROOT, "weights", "detector", "latest.pth")
    ap.add_argument("--videos", nargs="*", default=None, help="daftar video (bawaan: semua video di --samples-dir)")
    ap.add_argument("--samples-dir", default=os.path.join(ROOT, "samples"))
    ap.add_argument("--only", default=None, help="saring nama berkas: potongan nama dipisah koma (mis. ccd_pos,dota)")
    ap.add_argument("--maks-klip", type=int, default=0, help="batasi jumlah klip (0 = semua)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "demo_presentasi"))
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT") or (_loc if os.path.exists(_loc) else None))
    ap.add_argument("--det-model", default="cascade_rcnn", choices=("cascade_rcnn", "tv_fasterrcnn", "tv_ssdlite", "yolov8"))
    ap.add_argument("--device", default=None, help="cuda atau cpu (bawaan: cuda bila ada)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--viz", default="full")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--max-windows", type=int, default=0, help="batasi jendela 50 frame per klip (0 = semua)")
    ap.add_argument("--tanpa-ketidakpastian", action="store_true", help="matikan --uncertainty (lebih cepat)")
    ap.add_argument("--ulang", action="store_true", help="hitung ulang klip yang sudah punya scores.csv")
    a = ap.parse_args(argv)

    if a.det_model in ("cascade_rcnn", "yolov8") and (not a.detector or not os.path.exists(a.detector)):
        raise SystemExit("bobot detektor tidak ditemukan: beri --detector <latest.pth|yolov8s.pt> atau set DETECTOR_CKPT")
    klip = daftar_klip(a)
    manifest = label_manifest(a.samples_dir)
    os.makedirs(a.out, exist_ok=True)
    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
    except Exception:
        commit = "?"

    import demo  # impor di sini agar --help tidak memuat torch
    hasil = []
    mulai = time.time()
    for i, p in enumerate(klip, 1):
        base = os.path.basename(p)
        nama = os.path.splitext(base)[0]
        out = os.path.join(a.out, nama)
        role, label, _ = manifest.get(base, ("", "", ""))
        h = {"nama": nama, "label": label}
        sc_csv = os.path.join(out, "scores.csv")
        if os.path.exists(sc_csv) and not a.ulang:
            print("[%d/%d] %s: sudah ada, dilewati" % (i, len(klip), nama), flush=True)
        else:
            argv_demo = ["--video", p, "--out", out, "--viz", a.viz, "--html", "--threshold", str(a.threshold),
                         "--width", str(a.width), "--det-model", a.det_model, "--max-windows", str(a.max_windows)]
            if a.detector and a.det_model in ("cascade_rcnn", "yolov8"):
                argv_demo += ["--detector", a.detector]
            if a.device:
                argv_demo += ["--device", a.device]
            if not a.tanpa_ketidakpastian:
                argv_demo.append("--uncertainty")
            t0 = time.time()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    demo.main(argv_demo)
                print("[%d/%d] %s: selesai (%.0f dtk)" % (i, len(klip), nama, time.time() - t0), flush=True)
            except BaseException as e:  # SystemExit dari demo.py juga ditangkap agar klip lain tetap jalan
                h["galat"] = "%s: %s" % (type(e).__name__, e)
                print("[%d/%d] %s: GAGAL %s" % (i, len(klip), nama, h["galat"]), flush=True)
                hasil.append(h)
                continue
        h["ringkas"] = ringkas(sc_csv, a.threshold)
        hasil.append(h)

    # index memuat SEMUA klip yang sudah dihitung di folder keluaran (bukan hanya pilihan kali ini)
    sudah = {h['nama'] for h in hasil}
    label_dari_nama = {os.path.splitext(f)[0]: v[1] for f, v in manifest.items()}
    for nama in sorted(os.listdir(a.out)):
        sc_csv = os.path.join(a.out, nama, 'scores.csv')
        if nama not in sudah and os.path.exists(sc_csv):
            hasil.append({'nama': nama, 'label': label_dari_nama.get(nama, ''), 'ringkas': ringkas(sc_csv, a.threshold)})
    hasil.sort(key=lambda h: h['nama'])
    tulis_index(a.out, hasil, a, commit)
    ok = sum(1 for h in hasil if h.get("ringkas"))
    print("\nselesai: %d/%d klip, %.1f menit." % (ok, len(hasil), (time.time() - mulai) / 60))
    print("buka di peramban:", os.path.join(os.path.abspath(a.out), "index.html"))
    print("bawa SELURUH folder itu ke laptop presentasi (tautan relatif; tidak butuh internet).")
    return 0 if all(h.get("ringkas") for h in hasil) else 1


if __name__ == "__main__":
    sys.exit(main())
