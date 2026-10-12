"""Demo GSTL pada video dashcam mentah: video -> deteksi objek -> fitur -> skor kemungkinan kecelakaan per frame.

  python demo.py --video contoh.mp4 --detector <latest.pth>
  python demo.py --video contoh.mp4 --detector <latest.pth> --viz full --uncertainty --html

Keluaran di folder --out: scores.csv, skor.png (grafik), overlay.mp4 (bergantung --viz), laporan.html (bila --html).

Cara kerja: video disampel ke ~10 fps (rentang frame per sampel = round(fps/10)), dipotong menjadi jendela 50 frame
(5 detik) yang diskor terpisah, seperti saat model dilatih pada klip CCD (50 frame, 10 fps).

Visualisasi (--viz, daftar dipisah koma atau preset):
  boxes  kotak deteksi (warna per kelas: Car hijau, Pedestrian biru, Cyclist oranye; ambang --box-conf)
  bar    bilah skor di atas video
  curve  grafik kemungkinan tabrakan yang bergerak di bawah video (jendela --curve-seconds)
  glow   bingkai berwarna sesuai skor (hijau -> oranye -> merah relatif terhadap --threshold)
  alert  banner "RISIKO TINGGI" saat skor >= --threshold, dan waktu peringatan pertama
  preset: minimal = boxes,bar (bawaan) | full = semua | clean = video tanpa tambahan | none = tanpa video

CATATAN TENTANG "KEMUNGKINAN TABRAKAN": GSTL hanya menghasilkan SATU skor per frame (probabilitas kelas "kecelakaan" dari softmax),
bukan risiko per objek. Skor itu belum terkalibrasi menjadi probabilitas kejadian sebenarnya.

KETERBATASAN (baca sebelum menafsirkan skor):
  * Detektor Cascade R-CNN dilatih di KITTI dengan 3 kelas (Car, Pedestrian, Cyclist), TANPA kelas motor; motor sering tidak terdeteksi.
  * Pipeline fitur ini adalah port PyTorch murni dari kode UString; mendekati tetapi tidak identik dengan fitur CCD rilis (lihat docs/demo_validasi.md).
  * Model dilatih pada CCD (video YouTube, 1280x720). Video lain (resolusi/rasio aspek/gaya berbeda, terutama malam hari) berada di luar
    distribusi latih; geometri kotak diskalakan ke kerangka 1280x720, rasio aspek yang berbeda akan terdistorsi.
  * Jendela 50 frame diskor TERPISAH (keadaan GRU di-reset): skor hampir selalu rendah pada ±1 detik pertama tiap jendela lalu naik, sehingga
    pada video panjang grafik turun mendadak tiap 5 detik. Itu artefak jendela, bukan perubahan risiko. "Peringatan pertama" pada
    ±0,5 sampai 1 detik awal sebaiknya tidak ditafsirkan sebagai deteksi kejadian.
  * Skor adalah keluaran model, bukan kalibrasi risiko nyata; jangan dipakai untuk keputusan keselamatan.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "gstl_ccd_seed123_epoch10.pth")
N_FRAMES, TARGET_FPS, REF_W, REF_H = 50, 10.0, 1280.0, 720.0
VIZ_ALL = ("boxes", "bar", "curve", "glow", "alert")
VIZ_PRESETS = {"minimal": ("boxes", "bar"), "full": VIZ_ALL, "clean": (), "none": None}
CLASS_COLORS = ((0, 200, 0), (255, 150, 0), (0, 150, 255))  # BGR: Car hijau, Pedestrian biru, Cyclist oranye


def parse_viz(text):
    """'full' / 'minimal' / 'clean' / 'none' atau daftar elemen dipisah koma. None = tanpa video; () = video tanpa tambahan."""
    t = text.strip().lower()
    if t in VIZ_PRESETS:
        return VIZ_PRESETS[t]
    items = tuple(x.strip() for x in t.split(",") if x.strip())
    bad = [x for x in items if x not in VIZ_ALL]
    if bad:
        raise ValueError("elemen --viz tidak dikenal: %s (pilihan: %s, atau preset %s)" % (bad, ", ".join(VIZ_ALL), ", ".join(VIZ_PRESETS)))
    return items


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT"),
                    help="bobot Cascade R-CNN (latest.pth dari Drive UString); bawaan: variabel DETECTOR_CKPT")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="checkpoint GSTL")
    ap.add_argument("--out", default=None, help="folder keluaran (bawaan: outputs/demo/<nama video>)")
    ap.add_argument("--threshold", type=float, default=0.5, help="ambang peringatan (untuk penanda, warna, dan ringkasan)")
    ap.add_argument("--viz", default="minimal", help="elemen overlay: boxes,bar,curve,glow,alert atau preset minimal|full|clean|none")
    ap.add_argument("--box-conf", type=float, default=0.5, help="ambang skor deteksi untuk menggambar kotak")
    ap.add_argument("--curve-seconds", type=float, default=30.0, help="lebar jendela waktu pada grafik bergerak (detik)")
    ap.add_argument("--width", type=int, default=960, help="lebar video keluaran (piksel)")
    ap.add_argument("--uncertainty", action="store_true", help="hitung dan tampilkan ketidakpastian model (aleatorik, epistemik) di grafik")
    ap.add_argument("--html", action="store_true", help="tulis laporan HTML (video tersinkron dengan grafik); butuh overlay.mp4")
    ap.add_argument("--no-plot", action="store_true", help="jangan tulis skor.png")
    ap.add_argument("--max-windows", type=int, default=0, help="batasi jumlah jendela 50 frame (0 = semua)")
    ap.add_argument("--no-video", action="store_true", help="sama dengan --viz none")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    a.viz_items = None if a.no_video else parse_viz(a.viz)
    return a


def sampled_windows(path, n_frames=N_FRAMES, target_fps=TARGET_FPS):
    """Baca video berurutan; hasilkan jendela berisi (indeks_frame_sumber, frame_bgr) dari frame yang disampel ~10 fps."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("video tidak dapat dibuka: %s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(fps / target_fps)))
    meta = {"fps": fps, "step": step, "sampled_fps": fps / step, "width": int(cap.get(3)), "height": int(cap.get(4)),
            "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    yield meta
    win, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            win.append((i, frame))
            if len(win) == n_frames:
                yield win
                win = []
        i += 1
    cap.release()
    if win:
        yield win


def score_window_full(model, feats, det, device, seed, size, uncertainty=False):
    """Skor per frame (n_frames,) untuk satu jendela, dan (bila uncertainty) ketidakpastian (n_frames, 2) = [aleatorik, epistemik].
    det diskalakan ke kerangka 1280x720 untuk geometri."""
    from gstl.graph import generate_st_graph
    from train import set_seed
    d = det.astype(np.float32).copy()
    w, h = size
    d[..., [0, 2]] *= REF_W / w
    d[..., [1, 3]] *= REF_H / h
    ge, ew = generate_st_graph(d)
    x = torch.from_numpy(np.ascontiguousarray(feats[:, :1], dtype=np.float32))[None].to(device)  # hanya fitur frame (GSTL)
    y = torch.tensor([[1.0, 0.0]], device=device)
    toa = torch.tensor([[N_FRAMES + 1.0]], device=device)
    set_seed(seed)
    with torch.no_grad():
        _, outs, _ = model(x, y, toa, torch.from_numpy(ge)[None].to(device), hidden_in=None,
                           edge_weights=torch.from_numpy(ew)[None].to(device), npass=10, det=torch.from_numpy(d)[None].to(device),
                           nbatch=1, testing=False, eval_uncertain=uncertainty)
        logits = torch.stack([o["pred_mean"] for o in outs], dim=1)
        scores = torch.softmax(logits, dim=-1)[0, :, 1].float().cpu().numpy()
        unc = None
        if uncertainty:
            alea = torch.stack([o["aleatoric"] for o in outs], dim=1)
            epis = torch.stack([o["epistemic"] for o in outs], dim=1)
            unc = torch.stack([alea[..., 0, 0] + alea[..., 1, 1], epis[..., 0, 0] + epis[..., 1, 1]], dim=-1)[0].float().cpu().numpy()
    return scores, unc


def score_window(model, feats, det, device, seed, size):
    """Hanya skor per frame (dipakai skrip validasi)."""
    return score_window_full(model, feats, det, device, seed, size, uncertainty=False)[0]


# ----------------------------------------------------------------------------- visualisasi
def risk_color(score, thr):
    """BGR: hijau (skor <= 0,5 thr) -> oranye (thr) -> merah (>= 1,5 thr); kontinu."""
    s = score / max(thr, 1e-6)
    stops = [(0.0, (0, 200, 0)), (0.5, (0, 200, 0)), (1.0, (0, 140, 255)), (1.5, (0, 0, 255))]
    s = min(max(s, 0.0), 1.5)
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if s <= b:
            u = 0.0 if b == a else (s - a) / (b - a)
            return tuple(int(round(ca[i] + u * (cb[i] - ca[i]))) for i in range(3))
    return stops[-1][1]


def draw_curve_panel(width, height, hist, fps, thr, curve_seconds):
    """Panel grafik (BGR) berisi kemungkinan tabrakan (skor) pada jendela waktu terakhir; titik terakhir = frame sekarang."""
    import cv2
    p = np.full((height, width, 3), 28, np.uint8)
    ml, mr, mt, mb = 46, 14, 22, 20
    x0, x1, y0, y1 = ml, width - mr, mt, height - mb
    n_win = max(2, int(curve_seconds * fps))
    h = hist[-n_win:]
    start = len(hist) - len(h)
    cv2.rectangle(p, (x0, y0), (x1, y1), (60, 60, 60), 1)
    for v in (0.0, 0.5, 1.0):
        y = int(y1 - v * (y1 - y0))
        cv2.putText(p, "%.1f" % v, (6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
        cv2.line(p, (x0, y), (x1, y), (50, 50, 50), 1)
    yt = int(y1 - min(thr, 1.0) * (y1 - y0))
    for xx in range(x0, x1, 10):
        cv2.line(p, (xx, yt), (min(xx + 5, x1), yt), (60, 60, 230), 1)
    cv2.putText(p, "ambang %.2f" % thr, (x1 - 96, max(yt - 4, y0 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 240), 1)
    span = max(n_win - 1, 1)
    pts = [(int(x0 + (k / span) * (x1 - x0)), int(y1 - float(v) * (y1 - y0))) for k, v in enumerate(h)]
    if len(pts) >= 2:
        poly = np.array([(pts[0][0], y1)] + pts + [(pts[-1][0], y1)], np.int32)
        fill = p.copy()
        cv2.fillPoly(fill, [poly], risk_color(h[-1], thr))
        p = cv2.addWeighted(fill, 0.25, p, 0.75, 0)
        cv2.polylines(p, [np.array(pts, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(p, pts[-1], 5, risk_color(h[-1], thr), -1, cv2.LINE_AA)
    cv2.putText(p, "kemungkinan tabrakan (skor model, belum terkalibrasi)", (ml, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    t_end = (start + len(h) - 1) / fps
    cv2.putText(p, "t=%.1fs" % t_end, (x1 - 58, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    return p


def compose_frame(frame, boxes, hist, t, thr, viz, fps, width=960, curve_seconds=30.0, first_alert=None):
    """Satu frame keluaran. boxes = daftar (x1,y1,x2,y2,skor,kelas). hist = semua skor sampai frame ini (termasuk frame ini)."""
    import cv2
    score = float(hist[-1])
    im = frame.copy()
    if "boxes" in viz:
        for b in boxes:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), CLASS_COLORS[int(b[5]) % 3], 3)
    s = width / im.shape[1]
    im = cv2.resize(im, (width, int(im.shape[0] * s)))
    col = risk_color(score, thr)
    if "glow" in viz:
        ov = im.copy()
        cv2.rectangle(ov, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), col, int(4 + 22 * min(score, 1.0)))
        im = cv2.addWeighted(ov, 0.55, im, 0.45, 0)
    if "bar" in viz:
        cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
        cv2.rectangle(im, (0, 0), (int(im.shape[1] * min(score, 1.0)), 34), col, -1)
        cv2.putText(im, "t=%.1fs  skor=%.2f" % (t, score), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if "alert" in viz:
        if score >= thr:
            h0 = im.shape[0]
            ov = im.copy()
            cv2.rectangle(ov, (0, h0 - 52), (im.shape[1], h0), (0, 0, 200), -1)
            im = cv2.addWeighted(ov, 0.7, im, 0.3, 0)
            cv2.putText(im, "RISIKO TINGGI  skor %.2f" % score, (12, h0 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        if first_alert is not None:
            txt = "peringatan pertama: t=%.1fs" % first_alert
            cv2.putText(im, txt, (im.shape[1] - 330, 62 if "bar" in viz else 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if "curve" in viz:
        im = np.vstack([im, draw_curve_panel(im.shape[1], 150, hist, fps, thr, curve_seconds)])
    return im


def finalize_video(tmp_path, final_path):
    """OpenCV hanya menulis MPEG-4 Part 2 ('mp4v'), yang tidak diputar oleh sebagian besar peramban. Bila ffmpeg ada di PATH,
    ubah ke H.264 (yuv420p, faststart); bila tidak, simpan apa adanya. Mengembalikan True bila hasilnya H.264."""
    ff = shutil.which("ffmpeg")
    if ff:
        r = subprocess.run([ff, "-loglevel", "error", "-y", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                            "-movflags", "+faststart", final_path], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            os.remove(tmp_path)
            return True
        print("PERINGATAN: ffmpeg gagal mengubah ke H.264 (%s); video disimpan sebagai mp4v." % r.stderr.strip()[:120])
    if os.path.exists(final_path):
        os.remove(final_path)
    os.replace(tmp_path, final_path)
    return False


def write_plot(path, scores, unc, fps, thr, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc = np.asarray(scores)
    tt = np.arange(len(sc)) / fps
    rows = 2 if unc is not None else 1
    fig, axs = plt.subplots(rows, 1, figsize=(10, 3.2 if rows == 1 else 5.2), sharex=True, squeeze=False,
                            gridspec_kw={"height_ratios": [3, 1.6][:rows]})
    ax = axs[0, 0]
    ax.plot(tt, sc, lw=1.6, color="#1f77b4")
    ax.fill_between(tt, 0, sc, where=sc >= thr, color="#d62728", alpha=0.25, label="skor >= ambang")
    ax.axhline(thr, color="#d62728", ls="--", lw=1, label="ambang %.2f" % thr)
    above = np.where(sc >= thr)[0]
    if len(above):
        ax.axvline(above[0] / fps, color="#d62728", lw=1)
        ax.annotate("peringatan pertama %.1fs" % (above[0] / fps), (above[0] / fps, 0.97), xytext=(4, -2), textcoords="offset points",
                    fontsize=8, va="top", color="#d62728")
    for k in range(N_FRAMES, len(sc), N_FRAMES):
        ax.axvline(k / fps, color="0.85", lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("kemungkinan tabrakan\n(skor GSTL)")
    ax.set_title(title + "  (garis abu-abu = batas jendela 50 frame; skor belum terkalibrasi)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    if unc is not None:
        ax2 = axs[1, 0]
        ax2.plot(tt, unc[:, 0], lw=1.2, label="aleatorik (data)", color="#2ca02c")
        ax2.plot(tt, unc[:, 1], lw=1.2, label="epistemik (model)", color="#9467bd")
        ax2.set_ylabel("ketidakpastian")
        ax2.legend(loc="upper right", fontsize=8)
    axs[-1, 0].set_xlabel("waktu (detik)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


HTML_TEMPLATE = """<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laporan demo GSTL</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#d8d8d2;--acc:#1f6feb;--bad:#d1242f}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ececea;--mut:#9a9a94;--card:#1f1f1c;--line:#34342f;--acc:#58a6ff;--bad:#ff6b6b}}
*{box-sizing:border-box}body{margin:0;padding:20px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}p.sub{margin:0 0 16px;color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:14px}
video{width:100%;border-radius:6px;background:#000}canvas{width:100%;height:auto;display:block;cursor:pointer}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;color:var(--mut);font-size:14px}.row b{color:var(--fg)}
.note{font-size:13px;color:var(--mut)}input[type=range]{width:180px}
</style></head><body><main>
<h1 id="judul"></h1><p class="sub">Skor GSTL per frame (model dilatih hanya di CCD). Klik grafik untuk melompat ke waktu itu.</p>
<div class="card" id="vcard"><video id="v" controls preload="metadata" src="__VIDEO__"></video></div>
<div class="card"><canvas id="c" width="1200" height="260"></canvas>
<div class="row" style="margin-top:8px"><span>waktu <b id="tt">0.0</b> s</span><span>skor <b id="ss">-</b></span>
<span>ambang <b id="th">-</b><input id="rg" type="range" min="0.05" max="0.95" step="0.01"></span>
<span id="al"></span></div></div>
<p class="note">Catatan: ini skor model (probabilitas kelas &ldquo;kecelakaan&rdquo; per frame), belum terkalibrasi menjadi probabilitas kejadian sebenarnya,
dan bukan risiko per objek. Detektor tidak mengenal motor; video di luar CCD (terutama malam hari) di luar distribusi latih.</p>
</main><script>
const D=__DATA__;const cv=document.getElementById('c'),g=cv.getContext('2d'),v=document.getElementById('v');
document.getElementById('judul').textContent=D.title;if(!D.video)document.getElementById('vcard').style.display='none';
const rg=document.getElementById('rg');rg.value=D.thr;let thr=D.thr;cv.height=D.u?360:260;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n);
function draw(t){const W=cv.width,H=cv.height,L=46,R=12,T=14,B=26,HU=D.u?96:0,PB=H-B-HU,x=i=>L+(i/(D.s.length-1))*(W-L-R),y=s=>PB-s*(PB-T);
g.clearRect(0,0,W,H);g.font='13px system-ui';g.fillStyle=css('--mut');g.strokeStyle=css('--line');g.lineWidth=1;
for(const s of [0,.5,1]){g.beginPath();g.moveTo(L,y(s));g.lineTo(W-R,y(s));g.stroke();g.fillText(s.toFixed(1),8,y(s)+4)}
g.fillText('kemungkinan tabrakan (skor)',L+6,T+12);
const nt=Math.min(8,Math.floor(D.s.length/D.fps));for(let k=0;k<=nt;k++){const i=k*(D.s.length-1)/nt;g.fillText((i/D.fps).toFixed(0)+'s',x(i)-8,H-6)}
g.fillStyle='rgba(209,36,47,.18)';for(let i=0;i<D.s.length-1;i++){if(D.s[i]>=thr){g.fillRect(x(i),T,x(i+1)-x(i),PB-T)}}
g.strokeStyle=css('--bad');g.setLineDash([6,4]);g.beginPath();g.moveTo(L,y(thr));g.lineTo(W-R,y(thr));g.stroke();g.setLineDash([]);
g.strokeStyle=css('--acc');g.lineWidth=2;g.beginPath();D.s.forEach((s,i)=>{i?g.lineTo(x(i),y(s)):g.moveTo(x(i),y(s))});g.stroke();
if(D.u){const um=Math.max(1e-6,...D.u.map(p=>Math.max(p[0],p[1]))),y0=H-B-6,y1=PB+22,yu=u=>y0-(u/um)*(y0-y1);
g.lineWidth=1;g.strokeStyle=css('--line');g.beginPath();g.moveTo(L,y0);g.lineTo(W-R,y0);g.stroke();g.beginPath();g.moveTo(L,y1);g.lineTo(W-R,y1);g.stroke();
g.fillStyle=css('--mut');g.fillText('ketidakpastian model, sumbu sendiri 0 sampai '+um.toFixed(2)+': aleatorik (hijau, data), epistemik (ungu, model)',L+6,PB+16);
g.fillText('0',24,y0+4);g.lineWidth=1.6;
[['#2ca02c',0],['#9467bd',1]].forEach(([c,k])=>{g.strokeStyle=c;g.beginPath();D.u.forEach((p,i)=>{i?g.lineTo(x(i),yu(p[k])):g.moveTo(x(i),yu(p[k]))});g.stroke()})}
const i=Math.max(0,Math.min(D.s.length-1,Math.round(t*D.fps)));g.strokeStyle=css('--fg');g.lineWidth=1.5;g.beginPath();g.moveTo(x(i),T);g.lineTo(x(i),H-B);g.stroke();
document.getElementById('tt').textContent=t.toFixed(1);document.getElementById('ss').textContent=D.s[i].toFixed(3);
document.getElementById('th').textContent=thr.toFixed(2);
const fa=D.s.findIndex(s=>s>=thr);document.getElementById('al').textContent=fa<0?'ambang tidak pernah terlampaui':'terlampaui pertama pada '+(fa/D.fps).toFixed(1)+' s';}
function now(){return D.video?v.currentTime:0}
v.addEventListener('timeupdate',()=>draw(now()));v.addEventListener('seeked',()=>draw(now()));
cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect(),f=(e.clientX-r.left)/r.width,px=f*cv.width,L=46,R=12;
const t=Math.max(0,Math.min(1,(px-L)/(cv.width-L-R)))*(D.s.length-1)/D.fps;if(D.video)v.currentTime=t;draw(t)});
rg.addEventListener('input',()=>{thr=parseFloat(rg.value);draw(now())});draw(0);
</script></body></html>
"""


def write_html(path, title, scores, unc, fps, thr, video_name):
    data = {"title": title, "fps": fps, "thr": thr, "s": [round(float(s), 4) for s in scores],
            "u": None if unc is None else [[round(float(a), 5), round(float(b), 5)] for a, b in unc], "video": bool(video_name)}
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__VIDEO__", video_name or "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main(argv=None):
    a = parse_args(argv)
    if not a.detector or not os.path.exists(a.detector):
        raise SystemExit("bobot detektor tidak ditemukan. Beri --detector atau set DETECTOR_CKPT (lihat README, bagian Demo).")
    import cv2
    from gstl.detector import load_detector
    from gstl.features import VGG16FC7, extract_frames_features
    from gstl.loader import load_gstl_model

    viz = a.viz_items
    out_dir = a.out or os.path.join(HERE, "outputs", "demo", os.path.splitext(os.path.basename(a.video))[0])
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(a.device)
    print("memuat detektor dan VGG16 (pertama kali VGG16 ImageNet ~528 MB diunduh otomatis oleh torchvision)...", flush=True)
    detector = load_detector(a.detector, device=a.device)
    vgg = VGG16FC7().to(dev).eval()
    model, epoch = load_gstl_model(a.weights, dev)
    print("GSTL epoch %s | perangkat %s | viz: %s" % (epoch, dev, "tanpa video" if viz is None else (",".join(viz) or "bersih")))

    gen = sampled_windows(a.video)
    meta = next(gen)
    print("video: %dx%d, %.2f fps, %d frame -> disampel tiap %d frame (%.2f fps)" % (
        meta["width"], meta["height"], meta["fps"], meta["total"], meta["step"], meta["sampled_fps"]))
    if abs(meta["sampled_fps"] - TARGET_FPS) > 2.0:
        print("PERINGATAN: laju sampel %.1f fps jauh dari 10 fps (laju latih); skor kurang dapat dipercaya." % meta["sampled_fps"])
    if (meta["width"], meta["height"]) != (1280, 720):
        print("PERINGATAN: resolusi bukan 1280x720; geometri kotak diskalakan ke kerangka 1280x720.")
    fps_s = meta["sampled_fps"]
    writer, rows, all_scores, all_unc, first_alert, t0 = None, [], [], [], None, time.time()
    for wi, win in enumerate(gen):
        if a.max_windows and wi >= a.max_windows:
            break
        frames = [f for _, f in win]
        feat = extract_frames_features(frames, detector, vgg, n_frames=N_FRAMES, seed=a.seed, device=a.device)
        sc, unc = score_window_full(model, feat["data"], feat["det"], dev, a.seed, (meta["width"], meta["height"]), uncertainty=a.uncertainty)
        sc = sc[: feat["n_real"]]
        for k, ((src, fr), s) in enumerate(zip(win, sc)):
            t = len(all_scores) / fps_s
            rows.append({"window": wi, "sample": len(all_scores), "src_frame": src, "time_s": round(t, 3), "score": float(s),
                         **({"u_aleatoric": float(unc[k, 0]), "u_epistemic": float(unc[k, 1])} if unc is not None else {})})
            all_scores.append(float(s))
            if unc is not None:
                all_unc.append(unc[k])
            if first_alert is None and s >= a.threshold:
                first_alert = t
            if viz is not None:
                boxes = [(*b[:5], ci) for ci, cls in enumerate(feat["raw_dets"][k]) for b in cls if b[4] >= a.box_conf]
                im = compose_frame(fr, boxes, all_scores, t, a.threshold, viz, fps_s, a.width, a.curve_seconds, first_alert)
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(out_dir, "overlay.tmp.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_s, (im.shape[1], im.shape[0]))
                writer.write(im)
        print("jendela %d: %d frame, skor maks %.3f (%.0f dtk berlalu)" % (wi, len(win), sc.max(), time.time() - t0), flush=True)
    h264 = None
    if writer is not None:
        writer.release()
        h264 = finalize_video(os.path.join(out_dir, "overlay.tmp.mp4"), os.path.join(out_dir, "overlay.mp4"))
        if not h264:
            print("PERINGATAN: ffmpeg tidak ditemukan; overlay.mp4 berformat mp4v dan MUNGKIN tidak diputar di peramban (laporan.html). "
                  "Pasang ffmpeg (https://ffmpeg.org) lalu ulangi, atau putar dengan VLC/pemutar desktop.")
    if not rows:
        raise SystemExit("tidak ada frame yang diproses")
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sc = np.array(all_scores)
    unc_arr = np.array(all_unc) if all_unc else None
    above = np.where(sc >= a.threshold)[0]
    print("\nRingkasan: %d frame diskor | skor maks %.3f (t=%.1fs) | rerata %.3f" % (len(sc), sc.max(), sc.argmax() / fps_s, sc.mean()))
    print("Ambang %.2f: %s" % (a.threshold, ("pertama terlampaui pada t=%.1fs (%d dari %d frame di atas ambang)" % (above[0] / fps_s, len(above), len(sc)))
                                          if len(above) else "tidak pernah terlampaui"))
    title = os.path.basename(a.video)
    if not a.no_plot:
        try:
            write_plot(os.path.join(out_dir, "skor.png"), sc, unc_arr, fps_s, a.threshold, title)
        except ImportError:
            print("matplotlib tidak ada: grafik dilewati (scores.csv tetap ditulis)")
    if a.html:
        write_html(os.path.join(out_dir, "laporan.html"), title, sc, unc_arr, fps_s, a.threshold, "overlay.mp4" if viz is not None else None)
        print("laporan HTML:", os.path.join(out_dir, "laporan.html"))
    print("keluaran:", out_dir)


if __name__ == "__main__":
    main()
