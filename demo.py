"""Demo GSTL pada video dashcam mentah: video -> deteksi objek -> fitur -> skor risiko kecelakaan per frame.

  python demo.py --video contoh.mp4 --detector <latest.pth>
  python demo.py --video contoh.mp4 --detector <latest.pth> --threshold 0.5 --out outputs/demo/contoh

Keluaran di folder --out: scores.csv (skor per frame), skor.png (grafik, bila matplotlib ada), overlay.mp4 (video + kotak + skor; matikan dengan --no-video).

Cara kerja: video disampel ke ~10 fps (rentang frame per sampel = round(fps/10)), dipotong menjadi jendela 50 frame
(5 detik) yang diskor terpisah, seperti saat model dilatih pada klip CCD (50 frame, 10 fps).

KETERBATASAN (baca sebelum menafsirkan skor):
  * Detektor Cascade R-CNN dilatih di KITTI dengan 3 kelas (Car, Pedestrian, Cyclist), TANPA kelas motor; motor sering tidak terdeteksi.
  * Pipeline fitur ini adalah port PyTorch murni dari kode UString; mendekati tetapi tidak identik dengan fitur CCD rilis (lihat docs/demo_validasi.md).
  * Model dilatih pada CCD (video YouTube, 1280x720). Video lain (resolusi/rasio aspek/gaya berbeda) berada di luar distribusi latih;
    geometri kotak diskalakan ke kerangka 1280x720, rasio aspek yang berbeda akan terdistorsi.
  * Skor adalah keluaran model (probabilitas per frame), bukan kalibrasi risiko nyata; jangan dipakai untuk keputusan keselamatan.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "gstl_ccd_seed123_epoch10.pth")
N_FRAMES, TARGET_FPS, REF_W, REF_H = 50, 10.0, 1280.0, 720.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT"),
                    help="bobot Cascade R-CNN (latest.pth dari Drive UString); bawaan: variabel DETECTOR_CKPT")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="checkpoint GSTL")
    ap.add_argument("--out", default=None, help="folder keluaran (bawaan: outputs/demo/<nama video>)")
    ap.add_argument("--threshold", type=float, default=0.5, help="ambang peringatan (hanya untuk ringkasan/penanda)")
    ap.add_argument("--max-windows", type=int, default=0, help="batasi jumlah jendela 50 frame (0 = semua)")
    ap.add_argument("--no-video", action="store_true", help="jangan tulis overlay.mp4")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


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


def score_window(model, feats, det, device, seed, size):
    """Skor per frame (n_frames,) untuk satu jendela; det diskalakan ke kerangka 1280x720 untuk geometri."""
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
                           nbatch=1, testing=False, eval_uncertain=False)
        logits = torch.stack([o["pred_mean"] for o in outs], dim=1)
    return torch.softmax(logits, dim=-1)[0, :, 1].float().cpu().numpy()


def draw_overlay(frame, boxes, score, t, thr, width=960):
    import cv2
    im = frame.copy()
    for b in boxes:
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 3)
    s = width / im.shape[1]
    im = cv2.resize(im, (width, int(im.shape[0] * s)))
    col = (0, 0, 255) if score >= thr else (0, 200, 255) if score >= 0.5 * thr else (0, 200, 0)
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
    cv2.rectangle(im, (0, 0), (int(im.shape[1] * float(score)), 34), col, -1)
    cv2.putText(im, "t=%.1fs  skor=%.2f" % (t, score), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im


def main(argv=None):
    a = parse_args(argv)
    if not a.detector or not os.path.exists(a.detector):
        raise SystemExit("bobot detektor tidak ditemukan. Beri --detector atau set DETECTOR_CKPT (lihat README, bagian Demo).")
    import cv2
    from gstl.detector import load_detector
    from gstl.features import VGG16FC7, extract_frames_features
    from gstl.loader import load_gstl_model

    out_dir = a.out or os.path.join(HERE, "outputs", "demo", os.path.splitext(os.path.basename(a.video))[0])
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(a.device)
    print("memuat detektor dan VGG16 (pertama kali VGG16 ImageNet ~528 MB diunduh otomatis oleh torchvision)...", flush=True)
    detector = load_detector(a.detector, device=a.device)
    vgg = VGG16FC7().to(dev).eval()
    model, epoch = load_gstl_model(a.weights, dev)
    print("GSTL epoch %s | perangkat %s" % (epoch, dev))

    gen = sampled_windows(a.video)
    meta = next(gen)
    print("video: %dx%d, %.2f fps, %d frame -> disampel tiap %d frame (%.2f fps)" % (
        meta["width"], meta["height"], meta["fps"], meta["total"], meta["step"], meta["sampled_fps"]))
    if abs(meta["sampled_fps"] - TARGET_FPS) > 2.0:
        print("PERINGATAN: laju sampel %.1f fps jauh dari 10 fps (laju latih); skor kurang dapat dipercaya." % meta["sampled_fps"])
    if (meta["width"], meta["height"]) != (1280, 720):
        print("PERINGATAN: resolusi bukan 1280x720; geometri kotak diskalakan ke kerangka 1280x720.")
    writer, rows, all_scores, t0 = None, [], [], time.time()
    for wi, win in enumerate(gen):
        if a.max_windows and wi >= a.max_windows:
            break
        frames = [f for _, f in win]
        feat = extract_frames_features(frames, detector, vgg, n_frames=N_FRAMES, seed=a.seed, device=a.device)
        sc = score_window(model, feat["data"], feat["det"], dev, a.seed, (meta["width"], meta["height"]))[: feat["n_real"]]
        for k, ((src, fr), s) in enumerate(zip(win, sc)):
            t = len(all_scores) / meta["sampled_fps"]
            rows.append({"window": wi, "sample": len(all_scores), "src_frame": src, "time_s": round(t, 3), "score": float(s)})
            all_scores.append(float(s))
            if not a.no_video:
                boxes = [b for cls in feat["raw_dets"][k] for b in cls if b[4] >= 0.5]
                im = draw_overlay(fr, boxes, s, t, a.threshold)
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(out_dir, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["sampled_fps"],
                                             (im.shape[1], im.shape[0]))
                writer.write(im)
        print("jendela %d: %d frame, skor maks %.3f (%.0f dtk berlalu)" % (wi, len(win), sc.max(), time.time() - t0), flush=True)
    if writer is not None:
        writer.release()
    if not rows:
        raise SystemExit("tidak ada frame yang diproses")
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sc = np.array(all_scores)
    above = np.where(sc >= a.threshold)[0]
    print("\nRingkasan: %d frame diskor | skor maks %.3f (t=%.1fs) | rerata %.3f" % (len(sc), sc.max(), sc.argmax() / meta["sampled_fps"], sc.mean()))
    print("Ambang %.2f: %s" % (a.threshold, ("pertama terlampaui pada t=%.1fs (%d dari %d frame di atas ambang)" % (above[0] / meta["sampled_fps"], len(above), len(sc)))
                                          if len(above) else "tidak pernah terlampaui"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.2))
        tt = np.arange(len(sc)) / meta["sampled_fps"]
        ax.plot(tt, sc, lw=1.6)
        ax.axhline(a.threshold, color="r", ls="--", lw=1, label="ambang %.2f" % a.threshold)
        for k in range(N_FRAMES, len(sc), N_FRAMES):
            ax.axvline(k / meta["sampled_fps"], color="0.8", lw=0.8)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("waktu (detik)")
        ax.set_ylabel("skor GSTL")
        ax.set_title(os.path.basename(a.video) + " (garis abu-abu = batas jendela 50 frame)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "skor.png"), dpi=130)
    except ImportError:
        print("matplotlib tidak ada: grafik dilewati (scores.csv tetap ditulis)")
    print("keluaran:", out_dir)


if __name__ == "__main__":
    main()
