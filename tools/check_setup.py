"""Periksa kesiapan lingkungan, bobot, data, dan (opsional) detektor. Tidak mengubah apa pun.

  python tools/check_setup.py                         # lingkungan + bobot + forward GSTL
  python tools/check_setup.py --data-root <CCD>       # + struktur dan contoh berkas CCD (atau set CCD_ROOT)
  python tools/check_setup.py --detector <latest.pth> [--hash-detector]   # + bobot detektor untuk demo.py

Status: OK / PERINGATAN (boleh lanjut, ada batasan) / GAGAL (perlu diperbaiki). Kode keluar 1 bila ada GAGAL.
"""
import argparse
import hashlib
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DETECTOR_SHA256 = "C768907093DB3B5BC87C1D79E728EEBB6F90252017D7BCB2DF0D5B34F2BD60BF"  # latest.pth Cascade R-CNN (Drive UString)
results = []


def report(status, name, detail=""):
    results.append(status)
    print("[%-10s] %-26s %s" % (status, name, detail))


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check_env():
    v = sys.version_info
    report("OK" if (v.major, v.minor) >= (3, 9) else "PERINGATAN", "Python", "%d.%d.%d (diuji 3.10)" % v[:3])
    try:
        import numpy as np
        ok = int(np.__version__.split(".")[0]) < 2
        report("OK" if ok else "GAGAL", "numpy", np.__version__ + ("" if ok else " (torch 2.3 butuh numpy 1.x; pasang numpy==1.26.4)"))
    except ImportError:
        report("GAGAL", "numpy", "tidak terpasang")
    try:
        import torch
        report("OK", "torch", torch.__version__ + " (diuji 2.3.0+cu118)")
        if torch.cuda.is_available():
            report("OK", "GPU", "%s, %.1f GB" % (torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9))
        else:
            report("PERINGATAN", "GPU", "CUDA tidak tersedia; evaluasi/pelatihan berjalan di CPU (lambat); demo.py butuh GPU agar praktis")
    except ImportError:
        report("GAGAL", "torch", "tidak terpasang")
        return False
    for mod, why in (("cv2", "demo.py"), ("torchvision", "demo.py"), ("PIL", "demo.py"), ("matplotlib", "grafik demo.py (opsional)")):
        try:
            m = importlib.import_module(mod)
            report("OK", mod, getattr(m, "__version__", "") + " (" + why + ")")
        except ImportError:
            report("PERINGATAN", mod, "tidak terpasang (untuk %s; pip install -r requirements-demo.txt)" % why)
    return True


def check_weights():
    p = os.path.join(ROOT, "weights", "gstl_ccd_seed123_epoch10.pth")
    sums = os.path.join(ROOT, "weights", "SHA256SUMS.txt")
    if not os.path.exists(p):
        report("GAGAL", "bobot GSTL", "tidak ada: %s" % p)
        return None
    want = open(sums).read().split()[0].lower() if os.path.exists(sums) else None
    got = sha256(p)
    if want is None:
        report("PERINGATAN", "bobot GSTL", "SHA256SUMS.txt tidak ada; hash %s" % got[:16])
    else:
        ok = got == want
        report("OK" if ok else "GAGAL", "bobot GSTL", "hash cocok" if ok else "hash BERBEDA (berkas rusak atau diganti; perbarui weights/SHA256SUMS.txt bila disengaja)")
    return p


def check_forward(ckpt):
    try:
        import numpy as np
        import torch
        from gstl.graph import generate_st_graph
        from gstl.gstl import build_gstl
        cfg = torch.load(ckpt, map_location="cpu")["config"]
        ck = torch.load(ckpt, map_location="cpu")
        m = build_gstl(50, 10.0, x_dim=4096, h_dim=cfg["hidden_dim"], z_dim=cfg["latent_dim"], n_layers=cfg.get("num_rnn", 1), n_obj=19,
                       frame_embed=cfg["gstl_frame_embed"], box_mode=cfg["gstl_box_mode"], box_embed=cfg["gstl_box_embed"],
                       with_saa=not cfg.get("gstl_no_saa", False), uncertain_ranking=cfg.get("gstl_ranking", False),
                       rnn_trainable=cfg.get("rnn_trainable", False), exp_loss_mode=cfg.get("exp_loss", "faithful"))
        m.load_state_dict(ck["model"])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = m.to(dev).eval()
        rng = np.random.default_rng(0)
        x1 = rng.uniform(0, 1000, (50, 19))
        y1 = rng.uniform(0, 500, (50, 19))
        det = np.stack([x1, y1, x1 + 80, y1 + 60, np.ones((50, 19)), np.zeros((50, 19))], -1).astype(np.float32)
        g, w = generate_st_graph(det)
        x = torch.randn(1, 50, 1, 4096, device=dev)
        y = torch.tensor([[1.0, 0.0]], device=dev)
        toa = torch.tensor([[51.0]], device=dev)
        with torch.no_grad():
            _, outs, _ = m(x, y, toa, torch.from_numpy(g)[None].to(dev), hidden_in=None, edge_weights=torch.from_numpy(w)[None].to(dev),
                           npass=2, det=torch.from_numpy(det)[None].to(dev), nbatch=1, testing=False, eval_uncertain=False)
        s = torch.softmax(torch.stack([o["pred_mean"] for o in outs], 1), -1)[..., 1]
        ok = tuple(s.shape) == (1, 50) and bool(torch.isfinite(s).all())
        report("OK" if ok else "GAGAL", "forward GSTL (data acak)", "skor per frame %s di %s" % (tuple(s.shape), dev))
    except Exception as e:  # noqa: BLE001
        report("GAGAL", "forward GSTL", repr(e)[:150])


def check_ccd(root):
    if not root:
        report("PERINGATAN", "data CCD", "tidak diperiksa (beri --data-root atau set CCD_ROOT; data tidak disertakan)")
        return
    need = [os.path.join(root, "vgg16_features", "train.txt"), os.path.join(root, "vgg16_features", "test.txt"),
            os.path.join(root, "videos", "Crash-1500.txt")]
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        report("GAGAL", "data CCD", "berkas tidak ada: " + "; ".join(os.path.relpath(p, root) for p in miss))
        return
    n_tr = sum(1 for line in open(need[0]) if line.strip())
    n_te = sum(1 for line in open(need[1]) if line.strip())
    exp = (n_tr, n_te) == (3600, 900)
    report("OK" if exp else "PERINGATAN", "daftar CCD", "latih %d / uji %d%s" % (n_tr, n_te, "" if exp else " (CCD lengkap: 3.600 / 900)"))
    import numpy as np
    first = open(need[1]).readline().split()[0]
    p = os.path.join(root, "vgg16_features", first)
    try:
        with np.load(p) as d:
            okf = d["data"].shape == (50, 20, 4096) and d["det"].shape == (50, 19, 6) and d["labels"].shape == (2,)
        report("OK" if okf else "GAGAL", "contoh npz uji", "%s bentuk %s" % (first, "sesuai" if okf else "TIDAK sesuai"))
    except Exception as e:  # noqa: BLE001
        report("GAGAL", "contoh npz uji", repr(e)[:120])
    missing = sum(1 for line in open(need[1]) if line.strip() and not os.path.exists(os.path.join(root, "vgg16_features", line.split()[0])))
    report("OK" if missing == 0 else "GAGAL", "berkas npz uji", "semua ada" if missing == 0 else "%d berkas npz uji hilang" % missing)


def check_detector(path, do_hash):
    if not path:
        report("PERINGATAN", "bobot detektor", "tidak diperiksa (hanya untuk demo.py; beri --detector atau set DETECTOR_CKPT)")
        return
    if not os.path.exists(path):
        report("GAGAL", "bobot detektor", "tidak ada: %s" % path)
        return
    size = os.path.getsize(path) / 1e9
    if not do_hash:
        report("OK", "bobot detektor", "ada, %.2f GB (tambah --hash-detector untuk memeriksa SHA-256)" % size)
        return
    got = sha256(path).upper()
    ok = got == DETECTOR_SHA256
    report("OK" if ok else "GAGAL", "bobot detektor", "hash cocok" if ok else "hash BERBEDA dari yang diharapkan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    ap.add_argument("--detector", default=os.environ.get("DETECTOR_CKPT"))
    ap.add_argument("--hash-detector", action="store_true")
    a = ap.parse_args()
    has_torch = check_env()
    ckpt = check_weights()
    if has_torch and ckpt:
        check_forward(ckpt)
    check_ccd(a.data_root)
    check_detector(a.detector, a.hash_detector)
    n = {k: results.count(k) for k in ("OK", "PERINGATAN", "GAGAL")}
    print("\nRingkas: %(OK)d OK, %(PERINGATAN)d peringatan, %(GAGAL)d gagal" % n)
    sys.exit(1 if n["GAGAL"] else 0)


if __name__ == "__main__":
    main()
