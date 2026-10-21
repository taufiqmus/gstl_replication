"""Evaluasi checkpoint GSTL pada set uji CCD (protokol uji asli: batch 10, drop_last, sampling BNN npass=10).

  python evaluate.py --data-root <folder CCD> [--ckpts weights/gstl_ccd_seed123_epoch10.pth] [--seeds 123 1 2 3 4 5]

Keluaran: results/<tag>.csv (satu baris per checkpoint x seed) dan results/<tag>.json (ringkasan).
Metrik: AP_orig / mTTA_orig (gaya UString, termasuk bug AP>1 bila ada), AP_origfix, AP_video (eksak), mTTA_sec (detik sungguhan).
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gstl.data import CrashCompactDataset, CrashDataset  # noqa: E402
from gstl.gstl import build_gstl  # noqa: E402
from train import compute_metrics, configure_for_model, run_inference, set_seed  # noqa: E402

DEFAULT_CKPT = os.path.join(HERE, "weights", "gstl_ccd_seed123_epoch10.pth")


class _A:  # bobot loss untuk run_inference (sama dengan bawaan train.py)
    loss_alpha, loss_beta, loss_yita = 1e-3, 10.0, 10.0


def load_gstl(path, ds, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if cfg.get("model", "gstl") != "gstl":
        raise ValueError("%s bukan checkpoint GSTL" % path)
    m = build_gstl(ds.n_frames, ds.fps, x_dim=ds.dim_feature, h_dim=cfg["hidden_dim"], z_dim=cfg["latent_dim"],
                   n_layers=cfg.get("num_rnn", 1), n_obj=ds.n_obj, frame_embed=cfg["gstl_frame_embed"],
                   box_mode=cfg["gstl_box_mode"], box_embed=cfg["gstl_box_embed"],
                   with_saa=not cfg.get("gstl_no_saa", False), uncertain_ranking=cfg.get("gstl_ranking", False),
                   rnn_trainable=cfg.get("rnn_trainable", False), exp_loss_mode=cfg.get("exp_loss", "faithful"), no_frame=cfg.get("gstl_no_frame", False), no_box=cfg.get("gstl_no_box", False))
    m.load_state_dict(ck["model"])
    return m.to(device), ck.get("epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"), help="folder CCD (berisi vgg16_features/ dan videos/); bawaan: CCD_ROOT")
    ap.add_argument("--dataset", choices=("crash", "crash_compact"), default="crash", help="crash = fitur rilis (.npz penuh); crash_compact = format ringkas")
    ap.add_argument("--ckpts", nargs="*", default=None, help="checkpoint (bawaan: weights/gstl_ccd_seed123_epoch10.pth)")
    ap.add_argument("--exp-dir", default=None, help="atau: folder eksperimen hasil train.py (semua snapshot/epoch_*.pth)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[123], help="seed sampling BNN saat evaluasi")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--no-drop-last", action="store_true", help="evaluasi semua video (bawaan: seperti loader uji asli, batch penuh saja)")
    a = ap.parse_args()
    if not a.data_root:
        raise SystemExit("beri --data-root atau set CCD_ROOT (lihat README: data CCD tidak disertakan)")
    ckpts = a.ckpts or (sorted(glob.glob(os.path.join(a.exp_dir, "snapshot", "epoch_*.pth"))) if a.exp_dir else [DEFAULT_CKPT])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = (CrashCompactDataset if a.dataset == "crash_compact" else CrashDataset)(a.data_root, "test")
    configure_for_model("gstl", ds)
    loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=not a.no_drop_last, num_workers=a.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=a.num_workers > 0)
    os.makedirs(a.out, exist_ok=True)
    rows, t0 = [], time.time()
    for k, path in enumerate(ckpts):
        model, epoch = load_gstl(path, ds, device)
        for s in a.seeds:
            set_seed(s)
            res, mloss = run_inference(model, loader, device, _A, eval_uncertain=True)
            m = compute_metrics(res, ds.fps)
            m["mAU"], m["mEU"] = [float(v) for v in res["uncertainty"].mean(axis=(0, 1))]
            rows.append({"ckpt": os.path.basename(path), "epoch": epoch, "sample_seed": s, **m})
            print("[%d/%d] %s seed %d | AP_video %.4f AP_orig %.4f mTTA_orig %.2f mTTA_sec %.2f (%.0f dtk)" % (
                k + 1, len(ckpts), os.path.basename(path), s, m["AP_video"], m["AP_orig"], m["mTTA_orig"], m["mTTA_sec"],
                time.time() - t0), flush=True)
    with open(os.path.join(a.out, a.tag + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({"rows": rows, "n_test": int(len(ds)), "torch": torch.__version__}, open(os.path.join(a.out, a.tag + ".json"), "w"), indent=1)
    print("selesai:", os.path.join(a.out, a.tag + ".csv"))


if __name__ == "__main__":
    main()
