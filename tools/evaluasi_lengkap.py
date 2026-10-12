"""Evaluasi lengkap satu checkpoint GSTL: metrik utama, metrik tingkat-FPR, alarm frame 0, dan ablasi geometri (tanpa pelatihan ulang).

  python tools/evaluasi_lengkap.py --ckpt <pth> --dataset crash_compact --data-root <folder> --tag M1_cascade [--ablasi] [--seeds 3]

Keluaran: results/evaluasi_<tag>.json. Ablasi: tanpa_geometri, tukar_kotak, kotak_acak, tukar_fitur_frame (acak: rata-rata --seeds) dan psi_nol.
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from collision_eval import alarm_rates, tta_at_fpr, video_scores  # noqa: E402
from gstl.ablation import AblatedDataset, psi_nol  # noqa: E402
from gstl.data import CrashCompactDataset, CrashDataset, DotaCompactDataset  # noqa: E402
from gstl.loader import load_gstl_model  # noqa: E402
from train import compute_metrics, configure_for_model, run_inference, set_seed  # noqa: E402


class _A:
    loss_alpha, loss_beta, loss_yita = 1e-3, 10.0, 10.0


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    allv = np.concatenate([pos, neg])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(res, fps):
    m = compute_metrics(res, fps)
    vs = video_scores(res["pred"], res["label"], res["toa"])
    out = {k: float(m[k]) for k in ("AP_orig", "AP_video", "mTTA_orig", "mTTA_sec")}
    out["AUC"] = auc(vs, res["label"])
    for f in (0.01, 0.05):
        r = tta_at_fpr(res["pred"], res["label"], res["toa"], fps, f)
        out["tta@fpr%d" % round(100 * f)] = {k: (float(v) if v == v else None) for k, v in r.items()}
    out["alarm@0.5"] = alarm_rates(res["pred"], res["label"], res["toa"], fps, 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=("crash", "crash_compact", "dota_compact"), default="crash_compact")
    ap.add_argument("--toa", choices=("V0", "V3"), default="V3", help="dota_compact: toa asli atau terkoreksi")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ablasi", action="store_true")
    ap.add_argument("--modes", default=None, help="subset mode ablasi dipisah koma (bawaan: semua; psi_nol dipakai bila disebut atau semua)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_gstl_model(a.ckpt, dev)
    if a.dataset == "dota_compact":
        base = DotaCompactDataset(a.data_root, toa=a.toa)
    else:
        base = (CrashCompactDataset if a.dataset == "crash_compact" else CrashDataset)(a.data_root, "test")
    drop_last = a.dataset != "dota_compact"
    configure_for_model("gstl", base)
    t0 = time.time()

    def run(ds, ctx=None):
        loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=drop_last, num_workers=a.workers, pin_memory=dev.type == "cuda",
                            persistent_workers=a.workers > 0)
        set_seed(123)
        with (ctx if ctx is not None else contextlib.nullcontext()):
            res, _ = run_inference(model, loader, dev, _A, eval_uncertain=False)
        return summarize(res, base.fps)

    result = {"ckpt": os.path.basename(a.ckpt), "epoch": epoch, "dataset": a.dataset, "toa": a.toa if a.dataset == "dota_compact" else None, "n_klip": len(base), "asli": run(base)}
    print("asli:", json.dumps({k: round(v, 4) for k, v in result["asli"].items() if isinstance(v, float)}), "(%.0f s)" % (time.time() - t0), flush=True)
    if a.ablasi:
        plan = [("tanpa_geometri", 0.0, 1), ("tukar_kotak", 0.0, a.seeds), ("kotak_acak", 0.0, a.seeds), ("tukar_fitur_frame", 0.0, a.seeds),
                ("acak_waktu", 0.0, a.seeds), ("acak_paruh", 0.0, a.seeds), ("balik_waktu", 0.0, 1)]
        want = set(a.modes.split(",")) if a.modes else None
        if want:
            plan = [x for x in plan if x[0] in want]
        abl = {}
        for mode, p, n in plan:
            runs = [run(AblatedDataset(base, mode, p, s)) for s in range(n)]
            abl[mode] = {k: float(np.mean([r[k] for r in runs])) for k in ("AP_orig", "AP_video", "AUC", "mTTA_sec")}
            print("  %-18s AP_orig %.4f AP_video %.4f AUC %.4f (%.0f s)" % (mode, abl[mode]["AP_orig"], abl[mode]["AP_video"], abl[mode]["AUC"], time.time() - t0), flush=True)
        if want is None or "psi_nol" in want:
            r = run(base, psi_nol(model))
            abl["psi_nol"] = {k: r[k] for k in ("AP_orig", "AP_video", "AUC", "mTTA_sec")}
            print("  %-18s AP_orig %.4f AP_video %.4f AUC %.4f" % ("psi_nol", abl["psi_nol"]["AP_orig"], abl["psi_nol"]["AP_video"], abl["psi_nol"]["AUC"]), flush=True)
        result["ablasi"] = abl
    os.makedirs(a.out, exist_ok=True)
    json.dump(result, open(os.path.join(a.out, "evaluasi_%s.json" % a.tag), "w"), indent=1)
    print("ditulis: results/evaluasi_%s.json" % a.tag)


if __name__ == "__main__":
    main()
