"""Uji langsung bagian mana dari masukan simpul GSTL yang membawa sinyal (tanpa pelatihan ulang).

Simpul GSTL = [Psi (4 koordinat kotak ternormalisasi), Phi(F) (fitur frame 256-d, SAMA untuk semua 19 simpul)].
Dengan mengganti node_features pada model yang dimuat, diukur pada set uji CCD:
  asli          acuan
  psi_nol       Psi = 0 (geometri dihapus dari simpul; graf tetap dari kotak asli)
  psi_acak      Psi tiap simpul diganti Psi simpul lain pada frame yang sama (permutasi acak; himpunan kotak tetap, korespondensi simpul-kotak rusak)
  fitur_nol     Phi(F) = 0 (fitur frame dihapus)
  keduanya_nol  Psi = 0 dan Phi(F) = 0
Juga: besar bobot lapisan GCN pertama untuk Psi vs Phi(F), skala nilai masukan, dan selisih skor per frame terhadap acuan.
Keluaran: docs/ablasi_geometri_dalam.md, results/ablasi_geometri_dalam.json.
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from collision_eval.metrics import video_scores  # noqa: E402
from gstl.data import CrashDataset  # noqa: E402
from gstl.gstl import box_geometry  # noqa: E402
from gstl.loader import load_gstl_model  # noqa: E402
from train import compute_metrics, configure_for_model, run_inference, set_seed  # noqa: E402


class _A:
    loss_alpha, loss_beta, loss_yita = 1e-3, 10.0, 10.0


@contextlib.contextmanager
def patched(model, mode, seed=0):
    orig = model.node_features
    gen = torch.Generator().manual_seed(seed)

    def nf(x_t, det_t):
        f = model.phi_f(x_t[:, 0])
        g = box_geometry(det_t, model.box_mode)
        if mode in ("psi_nol", "keduanya_nol"):
            g = torch.zeros_like(g)
        if mode == "psi_acak":
            perm = torch.stack([torch.randperm(g.size(1), generator=gen) for _ in range(g.size(0))]).to(g.device)
            g = torch.gather(g, 1, perm.unsqueeze(-1).expand_as(g))
        if mode in ("fitur_nol", "keduanya_nol"):
            f = torch.zeros_like(f)
        return torch.cat([g, f.unsqueeze(1).expand(-1, model.n_obj, -1)], dim=-1)

    model.node_features = nf
    try:
        yield
    finally:
        model.node_features = orig


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    allv = np.concatenate([pos, neg])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("CCD_ROOT"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "gstl_ccd_seed123_epoch10.pth"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_gstl_model(a.weights, dev)
    ds = CrashDataset(a.data_root, "test")
    configure_for_model("gstl", ds)
    loader = DataLoader(ds, batch_size=10, shuffle=False, drop_last=True, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)

    # ---- besar bobot dan skala masukan
    W = model.enc_gcn1.weight if hasattr(model.enc_gcn1, "weight") else next(p for p in model.enc_gcn1.parameters() if p.dim() == 2)
    W = W.detach().cpu().numpy()
    if W.shape[0] != model.node_dim:
        W = W.T
    wpsi, wphi = W[:4], W[4:]
    info = {"bobot_gcn1_psi_rerata_abs": float(np.abs(wpsi).mean()), "bobot_gcn1_phi_rerata_abs": float(np.abs(wphi).mean()),
            "bobot_gcn1_psi_norm_baris_rerata": float(np.linalg.norm(wpsi, axis=1).mean()),
            "bobot_gcn1_phi_norm_baris_rerata": float(np.linalg.norm(wphi, axis=1).mean())}
    xs, ds_tmp = [], []
    with torch.no_grad():
        for k, batch in enumerate(loader):
            if k >= 5:
                break
            x = batch[0].to(dev)
            det = batch[5].to(dev)
            f = model.phi_f(x[:, :, 0])
            g = box_geometry(det, model.box_mode)
            xs.append(float(f.abs().mean()))
            ds_tmp.append(float(g.abs().mean()))
    info["skala_Phi_rerata_abs"], info["skala_Psi_rerata_abs"] = float(np.mean(xs)), float(np.mean(ds_tmp))
    info["kontribusi_kasar_Psi_vs_Phi"] = (info["bobot_gcn1_psi_rerata_abs"] * info["skala_Psi_rerata_abs"] * 4) / (
        info["bobot_gcn1_phi_rerata_abs"] * info["skala_Phi_rerata_abs"] * 256)
    print(json.dumps(info, indent=1), flush=True)

    # ---- ablasi dalam
    res, base = {}, None
    for mode in ("asli", "psi_nol", "psi_acak", "fitur_nol", "keduanya_nol"):
        set_seed(123)
        with patched(model, mode):
            out, _ = run_inference(model, loader, dev, _A, eval_uncertain=False)
        m = compute_metrics(out, ds.fps)
        vs = video_scores(out["pred"], out["label"], out["toa"])
        row = dict(AP_orig=m["AP_orig"], AP_video=m["AP_video"], mTTA_sec=m["mTTA_sec"], AUC=auc(vs, out["label"]),
                   skor_maks_rerata=float(out["pred"].max(1).mean()))
        if base is None:
            base = out["pred"]
        else:
            d = np.abs(out["pred"] - base)
            row["selisih_skor_per_frame_rerata"] = float(d.mean())
            row["selisih_skor_per_frame_maks"] = float(d.max())
            row["frame_berubah_lebih_dari_0.1_persen"] = float(100 * (d > 0.1).mean())
        res[mode] = row
        print("%-13s AP_orig %.4f AP_video %.4f AUC %.4f mTTA %.2f %s" % (mode, row["AP_orig"], row["AP_video"], row["AUC"], row["mTTA_sec"],
              ("| dskor rerata %.4f maks %.3f, frame berubah>0,1: %.1f%%" % (row["selisih_skor_per_frame_rerata"], row["selisih_skor_per_frame_maks"],
                                                                          row["frame_berubah_lebih_dari_0.1_persen"]) if "selisih_skor_per_frame_rerata" in row else "")), flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({"info": info, "hasil": res}, open(os.path.join(ROOT, "results", "ablasi_geometri_dalam.json"), "w"), indent=1)
    md = ["# Uji bagian dalam: mana yang membawa sinyal pada simpul GSTL (set uji CCD, tanpa pelatihan ulang)\n",
          "Simpul = [Psi: 4 koordinat kotak, Phi(F): fitur frame 256-d yang SAMA untuk semua 19 simpul]. Bobot `gstl_ccd_seed123_epoch10.pth`.\n",
          "| Mode | AP asli (%) | AP video (%) | AUC | mTTA (s) | selisih skor/frame (rerata) | frame berubah > 0,1 |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        md.append("| %s | %.2f | %.2f | %.3f | %.2f | %s | %s |" % (k, 100 * r["AP_orig"], 100 * r["AP_video"], r["AUC"], r["mTTA_sec"],
                  ("%.4f" % r["selisih_skor_per_frame_rerata"]) if "selisih_skor_per_frame_rerata" in r else "-",
                  ("%.1f%%" % r["frame_berubah_lebih_dari_0.1_persen"]) if "frame_berubah_lebih_dari_0.1_persen" in r else "-"))
    md += ["", "## Besar bobot dan skala", "```", json.dumps(info, indent=1), "```"]
    open(os.path.join(ROOT, "docs", "ablasi_geometri_dalam.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
