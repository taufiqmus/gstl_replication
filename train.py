"""Latih GSTL (Mahmood dkk. 2023) pada CCD; GSTL dibangun di atas port UString (Bao dkk. 2020, MIT). Lihat README.md dan REPLIKASI.md.

Contoh (konfigurasi yang menghasilkan bobot di weights/; seed 123):
  python train.py --dataset crash --data-root <folder CCD> --exp gstl_s123 --epochs 70 --val-frac 0.1       --test-iter 0 --eval-epoch-end --select-metric AP_video --early-stop-patience 10 --keep-all-epochs --seed 123
(model bawaan = gstl, hidden/latent 128; --model ustring melatih GCRNN dasar sebagai pembanding)

Yang setia ke asli (bawaan):
  seed 123; Adam lr 5e-4 (run_train_test.sh); ReduceLROnPlateau(min, 0.5, patience 5) di-step
  sekali per epoch dengan log_posterior batch TERAKHIR; batch 10; drop_last pada loader latih
  DAN uji; npass 2 (latih) / 10 (uji); clip grad 10; evaluasi tiap 64 iterasi; model terbaik
  dipilih dengan AP di SET UJI (perhatian: ini seleksi pada data uji, optimistis).
Epoch: kode asli bawaan 30 (main.py), paper menyebut 70. Pilih eksplisit dengan --epochs.

Tambahan kita: --val-frac untuk memisahkan validasi dari data latih (seleksi model jujur);
metrik dari paket collision_eval (gaya UString + versi eksak); log CSV/JSON; tidak ada
tensorboard.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from collision_eval import evaluate_ustring_style, evaluate_video_level  # noqa: E402
from gstl.data import CrashCompactDataset, CrashDataset, DADDataset, DotaCompactDataset  # noqa: E402
from gstl.model import build_model  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["crash", "crash_compact", "dad", "dota_compact"], default="crash")
    ap.add_argument("--data-root", default=None,
                    help="folder data (CCD: berisi vgg16_features/ dan videos/); bawaan: variabel lingkungan CCD_ROOT / DAD_ROOT")
    ap.add_argument("--exp", required=True, help="nama eksperimen → outputs/<exp>")
    ap.add_argument("--out-root", default="outputs", help="folder keluaran (bawaan: ./outputs)")
    ap.add_argument("--feature", default="vgg16", choices=["vgg16", "res101"])
    ap.add_argument("--epochs", type=int, required=True, help="kode asli 30, paper 70")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--model", choices=["gstl", "ustring"], default="gstl",
                    help="gstl = Mahmood 2023 (bawaan); ustring = GCRNN dasar (Bao 2020) sebagai pembanding")
    ap.add_argument("--gstl-frame-embed", type=int, default=256, help="GSTL: FC fitur frame 4096→n")
    ap.add_argument("--gstl-box-mode", choices=["xyxy", "xyxywh", "xyxywh+cls", "xyxywh+cls+score"], default="xyxy")
    ap.add_argument("--gstl-no-frame", action="store_true", help="GSTL varian geometri saja: simpul tanpa fitur frame")
    ap.add_argument("--gstl-no-box", action="store_true", help="GSTL varian fitur frame saja: simpul tanpa geometri kotak (graf tetap dari kotak)")
    ap.add_argument("--dota-toa", choices=["V0", "V3"], default="V0", help="dota_compact: toa asli (V0) atau terkoreksi (V3)")
    ap.add_argument("--gstl-box-embed", type=int, default=0, help="GSTL: 0 = geometri mentah, n = FC ke n")
    ap.add_argument("--gstl-no-saa", action="store_true", help="GSTL: tanpa cabang SAA/L_BCE video")
    ap.add_argument("--gstl-ranking", action="store_true", help="GSTL: nyalakan ranking loss (tidak ada di pers. 4)")
    ap.add_argument("--hidden-dim", type=int, default=128, help="GSTL paper: 128 (UString: 256)")
    ap.add_argument("--latent-dim", type=int, default=128, help="GSTL paper: 128 (UString: 256)")
    ap.add_argument("--num-rnn", type=int, default=1)
    ap.add_argument("--loss-alpha", type=float, default=1e-3)
    ap.add_argument("--loss-beta", type=float, default=10.0)
    ap.add_argument("--loss-yita", type=float, default=10.0)
    ap.add_argument("--test-iter", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--rnn-trainable", action="store_true", help="perbaikan: latih bobot GRU-GCN")
    ap.add_argument("--exp-loss", choices=["faithful", "per_sample"], default="faithful")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="0 = setia (seleksi di set uji); >0 = sisihkan validasi bertingkat dari latih")
    ap.add_argument("--no-test-drop-last", action="store_true", help="evaluasi semua video uji")
    ap.add_argument("--select-metric", default="AP_orig",
                    choices=["AP_orig", "AP_origfix", "AP_video", "AP_trapz", "neg_loss_cross_entropy"],
                    help="kriteria model terbaik (asli: AP_orig, termasuk bug AP>1); dievaluasi pada split seleksi")
    ap.add_argument("--eval-epoch-end", action="store_true",
                    help="evaluasi tepat di akhir tiap epoch (disarankan bersama --test-iter 0)")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="hentikan bila --select-metric tidak membaik selama n epoch (0 = mati, setia ke asli)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-workers", type=int, default=2, help="worker loader uji/validasi (hemat RAM)")
    ap.add_argument("--keep-all-epochs", action="store_true", help="simpan checkpoint tiap epoch")
    ap.add_argument("--max-train-iters", type=int, default=0, help="debug: hentikan epoch setelah n iterasi")
    ap.add_argument("--resume", default=None, help="checkpoint port untuk melanjutkan")
    ap.add_argument("--no-log-file", action="store_true",
                    help="jangan salin keluaran konsol ke <exp>/stdout.log (bawaan: tampil di layar DAN ditulis)")
    return ap.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_datasets(a):
    root = a.data_root or os.environ.get({"crash": "CCD_ROOT", "crash_compact": "CCD_COMPACT_ROOT", "dota_compact": "DOTA_COMPACT_ROOT"}.get(a.dataset, "DAD_ROOT"))
    if not root:
        raise SystemExit("data tidak ditemukan: beri --data-root atau set CCD_ROOT / DAD_ROOT")
    if a.dataset == "dota_compact":
        full_train = DotaCompactDataset(root, ("training",), a.dota_toa)
        test = DotaCompactDataset(root, ("testing",), a.dota_toa)
        if a.val_frac > 0:
            items = list(zip(range(len(full_train)), full_train.labels_list))
            tr, va = stratified_split(items, [l for _, l in items], a.val_frac, a.seed)
            return (DotaCompactDataset(root, ("training",), a.dota_toa, indices=[i for i, _ in tr]),
                    DotaCompactDataset(root, ("training",), a.dota_toa, indices=[i for i, _ in va]), test)
        return full_train, None, test
    if a.dataset in ("crash", "crash_compact"):
        DS = CrashCompactDataset if a.dataset == "crash_compact" else CrashDataset
        full_train = DS(root, "train", a.feature)
        test = DS(root, "test", a.feature)
        if a.val_frac > 0:
            items = list(zip(full_train.files_list, full_train.labels_list))
            tr, va = stratified_split(items, [l for _, l in items], a.val_frac, a.seed)
            return DS(root, "train", a.feature, file_list=tr), DS(root, "train", a.feature, file_list=va), test
        return full_train, None, test
    full_train = DADDataset(root, "training", a.feature)
    test = DADDataset(root, "testing", a.feature)
    if a.val_frac > 0:
        # label DAD hanya ada di dalam npz; nama file DAD asli tidak memuat label → split acak sederhana
        files = list(full_train.files_list)
        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(files))
        n_val = int(round(a.val_frac * len(files)))
        va = [files[i] for i in sorted(idx[:n_val])]
        tr = [files[i] for i in sorted(idx[n_val:])]
        return DADDataset(root, "training", a.feature, files=tr), DADDataset(root, "training", a.feature, files=va), test
    return full_train, None, test


def stratified_split(items, labels, frac, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    val_idx = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[: int(round(frac * len(idx)))].tolist())
    val_set = set(val_idx)
    tr = [items[i] for i in range(len(items)) if i not in val_set]
    va = [items[i] for i in sorted(val_set)]
    return tr, va


def to_device(batch, device):
    return [b.to(device, non_blocking=True) for b in batch]


def det_kw(model, batch, device):
    """GSTL butuh kotak deteksi (elemen ke-6 batch); UString tidak."""
    if getattr(model, "needs_det", False):
        return {"det": batch[5].to(device, non_blocking=True)}
    return {}


def make_model(a, ds):
    """Bangun GSTL (bawaan) atau UString dari argumen; seed tepat sebelum konstruksi."""
    if getattr(a, "model", "ustring") == "gstl":
        from gstl.gstl import build_gstl
        return build_gstl(ds.n_frames, ds.fps, seed=a.seed, x_dim=ds.dim_feature, h_dim=a.hidden_dim,
                          z_dim=a.latent_dim, n_layers=a.num_rnn, n_obj=ds.n_obj, frame_embed=a.gstl_frame_embed,
                          box_mode=a.gstl_box_mode, box_embed=a.gstl_box_embed, with_saa=not a.gstl_no_saa,
                          uncertain_ranking=a.gstl_ranking, rnn_trainable=a.rnn_trainable, exp_loss_mode=a.exp_loss, no_frame=a.gstl_no_frame, no_box=a.gstl_no_box)
    return build_model(ds.dim_feature, ds.n_obj, ds.n_frames, ds.fps, hidden_dim=a.hidden_dim,
                       latent_dim=a.latent_dim, num_rnn=a.num_rnn, rnn_trainable=a.rnn_trainable,
                       exp_loss_mode=a.exp_loss, seed=a.seed)


def configure_for_model(model_name, *datasets):
    if model_name == "gstl":
        for ds in datasets:
            if ds is not None:
                ds.return_det = True
                ds.frame_only = True  # GSTL hanya memakai fitur frame utuh


def total_loss(losses, a):
    complexity = losses["log_posterior"] - losses["log_prior"]
    t = a.loss_alpha * complexity + losses["cross_entropy"]
    t = t + a.loss_beta * losses.get("auxloss", 0)
    t = t + a.loss_yita * losses.get("ranking", 0)
    return t


@torch.no_grad()
def run_inference(model, loader, device, a, eval_uncertain=False):
    """Sama dengan test_all/test_all_vis asli: model.eval(), npass=10, softmax kelas 1 per frame."""
    model.eval()
    preds, labels, toas, uncs, loss_rows = [], [], [], [], []
    for batch in loader:
        xs, ys, ge, ew, toa = to_device(batch[:5], device)
        losses, outs, _ = model(xs, ys, toa, ge, hidden_in=None, edge_weights=ew, npass=10, **det_kw(model, batch, device),
                                nbatch=len(loader), testing=False, eval_uncertain=eval_uncertain)
        losses["total_loss"] = total_loss(losses, a)
        loss_rows.append({k: float(torch.as_tensor(v).mean()) for k, v in losses.items()})
        logits = torch.stack([o["pred_mean"] for o in outs], dim=1)          # B x T x 2
        preds.append(torch.softmax(logits, dim=-1)[..., 1].float().cpu().numpy())
        labels.append(ys[:, 1].cpu().numpy())
        toas.append(toa.view(-1).cpu().numpy())
        if eval_uncertain:
            alea = torch.stack([o["aleatoric"] for o in outs], dim=1)
            epis = torch.stack([o["epistemic"] for o in outs], dim=1)
            u = torch.stack([alea[..., 0, 0] + alea[..., 1, 1], epis[..., 0, 0] + epis[..., 1, 1]], dim=-1)
            uncs.append(u.cpu().numpy())
    out = {"pred": np.concatenate(preds), "label": np.concatenate(labels).astype(int), "toa": np.concatenate(toas)}
    if eval_uncertain:
        out["uncertainty"] = np.concatenate(uncs)
    mean_loss = {k: float(np.mean([r[k] for r in loss_rows])) for k in loss_rows[0]}
    return out, mean_loss


def compute_metrics(res, fps):
    m = {}
    u = evaluate_ustring_style(res["pred"], res["label"], res["toa"], fps=fps, fix_tp_bug=False)
    m.update({"AP_orig": u["AP"], "mTTA_orig": u["mTTA"], "TTA_R80_orig": u["TTA_R80"]})
    uf = evaluate_ustring_style(res["pred"], res["label"], res["toa"], fps=fps, fix_tp_bug=True)
    m.update({"AP_origfix": uf["AP"], "mTTA_origfix": uf["mTTA"]})
    e = evaluate_video_level(res["pred"], res["label"], res["toa"], fps)
    m.update({"AP_video": e.AP, "AP_trapz": e.AP_trapz, "mTTA_sec": e.mTTA, "TTA_R80_sec": e.TTA_R80,
              "mTTA_relative": e.mTTA_relative, "n_eval": int(len(res["label"]))})
    return m


def main(argv=None):
    a = parse_args(argv)
    out_dir = os.path.join(a.out_root, a.exp)
    os.makedirs(os.path.join(out_dir, "snapshot"), exist_ok=True)
    if not a.no_log_file:
        # tampil di layar (cmd/PowerShell) sekaligus ditambahkan ke stdout.log; jangan dialihkan lagi dengan >>
        log_fh = open(os.path.join(out_dir, "stdout.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_fh)
        sys.stderr = Tee(sys.__stderr__, log_fh)
    print("=== %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(sys.argv)), flush=True)
    try:  # cegah Windows sleep selama pelatihan (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); dilepas saat proses selesai
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:  # noqa: BLE001
        pass
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set, test_set = make_datasets(a)
    configure_for_model(a.model, train_set, val_set, test_set)
    loader_kw = dict(num_workers=a.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=a.num_workers > 0)
    eval_kw = dict(num_workers=a.eval_workers, pin_memory=device.type == "cuda",
                   persistent_workers=a.eval_workers > 0)
    # Generator sampler terpisah dari generator DataLoader (yang hanya dipakai untuk base_seed worker),
    # agar urutan data dapat direkonstruksi saat --resume. Satu tarikan awal meniru base_seed pada
    # versi sebelumnya (DataLoader(generator=g, shuffle=True)) → urutan epoch identik dengan run lama.
    sg = torch.Generator()
    sg.manual_seed(a.seed)
    torch.empty((), dtype=torch.int64).random_(generator=sg)
    gw = torch.Generator()
    gw.manual_seed(a.seed + 1)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, sampler=RandomSampler(train_set, generator=sg),
                              drop_last=True, generator=gw, **loader_kw)
    test_loader = DataLoader(test_set, batch_size=a.batch_size, shuffle=False, drop_last=not a.no_test_drop_last, **eval_kw)
    val_loader = (DataLoader(val_set, batch_size=a.batch_size, shuffle=False, drop_last=False, **eval_kw)
                  if val_set is not None else None)

    # model dibangun dengan seed tepat sebelum konstruksi (urutan RNG sama dengan asli)
    model = make_model(a, train_set).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=a.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 0
    resumed_exact_rng = False
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)  # snapshot penuh buatan train.py (memuat status RNG numpy); jangan --resume berkas tak tepercaya
        if "optimizer" not in ck:
            raise ValueError("%s hanya berisi bobot (checkpoint hemat); --resume butuh last.pth atau "
                             "checkpoint penuh" % a.resume)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        if "rng" in ck:  # checkpoint baru: semua state acak dipulihkan (lanjutan identik)
            torch.set_rng_state(ck["rng"]["torch"].cpu())
            if torch.cuda.is_available() and ck["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
            np.random.set_state(ck["rng"]["numpy"])
            sg.set_state(ck["rng"]["sampler"].cpu())
            resumed_exact_rng = True
        else:
            # checkpoint lama: urutan data direkonstruksi. RandomSampler (torch 2.3) memanggil randperm DUA
            # kali per epoch (urutan + sisa num_samples % n yang kosong, dieksekusi saat iterator habis).
            # RNG global (sampling BNN, dropout) tidak dapat dipulihkan → tidak identik bit per bit.
            for _ in range(2 * start_epoch):
                torch.randperm(len(train_set), generator=sg)

    config = vars(a).copy()
    config.update({"n_train": len(train_set), "n_val": len(val_set) if val_set else 0, "n_test": len(test_set),
                   "device": str(device), "torch": str(torch.__version__),
                   "n_params_trainable": sum(p.numel() for p in params),
                   "n_params_total": sum(p.numel() for p in model.parameters())})
    if a.resume:
        # config awal tidak ditimpa; catat setiap lanjutan secara terpisah
        config.update({"resumed_from_epoch": start_epoch - 1, "resumed_exact_rng": resumed_exact_rng})
        with open(os.path.join(out_dir, "config_resume_ep%02d.json" % start_epoch), "w") as f:
            json.dump(config, f, indent=2)
    else:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))

    sel_loader = val_loader if val_loader is not None else test_loader
    sel_name = "val" if val_loader is not None else "test"
    train_log_path = os.path.join(out_dir, "train_log.csv")
    eval_log_path = os.path.join(out_dir, "eval_log.csv")
    eval_rows = []
    best, best_epoch = -float("inf"), -1
    metrics = None
    if a.resume:
        # 1) salinan utuh log apa adanya sebelum dirapikan (tidak pernah ditimpa)
        bak = os.path.join(out_dir, "logs_before_resume_ep%02d_%s" % (start_epoch, time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(bak, exist_ok=True)
        for name in ("train_log.csv", "eval_log.csv", "stdout.log"):
            if os.path.exists(os.path.join(out_dir, name)):
                shutil.copy2(os.path.join(out_dir, name), os.path.join(bak, name))
        # 2) baris dari epoch yang terputus (atau baris rusak/terpotong) dipindah ke *_interrupted_epXX.csv;
        #    riwayat epoch yang selesai dipertahankan
        for path in (train_log_path, eval_log_path):
            if os.path.exists(path):
                rows = list(csv.DictReader(open(path, newline="")))
                keep, drop = [], []
                for r in rows:
                    try:
                        ok = int(r["epoch"]) < start_epoch and None not in r.values() and "" not in r.values()
                    except (TypeError, ValueError):
                        ok = False
                    (keep if ok else drop).append(r)
                if drop:
                    write_rows(path.replace(".csv", "_interrupted_ep%02d.csv" % start_epoch), drop)
                if rows:
                    write_rows(path, keep) if keep else os.remove(path)
                if path == eval_log_path:
                    eval_rows = [{k: (v if k == "split" else int(v) if k in ("epoch", "iter", "n_eval")
                                      else float(v)) for k, v in r.items()} for r in keep]
        # sama dengan loop: best = maks AP_orig evaluasi TERAKHIR tiap epoch yang selesai
        last_per_epoch = {}
        for r in eval_rows:
            last_per_epoch[int(r["epoch"])] = r
        if last_per_epoch:
            best_row = max(last_per_epoch.values(), key=lambda r: r[a.select_metric])
            best, best_epoch = best_row[a.select_metric], int(best_row["epoch"])
            metrics = last_per_epoch[max(last_per_epoch)]
        if not ck.get("scheduler_stepped", False):
            # checkpoint lama disimpan SEBELUM scheduler.step() epoch terakhirnya → terapkan langkah yang
            # tertinggal dengan log_posterior batch terakhir epoch itu (tercatat persis di train_log.csv)
            prev = [r for r in csv.DictReader(open(train_log_path, newline=""))
                    if int(r["epoch"]) == start_epoch - 1] if os.path.exists(train_log_path) else []
            if not prev:
                raise RuntimeError("train_log.csv tidak memuat epoch %d; langkah scheduler tidak bisa dipulihkan"
                                   % (start_epoch - 1))
            scheduler.step(float(prev[-1]["log_posterior"]))
            print("langkah scheduler epoch %d dipulihkan (log_posterior=%s); lr=%g"
                  % (start_epoch - 1, prev[-1]["log_posterior"], optimizer.param_groups[0]["lr"]))
        print("lanjut dari epoch %d; best %s sebelumnya %.4f; urutan data direkonstruksi: ya; "
              "RNG sampling BNN/dropout dipulihkan: %s%s"
              % (start_epoch, a.select_metric, best, "ya" if resumed_exact_rng else "tidak",
                 "" if resumed_exact_rng else " (checkpoint format lama)"))
    train_log = open(train_log_path, "a", newline="")
    tw = csv.writer(train_log)
    if os.path.getsize(train_log_path) == 0:
        tw.writerow(["epoch", "iter", "total_loss", "cross_entropy", "log_posterior", "log_prior", "auxloss",
                     "ranking", "lr", "sec_per_iter"])
    iter_cur = start_epoch * len(train_loader)
    epoch_secs = []

    def evaluate_and_log(epoch, it):
        res, mloss = run_inference(model, sel_loader, device, a)
        m = compute_metrics(res, test_set.fps)
        m["neg_loss_cross_entropy"] = -mloss["cross_entropy"]  # untuk seleksi/early stopping berbasis loss
        eval_rows.append({"epoch": epoch, "iter": it, "split": sel_name, **m,
                          **{"loss_" + k: v for k, v in mloss.items()}})
        print("[ep %d it %d] %s AP_orig=%.4f AP_video=%.4f mTTA_orig=%.3f mTTA_sec=%.3f" % (
            epoch, it, sel_name, m["AP_orig"], m["AP_video"], m["mTTA_orig"], m["mTTA_sec"]), flush=True)
        write_rows(eval_log_path, eval_rows)
        return m

    for epoch in range(start_epoch, a.epochs):
        model.train()
        t0 = time.time()
        print("--- epoch %d/%d mulai %s (lr %g)" % (epoch + 1, a.epochs, time.strftime("%H:%M"),
                                                   optimizer.param_groups[0]["lr"]), flush=True)
        for i, batch in enumerate(train_loader):
            if a.max_train_iters and i >= a.max_train_iters:
                break
            xs, ys, ge, ew, toa = to_device(batch[:5], device)
            optimizer.zero_grad(set_to_none=True)
            losses, _, _ = model(xs, ys, toa, ge, edge_weights=ew, npass=2, nbatch=len(train_loader),
                                 **det_kw(model, batch, device),
                                 eval_uncertain=True)
            losses["total_loss"] = total_loss(losses, a)
            losses["total_loss"].mean().backward()
            torch.nn.utils.clip_grad_norm_(params, 10)
            optimizer.step()
            iter_cur += 1
            dt = (time.time() - t0) / (i + 1)
            tw.writerow([epoch, iter_cur] + [float(torch.as_tensor(losses[k]).mean()) for k in
                        ("total_loss", "cross_entropy", "log_posterior", "log_prior", "auxloss", "ranking")]
                        + [optimizer.param_groups[0]["lr"], round(dt, 4)])
            if a.test_iter > 0 and iter_cur % a.test_iter == 0:
                metrics = evaluate_and_log(epoch, iter_cur)
                model.train()
        train_log.flush()
        if a.eval_epoch_end or metrics is None:  # evaluasi tepat di akhir epoch (atau debug: epoch sangat pendek)
            metrics = evaluate_and_log(epoch, iter_cur)
        # setia ke asli: scheduler di-step dengan log_posterior batch terakhir. Di-step SEBELUM checkpoint
        # disimpan agar state scheduler di checkpoint sudah memuat langkah epoch ini (penting untuk --resume).
        # DSTA asli men-step dengan total_loss batch terakhir (atribut model scheduler_key)
        scheduler.step(float(torch.as_tensor(losses[getattr(model, "scheduler_key", "log_posterior")]).mean()))
        model_only = {"epoch": epoch, "model": model.state_dict(), "config": config, "metrics_last_eval": metrics}
        full = dict(model_only, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scheduler_stepped=True,
                    rng={"torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                         "numpy": np.random.get_state(), "sampler": sg.get_state()})
        snap = os.path.join(out_dir, "snapshot")
        save_atomic(full, os.path.join(snap, "last.pth"))           # penuh: untuk --resume
        if a.keep_all_epochs:                                      # hemat: bobot saja (±11 MB) untuk evaluasi
            save_atomic(model_only, os.path.join(snap, "epoch_%02d.pth" % epoch))
        # setia ke asli (bawaan --select-metric AP_orig): pakai metrik evaluasi TERAKHIR dalam epoch ini
        if metrics[a.select_metric] > best:
            best, best_epoch = metrics[a.select_metric], epoch
            save_atomic(full, os.path.join(snap, "best.pth"))
        # format awal baris ini dibaca tools/progress.py: "epoch <i> selesai dalam <dtk> dtk"
        epoch_secs.append(time.time() - t0)
        left = (a.epochs - epoch - 1) * sum(epoch_secs) / len(epoch_secs)
        print("epoch %d selesai dalam %.1f dtk; best %s %s=%.4f (epoch %d) | AP_video %.4f mTTA_orig %.2f mTTA_sec %.2f"
              " | sisa %d epoch, perkiraan selesai %s (%.1f jam lagi)"
              % (epoch, epoch_secs[-1], sel_name, a.select_metric, best, best_epoch, metrics["AP_video"],
                 metrics["mTTA_orig"], metrics["mTTA_sec"], a.epochs - epoch - 1,
                 time.strftime("%d-%m %H:%M", time.localtime(time.time() + left)), left / 3600),
              flush=True)
        if a.early_stop_patience > 0 and epoch - best_epoch >= a.early_stop_patience:
            print("early stopping: %s tidak membaik selama %d epoch (terbaik epoch %d = %.4f)"
                  % (a.select_metric, a.early_stop_patience, best_epoch, best), flush=True)
            break
    train_log.close()

    # evaluasi akhir di set uji untuk last dan best (prediksi disimpan)
    for tag in ("last", "best"):
        ck = torch.load(os.path.join(out_dir, "snapshot", tag + ".pth"), map_location=device, weights_only=False)  # snapshot buatan run ini (status RNG numpy)
        model.load_state_dict(ck["model"])
        res, _ = run_inference(model, test_loader, device, a, eval_uncertain=True)
        m = compute_metrics(res, test_set.fps)
        m["mAU"], m["mEU"] = [float(v) for v in res["uncertainty"].mean(axis=(0, 1))]
        m["epoch"] = ck["epoch"]
        np.savez(os.path.join(out_dir, "pred_test_%s.npz" % tag), **res)
        with open(os.path.join(out_dir, "final_test_%s.json" % tag), "w") as f:
            json.dump(m, f, indent=2)
        print("FINAL test (%s, epoch %d): %s" % (tag, ck["epoch"], json.dumps(m)))


class Tee:
    """Tulis ke konsol dan file sekaligus (pengganti `tee`, juga jalan di cmd)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        for t in (self.stream, self.fh):
            try:
                t.write(s)
            except (ValueError, OSError):
                pass
        return len(s)

    def flush(self):
        for t in (self.stream, self.fh):
            try:
                t.flush()
            except (ValueError, OSError):
                pass

    def isatty(self):
        return False


def save_atomic(obj, path):
    """Tulis ke file sementara lalu ganti nama: file lama tetap utuh bila proses mati saat menyimpan."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_rows(path, rows):
    """Tulis CSV secara atomik (file lama tetap utuh bila proses mati saat menulis)."""
    keys = list(rows[0].keys())
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
