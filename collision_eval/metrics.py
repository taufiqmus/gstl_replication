"""Metrik antisipasi kecelakaan (AP, mTTA, TTA@R80, CD, ketidakpastian).

Konvensi masukan (sama untuk semua fungsi):
    all_pred : (N, T) skor/probabilitas kecelakaan per frame, nilai di [0, 1].
    labels   : (N,) 1 = video kecelakaan (positif), 0 = normal.
    toas     : (N,) indeks frame awal kecelakaan (0-based). Untuk video positif,
               hanya frame [0, toa) yang dievaluasi (frame sesudah kecelakaan
               tidak dihitung). Untuk video negatif nilainya diabaikan.
    fps      : frame per detik video fitur (CCD 10, DAD 20).

Dua keluarga fungsi:
  * `evaluate_ustring_style` : reimplementasi setia `src/eval_tools.py::evaluation`
    milik UString (sama dengan DSA/DSTA). Dipakai untuk membandingkan angka dengan
    paper. Opsi `fix_tp_bug` menutup bug "AP > 1" (lihat docstring).
  * `evaluate_video_level`   : versi terkoreksi dan eksak (ambang = skor unik,
    TTA dalam detik sungguhan). Dipakai sebagai protokol utama milik kita.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


# ---------------------------------------------------------------------------
# utilitas
# ---------------------------------------------------------------------------
def _check_inputs(all_pred, labels, toas):
    all_pred = np.asarray(all_pred, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    toas = np.asarray(toas, dtype=np.float64).reshape(-1)
    if all_pred.ndim != 2:
        raise ValueError("all_pred harus (N, T), didapat %s" % (all_pred.shape,))
    n = all_pred.shape[0]
    if labels.shape[0] != n or toas.shape[0] != n:
        raise ValueError("panjang labels/toas tidak sama dengan N")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels harus 0/1")
    if np.isnan(all_pred).any():
        raise ValueError("all_pred mengandung NaN")
    pos_toa = toas[labels == 1]
    if (pos_toa < 1).any():
        raise ValueError("toa video positif harus >= 1 (minimal satu frame dievaluasi)")
    return all_pred, labels, toas


def eval_segments(all_pred, labels, toas):
    """Daftar segmen skor yang dievaluasi: positif [:toa], negatif seluruh frame."""
    all_pred, labels, toas = _check_inputs(all_pred, labels, toas)
    segs = []
    for p, y, toa in zip(all_pred, labels, toas):
        segs.append(p[: int(toa)] if y > 0 else p)
    return segs


def video_scores(all_pred, labels, toas):
    """Skor per video = maksimum skor frame pada segmen yang dievaluasi."""
    return np.array([s.max() for s in eval_segments(all_pred, labels, toas)])


# ---------------------------------------------------------------------------
# 1) Reimplementasi setia UString / DSA
# ---------------------------------------------------------------------------
def evaluate_ustring_style(all_pred, labels, toas, fps=20.0, fix_tp_bug=True, step=0.001):
    """Reimplementasi baris-per-baris `evaluation()` dari UString (commit 0ef5e9c).

    Perilaku asli yang DIPERTAHANKAN (agar angka sebanding dengan paper):
      * ambang = grid np.arange(max(min_pred, 0), 1.0, 0.001);
      * "TP" per video = ada frame >= ambang sebelum toa; presisi/recall tingkat video;
      * per recall unik diambil presisi maks dan TTA maks; AP = trapesium;
      * TTA relatif: (1 - t/toa), lalu dikali panjang klip penuh T/fps (BUKAN (toa - t)/fps).
        Untuk CCD (T=50, toa < 50) ini menggelembungkan TTA dengan faktor T/toa >= 1.
      * level recall TERTINGGI memakai satu elemen hasil argsort tak-stabil
        (`new_Time[-1] = Time[rep_index[-1]]`), bukan maks seperti level lain;
      * kode asli crash (IndexError) bila total frame < jumlah ambang grid (hanya data sangat kecil).

    Bug asli yang bisa DITUTUP dengan `fix_tp_bug=True`:
      kode asli menghitung TP sebagai `pred * label >= Th`. Untuk video negatif
      (label 0) hasilnya 0, sehingga pada Th = 0 (terjadi bila ada skor tepat 0,
      mis. softmax underflow) semua video negatif ikut dihitung TP → recall > 1 →
      "AP larger than 1" (isu UString #28). Versi perbaikan mensyaratkan label = 1.
      Dengan `fix_tp_bug=False` hasilnya identik dengan kode asli.

    Mengembalikan dict: AP, mTTA (detik, gaya UString), TTA_R80 (detik, gaya UString).
    """
    all_pred, labels_arr, toas_arr = _check_inputs(all_pred, labels, toas)
    preds_eval = eval_segments(all_pred, labels_arr, toas_arr)
    n_frames = int(sum(len(p) for p in preds_eval))
    min_pred = min(float(np.min(p)) for p in preds_eval)
    total_seconds = all_pred.shape[1] / fps
    n_pos = float(np.sum(labels_arr))

    grid = np.arange(max(min_pred, 0), 1.0, step)
    # Asli: ukuran n_frames → IndexError bila n_frames < len(grid). Entri sisa bernilai Recall=0
    # (tak pernah diisi; entri ber-Recall 0 selalu di-`continue`) dan dibuang oleh rep_index[1:],
    # jadi memperbesar array tidak mengubah hasil.
    size = max(n_frames, len(grid))
    Precision = np.zeros(size)
    Recall = np.zeros(size)
    Time = np.zeros(size)
    cnt = 0
    for Th in grid:
        Tp = Tp_Fp = time = counter = 0.0
        for i in range(len(preds_eval)):
            if fix_tp_bug:
                tp = np.where(preds_eval[i] >= Th)[0] if labels_arr[i] > 0 else np.empty(0, dtype=np.int64)
            else:
                tp = np.where(preds_eval[i] * labels_arr[i] >= Th)[0]
            if len(tp) > 0:
                Tp += 1.0
                time += tp[0] / float(toas_arr[i])
                counter += 1
            Tp_Fp += float(len(np.where(preds_eval[i] >= Th)[0]) > 0)
        if Tp_Fp == 0:
            continue
        Precision[cnt] = Tp / Tp_Fp
        if n_pos == 0:
            continue
        Recall[cnt] = Tp / n_pos
        if counter == 0:
            continue
        Time[cnt] = 1 - time / counter
        cnt += 1

    new_index = np.argsort(Recall)
    Precision, Recall, Time = Precision[new_index], Recall[new_index], Time[new_index]
    _, rep_index = np.unique(Recall, return_index=True)
    rep_index = rep_index[1:]
    if len(rep_index) == 0:
        return {"AP": 0.0, "mTTA": 0.0, "TTA_R80": 0.0}
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]

    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    mTTA = float(np.mean(new_Time) * total_seconds)
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = float(sort_time[np.argmin(np.abs(sort_recall - 0.8))] * total_seconds)
    return {"AP": float(AP), "mTTA": mTTA, "TTA_R80": TTA_R80}


# ---------------------------------------------------------------------------
# 2) Versi terkoreksi dan eksak
# ---------------------------------------------------------------------------
@dataclass
class VideoLevelResult:
    AP: float            # AP tingkat video, definisi langkah (sama dengan sklearn average_precision_score)
    AP_trapz: float      # AP trapesium atas kurva P-R (presisi maks per recall), konvensi DSA
    mTTA: float          # detik; rerata atas level recall unik dari TTA maks per level
    TTA_R80: float       # detik; TTA pada level recall terdekat ke 0,8
    mTTA_relative: float  # sama dengan mTTA tetapi dihitung gaya UString (1 - t/toa) * T/fps
    n_pos: int
    n_neg: int

    def as_dict(self):
        return asdict(self)


def evaluate_video_level(all_pred, labels, toas, fps):
    """Evaluasi eksak tanpa grid ambang.

    Untuk tiap ambang th (= setiap skor video unik, dari besar ke kecil):
      video "menyala" bila ada frame >= th pada segmen evaluasinya;
      TP = positif yang menyala, FP = negatif yang menyala;
      P = TP/(TP+FP), R = TP/N_pos;
      TTA_i(th) = (toa_i - t_i(th)) / fps, t_i = frame pertama >= th (positif yang menyala).
    AP (langkah) = sum_k (R_k - R_{k-1}) * P_k.
    mTTA = rerata atas recall unik (>0) dari TTA maks pada recall itu (agregasi sama dengan UString).
    """
    all_pred, labels, toas = _check_inputs(all_pred, labels, toas)
    segs = eval_segments(all_pred, labels, toas)
    scores = np.array([s.max() for s in segs])
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0:
        raise ValueError("tidak ada video positif; AP tidak terdefinisi")
    total_seconds = all_pred.shape[1] / fps
    pos_idx = np.where(labels == 1)[0]

    thresholds = np.unique(scores)[::-1]  # turun
    P, R, TTA, TTArel = [], [], [], []
    for th in thresholds:
        fired = scores >= th
        tp = int(np.sum(fired & (labels == 1)))
        fp = int(np.sum(fired & (labels == 0)))
        P.append(tp / (tp + fp))
        R.append(tp / n_pos)
        if tp > 0:
            ttas, rels = [], []
            for i in pos_idx:
                if fired[i]:
                    t_first = int(np.argmax(segs[i] >= th))
                    ttas.append((toas[i] - t_first) / fps)
                    rels.append((1.0 - t_first / toas[i]) * total_seconds)
            TTA.append(float(np.mean(ttas)))
            TTArel.append(float(np.mean(rels)))
        else:
            TTA.append(0.0)
            TTArel.append(0.0)
    P, R, TTA, TTArel = map(np.asarray, (P, R, TTA, TTArel))

    # AP langkah
    r_prev = 0.0
    ap = 0.0
    for p, r in zip(P, R):
        ap += (r - r_prev) * p
        r_prev = r

    # agregasi per recall unik (> 0)
    uniq_r = np.unique(R[R > 0])
    p_max = np.array([P[R == r].max() for r in uniq_r])
    t_max = np.array([TTA[R == r].max() for r in uniq_r])
    trel_max = np.array([TTArel[R == r].max() for r in uniq_r])
    ap_trapz = p_max[0] * uniq_r[0]
    for i in range(1, len(uniq_r)):
        ap_trapz += (p_max[i - 1] + p_max[i]) * (uniq_r[i] - uniq_r[i - 1]) / 2
    tta_r80 = float(t_max[np.argmin(np.abs(uniq_r - 0.8))])

    return VideoLevelResult(AP=float(ap), AP_trapz=float(ap_trapz), mTTA=float(t_max.mean()),
                            TTA_R80=tta_r80, mTTA_relative=float(trel_max.mean()),
                            n_pos=n_pos, n_neg=n_neg)


def average_precision_video(scores, labels):
    """AP tingkat video dari skor per video (definisi langkah, setara sklearn)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    n_pos = labels.sum()
    if n_pos == 0:
        raise ValueError("tidak ada positif")
    ap, r_prev = 0.0, 0.0
    for th in np.unique(scores)[::-1]:
        fired = scores >= th
        tp = np.sum(fired & (labels == 1))
        p = tp / fired.sum()
        r = tp / n_pos
        ap += (r - r_prev) * p
        r_prev = r
    return float(ap)


# ---------------------------------------------------------------------------
# 3) Metrik lain
# ---------------------------------------------------------------------------
def mean_uncertainties(all_uncertains):
    """mAU, mEU: rerata ketidakpastian aleatorik/epistemik atas video dan frame.

    all_uncertains: (N, T, 2) dengan [..., 0] = trace aleatorik, [..., 1] = trace epistemik
    (format keluaran `test_all_vis` UString).
    """
    u = np.asarray(all_uncertains, dtype=np.float64)
    m = u.mean(axis=(0, 1))
    return {"mAU": float(m[0]), "mEU": float(m[1])}


def cross_dataset_cd(self_score, other_scores, scale=100.0):
    """Cross-dataset Discrepancy (Mahmood 2023): CD = 1 / (1 + exp(-(Self - mean(Others)) / scale)).

    Skor dalam persen (mis. AP 99,5). CD mendekati 0,5 = lebih tergeneralisasi.
    Dicek terhadap Mahmood 2023 pers. (8) dan Tabel 11 (AP dalam persen), 2026-10-05.
    """
    others = np.asarray(other_scores, dtype=np.float64)
    return 1.0 / (1.0 + math.exp(-(float(self_score) - float(others.mean())) / scale))
