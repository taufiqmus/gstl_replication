"""Uji metrik (collision_eval) yang berdiri sendiri. Jalankan: python -m unittest discover -s tests -v.

Disalin dari repo kerja (models/collision_eval/tests) tanpa bagian yang membandingkan dengan kode UString asli
(`eval_tools.py`), yang tidak disertakan di repo ini. Bug "AP > 1" tetap diuji lewat opsi `fix_tp_bug`.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from collision_eval.metrics import (  # noqa: E402
    average_precision_video, cross_dataset_cd, evaluate_ustring_style, evaluate_video_level,
    mean_uncertainties, video_scores,
)


def synth(rng, n_pos=30, n_neg=60, T=50, fps=10.0, sep=1.0, zeros_frac=0.0, toa_min=30):
    """Data sintetis bergaya CCD: positif naik menjelang toa, negatif acak."""
    N = n_pos + n_neg
    labels = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int)
    toas = np.where(labels == 1, rng.integers(toa_min, T, size=N), T + 1).astype(float)
    t = np.arange(T)[None, :]
    base = rng.normal(0, 1, size=(N, T))
    ramp = np.where(labels[:, None] == 1, sep * 3 * np.clip((t - (toas[:, None] - 20)) / 20, 0, None), 0)
    pred = 1 / (1 + np.exp(-(base + ramp - 1.0)))
    if zeros_frac > 0:
        mask = rng.random(pred.shape) < zeros_frac
        pred[mask] = 0.0
    return pred.astype(np.float32), labels, toas, fps


class TestVideoLevel(unittest.TestCase):
    def test_perfect_separation_ap_is_one(self):
        T, fps = 50, 10.0
        pred = np.zeros((4, T))
        pred[0, :40] = 0.9
        pred[1, :35] = 0.8
        pred[2] = 0.1
        pred[3] = 0.2
        labels = [1, 1, 0, 0]
        toas = [40, 35, 51, 51]
        r = evaluate_video_level(pred, labels, toas, fps)
        self.assertAlmostEqual(r.AP, 1.0)
        self.assertAlmostEqual(r.AP_trapz, 1.0)

    def test_frames_after_toa_ignored(self):
        T, fps = 50, 10.0
        pred = np.full((2, T), 0.1)
        labels, toas = [1, 0], [30, 51]
        pred[1] = 0.5
        before = evaluate_video_level(pred, labels, toas, fps)
        pred[0, 30:] = 1.0  # skor tinggi SETELAH kecelakaan tidak boleh membantu
        after = evaluate_video_level(pred, labels, toas, fps)
        self.assertEqual(before.AP, after.AP)
        self.assertAlmostEqual(after.AP, 0.5)  # positif di bawah negatif → P=1/2 pada R=1

    def test_tta_seconds_hand_computed(self):
        # satu positif, toa = 40, skor melompat di frame 20 → TTA = (40-20)/10 = 2,0 dtk.
        # Negatif = 0,5 agar ambang terendah (0,5) tetap di atas skor awal positif (0,05).
        T, fps = 50, 10.0
        pred = np.full((2, T), 0.05)
        pred[0, 20:] = 0.9
        pred[1] = 0.5
        r = evaluate_video_level(pred, [1, 0], [40, 51], fps)
        self.assertAlmostEqual(r.mTTA, 2.0)
        # gaya UString: (1 - 20/40) * 50/10 = 2,5 dtk (digelembungkan faktor T/toa)
        self.assertAlmostEqual(r.mTTA_relative, 2.5)

    def test_mtta_takes_max_tta_per_recall_level(self):
        """Konvensi DSA/UString: per level recall diambil TTA MAKS, dari ambang mana pun
        (termasuk ambang rendah berpresisi buruk). Di sini recall=1 dicapai pada th=0,9
        (TTA 2 dtk) dan th=0,01 (positif menyala di frame 0 → TTA 4 dtk) → mTTA = 4."""
        T, fps = 50, 10.0
        pred = np.full((2, T), 0.05)
        pred[0, 20:] = 0.9
        pred[1] = 0.01
        r = evaluate_video_level(pred, [1, 0], [40, 51], fps)
        self.assertAlmostEqual(r.mTTA, 4.0)

    def test_ap_matches_bruteforce_mean_precision(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(5, 60))
            labels = rng.integers(0, 2, size=n)
            labels[0] = 1
            scores = np.round(rng.random(n), 1)  # sengaja banyak seri
            # AP = rerata atas positif dari presisi pada ambang = skornya
            ref = np.mean([np.mean(labels[scores >= s]) for s in scores[labels == 1]])
            self.assertAlmostEqual(average_precision_video(scores, labels), ref, places=12)

    def test_evaluate_video_level_ap_equals_ap_of_max_scores(self):
        rng = np.random.default_rng(1)
        pred, labels, toas, fps = synth(rng)
        r = evaluate_video_level(pred, labels, toas, fps)
        self.assertAlmostEqual(r.AP, average_precision_video(video_scores(pred, labels, toas), labels), places=12)

    def test_bounds_random(self):
        rng = np.random.default_rng(2)
        for k in range(100):
            pred, labels, toas, fps = synth(rng, n_pos=int(rng.integers(1, 20)), n_neg=int(rng.integers(0, 30)),
                                            sep=float(rng.uniform(-1, 2)), zeros_frac=0.05 * (k % 3))
            r = evaluate_video_level(pred, labels, toas, fps)
            for v in (r.AP, r.AP_trapz):
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0 + 1e-12)
            self.assertGreaterEqual(r.mTTA, 0.0)
            self.assertLessEqual(r.mTTA, pred.shape[1] / fps)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            evaluate_video_level(np.zeros((2, 5)), [1, 0], [0, 6], 10.0)  # toa positif = 0
        with self.assertRaises(ValueError):
            evaluate_video_level(np.zeros((2, 5)), [2, 0], [3, 6], 10.0)


class TestUStringStyle(unittest.TestCase):
    def test_original_style_ap_exceeds_one_when_scores_contain_zero(self):
        """Mereproduksi isu 'AP larger than 1' (UString #28): ada skor tepat 0 -> ambang 0 -> negatif dihitung TP.
        fix_tp_bug=False meniru perilaku kode asli; fix_tp_bug=True mensyaratkan label = 1."""
        rng = np.random.default_rng(4)
        pred, labels, toas, fps = synth(rng, n_pos=10, n_neg=40, sep=1.5)
        pred[-1, 0] = 0.0  # cukup satu frame bernilai 0 di video negatif
        ap_orig = evaluate_ustring_style(pred, labels, toas, fps=fps, fix_tp_bug=False)["AP"]
        ap_fixed = evaluate_ustring_style(pred, labels, toas, fps=fps, fix_tp_bug=True)["AP"]
        self.assertGreater(ap_orig, 1.0)
        self.assertLessEqual(ap_fixed, 1.0)

    def test_fixed_equals_unfixed_when_no_zero_scores(self):
        """Tanpa skor nol, perbaikan tidak mengubah angka (bug hanya aktif pada ambang 0)."""
        rng = np.random.default_rng(5)
        pred, labels, toas, fps = synth(rng, n_pos=15, n_neg=30)
        pred = np.clip(pred, 1e-4, 1)
        a = evaluate_ustring_style(pred, labels, toas, fps=fps, fix_tp_bug=False)
        b = evaluate_ustring_style(pred, labels, toas, fps=fps, fix_tp_bug=True)
        self.assertAlmostEqual(a["AP"], b["AP"], places=12)
        self.assertAlmostEqual(a["mTTA"], b["mTTA"], places=12)

    def test_ustring_style_close_to_exact_trapz(self):
        """Grid 0,001 vs ambang eksak: AP trapesium harus berdekatan (bukan identik)."""
        rng = np.random.default_rng(6)
        pred, labels, toas, fps = synth(rng, n_pos=40, n_neg=80, sep=0.7)
        pred = np.clip(pred, 1e-4, 1)
        a = evaluate_ustring_style(pred, labels, toas, fps=fps)["AP"]
        b = evaluate_video_level(pred, labels, toas, fps).AP_trapz
        self.assertLess(abs(a - b), 0.02)


class TestOther(unittest.TestCase):
    def test_uncertainty_mean(self):
        u = np.zeros((2, 3, 2))
        u[..., 0] = 0.2
        u[..., 1] = 0.4
        m = mean_uncertainties(u)
        self.assertAlmostEqual(m["mAU"], 0.2)
        self.assertAlmostEqual(m["mEU"], 0.4)

    def test_cd(self):
        self.assertAlmostEqual(cross_dataset_cd(50, [50, 50]), 0.5)
        self.assertGreater(cross_dataset_cd(90, [50]), 0.5)

    def test_cd_matches_mahmood_table11(self):
        """Mahmood 2023 Tabel 11 (AP dalam persen): Self = AP in-dataset, Others = AP lintas dataset."""
        self.assertAlmostEqual(cross_dataset_cd(63.5, [60.2, 55.6]), 0.514, places=3)  # GSTL latih DAD
        self.assertAlmostEqual(cross_dataset_cd(99.6, [43.7, 39.4]), 0.641, places=3)  # DSTA latih CCD
        self.assertAlmostEqual(cross_dataset_cd(65.8, [57.5, 51.8]), 0.527, delta=0.0015)  # GCRNN latih DAD


if __name__ == "__main__":
    unittest.main()
