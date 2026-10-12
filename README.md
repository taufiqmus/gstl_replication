# gstl_replication

Reimplementasi **GSTL** (Mahmood, Jeong, Ryu 2023) untuk antisipasi kecelakaan dari dashcam, dengan pelatihan, evaluasi, dan bobot yang dapat dijalankan ulang.

> **Status yang jujur:** ini reimplementasi dari paper; penulis tidak merilis kode. Pada CCD, AP dekat dengan paper tetapi mTTA belum cocok. Rincian, asumsi, dan yang belum dikerjakan ada di [`REPLIKASI.md`](REPLIKASI.md).

*(English, one line: a from-paper reimplementation of GSTL, "A New Approach to Traffic Accident Anticipation With Geometric Features for Better Generalizability", IEEE Access 11:29263-29274, DOI 10.1109/ACCESS.2023.3259992, with train/eval scripts and a checkpoint; AP on CCD is close to the paper, mTTA is not yet matched.)*

## Hasil (CCD, set uji 900 video)

Tabel ini dibuat otomatis dari `results/ccd_test_*.csv` oleh `tools/make_results_table.py`.

<!-- TABEL:BEGIN -->
| hasil | n evaluasi | AP asli (%) | mTTA asli (s) | AP video eksak (%) | mTTA detik sungguhan |
|---|---|---|---|---|---|
| **Paper (Tabel 9, CCD)** | | 96.5 | 4.68 | | |
| epoch10_seed123 | 1 | 96.13 | 4.17 | 96.25 | 2.25 |
<!-- TABEL:END -->

- "AP asli" dan "mTTA asli" memakai metrik gaya kode UString (yang dipakai paper). "AP video eksak" dan "mTTA detik sungguhan" adalah versi yang kami tulis sendiri (`collision_eval/`), lebih stabil dan memakai detik sungguhan.
- Bobot di `weights/` berasal dari satu pelatihan (seed 123). Satu seed belum cukup untuk menyatakan selisih 0,4 poin AP sebagai nyata; lihat `REPLIKASI.md`.

## Mulai cepat

```bash
# 1. lingkungan (Python 3.10; diuji torch 2.3.0 + CUDA 11.8, numpy 1.26.4)
conda env create -f environment.yml && conda activate gstl

# 2. tes unit (tanpa data; ±1 detik)
python -m unittest discover -s tests

# 3. data CCD (TIDAK disertakan): unduh dari penulis CCD, lihat tautan Google Drive di README resmi CCD,
#    lalu arahkan ke folder yang berisi vgg16_features/ dan videos/
export CCD_ROOT=/path/ke/CarCrash          # Windows PowerShell: $env:CCD_ROOT = "D:\data\CarCrash"

# 4. evaluasi bobot yang disertakan (±1 menit di GPU)
python evaluate.py --tag ccd_test_epoch10_seed123

# 5. latih ulang (±5 menit per epoch pada RTX 3060; early stopping biasanya berhenti ±epoch 20)
python train.py --dataset crash --exp gstl_s123 --epochs 70 --val-frac 0.1 --test-iter 0 \
    --eval-epoch-end --select-metric AP_video --early-stop-patience 10 --keep-all-epochs --seed 123
python evaluate.py --exp-dir outputs/gstl_s123 --tag ccd_test_gstl_s123_allepochs
```

Evaluasi memakai fitur yang sudah dihitung dan dirilis bersama CCD (VGG16 + kotak deteksi), sehingga detektor tidak perlu dipasang.

### Uji asap tanpa CCD
`python tools/make_synthetic_ccd.py --out <folder> --n-pos 12 --n-neg 24` membuat data sintetis berformat CCD (bukan data sungguhan), cukup untuk memastikan `train.py` dan `evaluate.py` berjalan. Pada set sintetis kecil tambahkan `--no-test-drop-last` (train) atau `--no-drop-last` (evaluate).

## Isi repo

| Jalur | Isi |
|---|---|
| `gstl/` | model (GSTL di atas port UString), lapisan graf/Bayesian, dataset fitur CCD/DAD |
| `collision_eval/` | metrik (gaya UString, termasuk bug AP>1 yang direproduksi, dan versi eksak) |
| `train.py`, `evaluate.py` | pelatihan dan evaluasi |
| `weights/` | bobot GSTL (bobot saja) + asal dan hash |
| `results/` | hasil evaluasi (CSV/JSON), semua epoch, log pelatihan, nilai acuan paper |
| `docs/spec_gstl.md` | spesifikasi dari paper, asumsi, pertanyaan terbuka |
| `tests/` | tes unit GSTL |

## Verifikasi porting
Evaluasi bobot epoch 10 di repo ini menghasilkan **12 dari 12 metrik identik** (selisih 0) dengan repo kerja asal. Itu syarat lulus pemisahan kode; ia membuktikan bahwa pemindahan tidak mengubah perilaku, bukan bahwa implementasi sama dengan kode penulis.

## Atribusi dan lisensi
Lihat [`NOTICE.md`](NOTICE.md). Kode dasar dari UString (MIT, `LICENSE_UString`). Data CCD tidak didistribusikan di sini. Lisensi kode baru belum ditetapkan.
