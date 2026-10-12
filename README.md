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

# 2. periksa lingkungan, bobot, dan (bila ada) data/detektor; tidak mengubah apa pun
python tools/check_setup.py

# 2b. tes unit (29 tes: GSTL, metrik, visualisasi demo, klip contoh; tanpa data; ±3 detik)
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

## Demo pada video dashcam sendiri

`demo.py` mengubah video mentah menjadi skor risiko per frame (CSV, grafik, dan video overlay berisi kotak deteksi + bilah skor):

```bash
pip install -r requirements-demo.txt          # torchvision, opencv-python-headless, Pillow, matplotlib
# bobot detektor (±1,0 GB; TIDAK disertakan): folder "Cascade R-CNN" di Drive penulis UString:
#   https://drive.google.com/drive/folders/1vUz3HYoAFeu0qGQy7pe51oa41zd0OCQU
#   berkas work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_kitti2d/latest.pth, SHA-256 C768907093DB3B5BC87C1D79E728EEBB6F90252017D7BCB2DF0D5B34F2BD60BF
python tools/check_setup.py --detector <latest.pth> --hash-detector     # periksa hash SEBELUM memakainya
python demo.py --video contoh.mp4 --detector <latest.pth> --threshold 0.5
```

Satu baris (cmd Windows, dijalankan dari direktori repo; ganti `<latest.pth>` dengan path bobot detektor):
```bat
python -W ignore demo.py --video "samples\dota_pos_naik__eN6y6-4C1Rc_000726.avi" --detector "<latest.pth>" --weights "weights\gstl_ccd_seed123_epoch10.pth" --out "outputs\demo\dota_pos_naik" --viz full --uncertainty --html --threshold 0.5 --box-conf 0.5 --curve-seconds 30 --width 960 --seed 123 --device cuda
```
Hasil di `outputs\demo\dota_pos_naik\` (`overlay.mp4`, `laporan.html`, `skor.png`, `scores.csv`). Pastikan `python` milik env yang benar (`python -c "import torch; print(torch.cuda.is_available())"` harus `True`); jangan memakai `py` di Windows, karena peluncur itu mengabaikan env conda.

- Keluaran di `outputs/demo/<nama video>/`: `scores.csv`, `skor.png`, `overlay.mp4`, dan `laporan.html` (dengan `--html`). Kecepatan ±0,3 detik per frame di RTX 3060 (jendela 50 frame ±14 detik).
- **Opsi visualisasi** (`--viz`, daftar dipisah koma atau preset): `boxes` kotak deteksi (warna per kelas), `bar` bilah skor, `curve` grafik kemungkinan tabrakan yang bergerak di bawah video, `glow` bingkai berwarna sesuai risiko, `alert` banner "RISIKO TINGGI" dan waktu peringatan pertama. Preset: `minimal` (bawaan: boxes,bar), `full` (semua), `clean` (video tanpa tambahan), `none` (tanpa video). Opsi lain: `--threshold`, `--box-conf`, `--curve-seconds`, `--width`, `--uncertainty` (ketidakpastian aleatorik/epistemik di grafik dan CSV), `--html` (laporan HTML mandiri: video tersinkron dengan grafik, klik grafik untuk lompat waktu, penggeser ambang), `--no-plot`.
  ```bash
  python demo.py --video contoh.mp4 --detector <latest.pth> --viz full --uncertainty --html
  ```
- "Kemungkinan tabrakan" = **satu skor per frame** dari model (probabilitas kelas kecelakaan, belum terkalibrasi), bukan risiko per objek.
- **Video di peramban:** OpenCV hanya menulis `mp4v`, yang tidak diputar peramban. Bila `ffmpeg` ada di PATH, `overlay.mp4` otomatis diubah ke H.264; bila tidak, demo memberi peringatan (putar dengan VLC atau pasang ffmpeg). Berkas `laporan.html` paling aman dibuka langsung dari folder keluaran (video dan HTML satu folder).
- VGG16 ImageNet (±528 MB) diunduh otomatis oleh torchvision pada pemakaian pertama.
- **Keamanan:** `latest.pth` dimuat dengan `torch.load` (unpickle). Gunakan hanya berkas dari penulis dan pastikan hash cocok.
- **Artefak jendela:** tiap jendela 50 frame diskor terpisah (keadaan GRU di-reset), jadi skor rendah pada ±1 detik pertama tiap jendela lalu naik; pada video panjang grafik turun mendadak tiap 5 detik. Itu bukan perubahan risiko, dan "peringatan pertama" di awal jendela tidak berarti deteksi kejadian.
- **Keterbatasan penting** (rinci di `docs/demo_validasi.md` dan docstring `demo.py`): detektor tanpa kelas motor; model dilatih hanya pada CCD 1280x720 sehingga video lain di luar distribusi latih; skor bukan risiko terkalibrasi. Pada video negatif CCD, pipeline demo menghasilkan skor yang hampir identik dengan fitur rilis (korelasi per frame median 0,999); pada video lain kesesuaian **tidak diuji**.

### Klip contoh untuk uji visualisasi
`samples/` berisi 9 klip pendek (±14 MB; positif dan negatif, termasuk kasus gagal) berikut hash, manifest, dan skor acuan. Satu perintah menjalankan demo pada semuanya dan membandingkan skor dengan acuan:
```bash
python tools/run_samples.py --detector <latest.pth> --viz full --html     # hasil per klip di outputs/samples/<role>/
```
Rincian dan **catatan hak cipta** (klip berasal dari video YouTube; hapus `samples/` DAN riwayat Git-nya sebelum repo dibagikan keluar, lihat `samples/README.md`) ada di `samples/README.md`.

### Uji asap tanpa CCD
`python tools/make_synthetic_ccd.py --out <folder> --n-pos 12 --n-neg 24` membuat data sintetis berformat CCD (bukan data sungguhan), cukup untuk memastikan `train.py` dan `evaluate.py` berjalan. Pada set sintetis kecil tambahkan `--no-test-drop-last` (train) atau `--no-drop-last` (evaluate).

## Isi repo

| Jalur | Isi |
|---|---|
| `gstl/` | model (GSTL di atas port UString), lapisan graf/Bayesian, dataset fitur CCD/DAD, detektor Cascade R-CNN (PyTorch murni) dan ekstraksi fitur |
| `collision_eval/` | metrik (gaya UString, termasuk bug AP>1 yang direproduksi, dan versi eksak) |
| `train.py`, `evaluate.py` | pelatihan dan evaluasi |
| `demo.py` | video mentah -> skor risiko per frame (butuh bobot detektor, lihat bagian Demo) |
| `tools/check_setup.py` | pemeriksa lingkungan/bobot/data/detektor |
| `tools/validate_demo.py` | validasi pipeline demo terhadap fitur CCD rilis |
| `tools/run_samples.py`, `samples/` | klip contoh dan uji regresi skor/visualisasi |
| `weights/` | bobot GSTL (bobot saja) + asal dan hash |
| `results/` | hasil evaluasi (CSV/JSON), semua epoch, log pelatihan, nilai acuan paper |
| `docs/spec_gstl.md` | spesifikasi dari paper, asumsi, pertanyaan terbuka |
| `tests/` | 29 tes unit (GSTL, metrik, visualisasi demo, klip contoh) |

## Verifikasi porting
Evaluasi bobot epoch 10 di repo ini menghasilkan **12 dari 12 metrik identik** (selisih 0) dengan repo kerja asal. Itu syarat lulus pemisahan kode; ia membuktikan bahwa pemindahan tidak mengubah perilaku, bukan bahwa implementasi sama dengan kode penulis.

## Atribusi dan lisensi
Lihat [`NOTICE.md`](NOTICE.md). Kode dasar dari UString (MIT, `LICENSE_UString`). Data CCD tidak didistribusikan di sini. Lisensi kode baru belum ditetapkan.
