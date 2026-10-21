# Bobot

| Berkas | Isi | SHA-256 |
|---|---|---|
| `gstl_ccd_seed123_epoch10.pth` | bobot GSTL saja (bukan optimizer), konfigurasi pelatihan di dalamnya (`config`), epoch 10 | `92e082485776b6bb28ebcf32bfa276ba278751215752b589741eeae17a14a468` |

**Asal:** `train.py --model gstl --hidden-dim 128 --latent-dim 128 --epochs 70 --val-frac 0.1 --test-iter 0 --eval-epoch-end --select-metric AP_video --early-stop-patience 10 --keep-all-epochs --seed 123` pada CCD (3.240 latih, 360 validasi, 900 uji). Terpilih epoch 10 (AP_video validasi terbaik); berhenti di epoch 20 karena early stopping. Log lengkap: `results/run_seed123/`.

**Mengganti bobot** (mis. setelah pelatihan ulang beberapa seed): letakkan checkpoint baru di folder ini, jalankan `python evaluate.py --ckpts weights/<berkas>.pth --seeds 123 1 2 3 4 5 --tag <nama>`, lalu `python tools/make_results_table.py` untuk memperbarui tabel di README. Perbarui juga hash di tabel ini (`sha256sum`).
