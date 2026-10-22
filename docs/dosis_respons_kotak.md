# Dosis-respons masukan kotak GSTL pada set uji CCD (tanpa pelatihan ulang)

Bobot `gstl_ccd_seed123_epoch10.pth`, protokol = `tools/ablasi_kotak.py`. Rerata (± simpangan baku antar seed acakan). AUC memakai skor maks per video.

| Mode | seed | AP asli (%) | AP video (%) | AUC | mTTA detik |
|---|---|---|---|---|---|
| asli | 1 | 96.13 | 96.25 | 0.981 | 2.25 |
| jitter_5 | 3 | 96.14 ± 0.02 | 96.33 ± 0.04 | 0.981 ± 0.000 | 2.26 ± 0.03 |
| jitter_10 | 3 | 95.66 ± 0.20 | 96.35 ± 0.08 | 0.981 ± 0.000 | 2.41 ± 0.01 |
| jitter_20 | 3 | 94.35 ± 0.14 | 96.29 ± 0.06 | 0.980 ± 0.000 | 2.65 ± 0.02 |
| jitter_50 | 3 | 94.15 ± 0.06 | 96.48 ± 0.02 | 0.980 ± 0.000 | 2.80 ± 0.00 |
| jitter_100 | 3 | 94.08 ± 0.09 | 96.03 ± 0.03 | 0.980 ± 0.000 | 2.84 ± 0.01 |
| acak_kelas | 3 | 96.13 ± 0.00 | 96.25 ± 0.00 | 0.981 ± 0.000 | 2.25 ± 0.00 |
| acak_skor | 3 | 96.13 ± 0.00 | 96.25 ± 0.00 | 0.981 ± 0.000 | 2.25 ± 0.00 |
| acak_posisi | 3 | 93.86 ± 0.07 | 95.86 ± 0.10 | 0.978 ± 0.001 | 2.83 ± 0.01 |
| perm_slot_frame | 3 | 96.16 ± 0.04 | 96.25 ± 0.00 | 0.981 ± 0.000 | 2.25 ± 0.00 |
| perm_slot_klip | 3 | 96.15 ± 0.04 | 96.25 ± 0.00 | 0.981 ± 0.000 | 2.25 ± 0.02 |
| kotak_acak | 3 | 93.82 ± 0.16 | 95.82 ± 0.05 | 0.977 ± 0.000 | 2.85 ± 0.01 |
