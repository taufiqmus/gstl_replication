# Ablasi masukan GSTL pada set uji CCD (tanpa pelatihan ulang)

Bobot `gstl_ccd_seed123_epoch10.pth`, protokol = `evaluate.py`. Rerata (± simpangan baku antar seed acakan; '-' bila satu seed). AUC memakai skor maks per video (positif: sebelum toa).

| Mode | seed | AP asli (%) | AP video (%) | AUC | mTTA detik |
|---|---|---|---|---|---|
| asli | 1 | 96.13 | 96.25 | 0.981 | 2.25 |
| tanpa_geometri | 1 | 96.14 | 96.18 | 0.981 | 2.19 |
| tukar_kotak | 3 | 96.11 ± 0.03 | 96.22 ± 0.01 | 0.981 ± 0.000 | 2.24 ± 0.05 |
| buang_25 | 3 | 96.12 ± 0.03 | 96.17 ± 0.01 | 0.981 ± 0.000 | 2.20 ± 0.01 |
| buang_50 | 3 | 96.14 ± 0.03 | 96.18 ± 0.01 | 0.981 ± 0.000 | 2.20 ± 0.01 |
| buang_75 | 3 | 96.15 ± 0.01 | 96.16 ± 0.01 | 0.981 ± 0.000 | 2.20 ± 0.00 |
| kotak_acak | 3 | 93.82 ± 0.16 | 95.82 ± 0.05 | 0.977 ± 0.000 | 2.85 ± 0.01 |
| tukar_fitur_frame | 3 | 32.83 ± 0.95 | 31.72 ± 1.30 | 0.487 ± 0.033 | 3.09 ± 0.05 |
