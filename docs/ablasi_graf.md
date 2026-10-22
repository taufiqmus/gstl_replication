# Ablasi graf GSTL pada set uji CCD (tanpa pelatihan ulang)

Bobot `gstl_ccd_seed123_epoch10.pth`, protokol = `tools/ablasi_kotak.py`. Topologi graf tidak diubah; hanya bobot sisi (dan Psi pada `psi_tukar`). Rerata (± simpangan baku antar seed acakan). AUC memakai skor maks per video.

| Mode | seed | AP asli (%) | AP video (%) | AUC | mTTA detik |
|---|---|---|---|---|---|
| asli | 1 | 96.13 | 96.25 | 0.981 | 2.25 |
| graf_seragam | 1 | 96.16 | 96.17 | 0.981 | 2.18 |
| graf_norm | 1 | 96.16 | 96.18 | 0.981 | 2.19 |
| graf_bobot_acak | 3 | 96.16 ± 0.03 | 96.25 ± 0.00 | 0.981 ± 0.000 | 2.25 ± 0.01 |
| graf_tukar | 3 | 96.13 ± 0.03 | 96.22 ± 0.01 | 0.981 ± 0.000 | 2.25 ± 0.05 |
| psi_tukar | 3 | 96.13 ± 0.03 | 96.25 ± 0.01 | 0.981 ± 0.000 | 2.25 ± 0.01 |
| tukar_semua | 3 | 96.11 ± 0.03 | 96.22 ± 0.01 | 0.981 ± 0.000 | 2.24 ± 0.05 |
