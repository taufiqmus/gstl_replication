# Ablasi graf GSTL pada set uji CCD (tanpa pelatihan ulang)

Bobot `gstl_ccd_seed123_epoch10.pth`, protokol = `tools/ablasi_kotak.py`. Topologi graf tidak diubah; hanya bobot sisi (dan Psi pada `psi_tukar`). Rerata (± simpangan baku antar seed acakan). AUC memakai skor maks per video.

| Mode | seed | AP asli (%) | AP video (%) | AUC | mTTA detik |
|---|---|---|---|---|---|
| asli | 1 | 96.13 | 96.25 | 0.981 | 2.25 |
| psi_nol | 1 | 96.11 | 96.23 | 0.981 | 2.24 |
| psi_nol+graf_seragam | 1 | 96.08 | 96.15 | 0.981 | 2.20 |
