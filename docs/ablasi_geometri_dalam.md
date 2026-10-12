# Uji bagian dalam: mana yang membawa sinyal pada simpul GSTL (set uji CCD, tanpa pelatihan ulang)

Simpul = [Psi: 4 koordinat kotak, Phi(F): fitur frame 256-d yang SAMA untuk semua 19 simpul]. Bobot `gstl_ccd_seed123_epoch10.pth`.

| Mode | AP asli (%) | AP video (%) | AUC | mTTA (s) | selisih skor/frame (rerata) | frame berubah > 0,1 |
|---|---|---|---|---|---|---|
| asli | 96.13 | 96.25 | 0.981 | 2.25 | - | - |
| psi_nol | 96.11 | 96.23 | 0.981 | 2.24 | 0.0013 | 0.0% |
| psi_acak | 96.11 | 96.25 | 0.981 | 2.26 | 0.0002 | 0.0% |
| fitur_nol | 28.52 | 28.47 | 0.383 | 1.34 | 0.6693 | 80.2% |
| keduanya_nol | 29.54 | 29.52 | 0.422 | 2.17 | 0.4439 | 99.0% |

## Besar bobot dan skala
```
{
 "bobot_gcn1_psi_rerata_abs": 0.06569666415452957,
 "bobot_gcn1_phi_rerata_abs": 0.06202654913067818,
 "bobot_gcn1_psi_norm_baris_rerata": 0.8620886206626892,
 "bobot_gcn1_phi_norm_baris_rerata": 0.8132913112640381,
 "skala_Phi_rerata_abs": 1.2377943038940429,
 "skala_Psi_rerata_abs": 0.5320677399635315,
 "kontribusi_kasar_Psi_vs_Phi": 0.0071138413006540105
}
```
