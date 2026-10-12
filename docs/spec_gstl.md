# Spesifikasi GSTL (Mahmood et al., IEEE Access 2023)

Disusun 2026-10-06 dari PDF `referensi\baseline\Mahmood2023_accident-anticipation-geometric.pdf`
(§IV Proposed Method hal. 6–7, §V.B Implementation hal. 8–9, Gambar 2 kiri, Tabel 6–8).
Kode resmi **tidak ada**; implementasi: `models\ustring_port\ustring\gstl.py` (di atas port UString),
dijalankan dengan `train.py --model gstl`.

## 1. Yang dinyatakan paper (dipakai apa adanya)

| Aspek | Isi paper | Implementasi |
|---|---|---|
| Simpul graf | X_t^(i) = [Ψ(O_t^(i)), Φ(F_t)]: info kotak objek i + fitur visual VGG16 frame utuh (pers. 1) | `GSTL.node_features` |
| Fitur visual objek | **tidak dipakai** di GSTL (hanya di EGSTL lewat Y_t = [Φ(O), Φ(F)]) | diuji: mengubah fitur objek tidak mengubah keluaran |
| Adjacency | A_t^(ij) = exp(−d(r_i, r_j)) / Σ exp(−d), d = jarak Euklid antar objek dari geometri (pers. 3) | sama dengan UString (`graph.py`) |
| Arsitektur | "similar to the existing GCRNN"; GCN + GRU (GCRN) + BNN; hidden GRU dan GCN "similar to [2]" | kelas turunan `UString` |
| Hidden | GRU dan output GCN = **128** (§V.B) | `--hidden-dim 128 --latent-dim 128` |
| Loss | L = L_EXP + w1 (L_VPOS − L_PRI) + w2 L_BCE, w1 = 0,001, w2 = 10 (pers. 4–7) | **tanpa ranking loss** (bawaan); `--gstl-ranking` untuk menyalakan |
| L_BCE | pada skor video a = softmax([a(n), a(p)]) (pers. 7, tanpa indeks waktu) → butuh cabang video | cabang SAA UString (bawaan menyala) |
| Latih | batch 10, lr awal 0,0005, Adam, 70 epoch (§V.B) | bawaan train.py |
| Detektor | Cascade R-CNN ResNeXt-101 FPN, KITTI 2D (sama dengan UString) | memakai `det` di npz CCD |
| Parameter | **1,42 M** (Tabel 8; GCRNN 1,97 M) | 1.443.192 (lihat §3) |

## 2. Asumsi (tidak dirinci paper; dapat diatur)

| # | Asumsi | Alasan | Opsi |
|---|---|---|---|
| A1 | Ψ = koordinat kotak ternormalisasi (x1/1280, y1/720, x2/1280, y2/720), 4-d, **tanpa FC** | paling cocok dengan 1,42 M parameter; det CCD dalam piksel 1280×720 | `--gstl-box-mode xyxywh` (6-d), `--gstl-box-embed n` (FC) |
| A2 | Φ(F_t) = FC 4096→256 + ReLU (seperti `phi_x` UString) | kombinasi terdekat ke 1,42 M | `--gstl-frame-embed` |
| A3 | GRU-GCN **beku** (setia ke perilaku kode UString yang menjadi dasar; Tabel 8 GCRNN 1,97 M juga tidak menghitungnya) | konsisten dengan baseline; dugaan kuat Mahmood memakai kode UString | `--rnn-trainable` |
| A4 | Exponential loss gaya kode UString (ter-broadcast) | "training procedure adopted from [2]" | `--exp-loss per_sample` |
| A5 | Bobot sisi graf dari koordinat **piksel** (degeneratif seperti UString) | pers. 3 sama dengan UString | belum ada opsi koordinat ternormalisasi (kandidat perbaikan) |
| A6 | Seleksi model: paper tidak menyebut; GCRNN/DSTA di paper memakai "best AP" | baseline kita memakai validasi 10% + AP_video + early stopping agar adil | `--select-metric`, `--val-frac` |

## 3. Kecocokan jumlah parameter (hitungan seperti paper: parameter terdaftar)

| Konfigurasi (h = z = 128) | Parameter | Selisih dari 1,42 M |
|---|---|---|
| FC frame 256, Ψ 4-d mentah, tanpa SAA | 1.426.6xx | +0,5% |
| **FC frame 256, Ψ 4-d mentah, dengan SAA (bawaan)** | **1.443.192** | +1,6% |
| FC frame 256, Ψ 4-d → FC64, tanpa SAA | 1.434.6xx | +1,0% |
| UString port (h = z = 256) | 1.966.968 | Tabel 8 GCRNN 1,97 M → cocok |

Kecocokan UString 1,97 M menunjukkan cara hitung paper sama dengan kita (tanpa GRU-GCN). Untuk GSTL tidak
ada konfigurasi yang tepat 1,42 M; selisih ±7–23 ribu parameter kemungkinan dari detail yang tidak disebut
(mis. bias, dimensi aux). Varian tanpa SAA lebih dekat tetapi bertentangan dengan L_BCE video di pers. 4/7.

## 4. Pertanyaan terbuka (perlu dicek; mungkin tanya penulis)

1. Bentuk tepat Ψ (koordinat? ternormalisasi? termasuk skor/kelas deteksi?) dan apakah lewat FC.
2. Dimensi Φ(F_t) dan apakah dibagikan ke semua simpul tanpa FC.
3. Apakah ranking loss UString dibuang (pers. 4) atau tetap (kode diadopsi dari [2]).
4. Cabang video untuk L_BCE: SAA UString atau lainnya.
5. Seleksi model (epoch) untuk angka Tabel 9 (CCD GSTL AP 96,5, mTTA 4,68; DAD AP 51,6).
6. Apakah GSTL juga mewarisi bug kode UString (GRU-GCN tak terlatih, exp-loss ter-broadcast, AP > 1).

## 5. Target pembanding (Mahmood Tabel 9/10)

| Dataset | GSTL AP | GSTL mTTA | iTTA |
|---|---|---|---|
| CCD | 96,5 | 4,68 | 4,64 |
| DAD | 51,6 | 2,67 | 1,19 |
| DAD (best AP, Tabel 10) | 68,53 | 1,76 | TTA@80 1,89 |

Catatan: metrik Mahmood mengikuti kode UString/DSA (mTTA gaya asli). Laporkan berdampingan dengan versi eksak.

## 6. Status
- [x] Model + integrasi train/test/eval_epochs + tes (unit dan integrasi sintetis; resume identik).
- [x] Latih di CCD (2026-10-06 13:12–14:58; validasi 10%, evaluasi akhir epoch, seleksi AP_video, early stopping 10)
      → `outputs\ccd_gstl_val_es\`.
      - best epoch 10: **AP video 0,9623 / AP asli 0,9613 / mTTA asli 4,15** (paper 96,5 / 4,68);
      - EGSTL: AP video 0,9869.
      - Rincian: `hasil_replikasi_ccd.md` §5.
- [ ] Varian: tanpa SAA; Ψ 6-d; GRU-GCN dilatih; seleksi gaya asli (AP di set uji).
- [x] EGSTL (ensemble GSTL + GCRNN, rerata skor per frame; Gambar 2 kanan): AP video 0,9869, mTTA asli 4,17
      (`outputs\ccd_egstl_val_es\`).
