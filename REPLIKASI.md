# Laporan replikasi GSTL

Paper: Mahmood, Jeong, Ryu 2023, *A New Approach to Traffic Accident Anticipation With Geometric Features for Better Generalizability*, IEEE Access 11:29263-29274, DOI 10.1109/ACCESS.2023.3259992.

**Kesimpulan satu kalimat:** AP CCD mendekati paper (96,1 vs 96,5, satu seed), mTTA belum cocok (4,17 vs 4,68), parameter 1,44 M vs 1,42 M; kode penulis tidak tersedia, jadi ini reimplementasi dengan asumsi yang dicatat di bawah.

## 1. Perbandingan dengan paper (CCD, uji 900 video)

| Ukuran | Paper | Repo ini | Catatan |
|---|---|---|---|
| AP (gaya UString) | 96,5 | 96,13 | selisih 0,4 poin; satu seed |
| mTTA (gaya UString, detik) | 4,68 | 4,17 | selisih 0,5 detik; **belum dijelaskan** |
| Parameter | 1,42 M | 1,443 M | selisih +1,6%; tidak ada konfigurasi yang tepat 1,42 M |
| AP video eksak (versi kami) | – | 96,25 | – |
| mTTA detik sungguhan (versi kami) | – | 2,25 | metrik gaya UString menggelembungkan TTA (faktor T/toa) |

Sumber angka kami: `results/ccd_test_epoch10_seed123.csv`; seluruh epoch: `results/ccd_test_all_epochs_seed123.csv`; log latih: `results/run_seed123/`.

## 2. Protokol pelatihan yang menghasilkan bobot
Validasi 10% bertingkat dari data latih (3.240 latih / 360 validasi), seleksi epoch dengan AP video pada **validasi** (bukan set uji), early stopping sabar 10, seed 123, batch 10, Adam lr 5e-4, h=z=128. Terpilih epoch 10; berhenti di epoch 20. Hasil uji tidak dipakai untuk memilih.

Paper tidak menyebut cara memilih epoch. Kami mengevaluasi juga seleksi "gaya paper" (AP asli terbaik di set uji): memilih **epoch yang sama (10)**, sehingga selisih mTTA bukan karena pemilihan epoch.

## 3. Keadaan antar epoch (satu seed)
- AP asli berfluktuasi 0,85–0,96 antar epoch, sedangkan AP video eksak stabil 0,95–0,97. mTTA asli berkisar 3,12–4,61.
- Tidak ada epoch dengan AP asli ≥ 0,95 dan mTTA ≥ 4,5 sekaligus (mTTA tertinggi pada AP ≥ 0,95 adalah 4,40, epoch 7). Epoch bermTTA ≥ 4,5 punya AP asli 0,85–0,93.
- Akibatnya selisih AP 0,4 poin dan mTTA 0,5 detik dengan paper tidak dapat dinilai nyata atau derau dari satu seed.

## 4. Asumsi (tidak dirinci paper)
Rinci di `docs/spec_gstl.md`. Ringkas:
- A1: Ψ = koordinat kotak ternormalisasi 4-d tanpa FC. A2: Φ(F) = FC 4096→256 + ReLU. A3: GRU-GCN dibekukan (setia ke perilaku kode UString yang menjadi dasar). A4: exponential loss gaya UString. A5: bobot sisi graf dari koordinat piksel (seperti UString). A6: tanpa ranking loss; cabang video L_BCE memakai SAA UString.
- Pertanyaan terbuka untuk penulis: bentuk tepat Ψ, dimensi Φ, ranking loss, cabang video L_BCE, seleksi epoch pada tabel paper, dan apakah bug kode UString (GRU-GCN tak terlatih, exp-loss ter-broadcast, AP>1) diwarisi.

## 5. Yang BELUM dikerjakan
- Beberapa seed pelatihan (selang kepercayaan); saat ini satu seed.
- Ablasi varian: tanpa SAA, Ψ 6-d, GRU-GCN dilatih, ranking loss, koordinat ternormalisasi untuk adjacency.
- Evaluasi dengan beberapa seed sampling BNN (hitungan menit).
- DAD (paper: AP 51,6, mTTA 2,67): data belum tersedia. Klaim generalisasi lintas dataset belum diuji.
- Klaim bias kualitas video pada paper (checkpoint UString/GCRNN, kualitas video disamakan: AP 99,5 → 52,1) belum tereproduksi di repo kerja kami: dengan checkpoint resmi UString dan dua jalur ekstraksi fitur, AP hanya turun ±3–4 poin (bukan ±47). Penyebabnya belum diketahui. Itu tentang UString, bukan GSTL.

## 6. Catatan metrik
- Metrik asli UString dapat menghasilkan **AP > 1** (bug: TP dihitung `pred*label >= Th`; bila ada skor tepat 0, semua video negatif ikut terhitung). Pada GSTL kami tidak menemukan kasus itu (`AP_orig` = `AP_origfix` di semua epoch).
- mTTA gaya UString menghitung `(1 - t/toa)·T/fps`, bukan `(toa - t)/fps`; pada CCD itu menggelembungkan nilainya. Kami melaporkan kedua versi berdampingan.

## 7. Verifikasi pemindahan kode
Repo ini dipisahkan dari repo kerja kami. Evaluasi bobot epoch 10 menghasilkan 12 dari 12 metrik identik (selisih 0) dengan repo asal; `train.py` hasil pangkasan lolos uji asap pelatihan pada data sintetis. Ini membuktikan perilaku tidak berubah, **tidak** membuktikan kesetaraan dengan kode penulis.

## 8. Ketergantungan pada geometri kotak (ditemukan 2026-10-12)
Uji pada bobot yang disertakan, set uji CCD, tanpa pelatihan ulang (`tools/ablasi_geometri_dalam.py`, `docs/ablasi_geometri_dalam.md`; `tools/ablasi_kotak.py`, `docs/ablasi_kotak.md`):
- Menghapus geometri dari simpul (Psi = 0) atau mengacaknya antar objek: AP asli 96,13 → 96,11, selisih skor per frame rerata 0,001, tidak ada frame yang berubah > 0,1. Menukar kotak dengan milik klip lain atau menghapus tiga-perempat kotak: AP juga tidak berubah.
- Menghapus fitur frame (Phi(F) = 0): AP runtuh ke 28,5% (di bawah acuan acak 33%), 80% frame berubah > 0,1.
- Sebabnya struktural pada implementasi ini: tiap simpul = [4 koordinat kotak, fitur frame 256-d yang SAMA untuk semua 19 simpul]; kontribusi kasar geometri ≈ 0,7% dari sinyal masukan simpul. Bobot GCN untuk koordinat tidak kecil (norma baris 0,86 vs 0,81), tetapi hanya 4 dari 260 dimensi.
- **Artinya: pada reimplementasi ini, GSTL hampir sepenuhnya memutuskan dari fitur frame VGG16; kualitas detektor dan geometri hampir tidak berpengaruh.** Ini hasil untuk reimplementasi kami (asumsi A1: Psi 4-d mentah; A3: GRU-GCN dibekukan; A5), bukan bukti tentang model Mahmood yang asli; paper mungkin memakai geometri lebih kuat. Itu bisa ikut menjelaskan selisih mTTA dan generalisasi lintas dataset yang lemah (AP 0,60 pada klip DoTA).
