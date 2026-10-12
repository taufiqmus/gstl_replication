# Deteksi per frame pada klip contoh (apa yang terlihat vs apa yang diterima model)

Rerata per klip (50 frame). 'Semua' = deteksi dengan skor > 1e-6 (ambang yang dipakai pipeline); 'ke model' = kotak unik yang dikirim ke GSTL setelah `bbox_sampling` (maks 19).

| Kelompok | deteksi semua | skor >= 0,1 | skor >= 0,3 | skor >= 0,5 | skor >= 0,8 | kotak unik ke model | frame tanpa kotak (skor >= 0,5) |
|---|---|---|---|---|---|---|---|
| semua klip (n=25) | 16.5 | 1.4 | 1.3 | 1.2 | 1.1 | 12.4 | 51% |
| DoTA (n=7) | 11.5 | 1.2 | 1.1 | 1.0 | 0.9 | 8.7 | 55% |
| CCD (n=18) | 18.5 | 1.5 | 1.3 | 1.3 | 1.2 | 13.8 | 49% |
| positif (n=14) | 14.8 | 0.6 | 0.5 | 0.4 | 0.4 | 11.0 | 73% |
| negatif (n=11) | 18.7 | 2.5 | 2.3 | 2.2 | 2.0 | 14.1 | 22% |

## Catatan tafsir (baca sebelum menyimpulkan)
- **Yang terlihat vs yang diterima model:** overlay `demo.py` hanya menggambar kotak berskor >= `--box-conf` (0,5). Model menerima semua deteksi berskor > 1e-6 (hingga 19 kotak unik): pada klip contoh rata-rata 16,5 deteksi per frame, tetapi hanya 1,4 yang berskor >= 0,1. Jadi sekitar 90% masukan kotak GSTL adalah deteksi berkepercayaan sangat rendah, dan banyak frame tampak "tanpa kotak" di overlay padahal model tetap menerima ±12 kotak.
- **Pada fitur CCD rilis** (`docs/analisis_kotak.md`, 900 klip uji): rerata ±12,7 kotak unik per frame, 0% frame tanpa deteksi, dan hampir sama antara positif (12,75) dan negatif (12,59). Skor deteksi di npz rilis dipotong menjadi bilangan bulat, sehingga tingkat kepercayaan tidak dapat dipulihkan dari berkas itu.
- **Perbedaan positif vs negatif pada klip contoh** (kotak skor >= 0,5: 0,4 vs 2,2 per frame) **tidak boleh digeneralisasi**: klip contoh dipilih tidak acak (sengaja memuat kasus gagal), dan fitur CCD rilis untuk klip positif berasal dari frame sumber yang berbeda dari mp4-nya. Pembandingan yang adil butuh sampel acak yang lebih besar.
