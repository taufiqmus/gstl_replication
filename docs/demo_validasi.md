# Validasi pipeline demo.py terhadap fitur CCD rilis

Skrip: `tools/validate_demo.py` (butuh data CCD dan bobot detektor). Hasil mentah: `results/demo_validasi.json`.

**Pertanyaan:** apakah skor GSTL dari fitur yang diekstrak ulang oleh pipeline demo (detektor + VGG16 dari mp4) mendekati skor dari fitur `.npz` yang dirilis bersama CCD?

**Cara:** 8 video negatif + 8 video positif dari set uji CCD (acak, seed 0). Skor dihitung dengan bobot yang sama dan fungsi yang sama (`demo.score_window`), sekali dari npz rilis, sekali dari mp4 lewat pipeline demo.

| Kelompok | selisih mutlak rerata per frame | korelasi per frame (median) | skor maks rerata: npz rilis / demo |
|---|---|---|---|
| Negatif (n=8) | 0,00014 | 0,9993 | 0,061 / 0,061 |
| Positif (n=8) | 0,023 | 0,90 | 0,406 / 0,396 |

AUC positif-vs-negatif pada 16 video (skor maks sebelum `toa`): 0,906 (npz rilis) vs 0,875 (demo). Sampel sangat kecil: perbedaan ini bukan bukti bahwa demo lebih buruk atau sama.

## Tafsiran dan batas
- Pada video **negatif** pipeline demo menghasilkan skor yang hampir identik dengan fitur rilis. Fitur frame dan kotak deteksi port sesuai dengan yang dipakai CCD.
- Pada video **positif** kesesuaiannya lebih rendah. Di repo kerja kami ditemukan bahwa fitur rilis positif berasal dari frame sumber yang berbeda dari mp4 rilis, sehingga perbandingan per-frame pada positif tidak adil; ini bukan terutama kesalahan pipeline demo.
- Validasi ini hanya mencakup video CCD (1280x720, 10 fps). Kesesuaian pada video lain (resolusi, laju frame, gaya kamera, motor) **tidak diuji**; detektor tidak mengenal kelas motor.
- Detektor adalah port PyTorch murni dari Cascade R-CNN mmdetection v1; kesetaraan penuh dengan kode asli belum diverifikasi (hanya kesesuaian dengan kotak `det` rilis, lihat docstring `gstl/detector.py`).

## Uji pada video non-CCD (klip DoTA; 2026-10-12)
Bobot GSTL (dilatih hanya di CCD) dijalankan lewat pipeline `demo.py` pada 80 klip acak (40 positif + 40 negatif) dari split uji DoTA rilis STAGNet (50 frame, 10 fps, 1280x720). Positif dievaluasi pada [:toa] (toa terkoreksi V3), negatif seluruh frame. Data dan skrip ini ada di repo kerja kami, bukan di repo ini.

| Ukuran | Hasil |
|---|---|
| AP video eksak | 0,60 (V3), 0,57 (toa asli V0); acuan acak untuk kelas seimbang 0,5 |
| AUC (skor maks) | 0,65 (95% bootstrap 0,53 sampai 0,77), 0,62 dengan toa V0 |
| mTTA detik sungguhan | 2,04 (V3) |
| AUC memakai 3 frame pertama saja / seluruh klip | 0,67 / 0,73 |
| Negatif dengan skor maks >= 0,5 | 22 dari 40 (17 klip >= 0,99) |
| Positif dengan skor >= 0,5 sudah di frame 0 | 18 dari 40 |
| Skor maks positif rerata, siang vs malam | 0,90 (n=30) vs 0,31 (n=10) |
| Kotak terdeteksi (skor >= 0,5) per frame, siang vs malam | 1,23 vs 0,57 |

**Tafsiran hati-hati:** generalisasi ke DoTA lemah (AP 0,60 vs 0,96 di CCD), dan sebagian pembedaan sudah muncul di 3 frame pertama. Pada malam hari detektor menemukan lebih sedikit objek dan skor positif jauh lebih rendah (n malam hanya 10). Banyak negatif berskor tinggi; klip negatif DoTA sering memuat sisa kecelakaan (segmen post-anomali). Sampel kecil dan selang kepercayaan lebar; ini bukan protokol lintas dataset Mahmood dkk. dan bukan perbandingan dengan model lain. Skor demo tidak boleh ditafsirkan sebagai risiko kecelakaan terkalibrasi.
