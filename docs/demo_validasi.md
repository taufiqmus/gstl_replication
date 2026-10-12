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
