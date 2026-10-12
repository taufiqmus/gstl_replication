# Seberapa baik kotak deteksi? (plot dan recall; 2026-10-12)

Alat: `tools/plot_kotak_ccd.py` (CCD; di repo) dan `outputs/gstl_demo_dota/plot_gt_vs_det.py` di repo kerja kami (DoTA; butuh anotasi dan pemetaan `toa` proyek, tidak ada di repo ini). Gambar tidak disertakan di repo karena berisi frame video berhak cipta.

## Pengamatan visual pada klip CCD uji (detektor kita, warna menurut skor)
- **Siang, adegan biasa (negatif):** mobil terdeteksi yakin (skor >= 0,5), kotak sekitar benar. Namun frame yang sama memuat banyak kotak merah tipis berskor < 0,1, termasuk kotak besar yang menutupi jalan.
- **Kotak yang DITERIMA model (.npz rilis)** mencakup kotak-kotak "sampah" itu: kotak besar bukan-objek (mis. kotak kelas Pedestrian seluas jalan) ikut di antara 19 kotak.
- **Malam/hujan:** 0 sampai 1 kotak yakin per frame; mobil yang jelas terlihat sering tidak terdeteksi.
- **Klip kecelakaan:** pada saat paling penting (kendaraan sangat dekat atau terpotong di tepi, motor, kendaraan terguling, kaca depan retak/malam) tidak ada kotak yakin sama sekali; motor sport pada klip `ccd_pos_siang_nonego` tidak terdeteksi.

## Recall terhadap kotak anotasi manusia DoTA (objek penyebab kecelakaan)
40 klip positif uji (acak, seed 5), 240 frame beranotasi, IoU >= 0,5. 'Model' = 19 kotak yang dikirim ke GSTL (termasuk kotak berskor sangat rendah).

| Kategori GT | n kotak GT | model (19 kotak) | semua deteksi | skor >= 0,1 | skor >= 0,5 |
|---|---|---|---|---|---|
| car | 205 | 65% | 65% | 51% | 44% |
| truck | 24 | 33% | 33% | 12% | 8% |
| bus | 12 | 92% | 92% | 33% | 0% |
| rider | 12 | 42% | 42% | 0% | 0% |
| motor | 12 | 33% | 33% | 17% | 17% |
| person | 6 | 100% | 100% | 100% | 100% |
| **TOTAL** | 271 | **62%** | 62% | **44%** | **37%** |

## Batasan
- 40 klip, 240 frame; kategori selain `car` hanya 6 sampai 24 kotak (angka kasar).
- "Model/semua deteksi" memasukkan kotak berkepercayaan sangat rendah; sebagian "kena" bisa kebetulan oleh kotak besar. Yang lebih jujur untuk "terdeteksi" adalah kolom skor >= 0,5 (37%) atau >= 0,1 (44%).
- GT hanya memuat objek penyebab kecelakaan (bukan semua objek), dan sebagian terpotong di tepi frame.
- Detektor dilatih di KITTI tanpa kelas motor/rider; sebagian rendahnya recall adalah keterbatasan kelas, bukan hanya kualitas.
