# Pilihan detektor dan pengaruhnya (2026-10-12)

`demo.py`, `tools/run_samples.py`, dan `tools/analisis_deteksi_sampel.py` menerima `--det-model {cascade_rcnn, tv_fasterrcnn, tv_ssdlite}` dan `--det-conf` (lihat `gstl/detectors.py`). Bawaan tetap `cascade_rcnn` (bobot penulis UString), satu-satunya yang mereproduksi pipeline CCD.

## 1. Apakah detektor yang lebih baik mengubah skor GSTL? (tanpa pelatihan ulang)
25 klip contoh (`samples/`), bobot GSTL yang sama, detektor Cascade (acuan) vs torchvision Faster R-CNN ResNet50-FPN v2 bobot COCO (ambang skor 0,3):
- selisih skor per frame: **maksimum 0,039** (klip `ccd_pos_siang_salju_2`), rerata seluruh klip **0,0006**; tidak ada klip dengan selisih maks > 0,05;
- keputusan ambang 0,5 dan waktu peringatan pertama sama pada semua klip.
Konsisten dengan ablasi (`docs/ablasi_kotak.md`): pada model ini kotak tidak menentukan skor.

## 2. Seberapa lebih baik detektornya? Recall pada kotak anotasi DoTA (objek penyebab kecelakaan)
40 klip positif uji, 240 frame beranotasi, IoU >= 0,5 (skrip: repo kerja proyek, `outputs/gstl_demo_dota/plot_gt_vs_det.py`):

| | Cascade (bawaan) | Faster R-CNN COCO |
|---|---|---|
| skor >= 0,5 | **37%** | **91%** |
| skor >= 0,1 | 44% | 96% |
| car | 44% | 92% |
| motor | 17% | **75%** |
| rider | 0% | **92%** |
| truck | 8% | 88% |

(Catatan: hanya "skor >= 0,5" dan "skor >= 0,1" yang bermakna sebagai kualitas; "semua deteksi" memasukkan kotak berkepercayaan sangat rendah.)
Kasus visual (`tools/bandingkan_detektor.py`): motor dan pengendara, motor sport, mobil di malam hari berlampu, mobil di malam hujan, dan gerobak motor terdeteksi oleh Faster R-CNN dan terlewat oleh Cascade. Yang gagal pada keduanya: mobil yang sangat dekat dan terpotong di tepi frame (klip `ccd_pos_siang_ego` f=30).

## 3. Catatan penting
- Pada pipeline `bbox_sampling` (warisan UString), kotak dikirim ke model menurut urutan kelas (Car dulu, lalu Pedestrian, lalu Cyclist) dan dipotong di 19 kotak. Dengan detektor yang menghasilkan banyak deteksi (ambang rendah), motor/pengendara (kelas Cyclist/Pedestrian) bisa terpotong dari 19 kotak; recall "model (19 kotak)" untuk motor hanya 17% pada ambang 0,05 walau detektor menemukannya (75% pada >= 0,5). Pakai `--det-conf` yang lebih tinggi atau ubah aturan pemilihan bila geometri nanti dipakai sungguhan.
- GSTL tidak memakai kelas dan hampir tidak memakai kotak, jadi detektor yang lebih baik di sini terutama memperbaiki visualisasi dan membuka jalan bagi model yang memakai geometri.
- Bobot COCO torchvision diunduh otomatis ke cache torch hub (Faster R-CNN 167 MB, SSDlite 13 MB; hash diverifikasi oleh torchvision).
