"""Registri detektor objek untuk demo/analisis. Semua detektor mengembalikan format yang sama:

    detect(frame_bgr) -> daftar 3 larik (satu per kelas lama: Car, Pedestrian, Cyclist), tiap larik (k, 5 atau 6):
                         x1, y1, x2, y2, skor [, id_label_coco]

Format ini sama dengan keluaran Cascade R-CNN bawaan sehingga `bbox_sampling`, `features.py`, dan `demo.py` tidak berubah.
Kolom ke-6 (opsional) berisi id label COCO asli, dipakai untuk menampilkan nama kelas sebenarnya (mis. motorcycle).

Detektor:
  cascade_rcnn   bobot penulis UString (KITTI, 3 kelas, TANPA motor); satu-satunya yang mereproduksi pipeline CCD. Bawaan.
  tv_fasterrcnn  torchvision Faster R-CNN ResNet50-FPN v2, bobot COCO (80 kelas; termasuk motorcycle, bicycle, person, bus, truck)
  tv_ssdlite     torchvision SSDlite320 MobileNetV3-Large, bobot COCO (ringan, resolusi masukan 320; objek kecil buruk)
Pemetaan COCO -> kelas lama: car, truck, bus -> Car; person -> Pedestrian; bicycle, motorcycle -> Cyclist.
GSTL tidak memakai kelas; pemetaan hanya untuk kompatibilitas dan warna tampilan.
"""
import numpy as np
import torch

LEGACY_CLASSES = ("Car", "Pedestrian", "Cyclist")
COCO_NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
COCO_TO_LEGACY = {3: 0, 6: 0, 8: 0, 1: 1, 2: 2, 4: 2}
DETECTORS = ("cascade_rcnn", "tv_fasterrcnn", "tv_ssdlite")
DEFAULT_CONF = {"cascade_rcnn": None, "tv_fasterrcnn": 0.3, "tv_ssdlite": 0.3}


def coco_to_legacy(boxes, scores, labels):
    """Larik numpy dari model COCO -> 3 larik (k,6) [x1,y1,x2,y2,skor,id_label_coco], terurut skor menurun; label di luar pemetaan dibuang."""
    boxes, scores, labels = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float).reshape(-1), np.asarray(labels).reshape(-1)
    cls = np.array([COCO_TO_LEGACY.get(int(l), -1) for l in labels], int) if len(labels) else np.zeros(0, int)
    out = []
    for ci in range(3):
        m = cls == ci
        if m.any():
            arr = np.concatenate([boxes[m], scores[m, None], labels[m, None].astype(float)], axis=1)
            out.append(arr[np.argsort(-arr[:, 4])])
        else:
            out.append(np.zeros((0, 6)))
    return out


def box_name(b, class_index):
    """Nama kelas untuk satu deteksi: nama COCO asli bila ada kolom label, selain itu nama kelas lama."""
    if len(b) > 5 and int(b[5]) in COCO_NAMES:
        return COCO_NAMES[int(b[5])]
    return LEGACY_CLASSES[int(class_index) % 3]


class TorchvisionDetector:
    def __init__(self, name, device="cuda", conf=0.3, pretrained=True):
        from torchvision.models import detection as D
        self.name, self.device = name, torch.device(device)
        if name == "tv_fasterrcnn":
            w = D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
            self.model = D.fasterrcnn_resnet50_fpn_v2(weights=w, weights_backbone=None, box_score_thresh=conf, box_detections_per_img=100)
        elif name == "tv_ssdlite":
            w = D.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None
            self.model = D.ssdlite320_mobilenet_v3_large(weights=w, weights_backbone=None, score_thresh=conf, detections_per_img=100)
        else:
            raise ValueError(name)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def detect(self, img_bgr):
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        out = self.model([t])[0]
        return coco_to_legacy(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy())


class ConfFilter:
    """Membungkus detektor mana pun dan membuang deteksi berskor < conf (mis. memangkas 'sampah' Cascade bawaan, yang berambang 1e-6)."""

    def __init__(self, base, conf):
        self.base, self.conf = base, float(conf)
        self.name = getattr(base, "name", "detektor") + "+conf%.2f" % conf

    def detect(self, img_bgr):
        return [c[c[:, 4] >= self.conf] if len(c) else c for c in self.base.detect(img_bgr)]

    def __getattr__(self, k):   # teruskan atribut lain (mis. parameters) ke detektor dasar
        return getattr(self.base, k)


def build_detector(name="cascade_rcnn", device="cuda", ckpt=None, conf=None, pretrained=True):
    """conf=None -> bawaan per detektor (Cascade: tanpa penyaringan, ambang pipeline asli 1e-6; torchvision: 0,3)."""
    if name not in DETECTORS:
        raise ValueError("detektor tidak dikenal: %s (pilihan: %s)" % (name, ", ".join(DETECTORS)))
    if name == "cascade_rcnn":
        from .detector import load_detector
        if not ckpt:
            raise ValueError("cascade_rcnn butuh bobot (--detector <latest.pth>)")
        det = load_detector(ckpt, device=device)
        det.name = "cascade_rcnn"
        return ConfFilter(det, conf) if conf is not None else det
    return TorchvisionDetector(name, device, DEFAULT_CONF[name] if conf is None else conf, pretrained)
