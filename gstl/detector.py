"""Cascade R-CNN (ResNeXt-101 64x4d + FPN, KITTI 3 kelas) dalam PyTorch/torchvision murni.

Tujuan: menjalankan bobot detektor UString (mmdetection v1.1.0, `latest.pth` dari Drive README
UString) di PyTorch 2.x / RTX 3060 tanpa mmdetection/mmcv lama. Perilaku inferensi ditiru dari
sumber mmdetection v1.1.0 dan mmcv 0.4.2 (Apache-2.0), file-file:
  cascade_rcnn.simple_test, rpn_head.get_bboxes_single, anchor_generator, delta2bbox,
  bbox_head.get_det_bboxes/regress_by_class, multiclass_nms, nms_kernel.cu (GPU),
  single_level.map_roi_levels, roi_align_kernel.cu, fpn.forward, mmcv imrescale/imnormalize/impad.

Konvensi v1 yang ditiru:
  * lebar kotak = x2 - x1 + 1 (anchor, delta2bbox, NMS, level RoI, RoIAlign);
  * NMS (versi GPU) membuang kotak bila IoU > ambang (ketat) → dipakai torchvision.ops.nms
    pada kotak (x1, y1, x2+1, y2+1), yang setara dengan konvensi +1;
  * RoIAlign v1: ujung = (x2+1)*skala, tanpa offset 0,5 → torchvision roi_align(aligned=False)
    pada kotak (x2+1, y2+1). Beda kecil: torchvision memaksa lebar RoI >= 1 sel fitur,
    v1 >= 0 (hanya berpengaruh pada RoI yang sangat kecil);
  * skor kelas cascade = rerata LOGIT ketiga tahap, lalu softmax;
  * pemilihan label untuk regress_by_class = argmax termasuk latar.
[belum diverifikasi terhadap mmdetection asli] karena mmdet v1 tidak bisa dijalankan di sini;
uji kesesuaian dengan kotak `det` di npz CCD (lihat tools/compare_ccd_detections.py).
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext101_64x4d
from torchvision.ops import nms as tv_nms
from torchvision.ops import roi_align as tv_roi_align

CLASSES = ("Car", "Pedestrian", "Cyclist")
IMG_SCALE = (1696, 512)
MEAN = np.array([92.84, 97.80, 93.58], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
ANCHOR_STRIDES = (4, 8, 16, 32, 64)
ANCHOR_RATIOS = (0.33, 0.5, 1.0, 2.0, 3.0)
ANCHOR_SCALES = (8,)
STAGE_STDS = ((0.1, 0.1, 0.2, 0.2), (0.05, 0.05, 0.1, 0.1), (0.033, 0.033, 0.067, 0.067))
RPN_TEST = dict(nms_pre=1000, nms_post=1000, max_num=1000, nms_thr=0.7)
RCNN_TEST = dict(score_thr=1e-6, nms_thr=0.0, max_per_img=100)
WH_RATIO_CLIP = 16 / 1000


# --------------------------------------------------------------------------- pra-proses (mmcv 0.4.2)
def preprocess(img_bgr, img_scale=IMG_SCALE):
    """imrescale (keep_ratio, bilinear) → imnormalize (to_rgb) → impad_to_multiple(32)."""
    h, w = img_bgr.shape[:2]
    max_long, max_short = max(img_scale), min(img_scale)
    scale_factor = min(max_long / max(h, w), max_short / min(h, w))
    new_w, new_h = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_shape = img.shape
    img = img.astype(np.float32)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, np.float64(MEAN.reshape(1, -1)), img)
    cv2.multiply(img, 1 / np.float64(STD.reshape(1, -1)), img)
    pad_h = int(np.ceil(img.shape[0] / 32)) * 32
    pad_w = int(np.ceil(img.shape[1] / 32)) * 32
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[: img.shape[0], : img.shape[1]] = img
    return padded, img_shape, scale_factor


# --------------------------------------------------------------------------- utilitas kotak
def delta2bbox(rois, deltas, means, stds, max_shape=None, wh_ratio_clip=WH_RATIO_CLIP):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    d = deltas * stds + means
    dx, dy, dw, dh = d[:, 0::4], d[:, 1::4], d[:, 2::4], d[:, 3::4]
    max_ratio = abs(math.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = px + pw * dx
    gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)


def nms_v1(boxes, scores, iou_thr):
    """NMS mmdet v1 (GPU): konvensi +1, buang bila IoU > iou_thr. Mengembalikan indeks terurut skor."""
    b = boxes.clone()
    b[:, 2:] += 1.0
    return tv_nms(b.float(), scores.float(), float(iou_thr))


def base_anchors(base_size, scales=ANCHOR_SCALES, ratios=ANCHOR_RATIOS):
    w = h = base_size
    x_ctr, y_ctr = 0.5 * (w - 1), 0.5 * (h - 1)
    ratios = torch.tensor(ratios, dtype=torch.float32)
    scales = torch.tensor(scales, dtype=torch.float32)
    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
    return torch.stack([x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                        x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)], dim=-1).round()


def grid_anchors(base, featmap_size, stride, device):
    base = base.to(device)
    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w, device=device) * stride
    shift_y = torch.arange(0, feat_h, device=device) * stride
    xx = shift_x.repeat(len(shift_y))
    yy = shift_y.view(-1, 1).repeat(1, len(shift_x)).view(-1)
    shifts = torch.stack([xx, yy, xx, yy], dim=-1).type_as(base)
    return (base[None, :, :] + shifts[:, None, :]).view(-1, 4)


# --------------------------------------------------------------------------- modul
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        r = resnext101_64x4d(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class _ConvModule(nn.Module):
    """Nama kunci sama dengan ConvModule mmdet (…​.conv.weight), tanpa norm/aktivasi."""

    def __init__(self, cin, cout, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256, num_outs=5):
        super().__init__()
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList([_ConvModule(c, out_channels, 1) for c in in_channels])
        self.fpn_convs = nn.ModuleList([_ConvModule(out_channels, out_channels, 3, padding=1) for _ in in_channels])

    def forward(self, inputs):
        lat = [l(x) for l, x in zip(self.lateral_convs, inputs)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], scale_factor=2, mode="nearest")
        outs = [self.fpn_convs[i](lat[i]) for i in range(len(lat))]
        for _ in range(self.num_outs - len(outs)):
            outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs


class RPNHead(nn.Module):
    def __init__(self, cin=256, num_anchors=5):
        super().__init__()
        self.rpn_conv = nn.Conv2d(cin, cin, 3, padding=1)
        self.rpn_cls = nn.Conv2d(cin, num_anchors, 1)
        self.rpn_reg = nn.Conv2d(cin, num_anchors * 4, 1)

    def forward(self, feats):
        cls, reg = [], []
        for x in feats:
            x = F.relu(self.rpn_conv(x))
            cls.append(self.rpn_cls(x))
            reg.append(self.rpn_reg(x))
        return cls, reg


class SharedFCBBoxHead(nn.Module):
    def __init__(self, in_dim=256 * 7 * 7, fc=1024, num_classes=4):
        super().__init__()
        self.shared_fcs = nn.ModuleList([nn.Linear(in_dim, fc), nn.Linear(fc, fc)])
        self.fc_cls = nn.Linear(fc, num_classes)
        self.fc_reg = nn.Linear(fc, 4)

    def forward(self, x):
        x = x.flatten(1)
        for f in self.shared_fcs:
            x = F.relu(f(x))
        return self.fc_cls(x), self.fc_reg(x)


class CascadeRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = FPN()
        self.rpn_head = RPNHead()
        self.bbox_head = nn.ModuleList([SharedFCBBoxHead() for _ in range(3)])
        self._base = [base_anchors(s) for s in ANCHOR_STRIDES]

    # ---- RPN
    def _proposals(self, cls_scores, bbox_preds, img_shape):
        mlvl = []
        for lvl, (c, r) in enumerate(zip(cls_scores, bbox_preds)):
            c, r = c[0], r[0]
            anchors = grid_anchors(self._base[lvl], c.shape[-2:], ANCHOR_STRIDES[lvl], c.device)
            scores = c.permute(1, 2, 0).reshape(-1).sigmoid()
            deltas = r.permute(1, 2, 0).reshape(-1, 4)
            if RPN_TEST["nms_pre"] > 0 and scores.shape[0] > RPN_TEST["nms_pre"]:
                _, topk = scores.topk(RPN_TEST["nms_pre"])
                deltas, anchors, scores = deltas[topk], anchors[topk], scores[topk]
            props = delta2bbox(anchors, deltas, (0, 0, 0, 0), (1, 1, 1, 1), img_shape)
            keep = nms_v1(props, scores, RPN_TEST["nms_thr"])[: RPN_TEST["nms_post"]]
            mlvl.append(torch.cat([props[keep], scores[keep, None]], dim=-1))
        props = torch.cat(mlvl, 0)
        num = min(RPN_TEST["max_num"], props.shape[0])
        _, topk = props[:, 4].topk(num)
        return props[topk]

    # ---- RoI
    def _roi_feats(self, feats, rois):
        """rois: (n, 4) x1,y1,x2,y2 di koordinat input. Level dari map_roi_levels v1."""
        scale = torch.sqrt((rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
        lvls = torch.floor(torch.log2(scale / 56 + 1e-6)).clamp(min=0, max=3).long()
        out = feats[0].new_zeros(rois.size(0), feats[0].size(1), 7, 7)
        boxes = rois.clone()
        boxes[:, 2:] += 1.0  # ujung v1 = (x2 + 1) * skala
        for i in range(4):
            inds = lvls == i
            if inds.any():
                b = torch.cat([boxes.new_zeros(int(inds.sum()), 1), boxes[inds]], dim=1)
                out[inds] = tv_roi_align(feats[i], b, (7, 7), spatial_scale=1.0 / ANCHOR_STRIDES[i],
                                         sampling_ratio=2, aligned=False)
        return out

    @torch.no_grad()
    def detect(self, img_bgr):
        """Satu frame BGR uint8 → list per kelas [array (k,5) x1,y1,x2,y2,skor] di koordinat asli,
        sama dengan keluaran `inference_detector` mmdet v1 (bbox2result)."""
        dev = next(self.parameters()).device
        img, img_shape, scale_factor = preprocess(img_bgr)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(dev)
        feats = self.neck(self.backbone(x))
        cls, reg = self.rpn_head(feats)
        rois = self._proposals(cls, reg, img_shape)[:, :4]
        ms_scores = []
        for i, head in enumerate(self.bbox_head):
            cls_score, bbox_pred = head(self._roi_feats(feats[:4], rois))
            ms_scores.append(cls_score)
            if i < 2:
                rois = delta2bbox(rois, bbox_pred, (0, 0, 0, 0), STAGE_STDS[i], img_shape)
        scores = F.softmax(sum(ms_scores) / 3, dim=1)
        bboxes = delta2bbox(rois, bbox_pred, (0, 0, 0, 0), STAGE_STDS[2], img_shape) / scale_factor
        return self._multiclass_nms(bboxes, scores)

    def _multiclass_nms(self, bboxes, scores):
        dets, labels = [], []
        for i in range(1, scores.shape[1]):
            inds = scores[:, i] > RCNN_TEST["score_thr"]
            if not inds.any():
                continue
            b, s = bboxes[inds], scores[inds, i]
            keep = nms_v1(b, s, RCNN_TEST["nms_thr"])
            dets.append(torch.cat([b[keep], s[keep, None]], dim=1))
            labels.append(torch.full((len(keep),), i - 1, dtype=torch.long, device=b.device))
        if dets:
            dets, labels = torch.cat(dets), torch.cat(labels)
            if dets.shape[0] > RCNN_TEST["max_per_img"]:
                _, order = dets[:, -1].sort(descending=True)
                order = order[: RCNN_TEST["max_per_img"]]
                dets, labels = dets[order], labels[order]
        else:
            dets = bboxes.new_zeros((0, 5))
            labels = bboxes.new_zeros((0,), dtype=torch.long)
        dets, labels = dets.cpu().numpy(), labels.cpu().numpy()
        return [dets[labels == c] for c in range(len(CLASSES))]


def load_detector(ckpt_path, device="cuda"):
    model = CascadeRCNN()
    # unpickle penuh: hanya muat bobot dari sumber tepercaya (penulis UString) dan periksa hash dulu
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # eksplisit: torch >= 2.6 membuat True bawaan
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError("bobot tidak cocok: missing=%s unexpected=%s" % (missing[:10], unexpected[:10]))
    return model.to(device).eval()


# --------------------------------------------------------------------------- pemilihan 19 kotak (demo.py)
def bbox_sampling(bbox_result, nbox=19, imsize=None, topN=5, rng=None):
    """Salinan perilaku `bbox_sampling` demo.py UString (termasuk pemotongan skor ke int dan
    pengisian kekurangan dengan duplikat acak dari topN kotak pertama). rng: np.random.RandomState."""
    rng = np.random if rng is None else rng
    bboxes = np.vstack(bbox_result)
    labels = np.concatenate([np.full(b.shape[0], i, dtype=np.int32) for i, b in enumerate(bbox_result)])
    new_boxes = []
    for box, label in zip(bboxes, labels):
        x1 = min(max(0, int(box[0])), imsize[1])
        y1 = min(max(0, int(box[1])), imsize[0])
        x2 = min(max(x1 + 1, int(box[2])), imsize[1])
        y2 = min(max(y1 + 1, int(box[3])), imsize[0])
        if (y2 - y1 + 1 > 2) and (x2 - x1 + 1 > 2):
            new_boxes.append([x1, y1, x2, y2, box[4], label])
    if len(new_boxes) == 0:
        new_boxes.append([0, 0, imsize[1] - 1, imsize[0] - 1, 1.0, 0])
    new_boxes = np.array(new_boxes, dtype=int)
    n_candidate = min(topN, len(new_boxes))
    if len(new_boxes) <= nbox - n_candidate:
        idx = rng.choice(n_candidate, nbox - len(new_boxes), replace=True)
        return np.vstack((new_boxes, new_boxes[idx]))
    if len(new_boxes) <= nbox:
        idx = rng.choice(n_candidate, nbox - len(new_boxes), replace=False)
        return np.vstack((new_boxes, new_boxes[idx]))
    return new_boxes[:nbox]
