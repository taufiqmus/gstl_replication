"""Ekstraksi fitur gaya UString (`demo.py::extract_features`), untuk video baru atau video yang
di-encode ulang (protokol "kualitas disamakan", uji transfer DASTRP).

Yang ditiru dari demo.py:
  * deteksi Cascade R-CNN per frame → `bbox_sampling` 19 kotak (skor terpotong int, isi
    kekurangan dengan duplikat acak); np.random di-seed 123 per video (demo: satu video per proses);
  * VGG16 ImageNet (torchvision), keluaran fc7 SEBELUM ReLU (classifier[:-3]), 4096-d;
  * crop kotak dan frame utuh → Resize(256) → CenterCrop(224) → ToTensor, TANPA Normalize;
  * fitur[t, 0] = frame utuh, fitur[t, 1:] = 19 objek; det (T, 19, 6) = x1,y1,x2,y2,skor(int),kelas;
  * bila video lebih pendek dari n_frames, frame sebelumnya disalin (catatan: di demo, frame
    salinan sudah berformat RGB lalu dikonversi lagi BGR→RGB → kanal tertukar; ditiru, bisa dimatikan).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .detector import bbox_sampling


class VGG16FC7(nn.Module):
    def __init__(self, weights="IMAGENET1K_V1"):
        super().__init__()
        vgg = models.vgg16(weights=weights)
        self.feature = vgg.features
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:-3])
        self.dim_feat = 4096

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x.view(x.size(0), -1))


TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


def read_frames(video_file, n_frames, frame_indices=None):
    """Baca frame BGR. frame_indices=None → frame 0..n_frames-1 (seperti mmcv.VideoReader.get_frame)."""
    cap = cv2.VideoCapture(video_file)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_indices is None:
        # baca berurutan (tanpa seek per frame; lebih cepat, frame sama)
        frames = []
        for _ in range(n_frames):
            ok, fr = cap.read()
            frames.append(fr if ok else None)
        cap.release()
        return frames, total
    idx = list(frame_indices)
    frames = []
    for i in idx:
        if i >= total:
            frames.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        frames.append(fr if ok else None)
    cap.release()
    return frames, total


@torch.no_grad()
def extract_video_features(video_file, detector, feat_extractor, n_frames=50, n_boxes=19, seed=123,
                           frame_indices=None, replicate_short_video_bug=True, device="cuda"):
    rng = np.random.RandomState(seed)
    frames, total = read_frames(video_file, n_frames, frame_indices)
    features = np.zeros((len(frames), n_boxes + 1, feat_extractor.dim_feat), dtype=np.float32)
    detections = np.zeros((len(frames), n_boxes, 6))
    frame_prev = None
    raw_dets = []
    for t, frame in enumerate(frames):
        if frame is None:
            frame = frame_prev.copy()
            if not replicate_short_video_bug:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bbox_result = detector.detect(frame)
        raw_dets.append(bbox_result)
        bboxes = bbox_sampling(bbox_result, nbox=n_boxes, imsize=frame.shape[:2], rng=rng)
        detections[t] = bboxes
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rois = torch.stack([TRANSFORM(Image.fromarray(frame[b[1]:b[3], b[0]:b[2], :])) for b in bboxes])
        f_roi = feat_extractor(rois.float().to(device))
        f_frame = feat_extractor(TRANSFORM(Image.fromarray(frame)).unsqueeze(0).float().to(device))
        features[t, 0] = f_frame.squeeze(0).cpu().numpy()
        features[t, 1:] = f_roi.cpu().numpy()
        frame_prev = frame
    return {"data": features, "det": detections, "n_video_frames": total, "raw_dets": raw_dets}


@torch.no_grad()
def extract_frames_features(frames, detector, feat_extractor, n_frames=50, n_boxes=19, seed=123, device="cuda"):
    """Fitur untuk satu jendela frame BGR yang sudah ada di memori (dipakai demo.py).

    Sama dengan extract_video_features (deteksi -> 19 kotak -> VGG16 fc7 kotak dan frame utuh), tetapi masukan berupa daftar frame.
    Bila len(frames) < n_frames, frame terakhir disalin sebagai pengisi (skornya dibuang oleh pemanggil).
    Mengembalikan dict: data (n_frames,20,4096), det (n_frames,19,6), raw_dets (per frame, hanya frame asli), n_real.
    """
    rng = np.random.RandomState(seed)
    n_real = len(frames)
    frames = list(frames) + [frames[-1]] * (n_frames - n_real)
    features = np.zeros((n_frames, n_boxes + 1, feat_extractor.dim_feat), dtype=np.float32)
    detections = np.zeros((n_frames, n_boxes, 6))
    raw_dets = []
    for t, frame in enumerate(frames):
        bbox_result = detector.detect(frame)
        if t < n_real:
            raw_dets.append(bbox_result)
        detections[t] = bbox_sampling(bbox_result, nbox=n_boxes, imsize=frame.shape[:2], rng=rng)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rois = torch.stack([TRANSFORM(Image.fromarray(rgb[b[1]:b[3], b[0]:b[2], :])) for b in detections[t].astype(int)])
        features[t, 1:] = feat_extractor(rois.float().to(device)).cpu().numpy()
        features[t, 0] = feat_extractor(TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).float().to(device)).squeeze(0).cpu().numpy()
    return {"data": features, "det": detections, "raw_dets": raw_dets, "n_real": n_real}
