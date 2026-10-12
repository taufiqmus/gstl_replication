"""Dataset fitur CCD dan DAD. Port dari UString `src/DataLoader.py` (MIT, Bao et al. 2020).

Perbedaan dari asli:
  * mengembalikan tensor CPU (asli: memindahkan ke GPU di __getitem__, sehingga tidak bisa
    num_workers > 0); pemindahan ke device dilakukan di loop latih;
  * `np.int` (dihapus di NumPy >= 1.24) diganti `int`;
  * graf dibangun tervektorisasi (`graph.py`), hasil sama dengan asli (diuji).
Format yang diasumsikan (dari kode asli; [perlu dicek di data nyata]):
  CCD: <root>/vgg16_features/{train,test}.txt berisi "<path relatif .npz> <label>";
       npz berisi data (50,20,4096), labels (2,), det (50,19,6), ID;
       <root>/videos/Crash-1500.txt: "vid,[50 label 0/1],startframe,vid_ytb,lighting,weather,ego_involve".
  DAD: <root>/vgg16_features/{training,testing}/*.npz (hasil script/split_dad.py),
       data (100,20,4096), labels (2,), det (100,19,6), ID; toa positif = 90.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph import generate_st_graph

FEATURE_DIMS = {"vgg16": 4096, "res101": 2048}


def read_crash_annotations(anno_file):
    """Parser Crash-1500.txt, sama dengan `CrashDataset.read_anno_file` asli."""
    result = []
    with open(anno_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = {"vid": line.split(",[")[0]}
            labels = line.split(",[")[1].split("],")[0]
            items["label"] = [int(v) for v in labels.split(",")]
            if sum(items["label"]) <= 0:
                raise ValueError("anotasi kecelakaan tidak valid: %s" % items["vid"])
            others = line.split(",[")[1].split("],")[1].split(",")
            (items["startframe"], items["vid_ytb"], items["lighting"], items["weather"],
             items["ego_involve"]) = others
            result.append(items)
    return result


def crash_toa_dict(anno_file, n_frames=50):
    toa = {}
    for a in read_crash_annotations(anno_file):
        labels = np.array(a["label"], dtype=int)
        t = int(np.where(labels == 1)[0][0])
        toa[a["vid"]] = min(max(1, t), n_frames - 1)
    return toa


class _FeatureDataset(Dataset):
    n_frames = None
    n_obj = 19
    fps = None

    def __init__(self, feature="vgg16", return_meta=False):
        self.feature = feature
        self.dim_feature = FEATURE_DIMS[feature]
        self.return_meta = return_meta
        # untuk GSTL (diatur setelah konstruksi): kembalikan det sebagai elemen ke-6, dan/atau hanya fitur frame
        self.return_det = False
        self.frame_only = False

    def _pack(self, features, labels, detections, toa, vid):
        graph_edges, edge_weights = generate_st_graph(detections)
        if self.frame_only:
            features = features[:, :1]
        out = (torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
               torch.from_numpy(np.asarray(labels, dtype=np.float32)),
               torch.from_numpy(graph_edges),
               torch.from_numpy(edge_weights),
               torch.tensor([toa], dtype=torch.float32))
        if self.return_det:
            out = out + (torch.from_numpy(np.asarray(detections, dtype=np.float32)),)
        if self.return_meta:
            return out + (torch.from_numpy(np.asarray(detections, dtype=np.float32)), str(vid))
        return out


class CrashDataset(_FeatureDataset):
    n_frames = 50
    fps = 10.0

    def __init__(self, root, phase="train", feature="vgg16", return_meta=False, file_list=None):
        """file_list: opsional daftar (path_relatif, label) untuk subset (mis. split validasi)."""
        super().__init__(feature, return_meta)
        self.root = root
        self.feat_dir = os.path.join(root, feature + "_features")
        if file_list is None:
            file_list = self.read_datalist(os.path.join(self.feat_dir, "%s.txt" % phase))
        self.files_list = [f for f, _ in file_list]
        self.labels_list = [l for _, l in file_list]
        self.toa_dict = crash_toa_dict(os.path.join(root, "videos", "Crash-1500.txt"), self.n_frames)

    @staticmethod
    def read_datalist(list_file):
        if not os.path.exists(list_file):
            raise FileNotFoundError(list_file)
        out = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    filename, label = line.split(" ")
                    out.append((filename, int(label)))
        return out

    def __len__(self):
        return len(self.files_list)

    def raw(self, index):
        """(fitur (50,20,4096), labels, det, vid, toa) mentah, untuk ablasi/analisis."""
        path = os.path.join(self.feat_dir, self.files_list[index])
        with np.load(path) as data:
            features = data["data"]
            labels = data["labels"]
            detections = data["det"]
            vid = str(data["ID"])
        toa = self.toa_dict[vid] if labels[1] > 0 else self.n_frames + 1
        return features, labels, detections, vid, toa

    def __getitem__(self, index):
        features, labels, detections, vid, toa = self.raw(index)
        return self._pack(features, labels, detections, toa, vid)


class CrashCompactDataset(_FeatureDataset):
    """CCD dalam format RINGKAS (hanya yang dipakai GSTL): per klip satu .npz berisi
         frame  (50, 4096) float16  fitur fc7 VGG16 frame utuh
         det    (50, 19, 6) float32 [x1,y1,x2,y2,skor,kelas_lama] piksel 1280x720; kotak kosong (padding) = 0
         labels (2,), ID
    root berisi train.txt/test.txt ("<path relatif .npz> <label>"), Crash-1500.txt (untuk toa), dan berkas .npz pada path relatif itu.
    Dibuat oleh tools/make_compact_ccd.py (dari .npz rilis) atau tools/ekstraksi (dari mp4 dengan detektor pilihan)."""
    n_frames = 50
    fps = 10.0

    def __init__(self, root, phase="train", feature="vgg16", return_meta=False, file_list=None):
        super().__init__(feature, return_meta)
        self.root = root
        if file_list is None:
            file_list = CrashDataset.read_datalist(os.path.join(root, "%s.txt" % phase))
        self.files_list = [f for f, _ in file_list]
        self.labels_list = [l for _, l in file_list]
        self.toa_dict = crash_toa_dict(os.path.join(root, "Crash-1500.txt"), self.n_frames)

    def __len__(self):
        return len(self.files_list)

    def raw(self, index):
        with np.load(os.path.join(self.root, self.files_list[index])) as data:
            features = data["frame"].astype(np.float32)[:, None, :]      # (50, 1, 4096): indeks 0 = frame utuh
            labels = data["labels"]
            detections = data["det"].astype(np.float32)
            vid = str(data["ID"])
        toa = self.toa_dict[vid] if labels[1] > 0 else self.n_frames + 1
        return features, labels, detections, vid, toa

    def __getitem__(self, index):
        features, labels, detections, vid, toa = self.raw(index)
        return self._pack(features, labels, detections, toa, vid)


class DADDataset(_FeatureDataset):
    n_frames = 100
    fps = 20.0

    def __init__(self, root, phase="training", feature="vgg16", return_meta=False, files=None):
        super().__init__(feature, return_meta)
        self.root = root
        self.phase = phase
        self.data_dir = os.path.join(root, feature + "_features", phase)
        if files is None:
            if not os.path.isdir(self.data_dir):
                raise FileNotFoundError(self.data_dir)
            files = sorted(os.listdir(self.data_dir))
        self.files_list = list(files)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.files_list[index])
        with np.load(path) as data:
            features = data["data"]
            labels = data["labels"]
            detections = data["det"]
            vid = str(data["ID"])
        toa = 90.0 if labels[1] > 0 else self.n_frames + 1
        return self._pack(features, labels, detections, toa, vid)



def collate_with_meta(batch):
    """Collate untuk return_meta=True (deteksi dan id video tidak di-stack sebagai tensor id)."""
    cols = list(zip(*batch))
    out = [torch.stack(c) for c in cols[:6]]
    out.append(list(cols[6]))
    return tuple(out)
