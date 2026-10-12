"""GSTL (Geometric Spatio-Temporal Learning; Mahmood et al., IEEE Access 2023) di atas port UString.

Kode resmi tidak tersedia; implementasi ini disusun dari paper (persamaan 1–7, Gambar 2 kiri, §IV–V.B) dan
arsitektur GCRNN/UString yang menurut paper menjadi dasarnya ("architecture ... similar to the existing
GCRNN", "Hidden states of GRU and the architecture of GCN layers is similar to [2]", "training procedure ...
adopted from [2]"). Spesifikasi, asumsi, dan pertanyaan terbuka: catatan/spec_gstl.md.

Inti perbedaan dengan UString:
  * simpul graf: X_t^(i) = [Ψ(O_t^(i)), Φ(F_t)] → geometri kotak objek i + fitur VGG16 FRAME UTUH
    (UString: fitur visual objek + frame). Fitur visual objek TIDAK dipakai;
  * hidden GRU dan GCN 128 (§V.B); batch 10, lr 5e-4, Adam, 70 epoch;
  * loss (pers. 4): L_EXP + w1 (L_VPOS − L_PRI) + w2 L_BCE, w1 = 0,001, w2 = 10 → TANPA ranking loss
    ketidakpastian (bisa dinyalakan dengan uncertain_ranking=True untuk perbandingan).
Asumsi yang dapat diatur (lihat spec): bentuk Ψ (koordinat kotak ternormalisasi, mentah atau lewat FC),
FC untuk Φ(F_t), skala koordinat untuk bobot sisi graf.
"""
import torch
import torch.nn as nn

from .layers import AccidentPredictor, BayesianPredictor, GCNConv, GraphGRUGCN, SelfAttAggregate, _identity
from .model import UString

IMG_W, IMG_H = 1280.0, 720.0  # koordinat det CCD dalam piksel 1280x720 (det maks 1278/718)


BOX_MODES = ("xyxy", "xyxywh", "xyxywh+cls", "xyxywh+cls+score")
_BOX_DIM = {"xyxy": 4, "xyxywh": 6, "xyxywh+cls": 9, "xyxywh+cls+score": 10}


def box_dim(mode):
    return _BOX_DIM[mode]


def box_geometry(det, mode="xyxy"):
    """det: (B, T, N, >=6) piksel [x1,y1,x2,y2,skor,kelas] → Ψ ternormalisasi (B, T, N, d).

    xyxy             : x1/W, y1/H, x2/W, y2/H (4-d)
    xyxywh           : + lebar, tinggi ternormalisasi (6-d)
    xyxywh+cls       : + one-hot kelas lama (Car, Pedestrian, Cyclist) (9-d); kotak kosong (padding, lebar 0) -> semua 0
    xyxywh+cls+score : + skor deteksi di [0,1] (10-d; skor di .npz CCD rilis terpotong jadi bilangan bulat, jadi bermakna hanya untuk fitur hasil ekstraksi baru)
    """
    x1, y1, x2, y2 = det[..., 0] / IMG_W, det[..., 1] / IMG_H, det[..., 2] / IMG_W, det[..., 3] / IMG_H
    feats = [x1, y1, x2, y2]
    if mode != "xyxy":
        feats += [x2 - x1, y2 - y1]
    if "cls" in mode:
        valid = ((det[..., 2] - det[..., 0]) > 0).to(det.dtype)
        cls = det[..., 5].long().clamp(0, 2)
        feats += [(cls == c).to(det.dtype) * valid for c in range(3)]
        if mode.endswith("+score"):
            feats.append(det[..., 4].clamp(0, 1) * valid)
    return torch.stack(feats, dim=-1)


class GSTL(UString):
    needs_det = True  # loop latih/uji mengirim det=(B, T, N, 6)

    def __init__(self, x_dim=4096, h_dim=128, z_dim=128, n_layers=1, n_obj=19, n_frames=50, fps=10.0,
                 frame_embed=256, box_mode="xyxy", box_embed=0, with_saa=True, uncertain_ranking=False,
                 rnn_trainable=False, exp_loss_mode="faithful", no_frame=False, no_box=False):
        nn.Module.__init__(self)
        if exp_loss_mode not in ("faithful", "per_sample"):
            raise ValueError(exp_loss_mode)
        self.x_dim, self.h_dim, self.z_dim = x_dim, h_dim, z_dim
        self.n_layers, self.n_obj, self.n_frames, self.fps = n_layers, n_obj, n_frames, fps
        self.with_saa, self.uncertain_ranking, self.exp_loss_mode = with_saa, uncertain_ranking, exp_loss_mode
        self.box_mode = box_mode
        if box_mode not in BOX_MODES:
            raise ValueError(box_mode)
        box_in = box_dim(box_mode)
        self.no_frame = no_frame
        self.no_box = no_box
        if no_frame and no_box:
            raise ValueError('no_frame dan no_box tidak boleh bersamaan')
        # Φ(F_t): FC + ReLU seperti phi_x UString (asumsi; paper tidak menyebut dimensinya)
        self.phi_f = None if no_frame else nn.Sequential(nn.Linear(x_dim, frame_embed), nn.ReLU())
        # Ψ(O_t^(i)): geometri mentah atau FC + ReLU (asumsi)
        self.psi_b = nn.Sequential(nn.Linear(box_in, box_embed), nn.ReLU()) if box_embed else None
        node_dim = (0 if no_box else (box_embed or box_in)) + (0 if no_frame else frame_embed)
        self.node_dim = node_dim
        self.enc_gcn1 = GCNConv(node_dim, h_dim)
        self.enc_gcn2 = GCNConv(h_dim + h_dim, z_dim, act=_identity)
        self.rnn = GraphGRUGCN(node_dim + z_dim, h_dim, n_layers, bias=True, trainable=rnn_trainable)
        self.predictor = BayesianPredictor(n_obj * z_dim, 2)
        if with_saa:
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=(0.5, 0.0))
            self.self_aggregation = SelfAttAggregate(n_frames)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def node_features(self, x_t, det_t):
        """x_t: (B, 1+N, D) atau (B, 1, D) (hanya frame); det_t: (B, N, >=4) piksel → (B, N, node_dim)."""
        if self.no_box:                                               # varian fitur frame saja (tanpa geometri di simpul; graf tetap dari kotak)
            f = self.phi_f(x_t[:, 0])
            return f.unsqueeze(1).expand(-1, self.n_obj, -1)
        g = box_geometry(det_t, self.box_mode)                        # B x N x box_in
        if self.psi_b is not None:
            g = self.psi_b(g)
        if self.no_frame:                                             # varian geometri saja (tanpa fitur frame)
            return g
        f = self.phi_f(x_t[:, 0])                                     # B x frame_embed
        return torch.cat([g, f.unsqueeze(1).expand(-1, self.n_obj, -1)], dim=-1)

    def forward(self, x, y, toa, graph, hidden_in=None, edge_weights=None, npass=2, nbatch=80, testing=False,
                eval_uncertain=False, det=None):
        if det is None:
            raise ValueError("GSTL butuh det (B, T, N, 6): kotak deteksi per frame")
        losses = {"cross_entropy": 0, "log_posterior": 0, "log_prior": 0, "total_loss": 0}
        if self.with_saa:
            losses["auxloss"] = 0
        losses["ranking"] = x.new_zeros(())  # selalu ada (tensor float) agar log dan total_loss seragam
        Ut = x.new_zeros(x.size(0))
        all_outputs, all_hidden = [], []
        h = x.new_zeros(self.n_layers, x.size(0), self.n_obj, self.h_dim) if hidden_in is None else hidden_in
        for t in range(x.size(1)):
            x_t = self.node_features(x[:, t], det[:, t])
            enc = self.enc_gcn1(x_t, graph[:, t], edge_weight=edge_weights[:, t])
            z_t = self.enc_gcn2(torch.cat([enc, h[-1]], -1), graph[:, t], edge_weight=edge_weights[:, t])
            out = self.predictor.sample_elbo(z_t.reshape(z_t.size(0), -1), npass=npass, testing=testing,
                                             eval_uncertain=eval_uncertain or self.uncertain_ranking)
            h = self.rnn(torch.cat([x_t, z_t], -1), graph[:, t], h, edge_weight=edge_weights[:, t])
            losses["log_posterior"] += out["log_posterior"] / nbatch
            losses["log_prior"] += out["log_prior"] / nbatch
            losses["cross_entropy"] += self._exp_loss(out["pred_mean"], y, t, toa=toa, fps=self.fps)
            if self.uncertain_ranking:
                L5, Ut = self._uncertainty_ranking(out, Ut)
                losses["ranking"] += L5
            all_outputs.append(out)
            all_hidden.append(h[-1])
        if self.with_saa:
            dec = self.predictor_aux(self.self_aggregation(torch.stack(all_hidden, dim=-1), "avg"))
            losses["auxloss"] = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
        return losses, all_outputs, all_hidden


def build_gstl(n_frames, fps, seed=None, **kw):
    if seed is not None:
        torch.manual_seed(seed)
    return GSTL(n_frames=n_frames, fps=fps, **kw)


def n_params_registered(model):
    """Hitungan seperti paper: parameter terlatih (GRU-GCN beku/tak terdaftar tidak dihitung)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
