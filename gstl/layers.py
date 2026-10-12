"""Lapisan graf dan prediktor. Port dari UString `src/Models.py` (MIT, Bao et al. 2020) ke PyTorch 2.x.

Pengganti dependensi lama:
  * `torch_scatter.scatter_add` dan `MessagePassing.propagate` → `Tensor.scatter_add_` tervektorisasi
    atas batch (asli: loop Python per sampel);
  * `torch_geometric.utils` tidak dipakai lagi (GCNConv asli sudah punya add_self_loops sendiri);
  * `torch.autograd.Variable` dan `inspect.getargspec` dihapus.
Semantika dipertahankan persis, termasuk:
  * pesan mengalir dari `col` (edge_index[1]) ke `row` (edge_index[0]); dengan daftar sisi
    `combinations` (i < j) dari DataLoader asli, informasi hanya mengalir dari objek berindeks
    besar ke kecil (ditambah self-loop). Ini perilaku asli, bukan pilihan kita;
  * derajat dihitung atas `row` dengan bobot sisi, norm = d^-1/2[row] * w * d^-1/2[col];
  * bias ditambahkan sebelum aktivasi.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayes import BayesianLinear


def glorot_(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    with torch.no_grad():
        tensor.uniform_(-stdv, stdv)


def _identity(x):
    return x


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=F.relu, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_(self.weight)
        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        """x: (B, N, C); edge_index: (B, 2, E) long; edge_weight: (B, E). Keluaran (B, N, out)."""
        B, N, _ = x.shape
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(0), edge_index.size(-1), dtype=x.dtype, device=x.device)
        assert edge_weight.size(-1) == edge_index.size(-1)
        loop = torch.arange(N, device=edge_index.device, dtype=edge_index.dtype)
        loop = loop.view(1, 1, N).expand(B, 2, N)
        ei = torch.cat([edge_index, loop], dim=-1)                       # B x 2 x E'
        ew = torch.cat([edge_weight, edge_weight.new_ones(B, N)], dim=-1)  # B x E'
        row, col = ei[:, 0], ei[:, 1]                                      # B x E'

        deg = ew.new_zeros(B, N).scatter_add_(1, row, ew)
        deg_inv = deg.pow(-0.5)
        deg_inv = deg_inv.masked_fill(torch.isinf(deg_inv), 0.0)
        norm = deg_inv.gather(1, row) * ew * deg_inv.gather(1, col)       # B x E'

        xw = torch.matmul(x, self.weight)                                  # B x N x out
        C = xw.size(-1)
        msg = norm.unsqueeze(-1) * xw.gather(1, col.unsqueeze(-1).expand(-1, -1, C))
        out = xw.new_zeros(B, N, C).scatter_add_(1, row.unsqueeze(-1).expand(-1, -1, C), msg)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class GraphGRUGCN(nn.Module):
    """GRU dengan gerbang GCN.

    PENTING (temuan dari kode asli): di UString asli keenam GCNConv gerbang disimpan dalam
    list Python biasa, bukan nn.ModuleList. Akibatnya bobotnya TIDAK terdaftar sebagai
    parameter: tidak ikut optimizer (tetap acak sesuai inisialisasi), tidak ikut
    `clip_grad_norm_`, dan tidak tersimpan di checkpoint (checkpoint asli hanya bisa
    dipakai ulang karena seed 123 membuat inisialisasi acaknya sama tiap kali model dibangun).

    trainable=False (bawaan, setia ke asli): bobot gerbang dibekukan (requires_grad=False)
        tetapi tetap tersimpan di state_dict sehingga checkpoint kita mandiri.
    trainable=True: varian perbaikan, bobot gerbang ikut dilatih.
    """

    GATES = ("xz", "hz", "xr", "hr", "xh", "hh")

    def __init__(self, input_size, hidden_size, n_layer, bias=True, trainable=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.trainable = trainable
        lists = {g: [] for g in self.GATES}
        for i in range(n_layer):
            for g in self.GATES:  # urutan sama dengan asli: xz, hz, xr, hr, xh, hh per lapis
                in_size = input_size if (i == 0 and g[0] == "x") else hidden_size
                lists[g].append(GCNConv(in_size, hidden_size, act=_identity, bias=bias))
        for g in self.GATES:
            setattr(self, "weight_" + g, nn.ModuleList(lists[g]))
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, inp, edgidx, h, edge_weight=None):
        h_out = []
        for i in range(self.n_layer):
            x_in = inp if i == 0 else h_out[i - 1]
            z_g = torch.sigmoid(self.weight_xz[i](x_in, edgidx, edge_weight) + self.weight_hz[i](h[i], edgidx, edge_weight))
            r_g = torch.sigmoid(self.weight_xr[i](x_in, edgidx, edge_weight) + self.weight_hr[i](h[i], edgidx, edge_weight))
            h_tilde = torch.tanh(self.weight_xh[i](x_in, edgidx, edge_weight) + self.weight_hh[i](r_g * h[i], edgidx, edge_weight))
            h_out.append(z_g * h[i] + (1 - z_g) * h_tilde)
        return torch.stack(h_out, dim=0)


class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=(0, 0)):
        super().__init__()
        self.act = act
        self.dropout = list(dropout)
        self.dense1 = nn.Linear(input_dim, 64)
        self.dense2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout[0], training=self.training)
        x = self.act(self.dense1(x))
        x = F.dropout(x, self.dropout[1], training=self.training)
        return self.dense2(x)


class SelfAttAggregate(nn.Module):
    def __init__(self, agg_dim):
        super().__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.empty(agg_dim, 1))
        self.softmax = nn.Softmax(dim=-1)
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hiddens, avgsum="sum"):
        """hiddens: (B, N, H, T) → (B, 2H)."""
        maxpool = torch.max(hiddens, dim=1)[0]
        avgpool = torch.sum(hiddens, dim=1) if avgsum == "sum" else torch.mean(hiddens, dim=1)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1)                      # B x 2H x T
        energy = torch.bmm(agg_spatial.permute(0, 2, 1), agg_spatial)            # B x T x T
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute(0, 2, 1))       # B x T x 2H
        weight = self.weight.unsqueeze(0).expand(hiddens.size(0), -1, -1)
        agg_feature = torch.bmm(weighted_feat.permute(0, 2, 1), weight)          # B x 2H x 1
        return agg_feature.squeeze(-1)


class BayesianPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, pi=0.5, sigma_1=None, sigma_2=None):
        super().__init__()
        self.act = act
        self.l1 = BayesianLinear(input_dim, 64, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)
        self.l2 = BayesianLinear(64, output_dim, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)

    def forward(self, x, sample=False):
        x = self.act(self.l1(x, sample))
        return self.l2(x, sample)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(self, x, out_dim=2, npass=2, testing=False, eval_uncertain=False):
        """npass sampel bobot → logit rata-rata (dirata-rata di ruang logit, sama dengan asli).

        Catatan: di kode asli cabang testing=True selalu crash (IndexError: tensor berukuran
        npass+1 diisi npass+1 sampel lalu indeks npass ditulis lagi). Cabang itu tidak pernah
        dipakai `main.py`. Di sini testing=True = npass sampel + 1 lintasan deterministik (mu).
        """
        outputs = []
        log_priors, log_posts = [], []
        for _ in range(npass):
            outputs.append(self(x, sample=True))
            log_priors.append(torch.as_tensor(self.log_prior(), dtype=x.dtype, device=x.device))
            log_posts.append(torch.as_tensor(self.log_variational_posterior(), dtype=x.dtype, device=x.device))
        if testing:
            outputs.append(self(x, sample=False))
        outputs = torch.stack(outputs, dim=0)          # npass x B x C
        output = outputs.mean(0)
        log_prior = torch.stack(log_priors).mean()
        log_post = torch.stack(log_posts).mean()
        B = x.size(0)
        uncertain_alea = x.new_zeros(B, out_dim, out_dim)
        uncertain_epis = x.new_zeros(B, out_dim, out_dim)
        if eval_uncertain:
            p = F.softmax(outputs, dim=-1)                                       # N x B x C
            p_diag = torch.diag_embed(p, offset=0, dim1=-2, dim2=-1)
            p_cov = torch.matmul(p.unsqueeze(-1), p.unsqueeze(-1).transpose(-1, -2))
            uncertain_alea = torch.mean(p_diag - p_cov, dim=0)
            p_bar = torch.mean(p, dim=0)
            d = (p - p_bar).unsqueeze(-1)
            uncertain_epis = torch.mean(torch.matmul(d, d.transpose(-1, -2)), dim=0)
        return {"pred_mean": output, "log_prior": log_prior, "log_posterior": log_post,
                "aleatoric": uncertain_alea, "epistemic": uncertain_epis}
