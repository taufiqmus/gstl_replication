"""Model UString (GCRNN + BNN + SAA). Port dari UString `src/Models.py` (MIT, Bao et al. 2020).

Dua opsi menandai perbedaan antara "replikasi setia" dan "perbaikan":

  rnn_trainable (bawaan False = setia):
      lihat `GraphGRUGCN`. Di kode asli bobot GCN di dalam GRU tidak pernah dilatih.

  exp_loss_mode (bawaan "faithful"):
      Di kode asli `toa` berbentuk (B, 1) (hasil collate DataLoader dari list [toa]),
      sedangkan cross-entropy berbentuk (B,). Perkalian keduanya ter-broadcast menjadi
      (B, B), sehingga bobot eksponensial video positif j BUKAN exp(-max(0, (toa_j - t - 1)/fps)),
      melainkan rerata batch: mean_i exp(-max(0, (toa_i - t - 1)/fps)) atas SEMUA video
      di batch (termasuk negatif, yang toa-nya = T+1). "faithful" mempertahankan perilaku ini;
      "per_sample" memakai toa milik masing-masing video seperti rumus di paper.

Urutan pembuatan sub-modul sama dengan asli (phi_x, enc_gcn1, enc_gcn2, rnn, predictor,
predictor_aux, self_aggregation) sehingga dengan seed yang sama inisialisasi acaknya
mengonsumsi RNG dalam urutan yang sama. Ini diperlukan untuk memakai checkpoint asli
(yang tidak menyimpan bobot GRU-GCN); lihat `load_original_checkpoint`.
"""
import torch
import torch.nn as nn

from .layers import AccidentPredictor, BayesianPredictor, GCNConv, GraphGRUGCN, SelfAttAggregate, _identity


class UString(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True,
                 uncertain_ranking=False, rnn_trainable=False, exp_loss_mode="faithful"):
        super().__init__()
        if exp_loss_mode not in ("faithful", "per_sample"):
            raise ValueError(exp_loss_mode)
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.with_saa = with_saa
        self.uncertain_ranking = uncertain_ranking
        self.exp_loss_mode = exp_loss_mode

        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.enc_gcn1 = GCNConv(h_dim + h_dim, h_dim)
        self.enc_gcn2 = GCNConv(h_dim + h_dim, z_dim, act=_identity)
        self.rnn = GraphGRUGCN(h_dim + h_dim + z_dim, h_dim, n_layers, bias=True, trainable=rnn_trainable)
        self.predictor = BayesianPredictor(n_obj * z_dim, 2)
        if self.with_saa:
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=(0.5, 0.0))
            self.self_aggregation = SelfAttAggregate(self.n_frames)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x, y, toa, graph, hidden_in=None, edge_weights=None, npass=2, nbatch=80,
                testing=False, eval_uncertain=False):
        """
        x: (B, T, 1+n_obj, D) fitur; indeks 0 = fitur frame utuh, 1.. = objek
        y: (B, 2) one-hot; toa: (B, 1) atau (B,); graph: (B, T, 2, E); edge_weights: (B, T, E)
        """
        losses = {"cross_entropy": 0, "log_posterior": 0, "log_prior": 0, "total_loss": 0}
        if self.with_saa:
            losses["auxloss"] = 0
        if self.uncertain_ranking:
            losses["ranking"] = 0
            Ut = x.new_zeros(x.size(0))
        all_outputs, all_hidden = [], []

        if hidden_in is None:
            h = x.new_zeros(self.n_layers, x.size(0), self.n_obj, self.h_dim)
        else:
            h = hidden_in.to(x.device)

        for t in range(x.size(1)):
            x_t = self.phi_x(x[:, t])
            img_embed = x_t[:, 0, :].unsqueeze(1).expand(-1, self.n_obj, -1)
            obj_embed = x_t[:, 1:, :]
            x_t = torch.cat([obj_embed, img_embed], dim=-1)

            enc = self.enc_gcn1(x_t, graph[:, t], edge_weight=edge_weights[:, t])
            z_t = self.enc_gcn2(torch.cat([enc, h[-1]], -1), graph[:, t], edge_weight=edge_weights[:, t])

            embed = z_t.reshape(z_t.size(0), -1)
            output_dict = self.predictor.sample_elbo(embed, npass=npass, testing=testing, eval_uncertain=eval_uncertain)
            dec_t = output_dict["pred_mean"]

            h = self.rnn(torch.cat([x_t, z_t], -1), graph[:, t], h, edge_weight=edge_weights[:, t])

            losses["log_posterior"] += output_dict["log_posterior"] / nbatch
            losses["log_prior"] += output_dict["log_prior"] / nbatch
            losses["cross_entropy"] += self._exp_loss(dec_t, y, t, toa=toa, fps=self.fps)
            if self.uncertain_ranking:
                L5, Ut = self._uncertainty_ranking(output_dict, Ut)
                losses["ranking"] += L5

            all_outputs.append(output_dict)
            all_hidden.append(h[-1])

        if self.with_saa:
            embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1), "avg")
            dec = self.predictor_aux(embed_video)
            losses["auxloss"] = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))

        return losses, all_outputs, all_hidden

    def _exp_loss(self, pred, target, time, toa, fps=20.0):
        target_cls = target[:, 1].to(torch.long)
        if self.exp_loss_mode == "per_sample":
            toa = toa.reshape(-1)
        else:
            # setia ke asli: toa (B, 1) → broadcast (B, B). Paksa bentuk (B, 1) agar perilaku
            # asli tetap terjadi walau pemanggil memberi (B,).
            toa = toa.reshape(-1, 1)
        penalty = -torch.max(torch.zeros_like(toa, dtype=pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        neg_loss = self.ce_loss(pred, target_cls)
        return torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))

    def _uncertainty_ranking(self, output_dict, Ut, eU_only=True):
        aleatoric = output_dict["aleatoric"]
        epistemic = output_dict["epistemic"]
        if eU_only:
            uncertainty = epistemic[:, 0, 0] + epistemic[:, 1, 1]
        else:
            uncertainty = aleatoric[:, 1, 1] + epistemic[:, 1, 1]
        loss = torch.mean(torch.clamp(uncertainty - Ut, min=0))
        return loss, uncertainty


def build_model(dim_feature, n_obj, n_frames, fps, hidden_dim=256, latent_dim=256, num_rnn=1,
                rnn_trainable=False, exp_loss_mode="faithful", seed=None):
    """Bangun model dengan hiperparameter bawaan main.py asli (with_saa=True, uncertain_ranking=True).

    seed: bila diberikan, torch.manual_seed(seed) dipanggil TEPAT sebelum konstruksi
    (asli: seed 123 di awal main.py, tanpa konsumsi RNG lain sebelum model dibangun).
    """
    if seed is not None:
        torch.manual_seed(seed)
    return UString(dim_feature, hidden_dim, latent_dim, n_layers=num_rnn, n_obj=n_obj, n_frames=n_frames, fps=fps,
                   with_saa=True, uncertain_ranking=True, rnn_trainable=rnn_trainable, exp_loss_mode=exp_loss_mode)


def load_original_checkpoint(model, path, map_location="cpu"):
    """Muat checkpoint dari kode UString asli (`{'epoch','model','optimizer'}`).

    Checkpoint asli tidak memuat bobot GRU-GCN (tidak terdaftar). Bobot itu harus berasal
    dari inisialisasi acak dengan seed 123 dan urutan konstruksi yang sama → bangun model
    dengan `build_model(..., seed=123)`. [belum diverifikasi] bahwa RNG CPU torch 2.3
    menghasilkan bilangan yang sama dengan torch 1.x untuk `uniform_`; uji dengan
    membandingkan AP model pretrained terhadap angka yang dilaporkan.
    """
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("rnn.")]
    if bad_missing or unexpected:
        raise RuntimeError("checkpoint tidak cocok: missing=%s unexpected=%s" % (bad_missing, unexpected))
    return ckpt.get("epoch") if isinstance(ckpt, dict) else None, missing
