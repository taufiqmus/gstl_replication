"""Pemuat checkpoint GSTL dari repo ini (konfigurasi pelatihan tertanam di checkpoint)."""
import torch

from .gstl import build_gstl


def load_gstl_model(path, device="cpu", n_frames=50, fps=10.0, x_dim=4096, n_obj=19):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if cfg.get("model", "gstl") != "gstl":
        raise ValueError("%s bukan checkpoint GSTL" % path)
    m = build_gstl(n_frames, fps, x_dim=x_dim, h_dim=cfg["hidden_dim"], z_dim=cfg["latent_dim"], n_layers=cfg.get("num_rnn", 1),
                   n_obj=n_obj, frame_embed=cfg["gstl_frame_embed"], box_mode=cfg["gstl_box_mode"], box_embed=cfg["gstl_box_embed"],
                   with_saa=not cfg.get("gstl_no_saa", False), uncertain_ranking=cfg.get("gstl_ranking", False),
                   rnn_trainable=cfg.get("rnn_trainable", False), exp_loss_mode=cfg.get("exp_loss", "faithful"), no_frame=cfg.get("gstl_no_frame", False), no_box=cfg.get("gstl_no_box", False))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck.get("epoch")
