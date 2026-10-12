"""Graf objek per frame. Port tervektorisasi dari `generate_st_graph` UString (src/DataLoader.py).

Perilaku asli yang dipertahankan:
  * sisi = semua pasangan (i, j) dengan i < j (itertools.combinations; urutan sama dengan
    networkx Graph.edges() di kode asli), jadi 171 sisi untuk 19 objek, TANPA arah balik;
  * bobot = exp(-d) dengan d = jarak Euklid KUADRAT antar pusat bbox dalam satuan koordinat
    deteksi apa adanya, disimpan float32, lalu dinormalisasi (jumlah = 1); bila semua bobot
    0 (underflow) diganti 1.

[perlu dicek di data nyata] bila koordinat bbox dalam piksel, exp(-d) underflow ke 0 untuk
hampir semua pasangan objek nyata (d > ~104 px^2 di float32), sehingga bobot efektif hanya
datang dari pasangan bbox kosong (0,0,0,0) atau jatuh ke fallback seragam. Lihat
`edge_weight_stats` untuk diagnosis.
"""
import itertools

import numpy as np


def edge_list(num_boxes):
    """(E, 2) pasangan i<j, urutan sama dengan kode asli."""
    return np.array(list(itertools.combinations(range(num_boxes), 2)), dtype=np.int64).reshape(-1, 2)


def compute_edge_weights(boxes, edges):
    """boxes: (T, N, 4) x1,y1,x2,y2; edges: (E, 2). Keluaran (T, E) float32."""
    boxes = np.asarray(boxes)
    cx = 0.5 * (boxes[..., 0] + boxes[..., 2])
    cy = 0.5 * (boxes[..., 1] + boxes[..., 3])
    # aritmetika float64 (numpy skalar x float Python di kode asli menghasilkan float64),
    # lalu disimpan ke array float32 seperti asli
    cx = cx.astype(np.float64)
    cy = cy.astype(np.float64)
    i, j = edges[:, 0], edges[:, 1]
    d = (cx[:, i] - cx[:, j]) ** 2 + (cy[:, i] - cy[:, j]) ** 2
    w = np.exp(-d).astype(np.float32)                      # T x E
    s = w.sum(axis=1, keepdims=True, dtype=np.float32)
    out = np.where(s > 0, w / np.where(s > 0, s, 1), np.float32(1.0)).astype(np.float32)
    return out


def generate_st_graph(detections):
    """detections: (T, N, >=4). Keluaran: graph_edges (T, 2, E) int64, edge_weights (T, E) float32."""
    num_frames, num_boxes = detections.shape[:2]
    edges = edge_list(num_boxes)
    graph_edges = np.broadcast_to(edges.T[None], (num_frames, 2, len(edges))).copy()
    edge_weights = compute_edge_weights(detections[:, :, :4], edges)
    return graph_edges, edge_weights


def edge_weight_stats(detections):
    """Diagnosis: seberapa sering bobot sisi degeneratif (fallback seragam / terkonsentrasi)."""
    _, w = generate_st_graph(detections)
    E = w.shape[1]
    uniform = np.all(np.isclose(w, 1.0), axis=1)
    nonzero = (w > 0).sum(axis=1)
    return {
        "frames": int(w.shape[0]),
        "frac_frames_uniform_fallback": float(uniform.mean()),
        "mean_nonzero_edges_per_frame": float(nonzero.mean()),
        "num_edges": int(E),
    }
