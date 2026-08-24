from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F


def set_torch_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested, but CUDA is not available.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("device='mps' requested, but MPS is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device!r}. Use 'auto', 'cpu', 'cuda', or 'mps'.")
    return requested


class STAGATEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        dims = (int(input_dim),) + tuple(int(dim) for dim in hidden_dims)
        if len(dims) < 2:
            raise ValueError("hidden_dims must contain at least one dimension.")
        self.n_layers = len(dims) - 1
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(dims[i], dims[i + 1])) for i in range(self.n_layers)]
        )
        self.att_src = nn.ParameterList(
            [nn.Parameter(torch.empty(dims[i + 1], 1)) for i in range(self.n_layers - 1)]
        )
        self.att_dst = nn.ParameterList(
            [nn.Parameter(torch.empty(dims[i + 1], 1)) for i in range(self.n_layers - 1)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        for param in list(self.att_src) + list(self.att_dst):
            nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        attentions: list[torch.Tensor] = []
        for layer in range(self.n_layers):
            h = h @ self.weights[layer]
            if layer == self.n_layers - 1:
                continue
            h, attention = self._attention_layer(h, edge_index, layer)
            attentions.append(attention)
            h = F.elu(h)
        embedding = h

        for layer in range(self.n_layers - 1, -1, -1):
            h = h @ self.weights[layer].T
            if layer == 0:
                continue
            h = aggregate_with_attention(h, edge_index, attentions[layer - 1])
            h = F.elu(h)
        return embedding, h

    def _attention_layer(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        src_score = h @ self.att_src[layer]
        dst_score = h @ self.att_dst[layer]
        logits = torch.sigmoid(src_score[src, 0] + dst_score[dst, 0])
        attention = segment_softmax(logits, dst, h.shape[0])
        return aggregate_with_attention(h, edge_index, attention), attention


def aggregate_with_attention(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    out = h.new_zeros(h.shape)
    out.index_add_(0, dst, h[src] * attention.unsqueeze(1))
    return out


def segment_softmax(values: torch.Tensor, segment_ids: torch.Tensor, n_segments: int) -> torch.Tensor:
    max_per_segment = values.new_full((int(n_segments),), -torch.inf)
    max_per_segment.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    centered = values - max_per_segment[segment_ids]
    exp_values = torch.exp(centered)
    sum_per_segment = values.new_zeros((int(n_segments),))
    sum_per_segment.index_add_(0, segment_ids, exp_values)
    return exp_values / sum_per_segment[segment_ids].clamp_min(1e-12)


def build_stagate_edge_index(
    *,
    spatial: np.ndarray,
    mode: str,
    k: int,
    radius_cutoff: float,
) -> np.ndarray:
    n_spots = int(spatial.shape[0])
    if n_spots < 2:
        raise ValueError("At least two spots are required to build a STAGATE graph.")
    edges: set[tuple[int, int]] = {(i, i) for i in range(n_spots)}
    if mode == "radius":
        nn = NearestNeighbors(radius=float(radius_cutoff)).fit(spatial)
        distances, indices = nn.radius_neighbors(spatial, return_distance=True)
        for i in range(n_spots):
            for j, dist in zip(indices[i], distances[i], strict=False):
                if float(dist) <= 0:
                    continue
                edges.add((int(j), i))
    elif mode == "knn":
        k_eff = max(1, min(int(k), n_spots - 1))
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(spatial)
        indices = nn.kneighbors(spatial, return_distance=False)
        for i in range(n_spots):
            for j in indices[i, 1:]:
                edges.add((int(j), i))
    else:
        raise ValueError(f"Unsupported STAGATE graph mode: {mode!r}. Use 'radius' or 'knn'.")
    return np.asarray(sorted(edges), dtype=np.int64).T


def train_stagate_latent(
    *,
    features: np.ndarray,
    spatial: np.ndarray,
    hidden_dims: tuple[int, ...],
    epochs: int,
    lr: float,
    weight_decay: float,
    gradient_clipping: float,
    graph_mode: str,
    stagate_graph_k: int,
    radius_cutoff: float,
    random_seed: int,
    device: str,
    verbose: bool,
    progress_interval: int,
) -> tuple[np.ndarray, list[float], dict]:
    if features.ndim != 2:
        raise ValueError(f"STAGATE features must be 2D, got shape={features.shape}.")
    set_torch_seed(random_seed)
    resolved_device = resolve_device(device)
    edge_index_np = build_stagate_edge_index(
        spatial=spatial,
        mode=graph_mode,
        k=stagate_graph_k,
        radius_cutoff=radius_cutoff,
    )
    if verbose:
        print(
            "[RSS STAGATE encoder] "
            f"input_shape={tuple(features.shape)} "
            f"hidden_dims={tuple(hidden_dims)} "
            f"graph_mode={graph_mode} "
            f"radius_cutoff={float(radius_cutoff):.3f} "
            f"stagate_graph_k={int(stagate_graph_k)} "
            f"n_edges={int(edge_index_np.shape[1])} "
            f"device={resolved_device} "
            f"epochs={int(epochs)}",
            flush=True,
        )
    model = STAGATEEncoder(features.shape[1], hidden_dims).to(resolved_device)
    x = torch.as_tensor(features, dtype=torch.float32, device=resolved_device)
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    losses: list[float] = []
    total_epochs = max(0, int(epochs))
    interval = max(1, int(progress_interval))
    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, reconstruction = model(x, edge_index)
        loss = torch.sqrt(torch.sum((x - reconstruction) ** 2))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clipping))
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        if verbose and (epoch == 0 or (epoch + 1) % interval == 0 or epoch + 1 == total_epochs):
            print(
                f"[RSS STAGATE encoder] epoch {epoch + 1}/{total_epochs} loss={loss_value:.6f}",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        embedding, _ = model(x, edge_index)
    meta = {
        "device": resolved_device,
        "graph_mode": str(graph_mode),
        "stagate_graph_k": int(stagate_graph_k),
        "radius_cutoff": float(radius_cutoff),
        "n_edges": int(edge_index_np.shape[1]),
        "final_loss": float(losses[-1]) if losses else None,
        "progress_interval": int(interval),
    }
    return embedding.detach().cpu().numpy().astype(np.float32, copy=False), losses, meta
