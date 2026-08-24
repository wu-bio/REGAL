from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from .stats import quantiles
    from .schema import NeighborsConfig
except ImportError:
    from RSS.stats import quantiles
    from RSS.schema import NeighborsConfig


def build_spatial_candidates(
    coords: np.ndarray,
    cfg: NeighborsConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[set[int]]]:
    n_spots = int(coords.shape[0])
    if n_spots < 2:
        return (
            [np.array([], dtype=np.int32) for _ in range(n_spots)],
            [np.array([], dtype=np.float32) for _ in range(n_spots)],
            [set() for _ in range(n_spots)],
        )

    k_query = max(2, min(cfg.candidate_k + 1, n_spots))
    nbrs = NearestNeighbors(n_neighbors=k_query)
    nbrs.fit(coords)

    distances_knn, indices_knn = nbrs.kneighbors(coords, return_distance=True)
    radius_neighbors = None
    radius_distances = None

    if cfg.candidate_mode in {"radius", "hybrid"}:
        if cfg.spatial_radius is None or cfg.spatial_radius <= 0:
            raise ValueError("spatial_radius must be > 0 when candidate_mode is radius or hybrid")
        radius_distances, radius_neighbors = nbrs.radius_neighbors(
            coords,
            radius=cfg.spatial_radius,
            return_distance=True,
        )

    candidates: list[np.ndarray] = []
    candidate_distances: list[np.ndarray] = []
    adjacency: list[set[int]] = [set() for _ in range(n_spots)]

    for i in range(n_spots):
        dist_map: dict[int, float] = {}

        for idx, dist in zip(indices_knn[i], distances_knn[i], strict=False):
            if idx == i:
                continue
            dist_map[int(idx)] = float(dist)
            adjacency[i].add(int(idx))
            adjacency[int(idx)].add(i)

        if radius_neighbors is not None:
            for idx, dist in zip(radius_neighbors[i], radius_distances[i], strict=False):
                if idx == i:
                    continue
                current = dist_map.get(int(idx))
                if current is None or dist < current:
                    dist_map[int(idx)] = float(dist)

        ordered = sorted(dist_map.items(), key=lambda x: x[1])
        candidates.append(np.array([x[0] for x in ordered], dtype=np.int32))
        candidate_distances.append(np.array([x[1] for x in ordered], dtype=np.float32))

    return candidates, candidate_distances, adjacency


def _nodes_within_hops(adjacency: list[set[int]], source: int, max_hops: int) -> set[int]:
    if max_hops <= 0:
        return set(adjacency[source])

    visited = {source}
    frontier = {source}
    within = set()

    for _ in range(max_hops):
        next_frontier: set[int] = set()
        for node in frontier:
            for nb in adjacency[node]:
                if nb in visited:
                    continue
                visited.add(nb)
                within.add(nb)
                next_frontier.add(nb)
        if not next_frontier:
            break
        frontier = next_frontier

    return within


def build_neighbors(
    latent: np.ndarray,
    candidate_idx: list[np.ndarray],
    candidate_dist: list[np.ndarray],
    spatial_adjacency: list[set[int]],
    cfg: NeighborsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[set[int]]]:
    n_spots = latent.shape[0]
    normal_k = max(0, min(int(cfg.normal_k), n_spots - 1))
    min_k = max(0, min(int(cfg.min_k), normal_k))
    include_self = bool(cfg.include_self)
    width = normal_k + (1 if include_self else 0)

    idx_out = np.full((n_spots, width), fill_value=-1, dtype=np.int32)
    sim_out = np.zeros((n_spots, width), dtype=np.float32)
    weight_out = np.zeros((n_spots, width), dtype=np.float32)

    latent_unit = _unit_latent_for_similarity(latent, cfg.similarity_metric)

    connectivity_sets = [
        _nodes_within_hops(spatial_adjacency, i, cfg.connectivity_hops) for i in range(n_spots)
    ]

    actual_neighbor_ks = []
    actual_microdomain_sizes = []
    for i in range(n_spots):
        base_candidates = candidate_idx[i]
        if base_candidates.size == 0:
            if include_self and width > 0:
                idx_out[i, 0] = i
                sim_out[i, 0] = 1.0
                weight_out[i, 0] = 1.0
                actual_microdomain_sizes.append(1)
            else:
                actual_microdomain_sizes.append(0)
            actual_neighbor_ks.append(0)
            continue

        connected = connectivity_sets[i]
        mask_connected = np.array([c in connected for c in base_candidates], dtype=bool)
        filtered_candidates = base_candidates[mask_connected]
        filtered_dist = candidate_dist[i][mask_connected]

        # Fallback semantics:
        # - decrease_k: strictly keep connectivity-filtered candidates (can be empty).
        # - spatial: allow fallback to spatial candidate pool if connectivity pool is empty.
        if filtered_candidates.size == 0 and cfg.fallback == "spatial":
            filtered_candidates = base_candidates
            filtered_dist = candidate_dist[i]

        if filtered_candidates.size == 0:
            if include_self and width > 0:
                idx_out[i, 0] = i
                sim_out[i, 0] = 1.0
                weight_out[i, 0] = 1.0
                actual_microdomain_sizes.append(1)
            else:
                actual_microdomain_sizes.append(0)
            actual_neighbor_ks.append(0)
            continue

        sims = latent_unit[filtered_candidates] @ latent_unit[i]
        candidate_records = [
            (int(node), float(sims[pos]), float(filtered_dist[pos]))
            for pos, node in enumerate(filtered_candidates)
        ]
        candidate_records.sort(key=lambda x: (-x[1], x[2], x[0]))

        core_take = min(min_k, len(candidate_records), normal_k)
        selected_nodes = [node for node, _, _ in candidate_records[:core_take]]
        selected_sims = [sim for _, sim, _ in candidate_records[:core_take]]
        selected_dist = [dist for _, _, dist in candidate_records[:core_take]]
        seen = set(selected_nodes)
        remaining = []
        if selected_sims:
            similarity_cutoff = float(np.median(np.asarray(selected_sims, dtype=np.float32)))
            similarity_cutoff -= max(0.0, float(cfg.similarity_margin))
        else:
            similarity_cutoff = -np.inf
        for node_int, sim_val, dist_val in candidate_records:
            if node_int in seen:
                continue
            if sim_val < similarity_cutoff:
                continue
            remaining.append((node_int, sim_val, dist_val))

        for node_int, sim_val, dist_val in remaining:
            if len(selected_nodes) >= normal_k:
                break
            selected_nodes.append(node_int)
            selected_sims.append(sim_val)
            selected_dist.append(dist_val)

        row_nodes: list[int] = []
        row_sims: list[float] = []
        row_weights: list[float] = []
        if include_self and width > 0:
            row_nodes.append(i)
            row_sims.append(1.0)

        neighbor_raw = np.asarray(
            [
                _neighbor_raw_weight(sim=sim, dist=dist, selected_distances=selected_dist, cfg=cfg)
                for sim, dist in zip(selected_sims, selected_dist, strict=False)
            ],
            dtype=np.float64,
        )
        if neighbor_raw.size > 0 and float(neighbor_raw.sum()) <= 0.0:
            neighbor_raw = np.ones_like(neighbor_raw, dtype=np.float64)

        if include_self:
            self_weight = float(np.clip(float(cfg.self_weight), 0.0, 1.0))
            if neighbor_raw.size == 0:
                row_weights.append(1.0)
            else:
                row_weights.append(self_weight)
                neighbor_weights = (1.0 - self_weight) * neighbor_raw / float(neighbor_raw.sum())
                row_weights.extend(float(x) for x in neighbor_weights.tolist())
        else:
            if neighbor_raw.size > 0:
                neighbor_weights = neighbor_raw / float(neighbor_raw.sum())
                row_weights.extend(float(x) for x in neighbor_weights.tolist())

        row_nodes.extend(selected_nodes)
        row_sims.extend(selected_sims)
        take = min(width, len(row_nodes))
        if take > 0:
            idx_out[i, :take] = np.asarray(row_nodes[:take], dtype=np.int32)
            sim_out[i, :take] = np.asarray(row_sims[:take], dtype=np.float32)
            weight_out[i, :take] = np.asarray(row_weights[:take], dtype=np.float32)

        actual_neighbor_ks.append(int(len(selected_nodes)))
        actual_microdomain_sizes.append(int(take))

    missing_ratio = float(np.mean(idx_out < 0)) if width > 0 else 0.0

    neighbors_meta = {
        "normal_k": cfg.normal_k,
        "normal_k_effective": normal_k,
        "min_k": cfg.min_k,
        "min_k_effective": min_k,
        "include_self": include_self,
        "self_weight": float(cfg.self_weight),
        "similarity_margin": float(cfg.similarity_margin),
        "distance_weight_power": float(cfg.distance_weight_power),
        "candidate_k": cfg.candidate_k,
        "candidate_mode": cfg.candidate_mode,
        "spatial_radius": cfg.spatial_radius,
        "similarity_metric": cfg.similarity_metric,
        "connectivity_hops": cfg.connectivity_hops,
        "fallback": cfg.fallback,
        "actual_neighbor_k_quantiles": quantiles(np.asarray(actual_neighbor_ks, dtype=np.float32)),
        "actual_neighbor_k_min": int(min(actual_neighbor_ks) if actual_neighbor_ks else 0),
        "actual_neighbor_k_max": int(max(actual_neighbor_ks) if actual_neighbor_ks else 0),
        "actual_microdomain_size_quantiles": quantiles(
            np.asarray(actual_microdomain_sizes, dtype=np.float32)
        ),
        "actual_microdomain_size_min": int(min(actual_microdomain_sizes) if actual_microdomain_sizes else 0),
        "actual_microdomain_size_max": int(max(actual_microdomain_sizes) if actual_microdomain_sizes else 0),
        "missing_ratio": missing_ratio,
        "valid_neighbor_ratio": float(np.mean(idx_out >= 0)) if width > 0 else 1.0,
    }

    return idx_out, sim_out, weight_out, neighbors_meta, connectivity_sets


def _neighbor_raw_weight(
    *,
    sim: float,
    dist: float,
    selected_distances: list[float],
    cfg: NeighborsConfig,
) -> float:
    sim_component = max(float(sim), 0.0) + 1e-6
    positive_dist = np.asarray([x for x in selected_distances if x > 0], dtype=np.float64)
    dist_scale = float(np.median(positive_dist)) if positive_dist.size else 1.0
    dist_scale = max(dist_scale, 1e-6)
    power = max(0.0, float(cfg.distance_weight_power))
    dist_component = 1.0 / (1.0 + (max(float(dist), 0.0) / dist_scale) ** power)
    return float(sim_component * dist_component)


def _unit_latent_for_similarity(latent: np.ndarray, similarity_metric: str) -> np.ndarray:
    if similarity_metric == "cosine":
        latent_for_similarity = latent
    elif similarity_metric == "centered_cosine":
        latent_for_similarity = latent - np.mean(latent, axis=0, keepdims=True)
    else:
        raise ValueError(
            "similarity_metric must be one of {'cosine', 'centered_cosine'}"
        )

    latent_norm = np.linalg.norm(latent_for_similarity, axis=1, keepdims=True)
    latent_norm[latent_norm == 0] = 1.0
    return latent_for_similarity / latent_norm
