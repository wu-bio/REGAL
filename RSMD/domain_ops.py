from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .schema import (
    RSMDDomainAdjacencyConfig,
    RSMDDomainFilterConfig,
    RSMDFiltrationConfig,
    PotentialConfig,
)


def build_spot_graph_from_neighbors(neighbor_idx: np.ndarray) -> tuple[list[set[int]], np.ndarray]:
    n_spots = int(neighbor_idx.shape[0])
    adjacency: list[set[int]] = [set() for _ in range(n_spots)]
    edges: set[tuple[int, int]] = set()

    for i in range(n_spots):
        for nb in neighbor_idx[i]:
            j = int(nb)
            if j < 0 or j >= n_spots or j == i:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    edge_array = np.asarray(sorted(edges), dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    return adjacency, edge_array


def build_spot_graph_from_coords_local(
    coords: np.ndarray,
    k: int,
    max_distance_factor: float,
) -> tuple[list[set[int]], np.ndarray]:
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"coords must have shape [n_spots,2+], got {xy.shape}")

    n_spots = int(xy.shape[0])
    adjacency: list[set[int]] = [set() for _ in range(n_spots)]
    edges: set[tuple[int, int]] = set()
    if n_spots <= 1:
        return adjacency, np.empty((0, 2), dtype=np.int32)

    if not np.isfinite(xy[:, :2]).all():
        raise ValueError("coords contain non-finite values; cannot build spatial graph")

    tree = cKDTree(xy[:, :2])
    nn_dist, _ = tree.query(xy[:, :2], k=min(2, n_spots))
    nn_dist = np.asarray(nn_dist, dtype=np.float64)
    if nn_dist.ndim == 1:
        positive_nn = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
    else:
        positive_nn = nn_dist[:, 1][np.isfinite(nn_dist[:, 1]) & (nn_dist[:, 1] > 0)]
    if positive_nn.size == 0:
        return adjacency, np.empty((0, 2), dtype=np.int32)

    radius = float(np.median(positive_nn) * max(1.0, float(max_distance_factor)))
    k_eff = max(1, min(int(k), n_spots - 1))
    dist, idx = tree.query(xy[:, :2], k=k_eff + 1)
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int32)
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]

    for i in range(n_spots):
        for d, j in zip(dist[i], idx[i]):
            j = int(j)
            if j == i or j < 0 or j >= n_spots or (not np.isfinite(d)) or float(d) > radius:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))

    edge_array = np.asarray(sorted(edges), dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    return adjacency, edge_array


def resolve_min_domain_spots(n_spots: int, cfg: RSMDDomainFilterConfig) -> int:
    if cfg.min_domain_spots is not None:
        return max(1, int(cfg.min_domain_spots))
    return max(20, int(math.ceil(float(cfg.min_domain_spots_frac) * max(1, n_spots))))


def _compute_centroid(spot_indices: np.ndarray, coords: np.ndarray | None) -> tuple[float, float]:
    if coords is None or spot_indices.size == 0:
        return float("nan"), float("nan")
    xy = coords[spot_indices]
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return float("nan"), float("nan")
    center = np.mean(xy[finite], axis=0)
    return float(center[0]), float(center[1])


def _internal_density(n_nodes: int, n_internal_edges: int) -> float:
    if n_nodes <= 1:
        return 0.0
    denom = n_nodes * (n_nodes - 1) / 2.0
    return float(n_internal_edges / denom) if denom > 0 else 0.0


def _compactness(area_est: float, boundary_edges: int) -> float:
    perimeter = max(1.0, float(boundary_edges))
    val = (4.0 * math.pi * max(area_est, 0.0)) / (perimeter * perimeter)
    return float(max(0.0, min(1.0, val)))


def _smooth_on_graph(
    values: np.ndarray,
    adjacency: list[set[int]],
    active_mask: np.ndarray,
    cfg: PotentialConfig,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if (not cfg.smoothing_enabled) or int(cfg.smoothing_steps) <= 0:
        return out

    alpha = float(max(0.0, min(1.0, cfg.smoothing_alpha)))
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        return out

    for _ in range(int(cfg.smoothing_steps)):
        nxt = out.copy()
        for i in active_idx:
            nbs = [nb for nb in adjacency[int(i)] if active_mask[nb]]
            if not nbs:
                continue
            nb_mean = float(np.mean(out[np.asarray(nbs, dtype=np.int32)]))
            nxt[int(i)] = float((1.0 - alpha) * out[int(i)] + alpha * nb_mean)
        out = nxt
    return out


def _split_into_connected_components(spots: np.ndarray, adjacency: list[set[int]]) -> list[np.ndarray]:
    if spots.size == 0:
        return []
    spot_set = set(int(x) for x in spots.tolist())
    unvisited = set(spot_set)
    comps: list[np.ndarray] = []

    while unvisited:
        seed = int(next(iter(unvisited)))
        stack = [seed]
        unvisited.remove(seed)
        cur: list[int] = []
        while stack:
            u = int(stack.pop())
            cur.append(u)
            for v in adjacency[u]:
                if v in unvisited:
                    unvisited.remove(v)
                    stack.append(int(v))
        comps.append(np.asarray(sorted(cur), dtype=np.int32))
    return comps


def _select_rsp_active_mask(
    raw: np.ndarray,
    base_mask: np.ndarray,
    potential_cfg: PotentialConfig,
) -> tuple[np.ndarray, float, float]:
    base_idx = np.flatnonzero(base_mask)
    if base_idx.size == 0:
        return np.zeros_like(base_mask, dtype=bool), 0.0, 0.0

    vals = raw[base_idx]
    scale = float(np.median(vals)) if vals.size > 0 else 0.0
    floor = float(max(float(potential_cfg.active_floor_abs), float(potential_cfg.active_floor_scale_factor) * scale))
    active = base_mask & (raw > floor)

    min_active = max(1, int(potential_cfg.min_active_spots_per_rsp))
    if int(np.count_nonzero(active)) < min_active:
        keep_n = min(min_active, base_idx.size)
        if keep_n > 0:
            order = np.argpartition(-vals, keep_n - 1)[:keep_n]
            selected = base_idx[order]
            active = np.zeros_like(base_mask, dtype=bool)
            active[selected] = True
        else:
            active = np.zeros_like(base_mask, dtype=bool)
    return active, floor, scale


def _screen_candidate_domain(
    spot_count: int,
    density: float,
    cfg: RSMDDomainFilterConfig,
    min_domain_spots: int,
    require_density: bool = True,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    tags: list[str] = []
    if spot_count < int(min_domain_spots):
        reasons.append(f"spot_count<{min_domain_spots}")
        tags.append("rejected_by_size")
    if require_density and density < float(cfg.min_domain_internal_density):
        reasons.append(f"internal_density<{float(cfg.min_domain_internal_density)}")
        tags.append("rejected_by_density")
    return reasons, sorted(set(tags))


def _resolve_activation_threshold_quantile(cfg: RSMDFiltrationConfig) -> float:
    return float(min(1.0, max(0.0, cfg.activation_threshold_quantile)))


def _components_for_threshold(
    values: np.ndarray,
    active_mask: np.ndarray,
    threshold: float,
    adjacency: list[set[int]],
) -> list[np.ndarray]:
    support = active_mask & (np.asarray(values, dtype=np.float32) > float(threshold))
    comps = _split_into_connected_components(np.flatnonzero(support).astype(np.int32), adjacency=adjacency)
    return [np.asarray(c, dtype=np.int32) for c in comps if np.asarray(c).size > 0]


def _extract_filtration_entities(
    smooth: np.ndarray,
    active_mask: np.ndarray,
    adjacency: list[set[int]],
    filtration_cfg: RSMDFiltrationConfig,
    min_component_spots: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict], dict]:
    quantile = _resolve_activation_threshold_quantile(filtration_cfg)

    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        return {}, {}, {
            "enabled": True,
            "activation_threshold_quantile": float(quantile),
            "threshold_count": 1,
            "candidate_entity_count": 0,
            "retained_component_count": 0,
            "selected_threshold_quantile": None,
            "selected_threshold_value": None,
            "threshold_selection_reason": "no_active_spots",
        }

    min_spots = max(1, int(min_component_spots))
    active_values = np.asarray(smooth[active_idx], dtype=np.float32)
    threshold_value = float(np.quantile(active_values, quantile))
    comps = _components_for_threshold(
        values=smooth,
        active_mask=active_mask,
        threshold=threshold_value,
        adjacency=adjacency,
    )
    retained_component_count = int(sum(1 for comp in comps if int(comp.size) >= min_spots))
    retained_spot_count = int(sum(int(comp.size) for comp in comps if int(comp.size) >= min_spots))
    reason = "manual_quantile" if retained_component_count > 0 else "manual_quantile_no_retained_components"

    entities: dict[int, np.ndarray] = {}
    entity_meta: dict[int, dict] = {}
    selected_threshold_quantile = float(quantile)
    selected_threshold_value = float(threshold_value)
    for comp in comps:
        spots = np.asarray(comp, dtype=np.int32)
        if spots.size == 0:
            continue
        root = int(spots[int(np.argmax(smooth[spots]))])
        entities[root] = spots
        entity_meta[root] = {
            "segmentation_method": "activation_low_quantile_components",
            "selected_threshold_quantile": selected_threshold_quantile,
            "selected_threshold_value": selected_threshold_value,
            "threshold_selection_reason": reason,
            "threshold_level_count_total": 1,
            "threshold_component_count": int(len(comps)),
            "threshold_retained_component_count": retained_component_count,
            "threshold_retained_spot_count": int(retained_spot_count),
            "activation_threshold_quantile": selected_threshold_quantile,
        }

    stats = {
        "enabled": True,
        "activation_threshold_quantile": selected_threshold_quantile,
        "threshold_count": 1,
        "candidate_entity_count": int(len(entities)),
        "retained_component_count": retained_component_count,
        "selected_threshold_quantile": selected_threshold_quantile,
        "selected_threshold_value": selected_threshold_value,
        "threshold_selection_reason": reason,
    }
    return entities, entity_meta, stats


def extract_candidate_domains(
    dense_activation: np.ndarray,
    rsp_ids: np.ndarray,
    adjacency: list[set[int]],
    coords: np.ndarray | None,
    potential_cfg: PotentialConfig,
    filter_cfg: RSMDDomainFilterConfig,
    filtration_cfg: RSMDFiltrationConfig | None = None,
    rsp_score_info: dict[str, dict[str, float]] | None = None,
    active_mask: np.ndarray | None = None,
    min_domain_spots_override: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    n_spots, n_rsps = dense_activation.shape
    if n_rsps != int(rsp_ids.shape[0]):
        raise ValueError("rsp_ids length mismatch")

    if active_mask is None:
        active_mask = np.ones(n_spots, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape[0] != n_spots:
            raise ValueError("active_mask length mismatch")

    active_n = int(np.count_nonzero(active_mask))
    min_spots = (
        int(min_domain_spots_override)
        if min_domain_spots_override is not None
        else resolve_min_domain_spots(active_n, filter_cfg)
    )

    all_domains: list[dict] = []
    rsp_summary: list[dict] = []
    candidate_id = 0
    flow_graph_mode = str(potential_cfg.flow_graph_mode)
    rsp_score_info = rsp_score_info or {}
    filtration_cfg = filtration_cfg or RSMDFiltrationConfig()
    filtration_quantile = _resolve_activation_threshold_quantile(filtration_cfg)
    filtration_global_stats = {
        "enabled": True,
        "activation_threshold_quantile": float(filtration_quantile),
        "threshold_count": 1,
        "candidate_entity_count": 0,
        "retained_component_count": 0,
    }

    for j, pid_raw in enumerate(rsp_ids):
        pid = str(pid_raw)
        p_weight = rsp_score_info.get(pid, {})
        p_score = float(p_weight.get("rsp_score", 1.0))
        p_score_used = float(p_weight.get("rsp_score_used", 1.0))
        p_score_weight = float(p_weight.get("rsp_score_weight", 1.0))
        raw = np.asarray(dense_activation[:, j], dtype=np.float32)
        rsp_active_mask, active_floor, rsp_scale = _select_rsp_active_mask(
            raw=raw,
            base_mask=active_mask,
            potential_cfg=potential_cfg,
        )
        rsp_active_idx = np.flatnonzero(rsp_active_mask)
        if rsp_active_idx.size == 0:
            rsp_summary.append(
                {
                    "rsp_id": pid,
                    "candidate_domain_count": 0,
                    "screening_pass_count": 0,
                    "active_spot_count": 0,
                    "active_floor": float(active_floor),
                    "rsp_scale": float(rsp_scale),
                    "rsp_score": float(p_score),
                    "rsp_score_used": float(p_score_used),
                    "rsp_score_weight": float(p_score_weight),
                    "min_domain_spots": int(min_spots),
                    "min_domain_internal_density": float(filter_cfg.min_domain_internal_density),
                    "flow_graph_mode": flow_graph_mode,
                    "segmentation_method": "activation_low_quantile_components",
                    "filtration_threshold_quantile": float(filtration_quantile),
                    "retained_component_count": 0,
                }
            )
            continue

        smooth = _smooth_on_graph(raw, adjacency=adjacency, active_mask=rsp_active_mask, cfg=potential_cfg)
        candidate_entities, filtration_meta, filtration_stats = _extract_filtration_entities(
            smooth=smooth,
            active_mask=rsp_active_mask,
            adjacency=adjacency,
            filtration_cfg=filtration_cfg,
            min_component_spots=int(min_spots),
        )
        filtration_global_stats["candidate_entity_count"] += int(filtration_stats.get("candidate_entity_count", 0))
        filtration_global_stats["retained_component_count"] += int(
            filtration_stats.get("retained_component_count", 0)
        )
        global_baseline = float(np.quantile(smooth[rsp_active_idx], 0.50)) if rsp_active_idx.size > 0 else 0.0
        global_baseline_raw = float(np.quantile(raw[rsp_active_idx], 0.50)) if rsp_active_idx.size > 0 else 0.0

        kept = 0
        for root, spots in candidate_entities.items():
            meta = filtration_meta.get(int(root), {})
            candidate_id += 1
            spot_set = set(int(x) for x in spots.tolist())
            raw_vals = raw[spots]
            peak_value = float(np.max(smooth[spots])) if spots.size > 0 else 0.0
            root_local = int(spots[int(np.argmax(smooth[spots]))]) if spots.size > 0 else int(root)
            mean_activation = float(np.mean(raw_vals)) if raw_vals.size > 0 else 0.0
            sum_activation = float(np.sum(raw_vals)) if raw_vals.size > 0 else 0.0

            boundary_out: list[float] = []
            boundary_out_raw: list[float] = []
            for u in spot_set:
                for v in adjacency[u]:
                    if (not rsp_active_mask[v]) or (v in spot_set):
                        continue
                    boundary_out.append(float(smooth[v]))
                    boundary_out_raw.append(float(raw[v]))
            if boundary_out:
                outside_q = float(np.quantile(np.asarray(boundary_out, dtype=np.float32), filter_cfg.prominence_outside_quantile))
            else:
                outside_q = global_baseline
            prominence = float(max(0.0, peak_value - outside_q))
            outside_raw_mean = float(np.mean(np.asarray(boundary_out_raw, dtype=np.float32))) if boundary_out_raw else global_baseline_raw
            mean_enrichment_ratio = float(mean_activation / (outside_raw_mean + 1e-8))
            mean_enrichment_delta = float(mean_activation - outside_raw_mean)

            geom = compute_domain_geometry_metrics(spot_indices=spots, adjacency=adjacency, coords=coords)
            internal_edges = int(geom["internal_edge_count"])
            boundary_edges = int(geom["boundary_edge_count"])
            density = float(geom["internal_density"])
            cx = float(geom["geo_centroid_x"])
            cy = float(geom["geo_centroid_y"])
            area_est = float(geom["geo_area_est"])
            boundary_ratio = float(geom["geo_boundary_ratio"])
            compactness = float(geom["geo_compactness"])
            elongation = float(geom["geo_elongation"])
            leaf_ratio = float(geom["geo_leaf_ratio"])
            articulation_ratio = float(geom["geo_articulation_ratio"])

            reasons, reject_tags = _screen_candidate_domain(
                spot_count=int(spots.size),
                density=density,
                cfg=filter_cfg,
                min_domain_spots=int(min_spots),
                require_density=False,
            )
            passed = len(reasons) == 0
            if passed:
                kept += 1

            all_domains.append(
                {
                    "candidate_id": int(candidate_id),
                    "rsp_id": pid,
                    "root_spot_idx": int(root_local),
                    "spot_indices": spots,
                    "spot_set": spot_set,
                    "spot_count": int(spots.size),
                    "coverage": float(spots.size / max(1, int(rsp_active_idx.size))),
                    "rsp_seed_mean": mean_activation,
                    "rsp_seed_sum": sum_activation,
                    "rsp_mean_enrichment_ratio": mean_enrichment_ratio,
                    "rsp_mean_enrichment_delta": mean_enrichment_delta,
                    "rsp_peak_value": peak_value,
                    "rsp_prominence": prominence,
                    "rsp_outside_quantile": outside_q,
                    "rsp_outside_mean_raw": outside_raw_mean,
                    "geo_centroid_x": cx,
                    "geo_centroid_y": cy,
                    "geo_area_est": area_est,
                    "geo_compactness": compactness,
                    "geo_boundary_ratio": boundary_ratio,
                    "geo_elongation": elongation,
                    "geo_leaf_ratio": leaf_ratio,
                    "geo_articulation_ratio": articulation_ratio,
                    "components_count": int(geom.get("components_count", 1)),
                    "internal_edge_count": int(internal_edges),
                    "boundary_edge_count": int(boundary_edges),
                    "internal_density": density,
                    "screening_reasons": reasons,
                    "screening_reject_tags": reject_tags,
                    "screening_decision": "kept" if passed else "rejected",
                    "screening_pass": bool(passed),
                    "is_background": bool(not passed),
                    "segmentation_method": str(meta.get("segmentation_method", "activation_low_quantile_components")),
                    "selected_threshold_quantile": float(meta.get("selected_threshold_quantile", float("nan"))),
                    "selected_threshold_value": float(meta.get("selected_threshold_value", float("nan"))),
                    "threshold_selection_reason": str(meta.get("threshold_selection_reason", "")),
                    "threshold_level_count_total": int(meta.get("threshold_level_count_total", 0)),
                    "threshold_component_count": int(meta.get("threshold_component_count", 0)),
                    "threshold_retained_component_count": int(meta.get("threshold_retained_component_count", 0)),
                    "threshold_retained_spot_count": int(meta.get("threshold_retained_spot_count", 0)),
                    "activation_threshold_quantile": float(meta.get("activation_threshold_quantile", float("nan"))),
                    "rsp_scale": float(rsp_scale),
                    "rsp_score": float(p_score),
                    "rsp_score_used": float(p_score_used),
                    "rsp_score_weight": float(p_score_weight),
                    "flow_graph_mode": flow_graph_mode,
                    "active_floor": float(active_floor),
                    "min_domain_spots_effective": int(min_spots),
                    "min_domain_internal_density_effective": float(filter_cfg.min_domain_internal_density),
                }
            )

        rsp_summary.append(
            {
                "rsp_id": pid,
                "candidate_domain_count": int(len(candidate_entities)),
                "screening_pass_count": int(kept),
                "active_spot_count": int(rsp_active_idx.size),
                "active_floor": float(active_floor),
                "rsp_scale": float(rsp_scale),
                "rsp_score": float(p_score),
                "rsp_score_used": float(p_score_used),
                "rsp_score_weight": float(p_score_weight),
                "min_domain_spots": int(min_spots),
                "min_domain_internal_density": float(filter_cfg.min_domain_internal_density),
                "flow_graph_mode": flow_graph_mode,
                "segmentation_method": "activation_low_quantile_components",
                "filtration_threshold_quantile": float(filtration_quantile),
                "retained_component_count": int(kept),
                "selected_threshold_quantile": float(
                    next(iter(filtration_meta.values())).get("selected_threshold_quantile", float("nan"))
                )
                if filtration_meta
                else float("nan"),
                "selected_threshold_value": float(
                    next(iter(filtration_meta.values())).get("selected_threshold_value", float("nan"))
                )
                if filtration_meta
                else float("nan"),
            }
        )

    segmentation_stats = {
        "active_spot_count": int(active_n),
        "candidate_domain_count": int(len(all_domains)),
        "screening_pass_count": int(sum(1 for d in all_domains if d["screening_pass"])),
        "segmentation_method": "activation_low_quantile_components",
        "filtration": filtration_global_stats,
    }
    return all_domains, rsp_summary, segmentation_stats


def build_domain_membership_table(domains: list[dict], spot_ids: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for d in domains:
        dkey = str(d["domain_key"])
        for s in np.asarray(d["spot_indices"], dtype=np.int32):
            rows.append({"domain_key": dkey, "spot_idx": int(s), "spot_id": str(spot_ids[int(s)])})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["domain_key", "spot_idx", "spot_id"])
    return df


def build_domain_graph_table(
    domains: list[dict],
    spot_edges: np.ndarray,
    cfg: RSMDDomainAdjacencyConfig,
) -> pd.DataFrame:
    if len(domains) <= 1:
        return pd.DataFrame(
            columns=[
                "domain_key_i",
                "domain_key_j",
                "shared_boundary_edges",
                "spatial_overlap",
                "centroid_distance",
                "edge_weight",
            ]
        )

    spot_to_domains: dict[int, list[int]] = {}
    for di, d in enumerate(domains):
        for s in np.asarray(d["spot_indices"], dtype=np.int32):
            spot_to_domains.setdefault(int(s), []).append(di)

    boundary_counts: dict[tuple[int, int], int] = {}
    for u, v in np.asarray(spot_edges, dtype=np.int32):
        left = spot_to_domains.get(int(u), [])
        right = spot_to_domains.get(int(v), [])
        if (not left) or (not right):
            continue
        for i in left:
            for j in right:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                boundary_counts[(a, b)] = boundary_counts.get((a, b), 0) + 1

    overlap_counts: dict[tuple[int, int], int] = {}
    for domain_indices in spot_to_domains.values():
        if len(domain_indices) < 2:
            continue
        uniq = sorted(set(domain_indices))
        for i, j in combinations(uniq, 2):
            overlap_counts[(i, j)] = overlap_counts.get((i, j), 0) + 1

    pair_keys = set(boundary_counts.keys()) | set(overlap_counts.keys())
    if cfg.mode == "shared_boundary":
        pair_keys = {
            p
            for p in pair_keys
            if boundary_counts.get(p, 0) >= int(cfg.min_shared_boundary_edges) or overlap_counts.get(p, 0) > 0
        }

    rows: list[dict] = []
    eps = max(1e-12, float(cfg.centroid_distance_eps))

    for i, j in sorted(pair_keys):
        di = domains[i]
        dj = domains[j]
        overlap = int(overlap_counts.get((i, j), 0))
        n_i = int(di["spot_count"])
        n_j = int(dj["spot_count"])
        union = max(1, n_i + n_j - overlap)
        spatial_overlap = float(overlap / union)

        ci = np.array([float(di.get("geo_centroid_x", np.nan)), float(di.get("geo_centroid_y", np.nan))], dtype=np.float64)
        cj = np.array([float(dj.get("geo_centroid_x", np.nan)), float(dj.get("geo_centroid_y", np.nan))], dtype=np.float64)
        centroid_distance = float(np.linalg.norm(ci - cj)) if np.isfinite(ci).all() and np.isfinite(cj).all() else float("nan")

        shared_boundary = int(boundary_counts.get((i, j), 0))
        if cfg.mode == "shared_boundary":
            edge_weight = float(shared_boundary)
        elif cfg.mode == "inverse_centroid_distance":
            edge_weight = 0.0 if not np.isfinite(centroid_distance) else float(1.0 / (centroid_distance + eps))
        else:
            raise ValueError(f"Unsupported adjacency mode: {cfg.mode}")

        rows.append(
            {
                "domain_key_i": str(di["domain_key"]),
                "domain_key_j": str(dj["domain_key"]),
                "shared_boundary_edges": shared_boundary,
                "spatial_overlap": spatial_overlap,
                "centroid_distance": centroid_distance,
                "edge_weight": edge_weight,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "domain_key_i",
                "domain_key_j",
                "shared_boundary_edges",
                "spatial_overlap",
                "centroid_distance",
                "edge_weight",
            ]
        )
    return out


def _compute_elongation(spot_indices: np.ndarray, coords: np.ndarray | None) -> float:
    if coords is None or spot_indices.size < 3:
        return 1.0
    xy = np.asarray(coords[spot_indices], dtype=np.float64)
    finite = np.isfinite(xy).all(axis=1)
    if int(np.count_nonzero(finite)) < 3:
        return 1.0
    xy = xy[finite]
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return 1.0
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.clip(vals, 0.0, None))
    small = float(vals[0])
    large = float(vals[1])
    return float(math.sqrt((large + 1e-12) / (small + 1e-12)))


def _articulation_points_ratio(spot_indices: np.ndarray, adjacency: list[set[int]]) -> float:
    nodes = [int(x) for x in np.asarray(spot_indices, dtype=np.int32).tolist()]
    n = len(nodes)
    if n <= 2:
        return 0.0
    if sys.getrecursionlimit() < n + 100:
        sys.setrecursionlimit(n + 100)
    node_set = set(nodes)

    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int] = {}
    ap: set[int] = set()
    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        children = 0
        time += 1
        disc[u] = time
        low[u] = time

        for v in adjacency[u]:
            if v not in node_set:
                continue
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])

                if u not in parent and children > 1:
                    ap.add(u)
                if u in parent and low[v] >= disc[u]:
                    ap.add(u)
            elif parent.get(u, -1) != v:
                low[u] = min(low[u], disc[v])

    for u in nodes:
        if u not in disc:
            dfs(u)

    return float(len(ap) / max(1, n))


def _domain_internal_boundary_edges_all_graph(
    spot_indices: np.ndarray,
    adjacency: list[set[int]],
) -> tuple[int, int]:
    ss = set(int(x) for x in np.asarray(spot_indices, dtype=np.int32).tolist())
    if not ss:
        return 0, 0
    internal = 0
    boundary = 0
    for u in ss:
        for v in adjacency[u]:
            if v in ss:
                if u < v:
                    internal += 1
            else:
                boundary += 1
    return int(internal), int(boundary)


def compute_domain_geometry_metrics(
    spot_indices: np.ndarray,
    adjacency: list[set[int]],
    coords: np.ndarray | None,
) -> dict:
    idx = np.asarray(spot_indices, dtype=np.int32)
    n = int(idx.size)
    components_count = int(len(_split_into_connected_components(idx, adjacency)))
    internal, boundary = _domain_internal_boundary_edges_all_graph(idx, adjacency)
    density = _internal_density(n, internal)
    cx, cy = _compute_centroid(idx, coords)
    area = float(n)
    compactness = _compactness(area, boundary)
    boundary_ratio = float(boundary / max(1, boundary + internal))

    # degree-based shape heuristics on induced subgraph
    ss = set(int(x) for x in idx.tolist())
    deg = []
    for u in ss:
        deg.append(sum(1 for v in adjacency[u] if v in ss))
    deg_arr = np.asarray(deg, dtype=np.int32) if deg else np.zeros(0, dtype=np.int32)
    leaf_ratio = float(np.mean(deg_arr <= 1)) if deg_arr.size > 0 else 0.0
    articulation_ratio = _articulation_points_ratio(idx, adjacency)
    elongation = _compute_elongation(idx, coords)

    return {
        "spot_count": n,
        "geo_centroid_x": float(cx),
        "geo_centroid_y": float(cy),
        "geo_area_est": area,
        "geo_compactness": float(compactness),
        "geo_boundary_ratio": float(boundary_ratio),
        "geo_elongation": float(elongation),
        "geo_leaf_ratio": float(leaf_ratio),
        "geo_articulation_ratio": float(articulation_ratio),
        "components_count": int(components_count),
        "internal_edge_count": int(internal),
        "boundary_edge_count": int(boundary),
        "internal_density": float(density),
    }
