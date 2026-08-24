from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    from .stats import jaccard, neighbors_to_sets, quantiles
    from .schema import RSSConfig, NeighborsConfig
    from .neighbors import build_neighbors
    from .rss_compute import compute_top_sets
except ImportError:
    from RSS.stats import jaccard, neighbors_to_sets, quantiles
    from RSS.schema import RSSConfig, NeighborsConfig
    from RSS.neighbors import build_neighbors
    from RSS.rss_compute import compute_top_sets


def compute_neighbor_qc(
    coords: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_sim: np.ndarray,
    connectivity_sets: list[set[int]],
    radius: float | None,
    return_per_spot: bool = False,
) -> dict:
    n_spots, _ = neighbor_idx.shape
    locality = np.zeros(n_spots, dtype=np.float32)
    within_radius = np.zeros(n_spots, dtype=np.float32)
    homogeneity_mean = np.zeros(n_spots, dtype=np.float32)
    homogeneity_var = np.zeros(n_spots, dtype=np.float32)
    connectivity_ratio = np.zeros(n_spots, dtype=np.float32)

    for i in range(n_spots):
        valid = neighbor_idx[i] >= 0
        idx = neighbor_idx[i][valid]
        sims = neighbor_sim[i][valid]

        if idx.size == 0:
            continue

        delta = coords[idx] - coords[i]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        locality[i] = float(np.mean(dist))
        homogeneity_mean[i] = float(np.mean(sims))
        homogeneity_var[i] = float(np.var(sims))

        if radius is not None and radius > 0:
            within_radius[i] = float(np.mean(dist <= radius))
        else:
            within_radius[i] = np.nan

        connected = connectivity_sets[i]
        connectivity_ratio[i] = float(
            np.mean(np.array([int(x) == i or int(x) in connected for x in idx], dtype=np.float32))
        )

    payload = {
        "neighbor_locality": quantiles(locality),
        "neighbor_homogeneity_mean": quantiles(homogeneity_mean),
        "neighbor_homogeneity_var": quantiles(homogeneity_var),
        "neighbor_connectivity": quantiles(connectivity_ratio),
    }

    if radius is not None and radius > 0:
        payload["within_radius_ratio"] = quantiles(within_radius)

    if return_per_spot:
        per_spot = {
            "neighbor_locality": locality,
            "neighbor_homogeneity_mean": homogeneity_mean,
            "neighbor_homogeneity_var": homogeneity_var,
            "neighbor_connectivity": connectivity_ratio,
            "within_radius_ratio": within_radius,
        }
        return {"summary": payload, "per_spot": per_spot}

    return payload


def compute_stability_qc(
    latent: np.ndarray,
    expression: sp.csr_matrix,
    base_neighbor_idx: np.ndarray,
    base_top_sets: dict[int, list[set[int]]],
    candidate_idx: list[np.ndarray],
    candidate_dist: list[np.ndarray],
    spatial_adj: list[set[int]],
    neighbors_cfg: NeighborsConfig,
    rss_cfg: RSSConfig,
    top_ns: tuple[int, ...],
    repeats: int,
    noise_std: float,
    seed: int,
    remove_mt: bool,
    gene_names: np.ndarray,
    return_per_spot: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)

    base_neighbor_sets = neighbors_to_sets(base_neighbor_idx)
    neighbor_repeat_medians = []
    neighbor_per_spot_acc = np.zeros(base_neighbor_idx.shape[0], dtype=np.float32)

    rss_repeat_medians = {str(n): [] for n in top_ns}
    rss_per_spot_acc = {str(n): np.zeros(base_neighbor_idx.shape[0], dtype=np.float32) for n in top_ns}

    for _ in range(max(1, repeats)):
        latent_noisy = latent + rng.normal(0.0, noise_std, size=latent.shape).astype(np.float32)
        n_idx, _, n_weight, _, _ = build_neighbors(
            latent_noisy,
            candidate_idx,
            candidate_dist,
            spatial_adj,
            neighbors_cfg,
        )

        noisy_neighbor_sets = neighbors_to_sets(n_idx)
        nj = np.array(
            [jaccard(base_neighbor_sets[i], noisy_neighbor_sets[i]) for i in range(len(base_neighbor_sets))],
            dtype=np.float32,
        )
        neighbor_repeat_medians.append(float(np.median(nj)))
        neighbor_per_spot_acc += nj

        noisy_top_sets = compute_top_sets(
            expression=expression,
            gene_names=gene_names,
            neighbor_idx=n_idx,
            rss_cfg=rss_cfg,
            top_ns=top_ns,
            remove_mt=remove_mt,
            neighbor_weight=n_weight,
        )

        for n in top_ns:
            key = str(n)
            gj = np.array(
                [
                    jaccard(base_top_sets[int(n)][i], noisy_top_sets[int(n)][i])
                    for i in range(base_neighbor_idx.shape[0])
                ],
                dtype=np.float32,
            )
            rss_repeat_medians[key].append(float(np.median(gj)))
            rss_per_spot_acc[key] += gj

    denom = float(max(1, repeats))
    neighbor_per_spot = neighbor_per_spot_acc / denom

    neighbor_payload = {
        "repeats": int(repeats),
        "repeat_median": neighbor_repeat_medians,
        "quantiles": quantiles(neighbor_per_spot),
    }

    rss_payload: dict[str, dict] = {}
    for n in top_ns:
        key = str(n)
        rss_per_spot = rss_per_spot_acc[key] / denom
        rss_payload[key] = {
            "repeats": int(repeats),
            "repeat_median": rss_repeat_medians[key],
            "quantiles": quantiles(rss_per_spot),
        }

    payload = {
        "neighbor_stability_jaccard": neighbor_payload,
        "rss_topN_stability_jaccard": rss_payload,
    }

    if return_per_spot:
        payload["per_spot"] = {
            "neighbor_stability_jaccard": neighbor_per_spot,
            "rss_topN_stability_jaccard": {
                str(n): (rss_per_spot_acc[str(n)] / denom) for n in top_ns
            },
        }

    return payload
