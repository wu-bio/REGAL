from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .common import build_spatial_neighbors
from .core import (
    annotate_core_genes,
    build_rsp_core_gene_loadings_table,
    build_rsp_gene_loadings_table,
)
from .decomposition import (
    generate_decomposition_candidates,
    refresh_candidate_rss_support_on_retained_footprint,
)
from .evidence import build_rss_evidence
from .io import (
    build_candidate_diagnostics_table,
    build_rsp_family_member_activation_table,
    build_rsp_family_members_table,
    build_rsp_activation_table,
    build_rsp_qc_table,
    build_rsp_table,
)
from .prefilter import compute_gene_evidence_metrics, select_decomposition_gene_indices
from .schema import (
    RSPDiscoveryConfig,
    RSPScoringConfig,
    RSPStabilityConfig,
)
from .stability import select_stable_rsps


def discover_rsp_from_rss(
    *,
    rss_sparse_path: Path,
    spot_ids: np.ndarray,
    spatial: np.ndarray,
    discovery_cfg: RSPDiscoveryConfig,
    stability_cfg: RSPStabilityConfig,
    scoring_cfg: RSPScoringConfig,
    random_seed: int,
) -> dict:
    spot_ids = np.asarray(spot_ids, dtype=str)
    spatial = np.asarray(spatial, dtype=np.float64)[:, :2]
    spatial_neighbors = build_spatial_neighbors(spatial, discovery_cfg.spatial_knn)
    moran_neighbors = build_spatial_neighbors(spatial, discovery_cfg.moran_knn)

    evidence = build_rss_evidence(rss_sparse_path, spot_ids, discovery_cfg)
    gene_metrics = compute_gene_evidence_metrics(
        evidence=evidence,
        spatial_neighbors=spatial_neighbors,
        moran_neighbors=moran_neighbors,
        discovery_cfg=discovery_cfg,
        scoring_cfg=scoring_cfg,
    )
    selected_idx = select_decomposition_gene_indices(
        gene_metrics,
        adaptive_iqr_multiplier=discovery_cfg.adaptive_iqr_multiplier,
        enable_adaptive_gene_selection=discovery_cfg.enable_adaptive_gene_selection,
        max_genes=discovery_cfg.max_genes_for_decomposition,
    )
    if not gene_metrics.empty:
        gene_metrics = gene_metrics.copy()
        gene_metrics["is_selected_for_decomposition"] = gene_metrics["gene_index"].isin(
            set(selected_idx.tolist())
        )
    candidates, decomposition_runs = generate_decomposition_candidates(
        evidence=evidence,
        selected_idx=selected_idx,
        neighbors=moran_neighbors,
        discovery_cfg=discovery_cfg,
        scoring_cfg=scoring_cfg,
        random_seed=random_seed,
    )
    retained_candidates = select_stable_rsps(
        candidates,
        _stability_config_for_seed_grid(stability_cfg, discovery_cfg),
        scoring_cfg,
        moran_neighbors,
    )
    activation_df = build_rsp_activation_table(retained_candidates, spot_ids, scoring_cfg, spatial_neighbors)
    if retained_candidates and hasattr(evidence, "x_raw"):
        refresh_candidate_rss_support_on_retained_footprint(
            retained_candidates,
            x_raw=np.asarray(evidence.x_raw[selected_idx], dtype=np.float32),
        )
    annotate_core_genes(retained_candidates, scoring_cfg)

    family_members_df = build_rsp_family_members_table(candidates)
    family_member_activation_df = build_rsp_family_member_activation_table(
        candidates,
        spot_ids,
        scoring_cfg,
        spatial_neighbors,
    )
    rsp_df = build_rsp_table(retained_candidates)
    loadings_df = build_rsp_gene_loadings_table(retained_candidates)
    core_loadings_df = build_rsp_core_gene_loadings_table(retained_candidates)
    qc_df = build_rsp_qc_table(candidates)
    candidate_diagnostics_df = build_candidate_diagnostics_table(candidates)
    core_counts = [int(c.get("core_gene_count", 0)) for c in retained_candidates]

    return {
        "rsp": rsp_df,
        "rsp_activation": activation_df,
        "rsp_gene_loadings": loadings_df,
        "rsp_core_gene_loadings": core_loadings_df,
        "rsp_family_members": family_members_df,
        "rsp_family_member_activation": family_member_activation_df,
        "rsp_qc": qc_df,
        "candidate_diagnostics": candidate_diagnostics_df,
        "gene_filter_metrics": gene_metrics,
        "rss_matrix_metadata": evidence.metadata,
        "summary": {
            "discovery_backend": discovery_cfg.decomposition_backend,
            "rss_sparse_path": str(rss_sparse_path),
            "n_spots": int(len(spot_ids)),
            "n_candidate_genes": int(len(evidence.genes)),
            "n_rss_genes_kept": int(gene_metrics["is_kept_for_decomposition"].sum()) if not gene_metrics.empty else 0,
            "n_decomposition_genes": int(selected_idx.size),
            "enable_adaptive_gene_selection": bool(discovery_cfg.enable_adaptive_gene_selection),
            "adaptive_iqr_multiplier": float(discovery_cfg.adaptive_iqr_multiplier),
            "rsp_candidate_count": int(len(candidates)),
            "rsp_count": int(len(retained_candidates)),
            "core_gene_count_min": int(min(core_counts)) if core_counts else 0,
            "core_gene_count_median": float(np.median(core_counts)) if core_counts else 0.0,
            "core_gene_count_max": int(max(core_counts)) if core_counts else 0,
            "candidate_status_counts": qc_df["candidate_status"].value_counts().to_dict() if not qc_df.empty else {},
            "decomposition_runs": decomposition_runs,
            "rss_matrix_metadata": evidence.metadata,
            "config": {
                "discovery": asdict(discovery_cfg),
                "stability": asdict(stability_cfg),
                "scoring": asdict(scoring_cfg),
            },
        },
    }


def _stability_config_for_seed_grid(
    stability_cfg: RSPStabilityConfig,
    discovery_cfg: RSPDiscoveryConfig,
) -> RSPStabilityConfig:
    seeds = getattr(discovery_cfg, "seed_grid", (0,))
    if isinstance(seeds, int):
        n_seeds = 1
    else:
        n_seeds = len(tuple(seeds))
    effective_min_seed_count = max(1, min(int(stability_cfg.min_family_seed_count), max(1, n_seeds)))
    if effective_min_seed_count == int(stability_cfg.min_family_seed_count):
        return stability_cfg
    return replace(stability_cfg, min_family_seed_count=effective_min_seed_count)
