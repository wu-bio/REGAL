from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class RSPInputConfig:
    rss_sparse_relpath: str = "rss/rss_sparse.parquet"
    rss_manifest_relpath: str = "manifest.json"
    h5ad_path_override: Optional[str] = None
    spatial_obsm_key: str = "spatial"


@dataclass
class RSPDiscoveryConfig:
    rss_value_col: str = "rss_score"
    rss_transform: Literal["log1p", "none"] = "none"
    rss_rank_cutoff: Optional[int] = 1000
    max_genes_for_decomposition: Optional[int] = None
    enable_adaptive_gene_selection: bool = True
    decomposition_backend: Literal[
        "rss_anchored_cd_l1_sparse_factorization",
        "legacy_rss_spatial_nmf",
    ] = "rss_anchored_cd_l1_sparse_factorization"
    adaptive_iqr_multiplier: float = 0.5
    k_grid: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18)
    seed_grid: tuple[int, ...] = (0, 1, 2)
    spatial_knn: int = 6
    moran_knn: int = 6
    min_nonzero_rss_spot_fraction: float = 0.01
    max_nonzero_rss_spot_fraction: Optional[float] = None
    min_nonzero_rss_spot_count: int = 7
    min_spatial_autocorrelation_score: float = 0.30
    structured_component_min_size: int = 7
    min_structured_component_support: float = 0.30
    min_local_support_score: float = 0.20
    decomposition_tol: float = 1e-4
    loading_sparsity_grid: tuple[float, ...] = (0.0003,)
    activation_sparsity_grid: tuple[float, ...] = (0.0001,)
    l1_ratio_grid: tuple[float, ...] = (0.5,)
    elementwise_weight_floor: float = 0.05
    zero_evidence_weight: float = 0.50
    loading_refinement_max_iter: int = 30
    loading_refinement_lambda_multiplier: float = 10.0
    axis_anchor_init: str = "nndsvda"
    axis_anchor_solver: str = "cd"
    axis_anchor_max_iter: int = 1000
    axis_anchor_tol: float = 1e-4
    axis_anchor_alpha_w: float = 1e-4
    axis_anchor_alpha_h: float = 1e-4
    axis_anchor_l1_ratio: float = 0.5
    anchored_loading_support_power: float = 1.5
    anchored_loading_reliability_weight: float = 0.35
    anchored_adaptive_refinement_enabled: bool = True
    anchored_adaptive_refinement_strength: float = 2.0
    anchored_adaptive_refinement_min_factor: float = 0.6
    anchored_adaptive_refinement_max_factor: float = 1.8
    legacy_nmf_init: str = "nndsvda"
    legacy_nmf_solver: str = "cd"
    legacy_nmf_max_iter: int = 1000
    legacy_nmf_tol: float = 1e-4
    legacy_nmf_alpha_w: float = 1e-4
    legacy_nmf_alpha_h: float = 1e-4
    legacy_nmf_l1_ratio: float = 0.5
    reliability_columns: tuple[str, ...] = (
        "rss_support",
        "microdomain_detect_rate",
        "global_detect_rate",
    )


@dataclass
class RSPStabilityConfig:
    activation_cleanup_min_peak_fraction: float = 0.05
    activation_cleanup_min_abs: float = 1e-6
    activation_cleanup_min_component_size: int = 5
    min_candidate_active_spots: int = 12
    max_candidate_active_fraction: float = 0.75
    min_candidate_activation_moran_i: float = 0.40
    structured_active_component_min_size: int = 7
    min_candidate_structured_active_fraction: float = 0.50
    min_match_map_corr: float = 0.90
    min_match_active_jaccard: float = 0.70
    min_match_gene_jaccard_70: float = 0.80
    min_match_loading_corr: float = 0.80
    min_family_centroid_map_corr: float = 0.90
    min_family_centroid_active_jaccard: float = 0.70
    min_family_centroid_loading_corr: float = 0.70
    min_family_size: int = 2
    min_family_k_count: int = 2
    min_family_seed_count: int = 2
    match_score_map_weight: float = 0.45
    match_score_active_weight: float = 0.20
    match_score_gene_weight: float = 0.20
    match_score_loading_weight: float = 0.15


@dataclass
class RSPScoringConfig:
    min_active_spots: int = 12
    activation_active_min_peak_fraction: float = 0.10
    activation_active_quantile: float = 0.05
    activation_active_min_abs: float = 1e-6
    activation_active_min_component_size: int = 20
    activation_active_fallback_to_base: bool = False
    activation_trim_enabled: bool = True
    activation_trim_fraction_of_peak: float = 0.05
    activation_trim_min_abs: float = 1e-6
    activation_trim_min_component_size: int = 20
    blacklist_prefixes: tuple[str, ...] = ("MT-", "RPL", "RPS")
    blacklist_genes: tuple[str, ...] = ("MALAT1", "NEAT1", "XIST")
    housekeeping_prefixes: tuple[str, ...] = ("RPL", "RPS", "MT-")
    housekeeping_genes: tuple[str, ...] = ("ACTB", "GAPDH", "B2M", "TUBB", "MALAT1")
    low_info_prefixes: tuple[str, ...] = (
        "AC",
        "AL",
        "AP",
        "LINC",
        "LOC",
        "MIR",
        "SNORD",
        "SNORA",
        "RNU",
    )
    blacklist_loading_fraction_drop: float = 0.50
    blacklist_top10_fraction_drop: float = 0.50
    housekeeping_loading_fraction_drop: float = 0.60
    low_info_top10_fraction_drop: float = 0.50
    ideal_min_coverage: float = 0.02
    ideal_max_coverage: float = 0.50
    core_safety_min_genes: int = 1
    core_safety_max_genes: Optional[int] = None
    core_use_elbow_selection: bool = False
    core_elbow_sensitivity: float = 0.75
    core_min_core_score: float = 0.50
    core_min_local_rss_strength: float = 0.40
    core_min_local_rss_coverage: float = 0.50
    core_min_local_rss_lift: float = 1.10
    core_exclude_confounder_genes: bool = True


@dataclass
class RSPPipelineConfig:
    schema_version: str = "rspbundle.v1"
    random_seed: int = 2024
    code_version_override: Optional[str] = None
    input: RSPInputConfig = field(default_factory=RSPInputConfig)
    discovery: RSPDiscoveryConfig = field(default_factory=RSPDiscoveryConfig)
    stability: RSPStabilityConfig = field(default_factory=RSPStabilityConfig)
    scoring: RSPScoringConfig = field(default_factory=RSPScoringConfig)
