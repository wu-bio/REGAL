from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class RSMDInputConfig:
    rsp_manifest_relpath: str = "manifest.json"
    rsp_relpath: str = "rsp.parquet"
    rsp_activation_relpath: str = "rsp_activation.parquet"
    rsp_qc_relpath: str = "rsp_qc.parquet"
    rss_manifest_relpath: str = "manifest.json"
    microdomain_idx_relpath: str = "microdomains/microdomain.idx.npy"
    microdomain_meta_relpath: str = "microdomains/microdomain_meta.json"
    rss_out_path_override: Optional[str] = None
    try_load_coords_from_h5ad: bool = True
    rsp_allowlist: tuple[str, ...] = ()
    rsp_allowlist_file: Optional[str] = None
    use_rsp_score_filter: bool = False
    min_rsp_score: float = 0.0


@dataclass
class PotentialConfig:
    flow_graph_mode: Literal["spatial", "rss"] = "spatial"
    spatial_graph_k: int = 6
    spatial_graph_max_distance_factor: float = 1.5
    smoothing_enabled: bool = True
    smoothing_alpha: float = 0.60
    smoothing_steps: int = 5
    active_floor_abs: float = 1e-6
    active_floor_scale_factor: float = 0.01
    min_active_spots_per_rsp: int = 20


@dataclass
class RSMDDomainFilterConfig:
    min_domain_spots: Optional[int] = 20
    min_domain_spots_frac: float = 0.002
    prominence_outside_quantile: float = 0.80
    min_domain_internal_density: float = 0.01


@dataclass
class RSMDFiltrationConfig:
    activation_threshold_quantile: float = 0.10


@dataclass
class RSMDDomainAdjacencyConfig:
    mode: Literal["shared_boundary", "inverse_centroid_distance"] = "shared_boundary"
    min_shared_boundary_edges: int = 1
    centroid_distance_eps: float = 1e-6


@dataclass
class RSMDDomainReliabilityConfig:
    enabled: bool = True
    rsp_score_exponent: float = 1.0
    prominence_exponent: float = 0.5
    density_exponent: float = 0.5
    prominence_scale_quantile: float = 0.90
    density_scale: float = 0.15
    min_node_reliability: float = 0.05
    max_node_reliability: float = 1.0


@dataclass
class RSMDPipelineConfig:
    schema_version: str = "rsmdbundle.v1"
    random_seed: int = 2024
    code_version_override: Optional[str] = None
    input: RSMDInputConfig = field(default_factory=RSMDInputConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    filter: RSMDDomainFilterConfig = field(default_factory=RSMDDomainFilterConfig)
    filtration: RSMDFiltrationConfig = field(default_factory=RSMDFiltrationConfig)
    adjacency: RSMDDomainAdjacencyConfig = field(default_factory=RSMDDomainAdjacencyConfig)
    domain_reliability: RSMDDomainReliabilityConfig = field(default_factory=RSMDDomainReliabilityConfig)
