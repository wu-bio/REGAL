from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PreprocessConfig:
    data_layer: str = "X"
    min_cells: int = 10
    normalize_target_sum: float = 1e4
    hvg: int = 3000
    hvg_flavor: str = "seurat_v3"
    use_hvg_for_latent: bool = True
    remove_mt_genes: bool = True
    spot_id_field: Optional[str] = None
    gene_id_type: str = "symbol"


@dataclass
class LatentConfig:
    epochs: int = 300
    hidden_dims: tuple[int, ...] = (512, 30)
    lr: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clipping: float = 5.0
    spatial_graph_mode: Literal["radius", "knn"] = "knn"
    graph_k: int = 6
    radius_cutoff: float = 150.0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    verbose: bool = True
    progress_interval: int = 50


@dataclass
class NeighborsConfig:
    normal_k: int = 12
    min_k: int = 6
    include_self: bool = True
    self_weight: float = 0.40
    similarity_margin: float = 0.05
    distance_weight_power: float = 1.0
    candidate_k: int = 61
    candidate_mode: Literal["spatial_knn", "radius", "hybrid"] = "spatial_knn"
    spatial_radius: Optional[float] = None
    similarity_metric: Literal["cosine", "centered_cosine"] = "centered_cosine"
    connectivity_hops: int = 2
    fallback: Literal["spatial", "decrease_k"] = "spatial"


@dataclass
class RSSConfig:
    no_expression_fraction: bool = False
    projection: Literal["exp2", "softplus"] = "exp2"
    sparsify_rule: Literal["topM", "positive"] = "topM"
    top_m: int = 3000


@dataclass
class QCConfig:
    bootstrap_repeats: int = 5
    bootstrap_latent_noise_std: float = 0.02
    bootstrap_top_n: tuple[int, ...] = (50, 100)
    neighbor_stability_gate_p50: float = 0.60
    rss_stability_gate_p50: float = 0.50


@dataclass
class RSSPipelineConfig:
    schema_version: str = "rssbundle.v1"
    random_seed: int = 2024
    code_version_override: Optional[str] = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    neighbors: NeighborsConfig = field(default_factory=NeighborsConfig)
    rss: RSSConfig = field(default_factory=RSSConfig)
    qc: QCConfig = field(default_factory=QCConfig)
