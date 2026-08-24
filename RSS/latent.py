from __future__ import annotations

import logging
import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/regal_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/regal_mplconfig")

import numpy as np
import scanpy as sc
import scipy.sparse as sp

try:
    from .schema import LatentConfig, PreprocessConfig
except ImportError:
    from RSS.schema import LatentConfig, PreprocessConfig

logger = logging.getLogger(__name__)


def resolve_spot_ids(adata: sc.AnnData, spot_id_field: str | None) -> tuple[np.ndarray, str, bool]:
    if spot_id_field and spot_id_field in adata.obs.columns:
        raw_values = adata.obs[spot_id_field]
        candidate = raw_values.astype(str).to_numpy()
        if not raw_values.isna().any() and len(candidate) == len(set(candidate)):
            return candidate, spot_id_field, False

    obs_names = adata.obs_names.astype(str).to_numpy()
    if len(obs_names) == len(set(obs_names)):
        return obs_names, "obs_names", False

    generated = np.array([f"spot_{i:07d}" for i in range(adata.n_obs)], dtype=object)
    return generated, "generated", True


def prepare_expression_and_latent_input(
    adata: sc.AnnData,
    cfg: PreprocessConfig,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray | None]:
    if cfg.data_layer != "X":
        if cfg.data_layer not in adata.layers:
            raise ValueError(f"Data layer {cfg.data_layer} is not found in adata.layers")
        adata.X = adata.layers[cfg.data_layer].copy()

    if adata.n_vars == 0:
        raise ValueError("No genes available after filtering.")

    n_top_genes = max(1, min(cfg.hvg, adata.n_vars))
    try:
        sc.pp.highly_variable_genes(adata, flavor=cfg.hvg_flavor, n_top_genes=n_top_genes)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to compute HVG with flavor %s, fallback to seurat. reason=%s",
            cfg.hvg_flavor,
            exc,
        )
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)

    sc.pp.normalize_total(adata, target_sum=cfg.normalize_target_sum)
    sc.pp.log1p(adata)

    expression = adata.X
    if not sp.issparse(expression):
        expression = sp.csr_matrix(np.asarray(expression, dtype=np.float32))
    elif not isinstance(expression, sp.csr_matrix):
        expression = expression.tocsr()

    expression = expression.astype(np.float32)

    hvg_mask = adata.var["highly_variable"].to_numpy() if "highly_variable" in adata.var else None
    if cfg.use_hvg_for_latent and hvg_mask is not None and np.any(hvg_mask):
        latent_input = expression[:, hvg_mask]
    else:
        latent_input = expression

    latent_input_arr = latent_input.toarray() if sp.issparse(latent_input) else np.asarray(latent_input)
    latent_input_arr = latent_input_arr.astype(np.float32, copy=False)

    return expression, latent_input_arr, hvg_mask


def compute_latent(
    adata: sc.AnnData,
    latent_input: np.ndarray,
    latent_cfg: LatentConfig,
    random_seed: int,
) -> tuple[np.ndarray, dict]:
    try:
        from .spatial_graph_encoder import train_spatial_graph_latent
    except ImportError:
        from RSS.spatial_graph_encoder import train_spatial_graph_latent

    if "spatial" not in adata.obsm:
        raise ValueError("Missing adata.obsm['spatial']; required for spatial graph encoding.")
    latent, losses, train_meta = train_spatial_graph_latent(
        features=latent_input,
        spatial=np.asarray(adata.obsm["spatial"])[:, :2],
        hidden_dims=tuple(int(dim) for dim in latent_cfg.hidden_dims),
        epochs=latent_cfg.epochs,
        lr=latent_cfg.lr,
        weight_decay=latent_cfg.weight_decay,
        gradient_clipping=latent_cfg.gradient_clipping,
        graph_mode=latent_cfg.spatial_graph_mode,
        graph_k=latent_cfg.graph_k,
        radius_cutoff=latent_cfg.radius_cutoff,
        random_seed=random_seed,
        device=latent_cfg.device,
        verbose=latent_cfg.verbose,
        progress_interval=latent_cfg.progress_interval,
    )
    return latent, {
        "method": "spatial_graph_attention_autoencoder",
        "epochs": int(latent_cfg.epochs),
        "hidden_dims": list(map(int, latent_cfg.hidden_dims)),
        "training": {
            "loss_first": float(losses[0]) if losses else None,
            "loss_final": float(losses[-1]) if losses else None,
            "loss_count": int(len(losses)),
        },
        "graph": train_meta,
    }
