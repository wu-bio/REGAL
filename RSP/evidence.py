from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import RSPDiscoveryConfig


@dataclass
class RSPEvidence:
    x_raw: np.ndarray
    x_model: np.ndarray
    reliability: np.ndarray
    genes: np.ndarray
    spot_ids: np.ndarray
    metadata: dict
    local_support: np.ndarray | None = None


def build_rss_evidence(
    rss_sparse_path: Path,
    spot_ids: np.ndarray,
    cfg: RSPDiscoveryConfig,
) -> RSPEvidence:
    rss = load_rss_table(rss_sparse_path, cfg)
    spot_ids = np.asarray(spot_ids, dtype=str)
    genes = np.asarray(sorted(rss["gene"].astype(str).unique()), dtype=str)

    x_raw = np.zeros((genes.size, spot_ids.size), dtype=np.float32)
    reliability = np.zeros_like(x_raw, dtype=np.float32)
    local_support = np.zeros_like(x_raw, dtype=np.float32)
    row_reliability, reliability_source = _row_reliability(rss, cfg)
    row_local_support, local_support_source = _row_local_support(rss, cfg)

    if rss.shape[0] and genes.size and spot_ids.size:
        matrix_rows = pd.DataFrame(
            {
                "gene": rss["gene"].to_numpy(dtype=str, copy=False),
                "spot_id": rss["spot_id"].to_numpy(dtype=str, copy=False),
                "rss_value": np.clip(
                    pd.to_numeric(rss[cfg.rss_value_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32),
                    0.0,
                    None,
                ),
                "reliability": np.asarray(row_reliability, dtype=np.float32),
                "local_support": np.asarray(row_local_support, dtype=np.float32),
            }
        )
        matrix_rows = (
            matrix_rows.groupby(["gene", "spot_id"], sort=False, observed=True)
            .agg(
                rss_value=("rss_value", "max"),
                reliability=("reliability", "max"),
                local_support=("local_support", "max"),
            )
            .reset_index()
        )

        gene_idx = pd.Categorical(matrix_rows["gene"], categories=genes).codes.astype(np.int64, copy=False)
        spot_idx = pd.Index(spot_ids).get_indexer(matrix_rows["spot_id"].to_numpy(dtype=str, copy=False))
        valid = (gene_idx >= 0) & (spot_idx >= 0)
        if np.any(valid):
            gene_idx = gene_idx[valid]
            spot_idx = spot_idx[valid]
            x_raw[gene_idx, spot_idx] = matrix_rows.loc[valid, "rss_value"].to_numpy(dtype=np.float32)
            reliability[gene_idx, spot_idx] = matrix_rows.loc[valid, "reliability"].to_numpy(dtype=np.float32)
            local_support[gene_idx, spot_idx] = matrix_rows.loc[valid, "local_support"].to_numpy(dtype=np.float32)

    x_model = _build_model_view(x_raw, cfg)
    metadata = {
        "n_genes": int(genes.size),
        "n_spots": int(spot_ids.size),
        "rss_rows_used": int(rss.shape[0]),
        "rss_value_col": str(cfg.rss_value_col),
        "rss_transform": str(cfg.rss_transform),
        "rss_rank_cutoff": int(cfg.rss_rank_cutoff) if cfg.rss_rank_cutoff is not None else None,
        "rss_rank_filter_applied": cfg.rss_rank_cutoff is not None,
        "model_view": "robust_gene_scaled_rss",
        "reliability_source": reliability_source,
        "local_support_source": local_support_source,
        "rss_gene_index": genes.astype(str).tolist(),
        "rss_spot_index": spot_ids.astype(str).tolist(),
    }
    return RSPEvidence(
        x_raw=x_raw,
        x_model=x_model,
        reliability=np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False),
        genes=genes,
        spot_ids=spot_ids,
        metadata=metadata,
        local_support=np.clip(local_support, 0.0, 1.0).astype(np.float32, copy=False),
    )


def load_rss_table(rss_sparse_path: Path, cfg: RSPDiscoveryConfig) -> pd.DataFrame:
    if not rss_sparse_path.exists():
        raise FileNotFoundError(f"Missing required RSS sparse file: {rss_sparse_path}")
    rss = pd.read_parquet(rss_sparse_path)
    required = {"spot_id", "gene", cfg.rss_value_col}
    missing = required - set(rss.columns)
    if missing:
        raise ValueError(f"rss_sparse parquet missing required columns: {sorted(missing)}")
    columns = ["spot_id", "gene", cfg.rss_value_col]
    if "rss_rank_in_spot" in rss.columns:
        columns.append("rss_rank_in_spot")
    for column in cfg.reliability_columns:
        if column in rss.columns and column not in columns:
            columns.append(column)
    rss = rss.loc[:, columns].copy()
    rss["spot_id"] = rss["spot_id"].astype(str)
    rss["gene"] = rss["gene"].astype(str)
    rss[cfg.rss_value_col] = pd.to_numeric(rss[cfg.rss_value_col], errors="coerce").fillna(0.0)
    if cfg.rss_rank_cutoff is not None:
        if "rss_rank_in_spot" not in rss.columns:
            raise ValueError("RSPDiscoveryConfig.rss_rank_cutoff requires rss_rank_in_spot in rss_sparse parquet.")
        cutoff = int(cfg.rss_rank_cutoff)
        if cutoff <= 0:
            raise ValueError("RSPDiscoveryConfig.rss_rank_cutoff must be a positive integer or None.")
        rank = pd.to_numeric(rss["rss_rank_in_spot"], errors="coerce")
        rss = rss.loc[rank.le(cutoff).fillna(False)].copy()
    return rss.reset_index(drop=True)


def _row_reliability(rss: pd.DataFrame, cfg: RSPDiscoveryConfig) -> tuple[np.ndarray, str]:
    if "rss_support" in rss.columns:
        values = pd.to_numeric(rss["rss_support"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return np.clip(values, 0.0, 1.0), "rss_support"

    if {"microdomain_detect_rate", "global_detect_rate"}.issubset(rss.columns):
        local = pd.to_numeric(rss["microdomain_detect_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        global_rate = pd.to_numeric(rss["global_detect_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        values = local / np.maximum(global_rate, 1e-6)
        return np.clip(values, 0.0, 1.0), "detect_rate_ratio"

    score = np.clip(pd.to_numeric(rss[cfg.rss_value_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, None)
    scale = float(np.quantile(score[score > 0], 0.90)) if np.any(score > 0) else 1.0
    return np.clip(score / max(scale, 1e-6), 0.0, 1.0).astype(np.float32, copy=False), "rss_score_fallback"


def _row_local_support(rss: pd.DataFrame, cfg: RSPDiscoveryConfig) -> tuple[np.ndarray, str]:
    if "microdomain_detect_rate" in rss.columns:
        values = pd.to_numeric(rss["microdomain_detect_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return np.clip(values, 0.0, 1.0), "microdomain_detect_rate"

    if "rss_support" in rss.columns:
        values = pd.to_numeric(rss["rss_support"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return np.clip(values, 0.0, 1.0), "rss_support"

    values, source = _row_reliability(rss, cfg)
    return values, source


def _build_model_view(x_raw: np.ndarray, cfg: RSPDiscoveryConfig) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float32)
    if cfg.rss_transform == "log1p":
        x = np.log1p(np.clip(x, 0.0, None)).astype(np.float32, copy=False)
    else:
        x = np.clip(x, 0.0, None).astype(np.float32, copy=False)

    scale = np.zeros((x.shape[0], 1), dtype=np.float32)
    for idx, row in enumerate(x):
        positive = row[row > 0]
        if positive.size:
            scale[idx, 0] = np.float32(np.quantile(positive, 0.95))
    scale = np.maximum(scale, np.float32(1e-6))
    out = x / scale
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
