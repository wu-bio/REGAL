from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    connected_component_indices,
    gene_mask,
    low_info_gene_mask,
    moran_matrix,
)
from .evidence import RSPEvidence
from .schema import RSPDiscoveryConfig, RSPScoringConfig


def compute_gene_evidence_metrics(
    evidence: RSPEvidence,
    spatial_neighbors: np.ndarray,
    moran_neighbors: np.ndarray,
    discovery_cfg: RSPDiscoveryConfig,
    scoring_cfg: RSPScoringConfig,
) -> pd.DataFrame:
    x_raw = np.asarray(evidence.x_raw, dtype=np.float32)
    reliability = np.asarray(evidence.reliability, dtype=np.float32)
    local_support = getattr(evidence, "local_support", None)
    if local_support is None:
        local_support = reliability
    local_support = np.asarray(local_support, dtype=np.float32)
    genes = np.asarray(evidence.genes, dtype=str)

    if genes.size == 0:
        return _empty_metrics()

    nonzero_mask = x_raw > 0
    nonzero_rss_spot_count = nonzero_mask.sum(axis=1)
    nonzero_rss_spot_fraction = nonzero_rss_spot_count / max(int(x_raw.shape[1]), 1)
    spatial_autocorrelation_score = moran_matrix(x_raw, moran_neighbors)
    structured_component_support = _structured_component_support_matrix(
        nonzero_mask,
        spatial_neighbors,
        min_component_size=discovery_cfg.structured_component_min_size,
    )
    local_support_score = np.divide(
        (x_raw * local_support).sum(axis=1),
        np.maximum(x_raw.sum(axis=1), 1e-6),
        dtype=np.float32,
    )
    reliability_mean = np.divide(
        (reliability * nonzero_mask).sum(axis=1),
        np.maximum(nonzero_rss_spot_count, 1),
        dtype=np.float32,
    )
    confounder_flag = (
        gene_mask(genes, scoring_cfg.blacklist_prefixes, scoring_cfg.blacklist_genes)
        | gene_mask(genes, scoring_cfg.housekeeping_prefixes, scoring_cfg.housekeeping_genes)
        | low_info_gene_mask(genes, scoring_cfg.low_info_prefixes)
    )
    min_nonzero_spots = max(
        int(discovery_cfg.min_nonzero_rss_spot_count),
        int(np.ceil(float(discovery_cfg.min_nonzero_rss_spot_fraction) * int(x_raw.shape[1]))),
    )

    rows = []
    for idx, gene in enumerate(genes):
        reasons: list[str] = []
        if bool(confounder_flag[idx]):
            reasons.append("confounder_gene")
        if int(nonzero_rss_spot_count[idx]) < min_nonzero_spots:
            reasons.append("low_nonzero_rss_support")
        max_nonzero_fraction = getattr(discovery_cfg, "max_nonzero_rss_spot_fraction", None)
        if max_nonzero_fraction is not None and float(nonzero_rss_spot_fraction[idx]) > float(max_nonzero_fraction):
            reasons.append("high_nonzero_rss_coverage")
        if float(spatial_autocorrelation_score[idx]) < float(discovery_cfg.min_spatial_autocorrelation_score):
            reasons.append("low_spatial_autocorrelation")
        if float(structured_component_support[idx]) < float(discovery_cfg.min_structured_component_support):
            reasons.append("low_structured_component_support")
        if float(local_support_score[idx]) < float(discovery_cfg.min_local_support_score):
            reasons.append("low_local_support")

        deduped = _dedupe_reasons(reasons)
        rows.append(
            {
                "gene_index": int(idx),
                "gene": str(gene),
                "nonzero_rss_spot_count": int(nonzero_rss_spot_count[idx]),
                "nonzero_rss_spot_fraction": float(nonzero_rss_spot_fraction[idx]),
                "spatial_autocorrelation_score": float(spatial_autocorrelation_score[idx]),
                "structured_component_support": float(structured_component_support[idx]),
                "local_support_score": float(local_support_score[idx]),
                "reliability_mean": float(reliability_mean[idx]),
                "confounder_flag": bool(confounder_flag[idx]),
                "is_kept_for_decomposition": len(deduped) == 0,
                "drop_reason": ";".join(deduped),
            }
        )

    return (
        _add_gene_evidence_score(pd.DataFrame(rows))
        .sort_values(
            [
                "is_kept_for_decomposition",
                "gene_evidence_score",
                "spatial_autocorrelation_score",
                "structured_component_support",
                "local_support_score",
            ],
            ascending=[False, False, False, False, False],
        )
        .reset_index(drop=True)
    )


def select_decomposition_gene_indices(
    metrics: pd.DataFrame,
    adaptive_iqr_multiplier: float = 0.5,
    enable_adaptive_gene_selection: bool = True,
    max_genes: int | None = None,
) -> np.ndarray:
    if metrics.empty:
        return np.array([], dtype=np.int64)
    work = _add_gene_evidence_score(metrics.loc[metrics["is_kept_for_decomposition"].astype(bool)].copy())
    if work.empty:
        return np.array([], dtype=np.int64)
    sort_cols = [
        col
        for col in [
            "gene_evidence_score",
            "spatial_autocorrelation_score",
            "structured_component_support",
            "local_support_score",
        ]
        if col in work.columns
    ]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    if not bool(enable_adaptive_gene_selection):
        if max_genes is not None and int(max_genes) > 0:
            work = work.head(int(max_genes))
        return work["gene_index"].to_numpy(dtype=np.int64)

    threshold = _adaptive_evidence_threshold(
        work["gene_evidence_score"].to_numpy(dtype=np.float64),
        iqr_multiplier=adaptive_iqr_multiplier,
    )
    selected = work.loc[work["gene_evidence_score"].ge(threshold)].copy()
    if selected.empty:
        selected = work.head(1).copy()
    work = selected
    if max_genes is not None and int(max_genes) > 0:
        work = work.head(int(max_genes))
    return work["gene_index"].to_numpy(dtype=np.int64)


def _add_gene_evidence_score(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "gene_evidence_score" in metrics.columns:
        return metrics

    work = metrics.copy()
    kept_mask = work["is_kept_for_decomposition"].astype(bool)
    score = pd.Series(0.0, index=work.index, dtype=float)
    kept = work.loc[kept_mask].copy()
    if kept.empty:
        work["gene_evidence_score"] = score.astype(float)
        return work

    def robust01(column: str) -> pd.Series:
        if column not in kept.columns:
            return pd.Series(0.0, index=kept.index, dtype=float)
        values = pd.to_numeric(kept[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        low, high = np.quantile(values.to_numpy(dtype=np.float64), [0.05, 0.95])
        if high <= low:
            return pd.Series(0.0, index=kept.index, dtype=float)
        return ((values - low) / max(float(high - low), 1e-12)).clip(0.0, 1.0).astype(float)

    kept_score = (
        0.40 * robust01("spatial_autocorrelation_score")
        + 0.35 * robust01("structured_component_support")
        + 0.25 * robust01("local_support_score")
    )
    score.loc[kept.index] = kept_score.clip(0.0, 1.0)
    work["gene_evidence_score"] = score.astype(float)
    return work


def _adaptive_evidence_threshold(scores: np.ndarray, iqr_multiplier: float = 0.5) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    threshold = float(q50 + max(0.0, float(iqr_multiplier)) * (q75 - q25))
    return float(np.clip(threshold, float(values.min()), float(values.max())))


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "gene_index",
            "gene",
            "nonzero_rss_spot_count",
            "nonzero_rss_spot_fraction",
            "spatial_autocorrelation_score",
            "structured_component_support",
            "local_support_score",
            "reliability_mean",
            "confounder_flag",
            "is_kept_for_decomposition",
            "drop_reason",
            "gene_evidence_score",
        ]
    )


def _structured_component_support_matrix(
    mask_matrix: np.ndarray,
    neighbors: np.ndarray,
    *,
    min_component_size: int,
) -> np.ndarray:
    min_size = max(1, int(min_component_size))
    mask = np.asarray(mask_matrix, dtype=bool)
    return np.asarray(
        [_structured_component_support(row, neighbors, min_size) for row in mask],
        dtype=np.float32,
    )


def _structured_component_support(mask: np.ndarray, neighbors: np.ndarray, min_component_size: int) -> float:
    active = int(np.count_nonzero(mask))
    if active <= 0 or neighbors.size == 0:
        return 0.0
    supported = 0
    for component in connected_component_indices(mask, neighbors):
        if int(component.size) >= int(min_component_size):
            supported += int(component.size)
    return float(supported / active) if active else 0.0


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped
