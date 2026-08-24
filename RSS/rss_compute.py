from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import gmean, rankdata

try:
    from .stats import quantiles
    from .schema import RSSConfig
except ImportError:
    from RSS.stats import quantiles
    from RSS.schema import RSSConfig


def _prepare_rss_inputs(
    expression: sp.csr_matrix,
    gene_names: np.ndarray,
    remove_mt: bool,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    keep_mask = np.ones(expression.shape[1], dtype=bool)
    if remove_mt:
        lower = np.char.lower(gene_names)
        keep_mask = ~(np.char.startswith(lower, "mt-"))

    expression = expression[:, keep_mask]
    gene_names = gene_names[keep_mask]
    bool_expression = expression.astype(bool).astype(np.float32)
    return expression, gene_names, bool_expression


def _compute_ranks(expression: sp.csr_matrix) -> np.ndarray:
    n_spots, n_genes = expression.shape
    ranks = np.zeros((n_spots, n_genes), dtype=np.float32)
    for i in range(n_spots):
        row = expression.getrow(i).toarray().ravel()
        ranks[i] = rankdata(row, method="average")
    return ranks


def _project_score(raw: np.ndarray, projection: str) -> np.ndarray:
    if projection == "exp2":
        return np.exp2(np.clip(raw, 0.0, None)) - 1.0
    if projection == "softplus":
        return np.log1p(np.exp(raw))
    raise ValueError(f"Unknown projection method: {projection}")


def _top_indices(values: np.ndarray, n_top: int, *, positive_only: bool) -> np.ndarray:
    if values.size == 0 or n_top <= 0:
        return np.array([], dtype=np.int64)
    if positive_only:
        idx = np.flatnonzero(values > 0)
        if idx.size == 0:
            return idx
        return idx[np.argsort(-values[idx])]
    m = min(max(1, int(n_top)), values.shape[0])
    idx = np.argpartition(-values, m - 1)[:m]
    idx = idx[np.argsort(-values[idx])]
    return idx[values[idx] > 0]


def _valid_microdomain(
    neighbor_idx: np.ndarray,
    neighbor_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = neighbor_idx >= 0
    idx = neighbor_idx[valid].astype(np.int64, copy=False)
    if idx.size == 0:
        return idx, np.array([], dtype=np.float64)

    if neighbor_weight is None:
        weight = np.full(idx.shape[0], 1.0 / idx.shape[0], dtype=np.float64)
        return idx, weight

    raw_weight = np.asarray(neighbor_weight[valid], dtype=np.float64)
    raw_weight = np.clip(raw_weight, 0.0, None)
    total = float(raw_weight.sum())
    if total <= 0.0 or not np.isfinite(total):
        weight = np.full(idx.shape[0], 1.0 / idx.shape[0], dtype=np.float64)
    else:
        weight = raw_weight / total
    return idx, weight


def _weighted_gmean(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    safe = np.clip(np.asarray(matrix, dtype=np.float64), 1e-12, None)
    return np.exp(weight @ np.log(safe)).astype(np.float32, copy=False)


def _weighted_mean(matrix: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return (weight @ np.asarray(matrix, dtype=np.float64)).astype(np.float32, copy=False)


def _empty_rss_sparse_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "spot_id",
            "gene",
            "rss_score",
            "rss_rank_in_spot",
            "rss_raw",
            "rss_support",
            "microdomain_rank",
            "global_rank",
            "microdomain_detect_rate",
            "global_detect_rate",
        ]
    )


def compute_top_sets(
    expression: sp.csr_matrix,
    gene_names: np.ndarray,
    neighbor_idx: np.ndarray,
    rss_cfg: RSSConfig,
    top_ns: tuple[int, ...],
    remove_mt: bool,
    neighbor_weight: np.ndarray | None = None,
) -> dict[int, list[set[int]]]:
    expression_use, _, bool_expression = _prepare_rss_inputs(expression, gene_names, remove_mt)
    ranks = _compute_ranks(expression_use)
    global_rank = gmean(ranks, axis=0) + 1e-12
    global_frac = np.asarray(bool_expression.mean(axis=0)).ravel() + 1e-12

    top_sets = {int(n): [] for n in top_ns}
    for i in range(expression_use.shape[0]):
        nb, weight = _valid_microdomain(
            neighbor_idx[i],
            neighbor_weight[i] if neighbor_weight is not None else None,
        )
        if nb.size == 0:
            for n in top_sets:
                top_sets[n].append(set())
            continue

        microdomain_rank = _weighted_gmean(ranks[nb], weight) + 1e-12
        rss_raw = np.log2(microdomain_rank / global_rank)

        if rss_cfg.no_expression_fraction:
            rss_support = np.ones_like(rss_raw)
        else:
            local_bool = bool_expression[nb].toarray()
            local_frac = _weighted_mean(local_bool, weight)
            rss_support = np.clip(local_frac / global_frac, 0.0, 1.0)

        rss_score_row = _project_score(rss_raw * rss_support, rss_cfg.projection)

        for n in top_sets:
            n_use = min(int(n), rss_score_row.shape[0])
            if n_use == 0:
                top_sets[n].append(set())
                continue
            idx = np.argpartition(-rss_score_row, n_use - 1)[:n_use]
            top_sets[n].append(set(int(x) for x in idx.tolist()))

    return top_sets


def compute_and_sparsify_rss(
    expression: sp.csr_matrix,
    gene_names: np.ndarray,
    spot_ids: np.ndarray,
    neighbor_idx: np.ndarray,
    rss_cfg: RSSConfig,
    remove_mt: bool,
    top_n_for_qc: tuple[int, ...],
    neighbor_weight: np.ndarray | None = None,
) -> dict:
    expression_use, gene_names_use, bool_expression = _prepare_rss_inputs(
        expression,
        gene_names,
        remove_mt,
    )

    n_spots, n_genes = expression_use.shape
    ranks = _compute_ranks(expression_use)
    global_rank = gmean(ranks, axis=0) + 1e-12
    global_frac = np.asarray(bool_expression.mean(axis=0)).ravel() + 1e-12

    rss_rows: list[dict] = []
    top_sets = {int(n): [] for n in top_n_for_qc}

    nonzero_counts = np.zeros(n_spots, dtype=np.int32)
    top10_share = np.zeros(n_spots, dtype=np.float32)
    gene_nonzero_counts = np.zeros(n_genes, dtype=np.int32)

    for i in range(n_spots):
        nb, weight = _valid_microdomain(
            neighbor_idx[i],
            neighbor_weight[i] if neighbor_weight is not None else None,
        )

        if nb.size == 0:
            for n in top_sets:
                top_sets[n].append(set())
            continue

        microdomain_rank = _weighted_gmean(ranks[nb], weight) + 1e-12
        rss_raw = np.log2(microdomain_rank / global_rank)

        if rss_cfg.no_expression_fraction:
            rss_support = np.ones_like(rss_raw)
        else:
            local_bool = bool_expression[nb].toarray()
            local_frac = _weighted_mean(local_bool, weight)
            rss_support = np.clip(local_frac / global_frac, 0.0, 1.0)

        rss_score_row = _project_score(rss_raw * rss_support, rss_cfg.projection)

        positive_idx = np.flatnonzero(rss_score_row > 0)
        nonzero_counts[i] = int(positive_idx.size)
        if positive_idx.size > 0:
            gene_nonzero_counts[positive_idx] += 1

        total_positive = float(rss_score_row[positive_idx].sum()) if positive_idx.size else 0.0
        if total_positive > 0:
            top10 = min(10, positive_idx.size)
            top10_idx = np.argpartition(-rss_score_row[positive_idx], top10 - 1)[:top10]
            top10_share[i] = float(rss_score_row[positive_idx][top10_idx].sum() / total_positive)

        for n in top_sets:
            n_use = min(int(n), rss_score_row.shape[0])
            if n_use == 0:
                top_sets[n].append(set())
            else:
                idx_top = np.argpartition(-rss_score_row, n_use - 1)[:n_use]
                top_sets[n].append(set(int(x) for x in idx_top.tolist()))

        if rss_cfg.sparsify_rule not in {"positive", "topM"}:
            raise ValueError(f"Unknown sparsify_rule: {rss_cfg.sparsify_rule}")
        keep = _top_indices(
            rss_score_row,
            rss_cfg.top_m,
            positive_only=rss_cfg.sparsify_rule == "positive",
        )

        for rank, gene_idx in enumerate(keep.tolist(), start=1):
            rss_rows.append(
                {
                    "spot_id": str(spot_ids[i]),
                    "gene": str(gene_names_use[gene_idx]),
                    "rss_score": float(rss_score_row[gene_idx]),
                    "rss_rank_in_spot": int(rank),
                    "rss_raw": float(rss_raw[gene_idx]),
                    "rss_support": float(rss_support[gene_idx]),
                    "microdomain_rank": float(microdomain_rank[gene_idx]),
                    "global_rank": float(global_rank[gene_idx]),
                    "microdomain_detect_rate": float(local_frac[gene_idx])
                    if not rss_cfg.no_expression_fraction
                    else 1.0,
                    "global_detect_rate": float(global_frac[gene_idx]),
                }
            )

    rss_sparse_df = pd.DataFrame(rss_rows)
    if rss_sparse_df.empty:
        rss_sparse_df = _empty_rss_sparse_df()

    distribution = {
        "nonzero_genes_count": quantiles(nonzero_counts.astype(np.float32)),
        "top10_share": quantiles(top10_share.astype(np.float32)),
        "gene_level_nonzero_ratio": quantiles(
            (gene_nonzero_counts / max(1, n_spots)).astype(np.float32)
        ),
    }

    return {
        "rss_sparse_df": rss_sparse_df,
        "top_sets": top_sets,
        "distribution": distribution,
    }
