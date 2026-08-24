from __future__ import annotations

import numpy as np
import pandas as pd

from .common import gene_mask, low_info_gene_mask
from .schema import RSPScoringConfig


def annotate_core_genes(candidates: list[dict], cfg: RSPScoringConfig) -> None:
    specificity_lookup = _program_specificity_lookup(candidates)
    for candidate in candidates:
        genes = np.asarray(candidate.get("genes", []), dtype=str)
        loading = np.asarray(candidate.get("gene_loadings", np.zeros(genes.shape[0])), dtype=np.float64)
        if genes.size == 0 or loading.size == 0:
            _set_empty_core(candidate)
            continue

        loading = np.clip(loading, 0.0, None)
        total = max(float(loading.sum()), 1e-12)
        frac = loading / total

        local_rss_strength = _aligned_vector(candidate, "gene_local_rss_strength", genes.shape[0], fill=0.0)
        local_rss_coverage = _aligned_vector(candidate, "gene_local_rss_coverage", genes.shape[0], fill=0.0)
        local_rss_lift = _aligned_vector(candidate, "gene_local_rss_lift", genes.shape[0], fill=0.0)
        program_specificity = np.asarray(
            [specificity_lookup.get((str(candidate.get("candidate_id", "")), str(gene)), 0.0) for gene in genes],
            dtype=np.float64,
        )

        confounder = (
            gene_mask(genes, cfg.blacklist_prefixes, cfg.blacklist_genes)
            | gene_mask(genes, cfg.housekeeping_prefixes, cfg.housekeeping_genes)
            | low_info_gene_mask(genes, cfg.low_info_prefixes)
        )

        loading_strength = _robust01(frac)
        local_rss_lift_scaled = _lift01(local_rss_lift)
        core_score = np.clip(
            0.40 * loading_strength
            + 0.35 * np.clip(local_rss_strength, 0.0, 1.0)
            + 0.15 * np.clip(local_rss_coverage, 0.0, 1.0)
            + 0.10 * local_rss_lift_scaled,
            0.0,
            1.0,
        )
        min_local_rss_strength = max(0.0, float(cfg.core_min_local_rss_strength))
        min_local_rss_coverage = max(0.0, float(cfg.core_min_local_rss_coverage))
        min_local_rss_lift = max(0.0, float(cfg.core_min_local_rss_lift))
        eligible = (
            (loading > 0.0)
            & (local_rss_strength >= min_local_rss_strength)
            & (local_rss_coverage >= min_local_rss_coverage)
            & (local_rss_lift >= min_local_rss_lift)
        )
        if bool(cfg.core_exclude_confounder_genes):
            eligible &= ~confounder

        evidence_order = _core_evidence_order(core_score, frac, local_rss_strength, local_rss_coverage, local_rss_lift, genes)
        eligible_order = [int(idx) for idx in evidence_order if bool(eligible[idx])]
        strict_selection = _adaptive_core_selection(
            eligible_order=eligible_order,
            core_score=core_score,
            effective_gene_number=float(candidate.get("effective_gene_number", 0.0)),
            cfg=cfg,
        )
        selected = list(strict_selection["selected_indices"])
        if not selected and bool(cfg.core_use_elbow_selection) and bool(cfg.core_exclude_confounder_genes):
            relaxed = (
                (loading > 0.0)
                & (local_rss_strength >= min_local_rss_strength)
                & (local_rss_coverage >= min_local_rss_coverage)
                & (local_rss_lift >= min_local_rss_lift)
            )
            selected = _adaptive_core_indices(
                eligible_order=[int(idx) for idx in evidence_order if bool(relaxed[idx])],
                core_score=core_score,
                effective_gene_number=float(candidate.get("effective_gene_number", 0.0)),
                cfg=cfg,
            )
        if not selected and bool(cfg.core_use_elbow_selection):
            relaxed = (loading > 0.0) & (local_rss_strength > 0.0) & (local_rss_coverage > 0.0)
            selected = _adaptive_core_indices(
                eligible_order=[int(idx) for idx in evidence_order if bool(relaxed[idx])],
                core_score=core_score,
                effective_gene_number=float(candidate.get("effective_gene_number", 0.0)),
                cfg=cfg,
            )

        core_indices = np.asarray(selected, dtype=np.int64)
        core_rank = np.zeros(genes.shape[0], dtype=np.int32)
        for rank, idx in enumerate(core_indices, start=1):
            core_rank[int(idx)] = int(rank)

        candidate["program_specificity"] = program_specificity.astype(np.float32, copy=False)
        candidate["gene_local_rss_strength"] = local_rss_strength.astype(np.float32, copy=False)
        candidate["gene_local_rss_coverage"] = local_rss_coverage.astype(np.float32, copy=False)
        candidate["gene_local_rss_lift"] = local_rss_lift.astype(np.float32, copy=False)
        candidate["gene_loading_strength"] = loading_strength.astype(np.float32, copy=False)
        candidate["gene_local_rss_lift_scaled"] = local_rss_lift_scaled.astype(np.float32, copy=False)
        candidate["core_gene_eligible"] = eligible.astype(bool, copy=False)
        candidate["core_gene_drop_reason"] = _core_drop_reasons(
            loading=loading,
            local_rss_strength=local_rss_strength,
            local_rss_coverage=local_rss_coverage,
            local_rss_lift=local_rss_lift,
            core_score=core_score,
            confounder=confounder if bool(cfg.core_exclude_confounder_genes) else np.zeros(genes.shape[0], dtype=bool),
            selected_mask=core_rank > 0,
            eligible=eligible,
            min_local_rss_strength=min_local_rss_strength,
            min_local_rss_coverage=min_local_rss_coverage,
            min_local_rss_lift=min_local_rss_lift,
            min_core_score=float(cfg.core_min_core_score),
            selection_method=str(strict_selection["core_selection_method"]),
        )
        candidate["core_gene_indices"] = core_indices
        candidate["core_genes"] = genes[core_indices].astype(str).tolist()
        candidate["core_gene_count"] = int(core_indices.size)
        candidate["eligible_gene_count"] = int(np.count_nonzero(eligible))
        candidate["core_eligible_fraction"] = float(np.mean(eligible)) if eligible.size else 0.0
        candidate["core_selection_method"] = str(strict_selection["core_selection_method"])
        candidate["core_score_threshold"] = float(strict_selection["core_score_threshold"])
        candidate["raw_elbow_size"] = int(strict_selection["raw_elbow_size"])
        candidate["effective_gene_floor"] = int(strict_selection["effective_gene_floor"])
        candidate["core_score_at_elbow"] = float(strict_selection["core_score_at_elbow"])
        candidate["core_score_median_eligible"] = float(strict_selection["core_score_median_eligible"])
        candidate["core_score_median_core"] = float(np.median(core_score[core_indices])) if core_indices.size else 0.0
        candidate["core_loading_fraction"] = float(frac[core_indices].sum()) if core_indices.size else 0.0
        candidate["core_mean_local_rss_strength"] = (
            float(np.mean(local_rss_strength[core_indices])) if core_indices.size else 0.0
        )
        candidate["core_mean_local_rss_coverage"] = (
            float(np.mean(local_rss_coverage[core_indices])) if core_indices.size else 0.0
        )
        candidate["core_mean_local_rss_lift"] = float(np.mean(local_rss_lift[core_indices])) if core_indices.size else 0.0
        candidate["core_gene_score"] = core_score.astype(np.float32, copy=False)
        candidate["core_gene_rank"] = core_rank


def build_rsp_gene_loadings_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.extend(_gene_loading_rows(candidate, rsp_id=str(candidate["rsp_id"])))
    return pd.DataFrame(rows, columns=_loading_columns())


def build_rsp_core_gene_loadings_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.extend(row for row in _gene_loading_rows(candidate, rsp_id=str(candidate["rsp_id"])) if bool(row["is_core_gene"]))
    return pd.DataFrame(rows, columns=_loading_columns()).sort_values(
        ["rsp_id", "core_rank", "loading_rank"],
        ascending=[True, True, True],
        ignore_index=True,
    )


def _program_specificity_lookup(candidates: list[dict]) -> dict[tuple[str, str], float]:
    total_by_gene: dict[str, float] = {}
    loading_by_candidate_gene: dict[tuple[str, str], float] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        genes = np.asarray(candidate.get("genes", []), dtype=str)
        loading = np.asarray(candidate.get("gene_loadings", np.zeros(genes.shape[0])), dtype=np.float64)
        for gene, value in zip(genes, loading, strict=False):
            val = max(0.0, float(value))
            key = str(gene)
            loading_by_candidate_gene[(candidate_id, key)] = val
            total_by_gene[key] = total_by_gene.get(key, 0.0) + val
    return {
        key: float(value / max(total_by_gene.get(key[1], 0.0), 1e-12))
        for key, value in loading_by_candidate_gene.items()
    }


def _robust01(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    positive = arr[arr > 0]
    if positive.size == 0:
        return out
    scale = float(np.quantile(positive, 0.95))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = float(positive.max())
    if scale <= 0.0 or not np.isfinite(scale):
        return out
    out = arr / scale
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0)


def _lift01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    excess = np.clip(arr - 1.0, 0.0, None)
    positive = excess[excess > 0]
    if positive.size == 0:
        return out
    scale = float(np.quantile(positive, 0.95))
    if scale <= 0.0 or not np.isfinite(scale):
        scale = float(positive.max())
    if scale <= 0.0 or not np.isfinite(scale):
        return out
    out = excess / scale
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0)


def _core_evidence_order(
    core_score: np.ndarray,
    loading_fraction: np.ndarray,
    local_rss_strength: np.ndarray,
    local_rss_coverage: np.ndarray,
    local_rss_lift: np.ndarray,
    genes: np.ndarray,
) -> list[int]:
    order = list(range(np.asarray(core_score).shape[0]))
    return sorted(
        order,
        key=lambda idx: (
            -float(core_score[idx]),
            -float(local_rss_strength[idx]),
            -float(local_rss_coverage[idx]),
            -float(local_rss_lift[idx]),
            -float(loading_fraction[idx]),
            str(genes[idx]),
        ),
    )


def _adaptive_core_indices(
    *,
    eligible_order: list[int],
    core_score: np.ndarray,
    effective_gene_number: float,
    cfg: RSPScoringConfig,
) -> list[int]:
    return list(
        _adaptive_core_selection(
            eligible_order=eligible_order,
            core_score=core_score,
            effective_gene_number=effective_gene_number,
            cfg=cfg,
        )["selected_indices"]
    )


def _adaptive_core_selection(
    *,
    eligible_order: list[int],
    core_score: np.ndarray,
    effective_gene_number: float,
    cfg: RSPScoringConfig,
) -> dict:
    if not eligible_order:
        return _empty_core_selection_summary(method=_core_selection_method(cfg), threshold=float(cfg.core_min_core_score))
    scores = np.asarray([core_score[idx] for idx in eligible_order], dtype=np.float64)
    positive = scores > 0
    if np.any(positive):
        eligible_order = [idx for idx, keep in zip(eligible_order, positive, strict=False) if bool(keep)]
        scores = scores[positive]
    if not eligible_order:
        return _empty_core_selection_summary(method=_core_selection_method(cfg), threshold=float(cfg.core_min_core_score))

    raw_elbow_size = _loading_elbow_size(scores, sensitivity=float(cfg.core_elbow_sensitivity))
    effective_scale = float(np.sqrt(max(0.0, float(effective_gene_number))))
    effective_floor = int(np.ceil(effective_scale))
    threshold = max(0.0, float(cfg.core_min_core_score))
    if bool(cfg.core_use_elbow_selection):
        target_size = int(raw_elbow_size)
        min_size = max(0, int(cfg.core_safety_min_genes), effective_floor)
        target_size = max(target_size, min_size)
        if cfg.core_safety_max_genes is not None:
            target_size = min(target_size, int(cfg.core_safety_max_genes))
        target_size = min(target_size, len(eligible_order))
        selected = eligible_order[:target_size]
    else:
        selected = [idx for idx, score in zip(eligible_order, scores, strict=False) if float(score) >= threshold]
    elbow_idx = min(max(1, raw_elbow_size), scores.size) - 1
    return {
        "selected_indices": selected,
        "eligible_gene_count": int(len(eligible_order)),
        "core_selection_method": _core_selection_method(cfg),
        "core_score_threshold": float(threshold),
        "raw_elbow_size": int(raw_elbow_size),
        "effective_gene_floor": int(effective_floor),
        "core_score_at_elbow": float(scores[elbow_idx]) if scores.size else 0.0,
        "core_score_median_eligible": float(np.median(scores)) if scores.size else 0.0,
    }


def _core_selection_method(cfg: RSPScoringConfig) -> str:
    return "elbow" if bool(cfg.core_use_elbow_selection) else "threshold"


def _empty_core_selection_summary(*, method: str = "threshold", threshold: float = 0.0) -> dict:
    return {
        "selected_indices": [],
        "eligible_gene_count": 0,
        "core_selection_method": str(method),
        "core_score_threshold": float(threshold),
        "raw_elbow_size": 0,
        "effective_gene_floor": 0,
        "core_score_at_elbow": 0.0,
        "core_score_median_eligible": 0.0,
    }


def _loading_elbow_size(scores: np.ndarray, sensitivity: float) -> int:
    values = np.asarray(scores, dtype=np.float64)
    if values.size <= 2:
        return int(values.size)
    values = np.maximum(values, 0.0)
    if float(values[0]) <= 0.0:
        return 1
    normalized = values / max(float(values[0]), 1e-12)
    x = np.linspace(0.0, 1.0, values.size)
    line = 1.0 + (normalized[-1] - 1.0) * x
    distance = line - normalized
    knee = int(np.argmax(distance)) + 1
    drops = normalized[:-1] - normalized[1:]
    if drops.size:
        largest_drop_idx = int(np.argmax(drops))
        largest_drop = float(drops[largest_drop_idx])
        adaptive_drop = float(np.median(drops) + max(0.0, sensitivity) * np.std(drops))
        if largest_drop >= max(adaptive_drop, 0.05):
            knee = int(largest_drop_idx + 1)
    return max(1, min(knee, values.size))


def _gene_loading_rows(candidate: dict, rsp_id: str) -> list[dict]:
    genes = np.asarray(candidate["genes"], dtype=str)
    loading = np.asarray(candidate["gene_loadings"], dtype=np.float64)
    total = max(float(loading.sum()), 1e-12)
    frac = loading / total
    support_fraction = _aligned_vector(candidate, "gene_support_fraction", genes.shape[0], fill=0.0)
    local_rss_strength = _aligned_vector(candidate, "gene_local_rss_strength", genes.shape[0], fill=0.0)
    local_rss_coverage = _aligned_vector(candidate, "gene_local_rss_coverage", genes.shape[0], fill=0.0)
    local_rss_lift = _aligned_vector(candidate, "gene_local_rss_lift", genes.shape[0], fill=0.0)
    loading_strength = _aligned_vector(candidate, "gene_loading_strength", genes.shape[0], fill=0.0)
    local_rss_lift_scaled = _aligned_vector(candidate, "gene_local_rss_lift_scaled", genes.shape[0], fill=0.0)
    program_specificity = _aligned_vector(candidate, "program_specificity", genes.shape[0], fill=0.0)
    core_eligible = np.asarray(candidate.get("core_gene_eligible", np.zeros(genes.shape[0], dtype=bool)), dtype=bool)
    if core_eligible.shape[0] != genes.shape[0]:
        core_eligible = np.zeros(genes.shape[0], dtype=bool)
    core_drop_reason = np.asarray(candidate.get("core_gene_drop_reason", np.full(genes.shape[0], "", dtype=object)), dtype=object)
    if core_drop_reason.shape[0] != genes.shape[0]:
        core_drop_reason = np.full(genes.shape[0], "", dtype=object)
    core_rank = np.asarray(candidate.get("core_gene_rank", np.zeros(genes.shape[0])), dtype=np.int32)
    if core_rank.shape[0] != genes.shape[0]:
        core_rank = np.zeros(genes.shape[0], dtype=np.int32)
    core_score = _aligned_vector(candidate, "core_gene_score", genes.shape[0], fill=0.0)
    order = np.argsort(-loading)
    cumulative = np.cumsum(frac[order])
    blacklist = gene_mask(genes, ("MT-", "RPL", "RPS"), ("MALAT1", "NEAT1", "XIST"))
    housekeeping = gene_mask(genes, ("RPL", "RPS", "MT-"), ("ACTB", "GAPDH", "B2M", "TUBB", "MALAT1"))
    low_info = low_info_gene_mask(genes, ("AC", "AL", "AP", "LINC", "LOC", "MIR", "SNORD", "SNORA", "RNU"))
    rows = []
    prev = 0.0
    for rank_idx, gene_idx in enumerate(order):
        cur = float(cumulative[rank_idx])
        rows.append(
            {
                "rsp_id": rsp_id,
                "gene": str(genes[gene_idx]),
                "loading": float(loading[gene_idx]),
                "loading_rank": int(rank_idx + 1),
                "loading_fraction": float(frac[gene_idx]),
                "cumulative_fraction": cur,
                "in_contribution_50": bool(cur <= 0.50 or prev < 0.50),
                "in_contribution_70": bool(cur <= 0.70 or prev < 0.70),
                "in_contribution_90": bool(cur <= 0.90 or prev < 0.90),
                "gene_support_fraction": float(support_fraction[gene_idx]),
                "loading_strength": float(loading_strength[gene_idx]),
                "local_rss_strength": float(local_rss_strength[gene_idx]),
                "local_rss_coverage": float(local_rss_coverage[gene_idx]),
                "local_rss_lift": float(local_rss_lift[gene_idx]),
                "local_rss_lift_scaled": float(local_rss_lift_scaled[gene_idx]),
                "program_specificity": float(program_specificity[gene_idx]),
                "core_eligible": bool(core_eligible[gene_idx]),
                "is_core_gene": bool(core_rank[gene_idx] > 0),
                "core_rank": int(core_rank[gene_idx]),
                "core_score": float(core_score[gene_idx]),
                "core_drop_reason": str(core_drop_reason[gene_idx]),
                "is_blacklist_gene": bool(blacklist[gene_idx]),
                "is_housekeeping_gene": bool(housekeeping[gene_idx]),
                "is_low_info_gene": bool(low_info[gene_idx]),
            }
        )
        prev = cur
    return rows


def _aligned_vector(candidate: dict, key: str, size: int, fill: float | np.ndarray) -> np.ndarray:
    if isinstance(fill, np.ndarray):
        default = np.asarray(fill, dtype=np.float64)
    else:
        default = np.full(size, float(fill), dtype=np.float64)
    values = np.asarray(candidate.get(key, default), dtype=np.float64)
    if values.shape[0] != size:
        return default.copy()
    return values


def _set_empty_core(candidate: dict) -> None:
    candidate["core_gene_indices"] = np.array([], dtype=np.int64)
    candidate["core_genes"] = []
    candidate["core_gene_count"] = 0
    candidate["core_loading_fraction"] = 0.0
    candidate["core_mean_local_rss_strength"] = 0.0
    candidate["core_mean_local_rss_coverage"] = 0.0
    candidate["core_mean_local_rss_lift"] = 0.0
    candidate["eligible_gene_count"] = 0
    candidate["core_eligible_fraction"] = 0.0
    candidate["core_selection_method"] = "threshold"
    candidate["core_score_threshold"] = 0.0
    candidate["raw_elbow_size"] = 0
    candidate["effective_gene_floor"] = 0
    candidate["core_score_at_elbow"] = 0.0
    candidate["core_score_median_eligible"] = 0.0
    candidate["core_score_median_core"] = 0.0
    candidate["gene_local_rss_strength"] = np.array([], dtype=np.float32)
    candidate["gene_local_rss_coverage"] = np.array([], dtype=np.float32)
    candidate["gene_local_rss_lift"] = np.array([], dtype=np.float32)
    candidate["gene_loading_strength"] = np.array([], dtype=np.float32)
    candidate["gene_local_rss_lift_scaled"] = np.array([], dtype=np.float32)
    candidate["core_gene_eligible"] = np.array([], dtype=bool)
    candidate["core_gene_drop_reason"] = np.array([], dtype=object)
    candidate["core_gene_score"] = np.array([], dtype=np.float32)
    candidate["core_gene_rank"] = np.array([], dtype=np.int32)
    candidate["program_specificity"] = np.array([], dtype=np.float32)


def _core_drop_reasons(
    *,
    loading: np.ndarray,
    local_rss_strength: np.ndarray,
    local_rss_coverage: np.ndarray,
    local_rss_lift: np.ndarray,
    core_score: np.ndarray,
    confounder: np.ndarray,
    selected_mask: np.ndarray,
    eligible: np.ndarray,
    min_local_rss_strength: float,
    min_local_rss_coverage: float,
    min_local_rss_lift: float,
    min_core_score: float,
    selection_method: str,
) -> np.ndarray:
    reasons: list[str] = []
    for idx in range(loading.shape[0]):
        if bool(selected_mask[idx]):
            reasons.append("")
            continue
        gene_reasons = []
        if float(loading[idx]) <= 0.0:
            gene_reasons.append("zero_loading")
        if float(local_rss_strength[idx]) < min_local_rss_strength:
            gene_reasons.append("low_local_rss_strength")
        if float(local_rss_coverage[idx]) < min_local_rss_coverage:
            gene_reasons.append("low_local_rss_coverage")
        if float(local_rss_lift[idx]) < min_local_rss_lift:
            gene_reasons.append("low_local_rss_lift")
        if bool(confounder[idx]):
            gene_reasons.append("confounder_gene")
        if not gene_reasons and bool(eligible[idx]):
            if selection_method == "threshold" and float(core_score[idx]) < min_core_score:
                gene_reasons.append("below_core_score_threshold")
            elif selection_method == "threshold":
                gene_reasons.append("not_selected_by_core_threshold")
            else:
                gene_reasons.append("not_selected_by_elbow")
        reasons.append(";".join(gene_reasons))
    return np.asarray(reasons, dtype=object)


def _loading_columns() -> list[str]:
    return [
        "rsp_id",
        "gene",
        "loading",
        "loading_rank",
        "loading_fraction",
        "cumulative_fraction",
        "in_contribution_50",
        "in_contribution_70",
        "in_contribution_90",
        "gene_support_fraction",
        "loading_strength",
        "local_rss_strength",
        "local_rss_coverage",
        "local_rss_lift",
        "local_rss_lift_scaled",
        "program_specificity",
        "core_eligible",
        "is_core_gene",
        "core_rank",
        "core_score",
        "core_drop_reason",
        "is_blacklist_gene",
        "is_housekeeping_gene",
        "is_low_info_gene",
    ]
