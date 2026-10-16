from __future__ import annotations

import numpy as np
import pandas as pd

from .common import activation_active_mask, connected_component_indices
from .schema import RSPScoringConfig


FINAL_RSS_SUPPORT_FOOTPRINT = "retained_rsp_activation"


def build_rsp_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        tiers = candidate.get("contribution_tiers", {"50": [], "70": [], "90": []})
        core_genes = [str(gene) for gene in candidate.get("core_genes", [])]
        top_genes = core_genes if core_genes else tiers.get("70", [])
        rows.append(
            {
                "rsp_id": str(candidate["rsp_id"]),
                "source_axis": "RSS",
                "source_k": int(candidate["source_k"]),
                "source_seed": int(candidate.get("source_seed", 0)),
                "source_component": int(candidate["source_component"]),
                "family_id": str(candidate.get("family_id", "")),
                "primary_candidate_id": str(candidate.get("candidate_id", "")),
                "rsp_score": float(candidate.get("rsp_score", 0.0)),
                "quality_score": float(candidate.get("rsp_score", 0.0)),
                "top_genes": ",".join(top_genes),
                "n_genes_50": int(len(tiers.get("50", []))),
                "n_genes_70": int(len(tiers.get("70", []))),
                "n_genes_90": int(len(tiers.get("90", []))),
                "core_gene_count": int(candidate.get("core_gene_count", len(core_genes))),
                "n_core_genes": int(candidate.get("core_gene_count", len(core_genes))),
                "effective_gene_number": float(candidate.get("effective_gene_number", 0.0)),
                "core_loading_fraction": float(candidate.get("core_loading_fraction", 0.0)),
                "core_mean_local_rss_strength": float(candidate.get("core_mean_local_rss_strength", 0.0)),
                "core_mean_local_rss_coverage": float(candidate.get("core_mean_local_rss_coverage", 0.0)),
                "core_mean_local_rss_lift": float(candidate.get("core_mean_local_rss_lift", 0.0)),
                "eligible_gene_count": int(candidate.get("eligible_gene_count", 0)),
                "core_eligible_fraction": float(candidate.get("core_eligible_fraction", 0.0)),
                "core_selection_method": str(candidate.get("core_selection_method", "")),
                "core_score_threshold": float(candidate.get("core_score_threshold", 0.0)),
                "raw_elbow_size": int(candidate.get("raw_elbow_size", 0)),
                "effective_gene_floor": int(candidate.get("effective_gene_floor", 0)),
                "core_score_at_elbow": float(candidate.get("core_score_at_elbow", 0.0)),
                "core_score_median_eligible": float(candidate.get("core_score_median_eligible", 0.0)),
                "core_score_median_core": float(candidate.get("core_score_median_core", 0.0)),
                "spatial_score": float(candidate.get("spatial_score", 0.0)),
                "recurrence_score": float(candidate.get("recurrence_score", 0.0)),
                "family_size": int(candidate.get("family_size", 0)),
                "family_k_count": int(candidate.get("family_k_count", 0)),
                "family_seed_count": int(candidate.get("family_seed_count", 0)),
                "family_mean_map_corr": float(candidate.get("family_mean_map_corr", 0.0)),
                "family_mean_active_jaccard": float(candidate.get("family_mean_active_jaccard", 0.0)),
                "family_mean_gene_jaccard_70": float(candidate.get("family_mean_gene_jaccard_70", 0.0)),
                "family_mean_loading_corr": float(candidate.get("family_mean_loading_corr", 0.0)),
                "family_medoid_score": float(candidate.get("family_medoid_score", 0.0)),
                "reconstruction_score": float(candidate.get("reliability_weighted_explained", 0.0)),
                "redundancy_score": float(candidate.get("redundancy_score", 0.0)),
                "gene_quality_score": float(candidate.get("gene_quality_score", 0.0)),
                "activation_quality_score": float(candidate.get("activation_quality_score", 0.0)),
                "rss_support_score": _final_rss_support_value(candidate, "rss_support_score"),
                "weighted_local_rss_strength": _final_rss_support_value(candidate, "weighted_local_rss_strength"),
                "weighted_local_rss_coverage": _final_rss_support_value(candidate, "weighted_local_rss_coverage"),
                "weighted_local_rss_lift": _final_rss_support_value(candidate, "weighted_local_rss_lift"),
                "top10_local_rss_strength": _final_rss_support_value(candidate, "top10_local_rss_strength"),
                "top10_local_rss_coverage": _final_rss_support_value(candidate, "top10_local_rss_coverage"),
                "top10_local_rss_lift": _final_rss_support_value(candidate, "top10_local_rss_lift"),
                "confounder_penalty": float(candidate.get("confounder_penalty", 0.0)),
                "activation_coverage": float(candidate.get("activation_coverage", 0.0)),
                "activation_raw_positive_fraction": float(candidate.get("activation_raw_positive_fraction", 0.0)),
                "activation_active_fraction": float(candidate.get("activation_active_fraction", candidate.get("activation_coverage", 0.0))),
                "activation_active_spots": int(candidate.get("activation_active_spots", 0)),
                "activation_active_threshold": float(candidate.get("activation_active_threshold", 0.0)),
                "activation_peakiness": float(candidate.get("activation_peakiness", 0.0)),
                "activation_entropy": float(candidate.get("activation_entropy", 0.0)),
                "activation_sparsity": float(candidate.get("activation_sparsity", 0.0)),
                "activation_moran_i": float(candidate.get("activation_moran_i", candidate.get("pattern_moran", 0.0))),
                "pattern_moran": float(candidate.get("pattern_moran", candidate.get("activation_moran_i", 0.0))),
                "largest_component_fraction": float(candidate.get("largest_component_fraction", 0.0)),
                "structured_active_fraction": float(candidate.get("structured_active_fraction", 0.0)),
                "fragmentation_score": float(candidate.get("fragmentation_score", 0.0)),
            }
        )
    return pd.DataFrame(rows, columns=_rsp_columns())


def build_rsp_activation_table(
    candidates: list[dict],
    spot_ids: np.ndarray,
    scoring_cfg: RSPScoringConfig,
    neighbors: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        raw = np.asarray(candidate["activation_raw"], dtype=np.float32)
        stability_ready = np.asarray(candidate.get("activation_clean", raw), dtype=np.float32)
        activation = _trim_activation_for_output(stability_ready, scoring_cfg, neighbors)
        candidate["activation"] = activation
        active_mask, active_threshold = activation_active_mask(
            activation,
            neighbors,
            min_peak_fraction=float(scoring_cfg.activation_active_min_peak_fraction),
            quantile=float(scoring_cfg.activation_active_quantile),
            min_abs=float(scoring_cfg.activation_active_min_abs),
            min_component_size=int(scoring_cfg.activation_active_min_component_size),
            fallback_to_base=bool(scoring_cfg.activation_active_fallback_to_base),
        )
        candidate["activation_active_mask"] = active_mask
        candidate["activation_active_threshold"] = float(active_threshold)
        candidate["activation_active_spots"] = int(np.count_nonzero(active_mask))
        candidate["activation_active_fraction"] = float(np.mean(active_mask)) if active_mask.size else 0.0
        rank_order = np.argsort(-activation)
        ranks = np.zeros(activation.shape[0], dtype=np.int32)
        ranks[rank_order] = np.arange(1, activation.shape[0] + 1, dtype=np.int32)
        active_ranks = np.zeros(activation.shape[0], dtype=np.int32)
        active_order = [int(idx) for idx in rank_order if bool(active_mask[idx])]
        for rank, idx in enumerate(active_order, start=1):
            active_ranks[idx] = int(rank)
        for idx, spot_id in enumerate(np.asarray(spot_ids, dtype=str)):
            rows.append(
                {
                    "rsp_id": str(candidate["rsp_id"]),
                    "spot_id": str(spot_id),
                    "activation_raw": float(raw[idx]) if idx < raw.shape[0] else 0.0,
                    "activation": float(activation[idx]) if idx < activation.shape[0] else 0.0,
                    "rank_in_spot": int(ranks[idx]) if idx < ranks.shape[0] else 0,
                    "activation_active": bool(active_mask[idx]) if idx < active_mask.shape[0] else False,
                    "activation_active_rank": int(active_ranks[idx]) if idx < active_ranks.shape[0] else 0,
                    "activation_active_threshold": float(active_threshold),
                }
            )
    return pd.DataFrame(rows, columns=_activation_columns())


def build_rsp_qc_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(_candidate_qc_row(candidate, candidates))
    return pd.DataFrame(rows, columns=_qc_columns())


def build_candidate_diagnostics_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        row = _candidate_qc_row(candidate, candidates)
        row.update(
            {
                "initializer_type": str(candidate.get("initializer_type", "")),
                "initialization_strategy": str(candidate.get("initialization_strategy", "")),
                "backend_solver": str(candidate.get("backend_solver", "")),
                "backend_init": str(candidate.get("backend_init", "")),
                "factorization_path_id": str(candidate.get("factorization_path_id", "")),
                "loading_sparsity_penalty": float(candidate.get("loading_sparsity_penalty", 0.0)),
                "activation_sparsity_penalty": float(candidate.get("activation_sparsity_penalty", 0.0)),
                "l1_ratio": float(candidate.get("l1_ratio", 0.0)),
                "component_loading_lambda": float(candidate.get("component_loading_lambda", 0.0)),
                "component_activation_lambda": float(candidate.get("component_activation_lambda", 0.0)),
                "adaptive_refinement_factor": float(candidate.get("adaptive_refinement_factor", 1.0)),
                "adaptive_refinement_signal": float(candidate.get("adaptive_refinement_signal", 0.0)),
                "adaptive_effective_gene_number": float(candidate.get("adaptive_effective_gene_number", 0.0)),
                "adaptive_weighted_support": float(candidate.get("adaptive_weighted_support", 0.0)),
                "adaptive_loading_lambda": float(candidate.get("adaptive_loading_lambda", 0.0)),
                "adaptive_support_power": float(candidate.get("adaptive_support_power", 0.0)),
                "source_seed": int(candidate.get("source_seed", 0)),
                "source_seed_offset": int(candidate.get("source_seed_offset", 0)),
                "weighted_reconstruction_error": float(candidate.get("weighted_reconstruction_error", 0.0)),
                "rss_evidence_explained": float(candidate.get("rss_evidence_explained", 0.0)),
                "reliability_weighted_explained": float(candidate.get("reliability_weighted_explained", 0.0)),
                "effective_gene_number": float(candidate.get("effective_gene_number", 0.0)),
                "gene_loading_entropy": float(candidate.get("gene_loading_entropy", 0.0)),
                "fragmentation_score": float(candidate.get("fragmentation_score", 0.0)),
                "single_gene_dominance_score": float(candidate.get("single_gene_dominance_score", 0.0)),
                "redundancy_score": float(candidate.get("redundancy_score", 0.0)),
                "quality_flag": str(candidate.get("quality_flag", "")),
                "quality_flags": str(candidate.get("quality_flags", "")),
                "preliminary_core_genes": ",".join(str(g) for g in candidate.get("preliminary_core_genes", [])),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=_candidate_diagnostic_columns())


def build_rsp_family_members_table(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        if _is_dropped_or_unassigned(candidate):
            continue
        tiers = candidate.get("contribution_tiers", {"50": [], "70": [], "90": []})
        rows.append(
            {
                "family_id": str(candidate.get("family_id", "")),
                "primary_rsp_id": str(candidate.get("primary_rsp_id", "")),
                "rsp_id": str(candidate.get("rsp_id", "")),
                "candidate_id": str(candidate.get("candidate_id", "")),
                "member_status": str(candidate.get("candidate_status", "")),
                "is_primary": bool(str(candidate.get("candidate_status", "")) == "retained"),
                "source_k": int(candidate.get("source_k", 0)),
                "source_seed": int(candidate.get("source_seed", 0)),
                "source_seed_offset": int(candidate.get("source_seed_offset", 0)),
                "source_component": int(candidate.get("source_component", 0)),
                "backend_solver": str(candidate.get("backend_solver", "")),
                "backend_init": str(candidate.get("backend_init", "")),
                "rsp_score": float(candidate.get("rsp_score", 0.0)),
                "spatial_score": float(candidate.get("spatial_score", 0.0)),
                "gene_quality_score": float(candidate.get("gene_quality_score", 0.0)),
                "activation_quality_score": float(candidate.get("activation_quality_score", 0.0)),
                "rss_support_score": _final_rss_support_value(candidate, "rss_support_score"),
                "reconstruction_score": float(candidate.get("reliability_weighted_explained", 0.0)),
                "effective_gene_number": float(candidate.get("effective_gene_number", 0.0)),
                "n_genes_50": int(len(tiers.get("50", []))),
                "n_genes_70": int(len(tiers.get("70", []))),
                "n_genes_90": int(len(tiers.get("90", []))),
                "activation_moran_i": float(candidate.get("activation_moran_i", candidate.get("pattern_moran", 0.0))),
                "activation_active_fraction": float(candidate.get("activation_active_fraction", 0.0)),
                "activation_active_spots": int(candidate.get("activation_active_spots", 0)),
                "structured_active_fraction": float(candidate.get("structured_active_fraction", 0.0)),
                "largest_component_fraction": float(candidate.get("largest_component_fraction", 0.0)),
                "fragmentation_score": float(candidate.get("fragmentation_score", 0.0)),
                "family_size": int(candidate.get("family_size", 0)),
                "family_k_count": int(candidate.get("family_k_count", 0)),
                "family_seed_count": int(candidate.get("family_seed_count", 0)),
                "family_mean_map_corr": float(candidate.get("family_mean_map_corr", 0.0)),
                "family_mean_active_jaccard": float(candidate.get("family_mean_active_jaccard", 0.0)),
                "family_mean_gene_jaccard_70": float(candidate.get("family_mean_gene_jaccard_70", 0.0)),
                "family_mean_loading_corr": float(candidate.get("family_mean_loading_corr", 0.0)),
                "family_medoid_score": float(candidate.get("family_medoid_score", 0.0)),
                "adaptive_refinement_factor": float(candidate.get("adaptive_refinement_factor", 1.0)),
                "adaptive_weighted_support": float(candidate.get("adaptive_weighted_support", 0.0)),
                "adaptive_loading_lambda": float(candidate.get("adaptive_loading_lambda", 0.0)),
                "adaptive_support_power": float(candidate.get("adaptive_support_power", 0.0)),
                "contribution_50_genes": ",".join(str(gene) for gene in tiers.get("50", [])),
                "contribution_70_genes": ",".join(str(gene) for gene in tiers.get("70", [])),
            }
        )
    return pd.DataFrame(rows, columns=_family_member_columns())


def build_rsp_family_member_activation_table(
    candidates: list[dict],
    spot_ids: np.ndarray,
    scoring_cfg: RSPScoringConfig,
    neighbors: np.ndarray,
) -> pd.DataFrame:
    rows = []
    spot_ids = np.asarray(spot_ids, dtype=str)
    for candidate in candidates:
        if _is_dropped_or_unassigned(candidate):
            continue
        raw = np.asarray(candidate.get("activation_raw", []), dtype=np.float32)
        clean = np.asarray(candidate.get("activation_clean", candidate.get("activation", raw)), dtype=np.float32)
        activation = _trim_activation_for_output(clean, scoring_cfg, neighbors)
        active_mask, active_threshold = activation_active_mask(
            activation,
            neighbors,
            min_peak_fraction=float(scoring_cfg.activation_active_min_peak_fraction),
            quantile=float(scoring_cfg.activation_active_quantile),
            min_abs=float(scoring_cfg.activation_active_min_abs),
            min_component_size=int(scoring_cfg.activation_active_min_component_size),
            fallback_to_base=bool(scoring_cfg.activation_active_fallback_to_base),
        )
        rank_order = np.argsort(-activation)
        ranks = np.zeros(activation.shape[0], dtype=np.int32)
        ranks[rank_order] = np.arange(1, activation.shape[0] + 1, dtype=np.int32)
        for idx, spot_id in enumerate(spot_ids):
            rows.append(
                {
                    "family_id": str(candidate.get("family_id", "")),
                    "primary_rsp_id": str(candidate.get("primary_rsp_id", "")),
                    "rsp_id": str(candidate.get("rsp_id", "")),
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "member_status": str(candidate.get("candidate_status", "")),
                    "source_k": int(candidate.get("source_k", 0)),
                    "source_seed": int(candidate.get("source_seed", 0)),
                    "source_seed_offset": int(candidate.get("source_seed_offset", 0)),
                    "source_component": int(candidate.get("source_component", 0)),
                    "spot_id": str(spot_id),
                    "activation_raw": float(raw[idx]) if idx < raw.shape[0] else 0.0,
                    "activation_clean": float(clean[idx]) if idx < clean.shape[0] else 0.0,
                    "activation": float(activation[idx]) if idx < activation.shape[0] else 0.0,
                    "activation_active": bool(active_mask[idx]) if idx < active_mask.shape[0] else False,
                    "rank_in_candidate": int(ranks[idx]) if idx < ranks.shape[0] else 0,
                    "activation_active_threshold": float(active_threshold),
                }
            )
    return pd.DataFrame(rows, columns=_family_member_activation_columns())


def _candidate_qc_row(candidate: dict, candidates: list[dict]) -> dict:
    family_id = str(candidate.get("family_id", ""))
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "rsp_id": str(candidate.get("rsp_id", "")),
        "family_id": family_id,
        "candidate_status": str(candidate.get("candidate_status", "")),
        "drop_reason": str(candidate.get("drop_reason", "")),
        "primary_rsp_id": str(candidate.get("primary_rsp_id", "")),
        "source_k": int(candidate["source_k"]),
        "source_component": int(candidate["source_component"]),
        "rsp_score": float(candidate.get("rsp_score", 0.0)),
        "spatial_score": float(candidate.get("spatial_score", 0.0)),
        "recurrence_score": float(candidate.get("recurrence_score", 0.0)),
        "family_size": int(candidate.get("family_size", 0)),
        "family_k_count": int(candidate.get("family_k_count", 0)),
        "family_seed_count": int(candidate.get("family_seed_count", 0)),
        "family_mean_map_corr": float(candidate.get("family_mean_map_corr", 0.0)),
        "family_mean_active_jaccard": float(candidate.get("family_mean_active_jaccard", 0.0)),
        "family_mean_gene_jaccard_70": float(candidate.get("family_mean_gene_jaccard_70", 0.0)),
        "family_mean_loading_corr": float(candidate.get("family_mean_loading_corr", 0.0)),
        "family_medoid_score": float(candidate.get("family_medoid_score", 0.0)),
        "gene_quality_score": float(candidate.get("gene_quality_score", 0.0)),
        "activation_quality_score": float(candidate.get("activation_quality_score", 0.0)),
        "rss_support_score": _final_rss_support_value(candidate, "rss_support_score"),
        "eligible_gene_count": int(candidate.get("eligible_gene_count", 0)),
        "core_eligible_fraction": float(candidate.get("core_eligible_fraction", 0.0)),
        "core_selection_method": str(candidate.get("core_selection_method", "")),
        "core_score_threshold": float(candidate.get("core_score_threshold", 0.0)),
        "raw_elbow_size": int(candidate.get("raw_elbow_size", 0)),
        "effective_gene_floor": int(candidate.get("effective_gene_floor", 0)),
        "core_score_at_elbow": float(candidate.get("core_score_at_elbow", 0.0)),
        "core_score_median_eligible": float(candidate.get("core_score_median_eligible", 0.0)),
        "core_score_median_core": float(candidate.get("core_score_median_core", 0.0)),
        "weighted_local_rss_strength": _final_rss_support_value(candidate, "weighted_local_rss_strength"),
        "weighted_local_rss_coverage": _final_rss_support_value(candidate, "weighted_local_rss_coverage"),
        "weighted_local_rss_lift": _final_rss_support_value(candidate, "weighted_local_rss_lift"),
        "top10_local_rss_strength": _final_rss_support_value(candidate, "top10_local_rss_strength"),
        "top10_local_rss_coverage": _final_rss_support_value(candidate, "top10_local_rss_coverage"),
        "top10_local_rss_lift": _final_rss_support_value(candidate, "top10_local_rss_lift"),
        "confounder_penalty": float(candidate.get("confounder_penalty", 0.0)),
        "recurrence_count": int(candidate.get("recurrence_count", 0)),
        "recurrence_k_count": int(candidate.get("recurrence_k_count", 0)),
        "recurrence_seed_count": int(candidate.get("recurrence_seed_count", 0)),
        "best_match_score": float(candidate.get("best_match_score", 0.0)),
        "mean_match_score": float(candidate.get("mean_match_score", 0.0)),
        "activation_moran_i": float(candidate.get("activation_moran_i", candidate.get("pattern_moran", 0.0))),
        "pattern_moran": float(candidate.get("pattern_moran", candidate.get("activation_moran_i", 0.0))),
        "largest_component_fraction": float(candidate.get("largest_component_fraction", 0.0)),
        "structured_active_fraction": float(candidate.get("structured_active_fraction", 0.0)),
        "activation_coverage": float(candidate.get("activation_coverage", 0.0)),
        "activation_raw_positive_fraction": float(candidate.get("activation_raw_positive_fraction", 0.0)),
        "activation_active_fraction": float(candidate.get("activation_active_fraction", candidate.get("activation_coverage", 0.0))),
        "activation_active_spots": int(candidate.get("activation_active_spots", 0)),
        "activation_active_threshold": float(candidate.get("activation_active_threshold", 0.0)),
        "activation_peakiness": float(candidate.get("activation_peakiness", 0.0)),
        "activation_entropy": float(candidate.get("activation_entropy", 0.0)),
        "activation_sparsity": float(candidate.get("activation_sparsity", 0.0)),
        "blacklist_loading_fraction": float(candidate.get("blacklist_loading_fraction", 0.0)),
        "blacklist_top10_fraction": float(candidate.get("blacklist_top10_fraction", 0.0)),
        "housekeeping_loading_fraction": float(candidate.get("housekeeping_loading_fraction", 0.0)),
        "housekeeping_top10_fraction": float(candidate.get("housekeeping_top10_fraction", 0.0)),
        "low_info_loading_fraction": float(candidate.get("low_info_loading_fraction", 0.0)),
        "low_info_top10_fraction": float(candidate.get("low_info_top10_fraction", 0.0)),
        "redundancy_family_size": int(sum(1 for x in candidates if str(x.get("family_id", "")) == family_id)) if family_id else 0,
    }


def _is_dropped_or_unassigned(candidate: dict) -> bool:
    if str(candidate.get("candidate_status", "")) == "dropped":
        return True
    return not bool(str(candidate.get("family_id", "")).strip())


def _has_final_rss_support(candidate: dict) -> bool:
    return (
        str(candidate.get("candidate_status", "")) == "retained"
        and str(candidate.get("local_rss_support_footprint", "")) == FINAL_RSS_SUPPORT_FOOTPRINT
    )


def _final_rss_support_value(candidate: dict, key: str) -> float:
    if not _has_final_rss_support(candidate):
        return float("nan")
    return float(candidate.get(key, np.nan))


def _trim_activation_for_output(
    values: np.ndarray,
    cfg: RSPScoringConfig,
    neighbors: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    if not bool(getattr(cfg, "activation_trim_enabled", True)):
        return arr
    if arr.size == 0:
        return arr
    peak = float(np.max(arr))
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    frac = float(max(0.0, getattr(cfg, "activation_trim_fraction_of_peak", 0.10)))
    min_abs = float(max(0.0, getattr(cfg, "activation_trim_min_abs", 1e-6)))
    threshold = max(min_abs, frac * peak)
    arr[arr < threshold] = 0.0
    min_component_size = int(max(1, getattr(cfg, "activation_trim_min_component_size", 20)))
    if neighbors is not None and min_component_size > 1:
        keep = np.zeros(arr.shape[0], dtype=bool)
        for component in connected_component_indices(arr > 0, neighbors):
            if component.size >= min_component_size:
                keep[component] = True
        arr[~keep] = 0.0
    return arr.astype(np.float32, copy=False)


def _rsp_columns() -> list[str]:
    return [
        "rsp_id",
        "source_axis",
        "source_k",
        "source_seed",
        "source_component",
        "family_id",
        "primary_candidate_id",
        "rsp_score",
        "quality_score",
        "top_genes",
        "n_genes_50",
        "n_genes_70",
        "n_genes_90",
        "core_gene_count",
        "n_core_genes",
        "effective_gene_number",
        "core_loading_fraction",
        "core_mean_local_rss_strength",
        "core_mean_local_rss_coverage",
        "core_mean_local_rss_lift",
        "eligible_gene_count",
        "core_eligible_fraction",
        "core_selection_method",
        "core_score_threshold",
        "raw_elbow_size",
        "effective_gene_floor",
        "core_score_at_elbow",
        "core_score_median_eligible",
        "core_score_median_core",
        "spatial_score",
        "recurrence_score",
        "family_size",
        "family_k_count",
        "family_seed_count",
        "family_mean_map_corr",
        "family_mean_active_jaccard",
        "family_mean_gene_jaccard_70",
        "family_mean_loading_corr",
        "family_medoid_score",
        "reconstruction_score",
        "redundancy_score",
        "gene_quality_score",
        "activation_quality_score",
        "rss_support_score",
        "weighted_local_rss_strength",
        "weighted_local_rss_coverage",
        "weighted_local_rss_lift",
        "top10_local_rss_strength",
        "top10_local_rss_coverage",
        "top10_local_rss_lift",
        "confounder_penalty",
        "activation_coverage",
        "activation_raw_positive_fraction",
        "activation_active_fraction",
        "activation_active_spots",
        "activation_active_threshold",
        "activation_peakiness",
        "activation_entropy",
        "activation_sparsity",
        "activation_moran_i",
        "pattern_moran",
        "largest_component_fraction",
        "structured_active_fraction",
        "fragmentation_score",
    ]


def _activation_columns() -> list[str]:
    return [
        "rsp_id",
        "spot_id",
        "activation_raw",
        "activation",
        "rank_in_spot",
        "activation_active",
        "activation_active_rank",
        "activation_active_threshold",
    ]


def _qc_columns() -> list[str]:
    return [
        "candidate_id",
        "rsp_id",
        "family_id",
        "candidate_status",
        "drop_reason",
        "primary_rsp_id",
        "source_k",
        "source_component",
        "rsp_score",
        "spatial_score",
        "recurrence_score",
        "family_size",
        "family_k_count",
        "family_seed_count",
        "family_mean_map_corr",
        "family_mean_active_jaccard",
        "family_mean_gene_jaccard_70",
        "family_mean_loading_corr",
        "family_medoid_score",
        "gene_quality_score",
        "activation_quality_score",
        "rss_support_score",
        "eligible_gene_count",
        "core_eligible_fraction",
        "core_selection_method",
        "core_score_threshold",
        "raw_elbow_size",
        "effective_gene_floor",
        "core_score_at_elbow",
        "core_score_median_eligible",
        "core_score_median_core",
        "weighted_local_rss_strength",
        "weighted_local_rss_coverage",
        "weighted_local_rss_lift",
        "top10_local_rss_strength",
        "top10_local_rss_coverage",
        "top10_local_rss_lift",
        "confounder_penalty",
        "recurrence_count",
        "recurrence_k_count",
        "recurrence_seed_count",
        "best_match_score",
        "mean_match_score",
        "activation_moran_i",
        "pattern_moran",
        "largest_component_fraction",
        "structured_active_fraction",
        "activation_coverage",
        "activation_raw_positive_fraction",
        "activation_active_fraction",
        "activation_active_spots",
        "activation_active_threshold",
        "activation_peakiness",
        "activation_entropy",
        "activation_sparsity",
        "blacklist_loading_fraction",
        "blacklist_top10_fraction",
        "housekeeping_loading_fraction",
        "housekeeping_top10_fraction",
        "low_info_loading_fraction",
        "low_info_top10_fraction",
        "redundancy_family_size",
    ]


def _candidate_diagnostic_columns() -> list[str]:
    return _qc_columns() + [
        "initializer_type",
        "initialization_strategy",
        "backend_solver",
        "backend_init",
        "factorization_path_id",
        "loading_sparsity_penalty",
        "activation_sparsity_penalty",
        "l1_ratio",
        "component_loading_lambda",
        "component_activation_lambda",
        "adaptive_refinement_factor",
        "adaptive_refinement_signal",
        "adaptive_effective_gene_number",
        "adaptive_weighted_support",
        "adaptive_loading_lambda",
        "adaptive_support_power",
        "source_seed",
        "source_seed_offset",
        "weighted_reconstruction_error",
        "rss_evidence_explained",
        "reliability_weighted_explained",
        "effective_gene_number",
        "gene_loading_entropy",
        "fragmentation_score",
        "single_gene_dominance_score",
        "redundancy_score",
        "quality_flag",
        "quality_flags",
        "preliminary_core_genes",
    ]


def _family_member_columns() -> list[str]:
    return [
        "family_id",
        "primary_rsp_id",
        "rsp_id",
        "candidate_id",
        "member_status",
        "is_primary",
        "source_k",
        "source_seed",
        "source_seed_offset",
        "source_component",
        "backend_solver",
        "backend_init",
        "rsp_score",
        "spatial_score",
        "gene_quality_score",
        "activation_quality_score",
        "rss_support_score",
        "reconstruction_score",
        "effective_gene_number",
        "n_genes_50",
        "n_genes_70",
        "n_genes_90",
        "activation_moran_i",
        "activation_active_fraction",
        "activation_active_spots",
        "structured_active_fraction",
        "largest_component_fraction",
        "fragmentation_score",
        "family_size",
        "family_k_count",
        "family_seed_count",
        "family_mean_map_corr",
        "family_mean_active_jaccard",
        "family_mean_gene_jaccard_70",
        "family_mean_loading_corr",
        "family_medoid_score",
        "adaptive_refinement_factor",
        "adaptive_weighted_support",
        "adaptive_loading_lambda",
        "adaptive_support_power",
        "contribution_50_genes",
        "contribution_70_genes",
    ]


def _family_member_activation_columns() -> list[str]:
    return [
        "family_id",
        "primary_rsp_id",
        "rsp_id",
        "candidate_id",
        "member_status",
        "source_k",
        "source_seed",
        "source_seed_offset",
        "source_component",
        "spot_id",
        "activation_raw",
        "activation_clean",
        "activation",
        "activation_active",
        "rank_in_candidate",
        "activation_active_threshold",
    ]
