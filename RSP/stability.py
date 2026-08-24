from __future__ import annotations

import numpy as np

from .common import (
    clamp01,
    connected_component_indices,
    gene_mask,
    jaccard,
    low_info_gene_mask,
    moran_matrix,
    safe_corr,
)
from .schema import RSPScoringConfig, RSPStabilityConfig


def select_stable_rsps(
    candidates: list[dict],
    stability_cfg: RSPStabilityConfig,
    scoring_cfg: RSPScoringConfig,
    neighbors: np.ndarray | None = None,
) -> list[dict]:
    for candidate in candidates:
        if str(candidate.get("candidate_status", "")) != "dropped":
            _prepare_candidate_for_stability(candidate, stability_cfg, neighbors)
            _score_candidate(candidate, scoring_cfg)
            _screen_candidate_before_matching(candidate, stability_cfg, scoring_cfg)
    for candidate in candidates:
        _attach_similarity_matches(candidate, candidates, stability_cfg)
    families = _assign_families(candidates, stability_cfg)
    retained = _select_primary_rsps(families, stability_cfg)

    rsp_id_by_candidate = {
        id(candidate): f"RSP{idx:04d}" for idx, candidate in enumerate(retained)
    }
    for candidate in candidates:
        if id(candidate) in rsp_id_by_candidate:
            candidate["rsp_id"] = rsp_id_by_candidate[id(candidate)]
            candidate["candidate_status"] = "retained"
            candidate["primary_rsp_id"] = candidate["rsp_id"]
        elif str(candidate.get("candidate_status", "")) != "dropped":
            primary = candidate.get("primary_candidate")
            candidate["rsp_id"] = ""
            candidate["candidate_status"] = "redundant"
            candidate["drop_reason"] = "redundant_with_primary"
            candidate["primary_rsp_id"] = rsp_id_by_candidate.get(id(primary), "") if primary is not None else ""
    return sorted(retained, key=lambda c: str(c.get("family_id", "")))


def _prepare_candidate_for_stability(
    candidate: dict,
    cfg: RSPStabilityConfig,
    neighbors: np.ndarray | None,
) -> None:
    raw = np.asarray(
        candidate.get("activation", candidate.get("activation_raw", [])),
        dtype=np.float32,
    )
    clean = np.clip(raw.reshape(-1), 0.0, None).copy()
    if clean.size == 0:
        candidate["activation"] = clean
        candidate["activation_clean"] = clean
        candidate["activation_active_mask"] = np.zeros(0, dtype=bool)
        candidate["activation_active_threshold"] = 0.0
        candidate["activation_active_spots"] = 0
        candidate["activation_active_fraction"] = 0.0
        candidate["structured_active_fraction"] = 0.0
        candidate["largest_component_fraction"] = 0.0
        candidate["fragmentation_score"] = 0.0
        candidate["activation_moran_i"] = 0.0
        candidate["pattern_moran"] = 0.0
        return

    peak = float(np.max(clean))
    if not np.isfinite(peak) or peak <= 0.0:
        clean = np.zeros_like(clean, dtype=np.float32)
        threshold = 0.0
    else:
        threshold = max(
            float(max(0.0, cfg.activation_cleanup_min_abs)),
            float(max(0.0, cfg.activation_cleanup_min_peak_fraction)) * peak,
        )
        clean[clean < threshold] = 0.0

    neighbor_arr = np.asarray(neighbors) if neighbors is not None else np.empty((0, 0), dtype=np.int64)
    if neighbor_arr.size and int(cfg.activation_cleanup_min_component_size) > 1:
        keep = np.zeros(clean.shape[0], dtype=bool)
        for component in connected_component_indices(clean > 0, neighbor_arr):
            if component.size >= int(cfg.activation_cleanup_min_component_size):
                keep[component] = True
        clean[~keep] = 0.0

    active_mask = clean > 0
    active_count = int(np.count_nonzero(active_mask))
    components = connected_component_indices(active_mask, neighbor_arr) if neighbor_arr.size else []
    if active_count > 0 and components:
        largest = max(int(component.size) for component in components)
        structured = sum(
            int(component.size)
            for component in components
            if component.size >= int(cfg.structured_active_component_min_size)
        )
        largest_fraction = float(largest / active_count)
        structured_fraction = float(structured / active_count)
        fragmentation = float(len(components) / active_count)
    elif active_count > 0:
        largest_fraction = 1.0
        structured_fraction = 1.0
        fragmentation = 1.0 / active_count
    else:
        largest_fraction = 0.0
        structured_fraction = 0.0
        fragmentation = 0.0
    moran = float(moran_matrix(clean[None, :], neighbor_arr)[0]) if neighbor_arr.size and clean.size else 0.0

    candidate["activation"] = clean.astype(np.float32, copy=False)
    candidate["activation_clean"] = clean.astype(np.float32, copy=False)
    candidate["activation_active_mask"] = active_mask.astype(bool, copy=False)
    candidate["activation_active_threshold"] = float(threshold)
    candidate["activation_active_spots"] = int(active_count)
    candidate["activation_active_fraction"] = float(active_count / max(1, clean.size))
    candidate["activation_coverage"] = float(active_count / max(1, clean.size))
    candidate["structured_active_fraction"] = float(structured_fraction)
    candidate["largest_component_fraction"] = float(largest_fraction)
    candidate["fragmentation_score"] = float(fragmentation)
    candidate["activation_moran_i"] = float(moran)
    candidate["pattern_moran"] = float(moran)


def _screen_candidate_before_matching(
    candidate: dict,
    stability_cfg: RSPStabilityConfig,
    scoring_cfg: RSPScoringConfig,
) -> None:
    drop_reasons = []
    if int(candidate.get("activation_active_spots", 0)) < int(stability_cfg.min_candidate_active_spots):
        drop_reasons.append("too_few_active_spots")
    if float(candidate.get("activation_active_fraction", 0.0)) > float(stability_cfg.max_candidate_active_fraction):
        drop_reasons.append("diffuse_activation")
    if float(candidate.get("activation_moran_i", 0.0)) < float(stability_cfg.min_candidate_activation_moran_i):
        drop_reasons.append("low_activation_spatial_autocorrelation")
    if float(candidate.get("structured_active_fraction", 0.0)) < float(stability_cfg.min_candidate_structured_active_fraction):
        drop_reasons.append("low_structured_active_fraction")
    if float(candidate.get("blacklist_loading_fraction", 0.0)) >= float(scoring_cfg.blacklist_loading_fraction_drop):
        drop_reasons.append("blacklist_loading_dominated")
    if float(candidate.get("blacklist_top10_fraction", 0.0)) >= float(scoring_cfg.blacklist_top10_fraction_drop):
        drop_reasons.append("blacklist_top10_dominated")
    if float(candidate.get("housekeeping_loading_fraction", 0.0)) >= float(scoring_cfg.housekeeping_loading_fraction_drop):
        drop_reasons.append("housekeeping_loading_dominated")
    if float(candidate.get("low_info_top10_fraction", 0.0)) >= float(scoring_cfg.low_info_top10_fraction_drop):
        drop_reasons.append("low_info_top10_dominated")

    if drop_reasons:
        candidate["candidate_status"] = "dropped"
        existing = str(candidate.get("drop_reason", ""))
        candidate["drop_reason"] = ";".join([x for x in [existing, *drop_reasons] if x])
    else:
        candidate["candidate_status"] = "candidate"
        candidate["drop_reason"] = ""


def _attach_similarity_matches(candidate: dict, candidates: list[dict], cfg: RSPStabilityConfig) -> None:
    if str(candidate.get("candidate_status", "")) == "dropped":
        _set_empty_recurrence(candidate)
        return

    matches = []
    for other in candidates:
        if other is candidate or str(other.get("candidate_status", "")) == "dropped":
            continue
        same_run = (
            int(other.get("source_k", -1)) == int(candidate.get("source_k", -2))
            and int(other.get("source_seed", -1)) == int(candidate.get("source_seed", -2))
        )
        if same_run:
            continue
        match = _pairwise_match(candidate, other, cfg)
        matches.append(match)

    valid = [m for m in matches if m["is_valid"]]
    unique_k = {int(m["source_k"]) for m in valid}
    unique_seed = {int(m["source_seed"]) for m in valid}
    candidate["recurrence_count"] = int(len(valid))
    candidate["recurrence_k_count"] = int(len(unique_k))
    candidate["recurrence_seed_count"] = int(len(unique_seed))
    candidate["best_match_score"] = float(max([m["match_score"] for m in valid], default=0.0))
    candidate["mean_match_score"] = float(np.mean([m["match_score"] for m in valid])) if valid else 0.0
    candidate["recurrence_score"] = 1.0 if valid else 0.0
    candidate["recurrence_matches"] = matches


def _pairwise_match(a: dict, b: dict, cfg: RSPStabilityConfig) -> dict:
    map_corr = safe_corr(np.asarray(a.get("activation", [])), np.asarray(b.get("activation", [])))
    active_jaccard = _active_mask_jaccard(a, b)
    gene_jaccard_70 = jaccard(set(a.get("contribution_tiers", {}).get("70", [])), set(b.get("contribution_tiers", {}).get("70", [])))
    loading_corr = safe_corr(np.asarray(a.get("gene_loadings", [])), np.asarray(b.get("gene_loadings", [])))
    match_score = float(
        cfg.match_score_map_weight * max(0.0, map_corr)
        + cfg.match_score_active_weight * active_jaccard
        + cfg.match_score_gene_weight * gene_jaccard_70
        + cfg.match_score_loading_weight * max(0.0, loading_corr)
    )
    is_valid = (
        map_corr >= float(cfg.min_match_map_corr)
        and active_jaccard >= float(cfg.min_match_active_jaccard)
        and gene_jaccard_70 >= float(cfg.min_match_gene_jaccard_70)
        and loading_corr >= float(cfg.min_match_loading_corr)
    )
    return {
        "candidate_id": str(b.get("candidate_id", "")),
        "source_k": int(b.get("source_k", -1)),
        "source_seed": int(b.get("source_seed", 0)),
        "map_corr": float(map_corr),
        "active_jaccard": float(active_jaccard),
        "gene_jaccard_70": float(gene_jaccard_70),
        "loading_corr": float(loading_corr),
        "match_score": float(match_score),
        "is_valid": bool(is_valid),
    }


def _score_candidate(candidate: dict, cfg: RSPScoringConfig) -> None:
    if str(candidate.get("candidate_status", "")) == "dropped":
        candidate.setdefault("rsp_score", 0.0)
        candidate.setdefault("spatial_score", 0.0)
        candidate.setdefault("gene_quality_score", 0.0)
        candidate.setdefault("activation_quality_score", 0.0)
        candidate.setdefault("confounder_penalty", 0.0)
        return

    activation = np.asarray(candidate["activation"], dtype=np.float32)
    raw_loading = np.asarray(candidate["gene_loadings"], dtype=np.float64)
    genes = np.asarray(candidate["genes"], dtype=str)
    loading = raw_loading / max(float(raw_loading.sum()), 1e-12)
    order = np.argsort(-loading)
    top10 = order[: min(10, order.size)]
    active_mask = np.asarray(candidate.get("activation_active_mask", activation > 0), dtype=bool)
    if active_mask.shape[0] != activation.shape[0]:
        active_mask = activation > 0
    active_count = int(np.count_nonzero(active_mask))
    coverage = float(np.mean(active_mask)) if active_mask.size else 0.0
    pos = activation[active_mask]
    if pos.size:
        probs = pos.astype(np.float64) / max(float(pos.sum()), 1e-12)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(max(2, probs.size)))
        peakiness = float((float(pos.max()) - float(np.median(pos))) / max(float(pos.max()), 1e-12))
        sparsity = float(1.0 - (active_count / max(1, activation.size)))
    else:
        entropy = 0.0
        peakiness = 0.0
        sparsity = 1.0

    blacklist_mask = gene_mask(genes, cfg.blacklist_prefixes, cfg.blacklist_genes)
    housekeeping_mask = gene_mask(genes, cfg.housekeeping_prefixes, cfg.housekeeping_genes)
    low_info_mask = low_info_gene_mask(genes, cfg.low_info_prefixes)
    blacklist_loading_fraction = float(loading[blacklist_mask].sum())
    housekeeping_loading_fraction = float(loading[housekeeping_mask].sum())
    low_info_loading_fraction = float(loading[low_info_mask].sum())
    blacklist_top10_fraction = float(np.mean(blacklist_mask[top10])) if top10.size else 0.0
    housekeeping_top10_fraction = float(np.mean(housekeeping_mask[top10])) if top10.size else 0.0
    low_info_top10_fraction = float(np.mean(low_info_mask[top10])) if top10.size else 0.0
    confounder_penalty = float(max(blacklist_loading_fraction, housekeeping_loading_fraction, low_info_loading_fraction))

    concentration_top10 = float(loading[top10].sum()) if top10.size else 0.0
    loading_entropy = float(candidate.get("gene_loading_entropy", 0.0))
    effective_gene_number = float(candidate.get("effective_gene_number", 0.0))
    collapse_penalty = clamp01(float(candidate.get("single_gene_dominance_score", 0.0)))
    rss_support_score = clamp01(
        0.45 * float(candidate.get("top10_local_rss_strength", 0.0))
        + 0.35 * float(candidate.get("weighted_local_rss_strength", 0.0))
        + 0.20 * min(1.0, max(0.0, float(candidate.get("weighted_local_rss_lift", 0.0)) - 1.0))
    )
    gene_quality_score = clamp01(
        0.22 * concentration_top10
        + 0.18 * (1.0 - loading_entropy)
        + 0.20 * min(1.0, effective_gene_number / max(1.0, np.sqrt(max(1, genes.size))))
        + 0.20 * (1.0 - confounder_penalty)
        + 0.20 * rss_support_score
        - 0.25 * collapse_penalty
    )
    coverage_score = _coverage_score(coverage, cfg.ideal_min_coverage, cfg.ideal_max_coverage)
    activation_quality_score = clamp01(0.35 * coverage_score + 0.25 * peakiness + 0.25 * (1.0 - entropy) + 0.15 * sparsity)
    spatial_score = clamp01(
        0.40 * max(0.0, float(candidate.get("activation_moran_i", candidate.get("pattern_moran", 0.0))))
        + 0.25 * float(candidate.get("structured_active_fraction", candidate.get("largest_component_fraction", 0.0)))
        + 0.20 * coverage_score
        + 0.15 * (1.0 - float(candidate.get("fragmentation_score", 1.0)))
    )
    rsp_score = clamp01(
        0.35 * spatial_score
        + 0.25 * gene_quality_score
        + 0.15 * float(candidate.get("reliability_weighted_explained", 0.0))
        + 0.10 * activation_quality_score
        + 0.10 * rss_support_score
        - 0.20 * confounder_penalty
        - 0.10 * float(candidate.get("redundancy_score", 0.0))
    )

    candidate.update(
        {
            "activation_coverage": coverage,
            "activation_active_fraction": coverage,
            "activation_active_spots": int(active_count),
            "activation_peakiness": peakiness,
            "activation_entropy": entropy,
            "activation_sparsity": sparsity,
            "blacklist_loading_fraction": blacklist_loading_fraction,
            "blacklist_top10_fraction": blacklist_top10_fraction,
            "housekeeping_loading_fraction": housekeeping_loading_fraction,
            "housekeeping_top10_fraction": housekeeping_top10_fraction,
            "low_info_loading_fraction": low_info_loading_fraction,
            "low_info_top10_fraction": low_info_top10_fraction,
            "confounder_penalty": confounder_penalty,
            "rss_support_score": rss_support_score,
            "spatial_score": spatial_score,
            "gene_quality_score": gene_quality_score,
            "activation_quality_score": activation_quality_score,
            "rsp_score": rsp_score,
        }
    )
    candidate.setdefault("drop_reason", "")


def _assign_families(candidates: list[dict], cfg: RSPStabilityConfig) -> list[list[dict]]:
    eligible = [c for c in candidates if str(c.get("candidate_status", "")) != "dropped"]
    families: list[list[dict]] = []
    family_idx = 0
    assigned: set[str] = set()
    ordered = sorted(
        eligible,
        key=lambda c: (
            -float(c.get("spatial_score", 0.0)),
            -float(c.get("gene_quality_score", 0.0)),
            str(c.get("candidate_id", "")),
        ),
    )
    for seed in ordered:
        seed_id = str(seed["candidate_id"])
        if seed_id in assigned:
            continue
        members = [seed]
        assigned.add(seed_id)
        changed = True
        while changed:
            changed = False
            for candidate in ordered:
                candidate_id = str(candidate["candidate_id"])
                if candidate_id in assigned:
                    continue
                if not _has_direct_valid_match(candidate, members):
                    continue
                if not _passes_family_centroid(candidate, members, cfg):
                    continue
                members.append(candidate)
                assigned.add(candidate_id)
                changed = True
        k_count = len({int(member.get("source_k", -1)) for member in members})
        seed_count = len({int(member.get("source_seed", 0)) for member in members})
        if (
            len(members) < int(cfg.min_family_size)
            or k_count < int(cfg.min_family_k_count)
            or seed_count < int(cfg.min_family_seed_count)
        ):
            for member in members:
                member["candidate_status"] = "dropped"
                reason = str(member.get("drop_reason", ""))
                member["drop_reason"] = ";".join([x for x in [reason, "no_retained_stability_family"] if x])
                member["family_id"] = ""
                member["primary_rsp_id"] = ""
            continue
        family_id = f"F{family_idx:04d}"
        family_idx += 1
        for member in members:
            member["family_id"] = family_id
        _attach_family_summary(members, cfg)
        families.append(members)
    for candidate in candidates:
        if str(candidate.get("candidate_status", "")) == "dropped":
            candidate["family_id"] = ""
            candidate["primary_rsp_id"] = ""
    return families


def _has_direct_valid_match(candidate: dict, members: list[dict]) -> bool:
    member_ids = {str(member["candidate_id"]) for member in members}
    for match in candidate.get("recurrence_matches", []):
        if bool(match.get("is_valid", False)) and str(match.get("candidate_id", "")) in member_ids:
            return True
    candidate_id = str(candidate["candidate_id"])
    for member in members:
        for match in member.get("recurrence_matches", []):
            if bool(match.get("is_valid", False)) and str(match.get("candidate_id", "")) == candidate_id:
                return True
    return False


def _passes_family_centroid(candidate: dict, members: list[dict], cfg: RSPStabilityConfig) -> bool:
    activation = np.asarray(candidate.get("activation", []), dtype=np.float32)
    loading = np.asarray(candidate.get("gene_loadings", []), dtype=np.float32)
    member_activation = np.asarray([np.asarray(member.get("activation", []), dtype=np.float32) for member in members], dtype=np.float32)
    member_loading = np.asarray([np.asarray(member.get("gene_loadings", []), dtype=np.float32) for member in members], dtype=np.float32)
    if member_activation.ndim != 2 or member_loading.ndim != 2:
        return False
    member_masks = np.asarray(
        [np.asarray(member.get("activation_active_mask", np.asarray(member.get("activation", [])) > 0), dtype=bool) for member in members],
        dtype=bool,
    )
    centroid_activation = member_activation.mean(axis=0)
    centroid_loading = member_loading.mean(axis=0)
    centroid_mask = member_masks.mean(axis=0) >= 0.5 if member_masks.ndim == 2 else centroid_activation > 0
    map_corr = safe_corr(activation, centroid_activation)
    loading_corr = safe_corr(loading, centroid_loading)
    active_jaccard = _mask_jaccard(
        np.asarray(candidate.get("activation_active_mask", activation > 0), dtype=bool),
        centroid_mask,
    )
    candidate["family_centroid_map_corr"] = float(map_corr)
    candidate["family_centroid_loading_corr"] = float(loading_corr)
    candidate["family_centroid_active_jaccard"] = float(active_jaccard)
    return (
        map_corr >= float(cfg.min_family_centroid_map_corr)
        and active_jaccard >= float(cfg.min_family_centroid_active_jaccard)
        and loading_corr >= float(cfg.min_family_centroid_loading_corr)
    )


def _select_primary_rsps(families: list[list[dict]], cfg: RSPStabilityConfig) -> list[dict]:
    retained = []
    for family in families:
        medoid_scores = {id(member): _family_medoid_score(member, family, cfg) for member in family}
        primary = sorted(
            family,
            key=lambda c: (
                -float(medoid_scores.get(id(c), 0.0)),
                -float(c.get("spatial_score", 0.0)),
                -float(c.get("gene_quality_score", 0.0)),
                -float(c.get("activation_moran_i", 0.0)),
                int(c.get("source_k", 10**9)),
                str(c.get("candidate_id", "")),
            ),
        )[0]
        retained.append(primary)
        for member in family:
            member["primary_candidate"] = primary
            member["family_medoid_score"] = float(medoid_scores.get(id(member), 0.0))
    return retained


def _family_medoid_score(candidate: dict, family: list[dict], cfg: RSPStabilityConfig) -> float:
    others = [member for member in family if member is not candidate]
    if not others:
        return 0.0
    scores = [_pairwise_match(candidate, other, cfg)["match_score"] for other in others]
    return float(np.mean(scores)) if scores else 0.0


def _attach_family_summary(family: list[dict], cfg: RSPStabilityConfig) -> None:
    pair_matches = []
    for idx, left in enumerate(family):
        for right in family[idx + 1 :]:
            pair_matches.append(_pairwise_match(left, right, cfg))
    k_count = len({int(member.get("source_k", -1)) for member in family})
    seed_count = len({int(member.get("source_seed", 0)) for member in family})
    family_size = len(family)
    summary = {
        "family_size": int(family_size),
        "family_k_count": int(k_count),
        "family_seed_count": int(seed_count),
        "family_mean_map_corr": float(np.mean([m["map_corr"] for m in pair_matches])) if pair_matches else 0.0,
        "family_mean_active_jaccard": float(np.mean([m["active_jaccard"] for m in pair_matches])) if pair_matches else 0.0,
        "family_mean_gene_jaccard_70": float(np.mean([m["gene_jaccard_70"] for m in pair_matches])) if pair_matches else 0.0,
        "family_mean_loading_corr": float(np.mean([m["loading_corr"] for m in pair_matches])) if pair_matches else 0.0,
        "family_support_score": 1.0,
    }
    for member in family:
        member.update(summary)
        member["recurrence_score"] = float(summary["family_support_score"])


def _coverage_score(value: float, low: float, high: float) -> float:
    v = float(value)
    if low <= v <= high:
        return 1.0
    if v < low:
        return clamp01(v / max(low, 1e-8))
    return clamp01((1.0 - v) / max(1.0 - high, 1e-8))


def _active_mask_jaccard(a: dict, b: dict) -> float:
    return _mask_jaccard(
        np.asarray(a.get("activation_active_mask", np.asarray(a.get("activation", [])) > 0), dtype=bool),
        np.asarray(b.get("activation_active_mask", np.asarray(b.get("activation", [])) > 0), dtype=bool),
    )


def _mask_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=bool).reshape(-1)
    y = np.asarray(b, dtype=bool).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    union = np.logical_or(x, y)
    if not bool(np.any(union)):
        return 0.0
    return float(np.logical_and(x, y).sum() / union.sum())


def _set_empty_recurrence(candidate: dict) -> None:
    candidate["recurrence_count"] = 0
    candidate["recurrence_k_count"] = 0
    candidate["recurrence_seed_count"] = 0
    candidate["best_match_score"] = 0.0
    candidate["mean_match_score"] = 0.0
    candidate["recurrence_score"] = 0.0
    candidate["recurrence_matches"] = []
