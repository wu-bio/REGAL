from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from .common import (
    activation_active_mask,
    clamp01,
    connected_component_indices,
    largest_component_fraction,
    moran_matrix,
    positive_quantile,
    safe_corr,
)
from .evidence import RSPEvidence
from .schema import RSPDiscoveryConfig, RSPScoringConfig


DISCOVERY_BACKEND = "rss_anchored_cd_l1_sparse_factorization"


def generate_decomposition_candidates(
    *,
    evidence: RSPEvidence,
    selected_idx: np.ndarray,
    neighbors: np.ndarray,
    discovery_cfg: RSPDiscoveryConfig,
    scoring_cfg: RSPScoringConfig,
    random_seed: int,
) -> tuple[list[dict], list[dict]]:
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    if selected_idx.size == 0:
        return [], [{"status": "skipped", "reason": "no genes available for decomposition"}]

    x = np.asarray(evidence.x_model[selected_idx], dtype=np.float32)
    x_raw = np.asarray(evidence.x_raw[selected_idx], dtype=np.float32)
    reliability = np.asarray(evidence.reliability[selected_idx], dtype=np.float32)
    genes = np.asarray(evidence.genes[selected_idx], dtype=str)
    gene_support_fraction = (x_raw > 0).mean(axis=1).astype(np.float32, copy=False)

    if discovery_cfg.decomposition_backend == DISCOVERY_BACKEND:
        return _generate_rss_anchored_cd_l1_candidates(
            x=x,
            x_raw=x_raw,
            reliability=reliability,
            genes=genes,
            gene_support_fraction=gene_support_fraction,
            neighbors=neighbors,
            discovery_cfg=discovery_cfg,
            scoring_cfg=scoring_cfg,
            random_seed=random_seed,
        )
    if discovery_cfg.decomposition_backend == "legacy_rss_spatial_nmf":
        return _generate_legacy_rss_spatial_nmf_candidates(
            x_raw=x_raw,
            reliability=reliability,
            genes=genes,
            gene_support_fraction=gene_support_fraction,
            neighbors=neighbors,
            discovery_cfg=discovery_cfg,
            scoring_cfg=scoring_cfg,
            random_seed=random_seed,
        )
    raise ValueError(f"Unsupported RSP decomposition backend: {discovery_cfg.decomposition_backend}")


def refresh_candidate_rss_support_on_retained_footprint(
    candidates: list[dict],
    *,
    x_raw: np.ndarray,
) -> None:
    matrix = np.clip(np.asarray(x_raw, dtype=np.float32), 0.0, None)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return
    for candidate in candidates:
        genes = np.asarray(candidate.get("genes", []), dtype=str)
        loading = np.clip(np.asarray(candidate.get("gene_loadings", []), dtype=np.float32), 0.0, None)
        if "activation" not in candidate:
            continue
        activation = np.clip(
            np.asarray(candidate["activation"], dtype=np.float32).reshape(-1),
            0.0,
            None,
        )
        if genes.size == 0 or loading.size != genes.size or matrix.shape[0] != genes.size:
            continue
        if activation.size != matrix.shape[1]:
            continue
        active_mask = np.asarray(candidate.get("activation_active_mask", activation > 0), dtype=bool).reshape(-1)
        if active_mask.size != activation.size:
            active_mask = activation > 0
        support = _candidate_rss_support_summary(
            x_raw=matrix,
            loading=loading,
            activation=activation,
            active_mask=active_mask,
        )
        candidate.update(support)
        candidate["local_rss_support_footprint"] = "retained_rsp_activation"


def _candidate_rss_support_summary(
    *,
    x_raw: np.ndarray,
    loading: np.ndarray,
    activation: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> dict:
    matrix = np.clip(np.asarray(x_raw, dtype=np.float32), 0.0, None)
    loading = np.clip(np.asarray(loading, dtype=np.float32), 0.0, None)
    activation = np.clip(np.asarray(activation, dtype=np.float32).reshape(-1), 0.0, None)
    if matrix.ndim != 2 or loading.size != matrix.shape[0] or activation.size != matrix.shape[1]:
        n_genes = int(matrix.shape[0]) if matrix.ndim == 2 else int(loading.size)
        return _empty_candidate_rss_support_summary(n_genes)

    if active_mask is None:
        mask = activation > 0
    else:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if mask.size != activation.size:
            mask = activation > 0
    support_activation = np.where(mask, activation, 0.0).astype(np.float32, copy=False)
    local_rss_strength = _activation_weighted_local_strength(matrix, support_activation)
    local_rss_coverage = _activation_local_rss_coverage(matrix, support_activation)
    local_rss_lift = _activation_local_rss_lift(matrix, support_activation)
    loading_fraction = loading / max(float(loading.sum()), 1e-12)
    top_loading = np.argsort(-loading_fraction)[: min(10, loading_fraction.size)]
    weighted_local_rss_strength = float(loading_fraction @ local_rss_strength) if loading_fraction.size else 0.0
    weighted_local_rss_coverage = float(loading_fraction @ local_rss_coverage) if loading_fraction.size else 0.0
    weighted_local_rss_lift = float(loading_fraction @ local_rss_lift) if loading_fraction.size else 0.0
    top10_local_rss_strength = float(np.mean(local_rss_strength[top_loading])) if top_loading.size else 0.0
    top10_local_rss_coverage = float(np.mean(local_rss_coverage[top_loading])) if top_loading.size else 0.0
    top10_local_rss_lift = float(np.mean(local_rss_lift[top_loading])) if top_loading.size else 0.0
    return {
        "gene_local_rss_strength": local_rss_strength,
        "gene_local_rss_coverage": local_rss_coverage,
        "gene_local_rss_lift": local_rss_lift,
        "weighted_local_rss_strength": weighted_local_rss_strength,
        "weighted_local_rss_coverage": weighted_local_rss_coverage,
        "weighted_local_rss_lift": weighted_local_rss_lift,
        "top10_local_rss_strength": top10_local_rss_strength,
        "top10_local_rss_coverage": top10_local_rss_coverage,
        "top10_local_rss_lift": top10_local_rss_lift,
        "rss_support_score": clamp01(
            0.45 * top10_local_rss_strength
            + 0.35 * weighted_local_rss_strength
            + 0.20 * min(1.0, max(0.0, weighted_local_rss_lift - 1.0))
        ),
    }


def _empty_candidate_rss_support_summary(n_genes: int) -> dict:
    zeros = np.zeros(max(0, int(n_genes)), dtype=np.float32)
    return {
        "gene_local_rss_strength": zeros,
        "gene_local_rss_coverage": zeros.copy(),
        "gene_local_rss_lift": zeros.copy(),
        "weighted_local_rss_strength": 0.0,
        "weighted_local_rss_coverage": 0.0,
        "weighted_local_rss_lift": 0.0,
        "top10_local_rss_strength": 0.0,
        "top10_local_rss_coverage": 0.0,
        "top10_local_rss_lift": 0.0,
        "rss_support_score": 0.0,
    }


def _generate_rss_anchored_cd_l1_candidates(
    *,
    x: np.ndarray,
    x_raw: np.ndarray,
    reliability: np.ndarray,
    genes: np.ndarray,
    gene_support_fraction: np.ndarray,
    neighbors: np.ndarray,
    discovery_cfg: RSPDiscoveryConfig,
    scoring_cfg: RSPScoringConfig,
    random_seed: int,
) -> tuple[list[dict], list[dict]]:
    anchor_matrix = _legacy_factorization_matrix(x_raw, discovery_cfg)
    candidates: list[dict] = []
    runs: list[dict] = []
    if anchor_matrix.shape[0] == 0:
        return [], [{"status": "skipped", "reason": "no genes available for decomposition"}]

    loading_penalty = _first_nonnegative(discovery_cfg.loading_sparsity_grid, default=0.0003)
    activation_penalty = _first_nonnegative(discovery_cfg.activation_sparsity_grid, default=0.0001)
    l1_ratio = _first_nonnegative(discovery_cfg.l1_ratio_grid, default=0.5, upper=1.0)

    for k in tuple(int(k) for k in discovery_cfg.k_grid):
        for seed_offset in _seed_offsets(discovery_cfg):
            actual_seed = int(random_seed) + int(seed_offset)
            nmf_init = _seeded_nmf_init(str(discovery_cfg.axis_anchor_init), int(seed_offset))
            run = {
                "requested_k": int(k),
                "seed_offset": int(seed_offset),
                "random_seed": int(actual_seed),
                "n_genes_input": int(anchor_matrix.shape[0]),
                "status": "pending",
                "backend": DISCOVERY_BACKEND,
                "initialization_strategy": "cd_axis_anchor",
                "factorization_path_id": "axis_anchor_fixed_h_l1",
                "loading_sparsity_penalty": float(loading_penalty),
                "activation_sparsity_penalty": float(activation_penalty),
                "l1_ratio": float(l1_ratio),
                "backend_init": str(nmf_init),
            }
            if anchor_matrix.shape[0] < int(k) or anchor_matrix.shape[1] < int(k):
                run.update(
                    {
                        "status": "skipped",
                        "reason": f"matrix_shape={anchor_matrix.shape} < requested_k={k}",
                    }
                )
                runs.append(run)
                continue

            model = NMF(
                n_components=int(k),
                init=str(nmf_init),
                solver=str(discovery_cfg.axis_anchor_solver),
                beta_loss="frobenius",
                max_iter=int(discovery_cfg.axis_anchor_max_iter),
                tol=float(discovery_cfg.axis_anchor_tol),
                alpha_W=float(max(0.0, discovery_cfg.axis_anchor_alpha_w)),
                alpha_H=float(max(0.0, discovery_cfg.axis_anchor_alpha_h)),
                l1_ratio=float(np.clip(discovery_cfg.axis_anchor_l1_ratio, 0.0, 1.0)),
                random_state=int(actual_seed),
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    w_anchor = np.asarray(model.fit_transform(anchor_matrix), dtype=np.float32)
                h_anchor = np.asarray(model.components_, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                run.update({"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})
                runs.append(run)
                continue

            w_anchor, h_anchor = _normalize_components(w_anchor, h_anchor)
            refinement_cfg = RSPDiscoveryConfig(**{**discovery_cfg.__dict__})
            refinement_cfg.loading_refinement_max_iter = int(getattr(discovery_cfg, "loading_refinement_max_iter", 0))
            refinement_cfg.loading_refinement_lambda_multiplier = float(
                getattr(discovery_cfg, "loading_refinement_lambda_multiplier", 0.0)
            )
            adaptive_profile = _component_adaptive_refinement_profile(
                w=w_anchor,
                h=h_anchor,
                x_raw=x_raw,
                reliability=reliability,
                cfg=discovery_cfg,
            )
            component_lambdas = (
                np.full(int(k), np.float32(loading_penalty), dtype=np.float32)
                * adaptive_profile["factors"].astype(np.float32, copy=False)
            )
            w, loading_refinement = _sparse_loading_refinement_with_fixed_activation(
                x=x,
                reliability=reliability,
                w=w_anchor,
                h=h_anchor,
                cfg=refinement_cfg,
                component_loading_lambdas=component_lambdas,
                l1_ratio=float(l1_ratio),
            )
            for component_idx in range(int(k)):
                w[:, component_idx] = _local_support_loading_compression(
                    loading=w[:, component_idx],
                    x_raw=x_raw,
                    reliability=reliability,
                    activation=h_anchor[component_idx],
                    cfg=discovery_cfg,
                    support_power=float(adaptive_profile["support_powers"][component_idx]),
                )
            run.update(
                {
                    "status": "completed",
                    "completed_k": int(k),
                    "n_iter": int(getattr(model, "n_iter_", 0)),
                    "weighted_refinement_iter": 0,
                    "loading_refinement": str(loading_refinement["method"]),
                    "loading_refinement_iter": int(loading_refinement["n_iter"]),
                    "loading_refinement_lambda_multiplier": float(loading_refinement["lambda_multiplier"]),
                    "adaptive_refinement_enabled": bool(adaptive_profile["enabled"]),
                    "adaptive_refinement_factor_min": float(np.min(adaptive_profile["factors"])),
                    "adaptive_refinement_factor_max": float(np.max(adaptive_profile["factors"])),
                    "adaptive_refinement_factor_mean": float(np.mean(adaptive_profile["factors"])),
                    "objective_initial": float("nan"),
                    "objective_final": float(_weighted_factorization_objective(
                        x=x,
                        weights=_elementwise_loss_weights(x, reliability, discovery_cfg),
                        w=w,
                        h=h_anchor,
                    )),
                    "reconstruction_error": float(getattr(model, "reconstruction_err_", np.nan)),
                    "backend_solver": "cd_axis_anchor_fixed_h_l1",
                    "backend_init": str(nmf_init),
                    "weighted_refinement": "fixed_activation_loading_refinement",
                    "scale_anchor": "activation_peak",
                }
            )
            runs.append(run)

            component_order = _component_order(w, h_anchor, neighbors)
            for new_component_idx, old_component_idx in enumerate(component_order):
                loading = w[:, old_component_idx].astype(np.float32, copy=False)
                activation_raw = h_anchor[old_component_idx].astype(np.float32, copy=False)
                candidate = _candidate_from_component(
                    candidate_id=f"RSP_k{k}_seed{seed_offset}_anchored_c{new_component_idx}",
                    source_k=int(k),
                    source_seed=int(actual_seed),
                    source_seed_offset=int(seed_offset),
                    source_component=int(new_component_idx),
                    initializer_type="cd_axis_anchor",
                    initialization_strategy="cd_axis_anchor",
                    genes=genes,
                    loading=loading,
                    activation_raw=activation_raw,
                    x=x,
                    x_raw=x_raw,
                    reliability=reliability,
                    gene_support_fraction=gene_support_fraction,
                    neighbors=neighbors,
                    scoring_cfg=scoring_cfg,
                    reconstruction=_component_reconstruction(w, h_anchor, old_component_idx),
                    backend_solver="cd_axis_anchor_fixed_h_l1",
                    backend_init=str(nmf_init),
                    factorization_path_id="axis_anchor_fixed_h_l1",
                    loading_sparsity_penalty=float(loading_penalty),
                    activation_sparsity_penalty=float(activation_penalty),
                    l1_ratio=float(l1_ratio),
                    component_loading_lambda=float(loading_penalty),
                    component_activation_lambda=float(activation_penalty),
                    adaptive_refinement_factor=float(adaptive_profile["factors"][old_component_idx]),
                    adaptive_refinement_signal=float(adaptive_profile["signals"][old_component_idx]),
                    adaptive_effective_gene_number=float(adaptive_profile["effective_gene_numbers"][old_component_idx]),
                    adaptive_weighted_support=float(adaptive_profile["weighted_support"][old_component_idx]),
                    adaptive_loading_lambda=float(component_lambdas[old_component_idx]),
                    adaptive_support_power=float(adaptive_profile["support_powers"][old_component_idx]),
                )
                candidates.append(candidate)
    _attach_redundancy_scores(candidates)
    return candidates, runs


def _generate_legacy_rss_spatial_nmf_candidates(
    *,
    x_raw: np.ndarray,
    reliability: np.ndarray,
    genes: np.ndarray,
    gene_support_fraction: np.ndarray,
    neighbors: np.ndarray,
    discovery_cfg: RSPDiscoveryConfig,
    scoring_cfg: RSPScoringConfig,
    random_seed: int,
) -> tuple[list[dict], list[dict]]:
    matrix = _legacy_factorization_matrix(x_raw, discovery_cfg)
    candidates: list[dict] = []
    runs: list[dict] = []
    if matrix.shape[0] == 0:
        return [], [{"status": "skipped", "reason": "no genes available for decomposition"}]

    for k in tuple(int(k) for k in discovery_cfg.k_grid):
        for seed_offset in _seed_offsets(discovery_cfg):
            actual_seed = int(random_seed) + int(seed_offset)
            nmf_init = _seeded_nmf_init(str(discovery_cfg.legacy_nmf_init), int(seed_offset))
            run = {
                "requested_k": int(k),
                "seed_offset": int(seed_offset),
                "random_seed": int(actual_seed),
                "n_genes_input": int(matrix.shape[0]),
                "status": "pending",
                "backend": "legacy_rss_spatial_nmf",
                "initialization_strategy": "legacy_axis_nmf",
                "factorization_path_id": "legacy_cd_l1",
                "loading_sparsity_penalty": float(max(0.0, discovery_cfg.legacy_nmf_alpha_w)),
                "activation_sparsity_penalty": float(max(0.0, discovery_cfg.legacy_nmf_alpha_h)),
                "l1_ratio": float(np.clip(discovery_cfg.legacy_nmf_l1_ratio, 0.0, 1.0)),
                "backend_init": str(nmf_init),
            }
            if matrix.shape[0] < int(k) or matrix.shape[1] < int(k):
                run.update(
                    {
                        "status": "skipped",
                        "reason": f"matrix_shape={matrix.shape} < requested_k={k}",
                    }
                )
                runs.append(run)
                continue

            model = NMF(
                n_components=int(k),
                init=str(nmf_init),
                solver=str(discovery_cfg.legacy_nmf_solver),
                beta_loss="frobenius",
                max_iter=int(discovery_cfg.legacy_nmf_max_iter),
                tol=float(discovery_cfg.legacy_nmf_tol),
                alpha_W=float(max(0.0, discovery_cfg.legacy_nmf_alpha_w)),
                alpha_H=float(max(0.0, discovery_cfg.legacy_nmf_alpha_h)),
                l1_ratio=float(np.clip(discovery_cfg.legacy_nmf_l1_ratio, 0.0, 1.0)),
                random_state=int(actual_seed),
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    w = np.asarray(model.fit_transform(matrix), dtype=np.float32)
                h = np.asarray(model.components_, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                run.update({"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})
                runs.append(run)
                continue

            w, h = _normalize_components(w, h)
            run.update(
                {
                    "status": "completed",
                    "completed_k": int(k),
                    "n_iter": int(getattr(model, "n_iter_", 0)),
                    "weighted_refinement_iter": 0,
                    "objective_initial": float("nan"),
                    "objective_final": float(_frobenius_objective(matrix, w, h)),
                    "reconstruction_error": float(getattr(model, "reconstruction_err_", np.nan)),
                    "backend_solver": "legacy_nmf_cd",
                    "backend_init": str(nmf_init),
                    "weighted_refinement": "none",
                    "scale_anchor": "activation_peak",
                }
            )
            runs.append(run)

            component_order = _component_order(w, h, neighbors)
            for new_component_idx, old_component_idx in enumerate(component_order):
                loading = w[:, old_component_idx].astype(np.float32, copy=False)
                activation_raw = h[old_component_idx].astype(np.float32, copy=False)
                candidate = _candidate_from_component(
                    candidate_id=f"RSP_k{k}_seed{seed_offset}_legacy_c{new_component_idx}",
                    source_k=int(k),
                    source_seed=int(actual_seed),
                    source_seed_offset=int(seed_offset),
                    source_component=int(new_component_idx),
                    initializer_type="legacy_axis_nmf",
                    initialization_strategy="legacy_axis_nmf",
                    genes=genes,
                    loading=loading,
                    activation_raw=activation_raw,
                    x=matrix,
                    x_raw=x_raw,
                    reliability=reliability,
                    gene_support_fraction=gene_support_fraction,
                    neighbors=neighbors,
                    scoring_cfg=scoring_cfg,
                    reconstruction=_component_reconstruction(w, h, old_component_idx),
                    backend_solver="legacy_nmf_cd",
                    backend_init=str(nmf_init),
                    factorization_path_id="legacy_cd_l1",
                    loading_sparsity_penalty=float(max(0.0, discovery_cfg.legacy_nmf_alpha_w)),
                    activation_sparsity_penalty=float(max(0.0, discovery_cfg.legacy_nmf_alpha_h)),
                    l1_ratio=float(np.clip(discovery_cfg.legacy_nmf_l1_ratio, 0.0, 1.0)),
                    component_loading_lambda=float(max(0.0, discovery_cfg.legacy_nmf_alpha_w)),
                    component_activation_lambda=float(max(0.0, discovery_cfg.legacy_nmf_alpha_h)),
                )
                candidates.append(candidate)
    _attach_redundancy_scores(candidates)
    return candidates, runs


def _legacy_factorization_matrix(x_raw: np.ndarray, cfg: RSPDiscoveryConfig) -> np.ndarray:
    matrix = np.clip(np.asarray(x_raw, dtype=np.float32), 0.0, None)
    if cfg.rss_transform == "log1p":
        matrix = np.log1p(matrix).astype(np.float32, copy=False)
    return np.clip(matrix, 0.0, None).astype(np.float32, copy=False)


def _seed_offsets(cfg: RSPDiscoveryConfig) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in getattr(cfg, "seed_grid", (0,)))
    if not values:
        return (0,)
    return tuple(dict.fromkeys(values))


def _seeded_nmf_init(base_init: str, seed_offset: int) -> str:
    init = str(base_init)
    if int(seed_offset) == 0:
        return init
    if init in {"nndsvd", "nndsvda"}:
        return "nndsvdar"
    return init


def _component_adaptive_refinement_profile(
    *,
    w: np.ndarray,
    h: np.ndarray,
    x_raw: np.ndarray,
    reliability: np.ndarray,
    cfg: RSPDiscoveryConfig,
) -> dict[str, np.ndarray | bool]:
    n_components = int(np.asarray(h).shape[0])
    if n_components <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return {
            "enabled": False,
            "factors": empty,
            "signals": empty,
            "effective_gene_numbers": empty,
            "weighted_support": empty,
            "support_powers": empty,
        }

    factors = np.ones(n_components, dtype=np.float32)
    signals = np.zeros(n_components, dtype=np.float32)
    effective_gene_numbers = np.zeros(n_components, dtype=np.float32)
    weighted_support = np.zeros(n_components, dtype=np.float32)
    base_power = float(max(0.0, getattr(cfg, "anchored_loading_support_power", 1.5)))
    support_powers = np.full(n_components, np.float32(base_power), dtype=np.float32)

    for component_idx in range(n_components):
        loading = np.clip(np.asarray(w[:, component_idx], dtype=np.float32), 0.0, None)
        total = float(loading.sum())
        if total <= 0.0:
            continue
        p = loading / np.float32(total)
        neff = float(_effective_gene_number(loading))
        support = _component_loading_support(
            x_raw=x_raw,
            reliability=reliability,
            activation=np.asarray(h[component_idx], dtype=np.float32),
            cfg=cfg,
        )
        support_mean = float(p @ support) if support.size == p.size else 0.0
        diffuseness = float(np.log1p(neff) / max(np.log1p(float(loading.size)), 1e-12))
        low_support_pressure = 1.0 - float(np.clip(support_mean, 0.0, 1.0))
        signals[component_idx] = np.float32(np.clip(0.65 * diffuseness + 0.35 * low_support_pressure, 0.0, 1.0))
        effective_gene_numbers[component_idx] = np.float32(neff)
        weighted_support[component_idx] = np.float32(np.clip(support_mean, 0.0, 1.0))

    enabled = bool(getattr(cfg, "anchored_adaptive_refinement_enabled", True))
    if not enabled or n_components <= 1:
        return {
            "enabled": False,
            "factors": factors,
            "signals": signals,
            "effective_gene_numbers": effective_gene_numbers,
            "weighted_support": weighted_support,
            "support_powers": support_powers,
        }

    strength = float(max(0.0, getattr(cfg, "anchored_adaptive_refinement_strength", 0.8)))
    min_factor = float(max(0.0, getattr(cfg, "anchored_adaptive_refinement_min_factor", 0.6)))
    max_factor = float(max(min_factor, getattr(cfg, "anchored_adaptive_refinement_max_factor", 1.8)))
    neutral_signal = np.float32(0.65)
    factors = np.clip(1.0 + np.float32(strength) * (signals - neutral_signal), min_factor, max_factor).astype(
        np.float32,
        copy=False,
    )
    support_powers = (np.float32(base_power) * factors).astype(np.float32, copy=False)
    return {
        "enabled": True,
        "factors": factors,
        "signals": signals,
        "effective_gene_numbers": effective_gene_numbers,
        "weighted_support": weighted_support,
        "support_powers": support_powers,
    }


def _sparse_loading_refinement_with_fixed_activation(
    *,
    x: np.ndarray,
    reliability: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    cfg: RSPDiscoveryConfig,
    component_loading_lambdas: np.ndarray,
    l1_ratio: float,
) -> tuple[np.ndarray, dict]:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    weights = _elementwise_loss_weights(x, reliability, cfg)
    out = np.clip(np.asarray(w, dtype=np.float32), 0.0, None).copy()
    fixed_h = np.clip(np.asarray(h, dtype=np.float32), 0.0, None)
    component_lambdas = np.asarray(component_loading_lambdas, dtype=np.float32).reshape(-1)
    if component_lambdas.size != fixed_h.shape[0]:
        component_lambdas = np.resize(component_lambdas, fixed_h.shape[0]).astype(np.float32, copy=False)

    max_iter = max(0, int(getattr(cfg, "loading_refinement_max_iter", 0)))
    lambda_multiplier = max(0.0, float(getattr(cfg, "loading_refinement_lambda_multiplier", 0.0)))
    if max_iter <= 0 or lambda_multiplier <= 0.0 or fixed_h.size == 0:
        return out.astype(np.float32, copy=False), {
            "method": "disabled",
            "n_iter": 0,
            "lambda_multiplier": float(lambda_multiplier),
        }

    eps = np.float32(1e-8)
    loading_l1 = (
        np.clip(component_lambdas, 0.0, None)
        * np.float32(x.shape[1])
        * np.float32(np.clip(l1_ratio, 0.0, 1.0))
        * np.float32(lambda_multiplier)
    )
    previous = _weighted_factorization_objective(x=x, weights=weights, w=out, h=fixed_h)
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        for component_idx in range(fixed_h.shape[0]):
            old_column = out[:, component_idx].copy()
            reconstruction = out @ fixed_h
            residual = x - reconstruction + np.outer(out[:, component_idx], fixed_h[component_idx])
            hr = fixed_h[component_idx]
            denom = np.sum(weights * (hr[None, :] * hr[None, :]), axis=1) + eps
            numer = np.sum(weights * hr[None, :] * residual, axis=1)
            out[:, component_idx] = np.maximum((numer - loading_l1[component_idx]) / denom, 0.0)
            if float(out[:, component_idx].sum()) <= 0.0:
                out[:, component_idx] = old_column

        current = _weighted_factorization_objective(x=x, weights=weights, w=out, h=fixed_h)
        if previous > 0 and abs(previous - current) / max(previous, 1e-8) < float(cfg.decomposition_tol):
            break
        previous = current

    return out.astype(np.float32, copy=False), {
        "method": "fixed_activation_coordinate_descent",
        "n_iter": int(n_iter),
        "lambda_multiplier": float(lambda_multiplier),
    }


def _elementwise_loss_weights(x: np.ndarray, reliability: np.ndarray, cfg: RSPDiscoveryConfig) -> np.ndarray:
    evidence = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    reliability_arr = np.clip(np.asarray(reliability, dtype=np.float32), 0.0, 1.0)
    floor = np.float32(max(0.0, float(getattr(cfg, "elementwise_weight_floor", 0.05))))
    zero_weight = np.float32(max(0.0, float(getattr(cfg, "zero_evidence_weight", 0.50))))
    positive = evidence > 0
    weights = np.where(positive, np.maximum(reliability_arr, floor), zero_weight)
    weights[~np.isfinite(weights)] = 0.0
    return np.clip(weights, 0.0, None).astype(np.float32, copy=False)


def _first_nonnegative(values: tuple[float, ...], *, default: float, upper: float | None = None) -> float:
    raw_values = tuple(float(value) for value in values)
    value = raw_values[0] if raw_values else float(default)
    value = max(0.0, value)
    if upper is not None:
        value = min(float(upper), value)
    return float(value)


def _candidate_from_component(
    *,
    candidate_id: str,
    source_k: int,
    source_seed: int,
    source_seed_offset: int,
    source_component: int,
    initializer_type: str,
    initialization_strategy: str,
    genes: np.ndarray,
    loading: np.ndarray,
    activation_raw: np.ndarray,
    x: np.ndarray,
    x_raw: np.ndarray,
    reliability: np.ndarray,
    gene_support_fraction: np.ndarray,
    neighbors: np.ndarray,
    scoring_cfg: RSPScoringConfig,
    reconstruction: np.ndarray,
    backend_solver: str,
    backend_init: str,
    factorization_path_id: str,
    loading_sparsity_penalty: float,
    activation_sparsity_penalty: float,
    l1_ratio: float,
    component_loading_lambda: float,
    component_activation_lambda: float,
    adaptive_refinement_factor: float = 1.0,
    adaptive_refinement_signal: float = 0.0,
    adaptive_effective_gene_number: float = 0.0,
    adaptive_weighted_support: float = 0.0,
    adaptive_loading_lambda: float = 0.0,
    adaptive_support_power: float = 0.0,
) -> dict:
    loading = np.clip(np.asarray(loading, dtype=np.float32), 0.0, None)
    activation_raw = np.clip(np.asarray(activation_raw, dtype=np.float32), 0.0, None)
    loading_fraction = loading / max(float(loading.sum()), 1e-12)
    effective_gene_number = _effective_gene_number(loading)
    loading_entropy = _loading_entropy(loading)
    active_mask, active_threshold = activation_active_mask(
        activation_raw,
        neighbors,
        min_peak_fraction=float(scoring_cfg.activation_active_min_peak_fraction),
        quantile=float(scoring_cfg.activation_active_quantile),
        min_abs=float(scoring_cfg.activation_active_min_abs),
        min_component_size=int(scoring_cfg.activation_active_min_component_size),
        fallback_to_base=bool(scoring_cfg.activation_active_fallback_to_base),
    )
    active_count = int(np.count_nonzero(active_mask))
    components = connected_component_indices(active_mask, neighbors)
    support = _candidate_rss_support_summary(
        x_raw=x_raw,
        loading=loading,
        activation=activation_raw,
        active_mask=active_mask,
    )
    tiers = _contribution_tiers_for_pattern(loading, genes)
    weighted_error, weighted_explained = _weighted_component_fit(x, reliability, reconstruction)
    unweighted_error, unweighted_explained = _weighted_component_fit(x, np.ones_like(reliability), reconstruction)
    single_gene_dominance = float(loading_fraction.max()) if loading_fraction.size else 0.0
    quality_flags = _quality_flags(
        active_count=active_count,
        effective_gene_number=effective_gene_number,
        single_gene_dominance=single_gene_dominance,
        weighted_explained=weighted_explained,
        scoring_cfg=scoring_cfg,
    )

    return {
        "candidate_id": candidate_id,
        "source_k": int(source_k),
        "source_seed": int(source_seed),
        "source_seed_offset": int(source_seed_offset),
        "source_component": int(source_component),
        "initializer_type": initializer_type,
        "initialization_strategy": str(initialization_strategy),
        "backend_solver": str(backend_solver),
        "backend_init": str(backend_init),
        "factorization_path_id": str(factorization_path_id),
        "loading_sparsity_penalty": float(loading_sparsity_penalty),
        "activation_sparsity_penalty": float(activation_sparsity_penalty),
        "l1_ratio": float(l1_ratio),
        "component_loading_lambda": float(component_loading_lambda),
        "component_activation_lambda": float(component_activation_lambda),
        "adaptive_refinement_factor": float(adaptive_refinement_factor),
        "adaptive_refinement_signal": float(adaptive_refinement_signal),
        "adaptive_effective_gene_number": float(adaptive_effective_gene_number),
        "adaptive_weighted_support": float(adaptive_weighted_support),
        "adaptive_loading_lambda": float(adaptive_loading_lambda),
        "adaptive_support_power": float(adaptive_support_power),
        "genes": genes.copy(),
        "gene_loadings": loading,
        "gene_support_fraction": np.asarray(gene_support_fraction, dtype=np.float32).copy(),
        "gene_local_rss_strength": support["gene_local_rss_strength"],
        "gene_local_rss_coverage": support["gene_local_rss_coverage"],
        "gene_local_rss_lift": support["gene_local_rss_lift"],
        "activation_raw": activation_raw.copy(),
        "activation": activation_raw.copy(),
        "activation_active_mask": active_mask.astype(bool, copy=False),
        "activation_active_threshold": float(active_threshold),
        "activation_active_spots": int(active_count),
        "activation_active_fraction": float(np.mean(active_mask)) if active_mask.size else 0.0,
        "activation_raw_positive_fraction": float(np.mean(np.asarray(activation_raw) > 0))
        if np.asarray(activation_raw).size
        else 0.0,
        "weighted_local_rss_strength": support["weighted_local_rss_strength"],
        "weighted_local_rss_coverage": support["weighted_local_rss_coverage"],
        "weighted_local_rss_lift": support["weighted_local_rss_lift"],
        "top10_local_rss_strength": support["top10_local_rss_strength"],
        "top10_local_rss_coverage": support["top10_local_rss_coverage"],
        "top10_local_rss_lift": support["top10_local_rss_lift"],
        "rss_support_score": support["rss_support_score"],
        "contribution_tiers": tiers,
        "preliminary_core_genes": tiers["70"],
        "weighted_reconstruction_error": float(weighted_error),
        "rss_evidence_explained": float(unweighted_explained),
        "reliability_weighted_explained": float(weighted_explained),
        "effective_gene_number": float(effective_gene_number),
        "gene_loading_entropy": float(loading_entropy),
        "activation_moran_i": float(moran_matrix(activation_raw[None, :], neighbors)[0]),
        "pattern_moran": float(moran_matrix(activation_raw[None, :], neighbors)[0]),
        "activation_coverage": float(np.mean(active_mask)) if active_mask.size else 0.0,
        "largest_component_fraction": float(largest_component_fraction(active_mask, neighbors)),
        "fragmentation_score": float(len(components) / max(1, active_count)),
        "single_gene_dominance_score": float(single_gene_dominance),
        "redundancy_score": 0.0,
        "quality_flag": "candidate" if not quality_flags else "review",
        "quality_flags": ";".join(quality_flags),
        "candidate_status": "candidate" if not quality_flags else "dropped",
        "drop_reason": ";".join(quality_flags),
    }


def _normalize_components(w: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=np.float32).copy()
    h = np.asarray(h, dtype=np.float32).copy()
    for component_idx in range(h.shape[0]):
        peak = float(np.max(h[component_idx]))
        if peak > 0:
            h[component_idx] /= peak
            w[:, component_idx] *= peak
        col_sum = float(np.sum(w[:, component_idx]))
        if col_sum <= 0:
            w[:, component_idx] = 1.0 / max(1, w.shape[0])
    return np.clip(w, 0.0, None), np.clip(h, 0.0, None)


def _component_order(w: np.ndarray, h: np.ndarray, neighbors: np.ndarray) -> list[int]:
    rows = []
    for component_idx in range(h.shape[0]):
        activation = np.asarray(h[component_idx], dtype=np.float32)
        active_mask = activation > positive_quantile(activation, 0.80)
        rows.append(
            (
                component_idx,
                float(moran_matrix(activation[None, :], neighbors)[0]),
                float(largest_component_fraction(active_mask, neighbors)),
                float(_effective_gene_number(w[:, component_idx])),
            )
        )
    return [int(row[0]) for row in sorted(rows, key=lambda row: (row[1], row[2], row[3]), reverse=True)]


def _component_reconstruction(w: np.ndarray, h: np.ndarray, component_idx: int) -> np.ndarray:
    return np.outer(w[:, component_idx], h[component_idx]).astype(np.float32, copy=False)


def _weighted_component_fit(x: np.ndarray, reliability: np.ndarray, reconstruction: np.ndarray) -> tuple[float, float]:
    weights = np.clip(np.asarray(reliability, dtype=np.float32), 0.0, 1.0)
    residual = np.asarray(x, dtype=np.float32) - np.asarray(reconstruction, dtype=np.float32)
    sse = float(np.sum(weights * residual * residual))
    total = float(np.sum(weights * x * x))
    rel_error = float(np.sqrt(sse / max(total, 1e-12)))
    explained = clamp01(1.0 - (sse / max(total, 1e-12)))
    return rel_error, explained


def _frobenius_objective(x: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    residual = np.asarray(x, dtype=np.float32) - np.asarray(w, dtype=np.float32) @ np.asarray(h, dtype=np.float32)
    return float(np.sum(residual * residual))


def _weighted_factorization_objective(*, x: np.ndarray, weights: np.ndarray, w: np.ndarray, h: np.ndarray) -> float:
    residual = np.asarray(x, dtype=np.float32) - np.asarray(w, dtype=np.float32) @ np.asarray(h, dtype=np.float32)
    return float(np.sum(np.asarray(weights, dtype=np.float32) * residual * residual))


def _effective_gene_number(loading: np.ndarray) -> float:
    values = np.clip(np.asarray(loading, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    p = values / total
    return float(1.0 / max(float(np.sum(p * p)), 1e-12))


def _loading_entropy(loading: np.ndarray) -> float:
    values = np.clip(np.asarray(loading, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    p = values[values > 0] / total
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(max(2, p.size)))


def _activation_weighted_local_strength(matrix: np.ndarray, activation: np.ndarray) -> np.ndarray:
    rss = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
    weights = _activation_weights(activation)
    if rss.size == 0 or weights.size == 0 or rss.shape[1] != weights.shape[0]:
        return np.zeros(rss.shape[0], dtype=np.float32)
    weighted_mean = rss @ weights
    scale = _row_positive_quantile(rss, 0.90)
    support = np.zeros(rss.shape[0], dtype=np.float32)
    valid = scale > 0
    support[valid] = weighted_mean[valid] / scale[valid]
    support[~np.isfinite(support)] = 0.0
    return np.clip(support, 0.0, 1.0).astype(np.float32, copy=False)


def _activation_local_rss_coverage(matrix: np.ndarray, activation: np.ndarray) -> np.ndarray:
    rss = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
    active = np.asarray(activation, dtype=np.float32).reshape(-1) > 0
    if rss.size == 0 or active.size == 0 or rss.shape[1] != active.shape[0] or not np.any(active):
        return np.zeros(rss.shape[0], dtype=np.float32)
    coverage = np.mean(rss[:, active] > 0, axis=1)
    coverage[~np.isfinite(coverage)] = 0.0
    return np.clip(coverage, 0.0, 1.0).astype(np.float32, copy=False)


def _activation_local_rss_lift(matrix: np.ndarray, activation: np.ndarray) -> np.ndarray:
    rss = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
    active = np.asarray(activation, dtype=np.float32).reshape(-1) > 0
    if rss.size == 0 or active.size == 0 or rss.shape[1] != active.shape[0] or not np.any(active):
        return np.zeros(rss.shape[0], dtype=np.float32)
    active_mean = np.mean(rss[:, active], axis=1)
    global_mean = np.mean(rss, axis=1)
    lift = np.zeros(rss.shape[0], dtype=np.float32)
    valid = global_mean > 0
    lift[valid] = active_mean[valid] / np.maximum(global_mean[valid], 1e-12)
    lift[~np.isfinite(lift)] = 0.0
    return np.clip(lift, 0.0, None).astype(np.float32, copy=False)


def _activation_weighted_local_reliability(
    matrix: np.ndarray,
    reliability: np.ndarray,
    activation: np.ndarray,
) -> np.ndarray:
    rss = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
    rel = np.clip(np.asarray(reliability, dtype=np.float32), 0.0, 1.0)
    weights = _activation_weights(activation)
    if (
        rss.size == 0
        or rel.size == 0
        or weights.size == 0
        or rss.shape != rel.shape
        or rss.shape[1] != weights.shape[0]
    ):
        return np.zeros(rss.shape[0], dtype=np.float32)
    local_evidence = rss * weights[None, :]
    denom = local_evidence.sum(axis=1)
    support = np.zeros(rss.shape[0], dtype=np.float32)
    valid = denom > 0
    support[valid] = (rel[valid] * local_evidence[valid]).sum(axis=1) / denom[valid]
    support[~np.isfinite(support)] = 0.0
    return np.clip(support, 0.0, 1.0).astype(np.float32, copy=False)


def _activation_weights(activation: np.ndarray) -> np.ndarray:
    act = np.clip(np.asarray(activation, dtype=np.float32).reshape(-1), 0.0, None)
    total = float(act.sum())
    if act.size == 0 or total <= 0.0 or not np.isfinite(total):
        return np.zeros(act.shape[0], dtype=np.float32)
    return (act / total).astype(np.float32, copy=False)


def _row_positive_quantile(matrix: np.ndarray, quantile: float) -> np.ndarray:
    values = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    scale = np.zeros(values.shape[0], dtype=np.float32)
    q = float(np.clip(quantile, 0.0, 1.0))
    for idx, row in enumerate(values):
        positive = row[row > 0]
        if positive.size:
            scale[idx] = float(np.quantile(positive, q))
    scale[~np.isfinite(scale)] = 0.0
    return scale


def _local_support_loading_compression(
    *,
    loading: np.ndarray,
    x_raw: np.ndarray,
    reliability: np.ndarray,
    activation: np.ndarray,
    cfg: RSPDiscoveryConfig,
    support_power: float | None = None,
) -> np.ndarray:
    values = np.clip(np.asarray(loading, dtype=np.float32), 0.0, None)
    if values.size == 0 or float(values.sum()) <= 0.0:
        return values.astype(np.float32, copy=False)
    support = _component_loading_support(
        x_raw=x_raw,
        reliability=reliability,
        activation=activation,
        cfg=cfg,
    )
    power = float(max(0.0, getattr(cfg, "anchored_loading_support_power", 1.5) if support_power is None else support_power))
    if power <= 0.0:
        return values.astype(np.float32, copy=False)
    compressed = values * np.power(support, np.float32(power))
    compressed[~np.isfinite(compressed)] = 0.0
    if float(compressed.sum()) <= 0.0:
        return values.astype(np.float32, copy=False)
    return compressed.astype(np.float32, copy=False)


def _component_loading_support(
    *,
    x_raw: np.ndarray,
    reliability: np.ndarray,
    activation: np.ndarray,
    cfg: RSPDiscoveryConfig,
) -> np.ndarray:
    local_rss_strength = _activation_weighted_local_strength(x_raw, activation)
    local_reliability = _activation_weighted_local_reliability(x_raw, reliability, activation)
    reliability_weight = float(np.clip(getattr(cfg, "anchored_loading_reliability_weight", 0.35), 0.0, 1.0))
    support = (1.0 - reliability_weight) * local_rss_strength + reliability_weight * local_reliability
    support[~np.isfinite(support)] = 0.0
    return np.clip(support, 0.0, 1.0).astype(np.float32, copy=False)


def _contribution_tiers_for_pattern(loading: np.ndarray, genes: np.ndarray) -> dict[str, list[str]]:
    order = np.argsort(-np.asarray(loading, dtype=np.float64))
    sorted_loading = np.asarray(loading, dtype=np.float64)[order]
    total = float(sorted_loading.sum())
    tiers = {"50": [], "70": [], "90": []}
    if total <= 0:
        return tiers
    cumulative = np.cumsum(sorted_loading) / total
    for threshold in (0.50, 0.70, 0.90):
        stop = int(np.searchsorted(cumulative, threshold, side="left")) + 1
        tiers[str(int(threshold * 100))] = genes[order[:stop]].astype(str).tolist()
    return tiers


def _attach_redundancy_scores(candidates: list[dict]) -> None:
    for idx, candidate in enumerate(candidates):
        others = candidates[:idx] + candidates[idx + 1 :]
        candidate["redundancy_score"] = float(
            max((safe_corr(candidate["activation"], other["activation"]) for other in others), default=0.0)
        )


def _quality_flags(
    *,
    active_count: int,
    effective_gene_number: float,
    single_gene_dominance: float,
    weighted_explained: float,
    scoring_cfg: RSPScoringConfig,
) -> list[str]:
    flags: list[str] = []
    if active_count < int(scoring_cfg.min_active_spots):
        flags.append("too_few_active_spots")
    if effective_gene_number <= 1.05:
        flags.append("single_gene_collapse")
    if single_gene_dominance >= 0.90:
        flags.append("single_gene_dominated")
    if weighted_explained <= 0.0:
        flags.append("low_weighted_explained")
    return flags
