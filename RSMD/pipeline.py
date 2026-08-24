from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle_io import (
    ensure_bundle_dirs,
    get_code_version,
    hash_array,
    hash_file,
    iso_now,
    promote_bundle,
    set_random_seed,
    write_json,
    write_parquet,
)
from .data_prep import load_rsmd_inputs
from .domain_ops import (
    build_domain_graph_table,
    build_domain_membership_table,
    extract_candidate_domains,
)
from .qc_ops import build_rsp_metadomain_summary_table, build_qc_report
from .schema import (
    RSMDDomainAdjacencyConfig,
    RSMDDomainFilterConfig,
    RSMDFiltrationConfig,
    RSMDInputConfig,
    RSMDPipelineConfig,
    RSMDDomainReliabilityConfig,
    PotentialConfig,
)

logger = logging.getLogger(__name__)

RSMD_OUTPUT_DIR = "rsmd_out"

__all__ = [
    "RSMDInputConfig",
    "PotentialConfig",
    "RSMDDomainFilterConfig",
    "RSMDFiltrationConfig",
    "RSMDDomainAdjacencyConfig",
    "RSMDDomainReliabilityConfig",
    "RSMDPipelineConfig",
    "RSMD_OUTPUT_DIR",
    "run_rsmd_pipeline",
]


def _stable_domain_key(sample_id: str, rsp_id: str, spot_ids_subset: np.ndarray) -> str:
    ordered = sorted(str(x) for x in spot_ids_subset.tolist())
    payload = f"{sample_id}|{rsp_id}|{'|'.join(ordered)}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"{sample_id}:{rsp_id}:{digest}"


def _uniq_reasons(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for r in values:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _assign_domain_ids(domains: list[dict]) -> None:
    ordered = sorted(
        domains,
        key=lambda d: (
            str(d.get("rsp_seed_id", "")),
            -int(d.get("spot_count", 0)),
            str(d.get("domain_key", "")),
        ),
    )
    for i, d in enumerate(ordered, start=1):
        d["domain_id"] = f"D{i:06d}"


def _attach_domain_reliability(
    domains_df: pd.DataFrame,
    cfg: RSMDPipelineConfig,
) -> tuple[pd.DataFrame, dict]:
    out = domains_df.copy()
    rel_cfg = cfg.domain_reliability

    if out.empty:
        out["domain_reliability"] = np.array([], dtype=np.float32)
        out["domain_rsp_score_component"] = np.array([], dtype=np.float32)
        out["domain_prominence_component"] = np.array([], dtype=np.float32)
        out["domain_density_component"] = np.array([], dtype=np.float32)
        return out, {
            "enabled": bool(rel_cfg.enabled),
            "prominence_scale": float("nan"),
            "density_scale": float(max(1e-8, rel_cfg.density_scale)),
            "reliability_min": float("nan"),
            "reliability_max": float("nan"),
            "reliability_mean": float("nan"),
            "low_reliability_frac": 0.0,
        }

    if not bool(rel_cfg.enabled):
        out["domain_reliability"] = np.ones(out.shape[0], dtype=np.float32)
        out["domain_rsp_score_component"] = np.ones(out.shape[0], dtype=np.float32)
        out["domain_prominence_component"] = np.ones(out.shape[0], dtype=np.float32)
        out["domain_density_component"] = np.ones(out.shape[0], dtype=np.float32)
        return out, {
            "enabled": False,
            "prominence_scale": float("nan"),
            "density_scale": float(max(1e-8, rel_cfg.density_scale)),
            "reliability_min": 1.0,
            "reliability_max": 1.0,
            "reliability_mean": 1.0,
            "low_reliability_frac": 0.0,
        }

    required = ["rsp_score_weight", "rsp_prominence", "internal_density"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            "Domain reliability requires columns missing from domains table: "
            f"{missing}. This is an internal RSMD error."
        )

    rsp_score = pd.to_numeric(out["rsp_score_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    rsp_score = np.clip(rsp_score, 0.0, 1.0)

    prom = pd.to_numeric(out["rsp_prominence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    prom = np.clip(prom, 0.0, None)
    prom_pos = prom[prom > 0]
    q = float(np.clip(rel_cfg.prominence_scale_quantile, 0.0, 1.0))
    prom_scale = float(np.quantile(prom_pos, q)) if prom_pos.size > 0 else 1.0
    prom_scale = max(1e-8, prom_scale)
    prom_norm = np.clip(prom / prom_scale, 0.0, 1.0)

    density = pd.to_numeric(out["internal_density"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    density = np.clip(density, 0.0, None)
    density_scale = float(max(1e-8, rel_cfg.density_scale))
    density_norm = np.clip(density / density_scale, 0.0, 1.0)

    a = float(max(0.0, rel_cfg.rsp_score_exponent))
    b = float(max(0.0, rel_cfg.prominence_exponent))
    c = float(max(0.0, rel_cfg.density_exponent))
    raw = (np.power(rsp_score, a) if a > 0 else np.ones_like(rsp_score)) * (
        np.power(prom_norm, b) if b > 0 else np.ones_like(prom_norm)
    ) * (
        np.power(density_norm, c) if c > 0 else np.ones_like(density_norm)
    )

    lo = float(np.clip(rel_cfg.min_node_reliability, 0.0, 1.0))
    hi = float(np.clip(rel_cfg.max_node_reliability, lo, 1.0))
    rel = np.clip(np.asarray(raw, dtype=np.float64), lo, hi)

    out["domain_reliability"] = rel.astype(np.float32)
    out["domain_rsp_score_component"] = rsp_score.astype(np.float32)
    out["domain_prominence_component"] = prom_norm.astype(np.float32)
    out["domain_density_component"] = density_norm.astype(np.float32)

    return out, {
        "enabled": True,
        "prominence_scale": float(prom_scale),
        "density_scale": float(density_scale),
        "reliability_min": float(np.min(rel)),
        "reliability_max": float(np.max(rel)),
        "reliability_mean": float(np.mean(rel)),
        "low_reliability_frac": float(np.mean(rel <= lo + 1e-8)),
    }


def refine_domains(
    candidate_domains: list[dict],
    sample_id: str,
    spot_ids: np.ndarray,
    cfg: RSMDPipelineConfig,
) -> dict:
    domains_all, _ = _finalize_domains(
        candidate_domains=candidate_domains,
        sample_id=sample_id,
        spot_ids=spot_ids,
        cfg=cfg,
    )
    _assign_domain_ids(domains_all)
    accepted_domains = [d for d in domains_all if bool(d.get("qc_pass", False))]

    return {
        "final_stage": "finalize_connected_rsd.v1",
        "domains_all": domains_all,
        "accepted_domains": accepted_domains,
    }


def _finalize_domains(
    candidate_domains: list[dict],
    sample_id: str,
    spot_ids: np.ndarray,
    cfg: RSMDPipelineConfig,
) -> tuple[list[dict], list[dict]]:
    domains_out: list[dict] = []
    accepted_domains: list[dict] = []

    for idx, d in enumerate(candidate_domains, start=1):
        reasons = list(d.get("screening_reasons", []))
        screening_tags = list(d.get("screening_reject_tags", []))
        reasons = _uniq_reasons(reasons)
        screening_tags = _uniq_reasons(screening_tags)
        qc_pass = len(reasons) == 0

        spot_idx = np.asarray(d["spot_indices"], dtype=np.int32)
        domain_key = _stable_domain_key(
            sample_id=sample_id,
            rsp_id=str(d["rsp_id"]),
            spot_ids_subset=spot_ids[spot_idx],
        )
        domain = {
            "domain_id": f"D{idx:06d}",
            "domain_key": domain_key,
            "sample_id": sample_id,
            "rsp_seed_id": str(d["rsp_id"]),
            "spot_indices": spot_idx,
            "spot_count": int(d["spot_count"]),
            "geo_centroid_x": float(d["geo_centroid_x"]),
            "geo_centroid_y": float(d["geo_centroid_y"]),
            "geo_area_est": float(d["geo_area_est"]),
            "geo_compactness": float(d["geo_compactness"]),
            "geo_boundary_ratio": float(d["geo_boundary_ratio"]),
            "geo_elongation": float(d.get("geo_elongation", float("nan"))),
            "geo_leaf_ratio": float(d.get("geo_leaf_ratio", float("nan"))),
            "geo_articulation_ratio": float(d.get("geo_articulation_ratio", float("nan"))),
            "components_count": int(d.get("components_count", 1)),
            "internal_edge_count": int(d["internal_edge_count"]),
            "boundary_edge_count": int(d["boundary_edge_count"]),
            "internal_density": float(d["internal_density"]),
            "rsp_seed_mean": float(d["rsp_seed_mean"]),
            "rsp_seed_sum": float(d["rsp_seed_sum"]),
            "rsp_mean_enrichment_ratio": float(d.get("rsp_mean_enrichment_ratio", 0.0)),
            "rsp_mean_enrichment_delta": float(d.get("rsp_mean_enrichment_delta", 0.0)),
            "rsp_peak_value": float(d["rsp_peak_value"]),
            "rsp_prominence": float(d["rsp_prominence"]),
            "rsp_scale_median": float(d.get("rsp_scale", 0.0)),
            "rsp_score": float(d.get("rsp_score", 1.0)),
            "rsp_score_used": float(d.get("rsp_score_used", 1.0)),
            "rsp_score_weight": float(d.get("rsp_score_weight", 1.0)),
            "flow_graph_mode": str(d.get("flow_graph_mode", str(cfg.potential.flow_graph_mode))),
            "segmentation_method": str(d.get("segmentation_method", "unknown")),
            "selected_threshold_quantile": float(d.get("selected_threshold_quantile", float("nan"))),
            "selected_threshold_value": float(d.get("selected_threshold_value", float("nan"))),
            "threshold_selection_reason": str(d.get("threshold_selection_reason", "")),
            "threshold_level_count_total": int(d.get("threshold_level_count_total", 0)),
            "threshold_component_count": int(d.get("threshold_component_count", 0)),
            "threshold_retained_component_count": int(d.get("threshold_retained_component_count", 0)),
            "threshold_retained_spot_count": int(d.get("threshold_retained_spot_count", 0)),
            "activation_threshold_quantile": float(d.get("activation_threshold_quantile", float("nan"))),
            "seg_active_floor": float(d.get("active_floor", 0.0)),
            "thr_min_domain_spots_effective": int(d.get("min_domain_spots_effective", 0)),
            "thr_min_domain_internal_density_effective": float(
                d.get("min_domain_internal_density_effective", 0.0)
            ),
            "qc_pass": bool(qc_pass),
            "qc_reject_reasons": reasons,
            "screening_reject_tags": screening_tags,
            "screening_decision": "kept" if qc_pass else "rejected",
            "is_background": bool(not qc_pass),
            "screening_pass": bool(d["screening_pass"]),
        }
        domains_out.append(domain)
        if qc_pass:
            accepted_domains.append(domain)

    return domains_out, accepted_domains


def _domain_rows_for_parquet(domains: list[dict]) -> pd.DataFrame:
    rows = []
    for d in domains:
        rows.append(
            {
                "domain_id": d["domain_id"],
                "domain_key": d["domain_key"],
                "sample_id": d["sample_id"],
                "rsp_seed_id": d["rsp_seed_id"],
                "spot_count": d["spot_count"],
                "geo_centroid_x": d["geo_centroid_x"],
                "geo_centroid_y": d["geo_centroid_y"],
                "geo_area_est": d["geo_area_est"],
                "geo_compactness": d["geo_compactness"],
                "geo_boundary_ratio": d["geo_boundary_ratio"],
                "geo_elongation": d.get("geo_elongation", float("nan")),
                "geo_leaf_ratio": d.get("geo_leaf_ratio", float("nan")),
                "geo_articulation_ratio": d.get("geo_articulation_ratio", float("nan")),
                "components_count": int(d.get("components_count", 1)),
                "internal_edge_count": d["internal_edge_count"],
                "boundary_edge_count": d["boundary_edge_count"],
                "internal_density": d["internal_density"],
                "rsp_seed_mean": d["rsp_seed_mean"],
                "rsp_seed_sum": d["rsp_seed_sum"],
                "rsp_mean_enrichment_ratio": d.get("rsp_mean_enrichment_ratio", 0.0),
                "rsp_mean_enrichment_delta": d.get("rsp_mean_enrichment_delta", 0.0),
                "rsp_peak_value": d["rsp_peak_value"],
                "rsp_prominence": d["rsp_prominence"],
                "rsp_scale_median": d["rsp_scale_median"],
                "rsp_score": d.get("rsp_score", 1.0),
                "rsp_score_used": d.get("rsp_score_used", 1.0),
                "rsp_score_weight": d.get("rsp_score_weight", 1.0),
                "flow_graph_mode": d.get("flow_graph_mode", str("unknown")),
                "segmentation_method": d.get("segmentation_method", str("unknown")),
                "selected_threshold_quantile": d.get("selected_threshold_quantile", float("nan")),
                "selected_threshold_value": d.get("selected_threshold_value", float("nan")),
                "threshold_selection_reason": d.get("threshold_selection_reason", ""),
                "threshold_level_count_total": int(d.get("threshold_level_count_total", 0)),
                "threshold_component_count": int(d.get("threshold_component_count", 0)),
                "threshold_retained_component_count": int(d.get("threshold_retained_component_count", 0)),
                "threshold_retained_spot_count": int(d.get("threshold_retained_spot_count", 0)),
                "activation_threshold_quantile": d.get("activation_threshold_quantile", float("nan")),
                "seg_active_floor": d["seg_active_floor"],
                "thr_min_domain_spots_effective": d["thr_min_domain_spots_effective"],
                "thr_min_domain_internal_density_effective": d["thr_min_domain_internal_density_effective"],
                "qc_pass": d["qc_pass"],
                "qc_reject_reasons": ";".join(d["qc_reject_reasons"]),
                "screening_reject_tags": ";".join(d.get("screening_reject_tags", [])),
                "screening_decision": str(d.get("screening_decision", "unknown")),
                "is_background": d["is_background"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "domain_id",
                "domain_key",
                "sample_id",
                "rsp_seed_id",
                "spot_count",
                "geo_centroid_x",
                "geo_centroid_y",
                "geo_area_est",
                "geo_compactness",
                "geo_boundary_ratio",
                "geo_elongation",
                "geo_leaf_ratio",
                "geo_articulation_ratio",
                "components_count",
                "internal_edge_count",
                "boundary_edge_count",
                "internal_density",
                "rsp_seed_mean",
                "rsp_seed_sum",
                "rsp_mean_enrichment_ratio",
                "rsp_mean_enrichment_delta",
                "rsp_peak_value",
                "rsp_prominence",
                "rsp_scale_median",
                "rsp_score",
                "rsp_score_used",
                "rsp_score_weight",
                "flow_graph_mode",
                "segmentation_method",
                "selected_threshold_quantile",
                "selected_threshold_value",
                "threshold_selection_reason",
                "threshold_level_count_total",
                "threshold_component_count",
                "threshold_retained_component_count",
                "threshold_retained_spot_count",
                "activation_threshold_quantile",
                "seg_active_floor",
                "thr_min_domain_spots_effective",
                "thr_min_domain_internal_density_effective",
                "qc_pass",
                "qc_reject_reasons",
                "screening_reject_tags",
                "screening_decision",
                "is_background",
            ]
        )
    return df


def _build_metadomains_table(
    domains_df: pd.DataFrame,
    rsp_score_table: pd.DataFrame,
    sample_id: str,
) -> pd.DataFrame:
    score_df = rsp_score_table.copy()
    if score_df.empty:
        return pd.DataFrame(
            columns=[
                "rsmd_id",
                "sample_id",
                "rsp_id",
                "rsp_score",
                "domain_count_total",
                "domain_count_pass",
                "total_domain_spots",
                "mean_domain_reliability",
                "max_domain_reliability",
            ]
        )

    score_df["rsp_id"] = score_df["rsp_id"].astype(str)
    rows: list[dict] = []
    for idx, row in enumerate(score_df.sort_values("rsp_id").itertuples(index=False), start=0):
        rsp_id = str(getattr(row, "rsp_id"))
        sub = (
            domains_df.loc[domains_df["rsp_seed_id"].astype(str) == rsp_id].copy()
            if (not domains_df.empty and "rsp_seed_id" in domains_df.columns)
            else pd.DataFrame()
        )
        passed = sub.loc[sub["qc_pass"].astype(bool)].copy() if (not sub.empty and "qc_pass" in sub.columns) else sub
        reliability = (
            pd.to_numeric(passed["domain_reliability"], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if (not passed.empty and "domain_reliability" in passed.columns)
            else np.zeros(0, dtype=np.float64)
        )
        rows.append(
            {
                "rsmd_id": _rsmd_id_from_rsp_id(rsp_id, fallback_index=idx),
                "sample_id": str(sample_id),
                "rsp_id": rsp_id,
                "rsp_score": float(getattr(row, "rsp_score", 1.0)),
                "rsp_score_weight": float(getattr(row, "rsp_score_weight", 1.0)),
                "domain_count_total": int(sub.shape[0]),
                "domain_count_pass": int(passed.shape[0]),
                "total_domain_spots": int(pd.to_numeric(passed.get("spot_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                if not passed.empty
                else 0,
                "mean_domain_reliability": float(np.mean(reliability)) if reliability.size else float("nan"),
                "max_domain_reliability": float(np.max(reliability)) if reliability.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _rsmd_id_from_rsp_id(rsp_id: str, fallback_index: int) -> str:
    text = str(rsp_id)
    prefix = "RSP"
    if text.startswith(prefix) and text[len(prefix) :].isdigit():
        return f"RSMD{text[len(prefix):]}"
    split_at = len(text)
    while split_at > 0 and text[split_at - 1].isdigit():
        split_at -= 1
    if split_at < len(text):
        return f"RSMD{text[split_at:]}"
    return f"RSMD{int(fallback_index):04d}"


def run_rsmd_pipeline(
    rsp_out_path: str | Path,
    out_root: str | Path,
    sample_id: str,
    config: RSMDPipelineConfig | None = None,
) -> Path:
    cfg = config or RSMDPipelineConfig()
    set_random_seed(cfg.random_seed)

    rsp_out_path = Path(rsp_out_path)
    out_root = Path(out_root)

    sample_dir = out_root / sample_id
    final_bundle = sample_dir / RSMD_OUTPUT_DIR
    tmp_bundle = sample_dir / f"{RSMD_OUTPUT_DIR}.__tmp__{int(time.time())}_{os.getpid()}"

    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)
    ensure_bundle_dirs(tmp_bundle)

    try:
        payload = load_rsmd_inputs(rsp_out_path=rsp_out_path, cfg=cfg)
        dense_activation = payload["dense_activation"]
        rsp_ids = payload["rsp_ids"]
        rsp_score_info = payload.get("rsp_score_info", {})
        rsp_score_summary = payload.get("rsp_score_summary", {})
        rsp_qc_selection_summary = payload.get("rsp_qc_selection_summary", {})
        spot_ids = payload["spot_ids"]
        n_spots = int(spot_ids.shape[0])
        n_rsps = int(rsp_ids.shape[0])

        candidate_domains, rsp_summary_rows, segmentation_summary = extract_candidate_domains(
            dense_activation=dense_activation,
            rsp_ids=rsp_ids,
            adjacency=payload["adjacency"],
            coords=payload["coords"],
            potential_cfg=cfg.potential,
            filter_cfg=cfg.filter,
            filtration_cfg=cfg.filtration,
            rsp_score_info=rsp_score_info,
            active_mask=None,
            min_domain_spots_override=None,
        )

        refinement = refine_domains(
            candidate_domains=candidate_domains,
            sample_id=sample_id,
            spot_ids=spot_ids,
            cfg=cfg,
        )
        domains_all = refinement["domains_all"]
        accepted_domains = refinement["accepted_domains"]
        final_stage = str(refinement["final_stage"])

        domain_qc_table = _domain_rows_for_parquet(domains_all)
        domain_qc_table, domain_reliability_summary = _attach_domain_reliability(
            domains_df=domain_qc_table,
            cfg=cfg,
        )
        domains_df = (
            domain_qc_table.loc[domain_qc_table["qc_pass"].astype(bool)].reset_index(drop=True)
            if (not domain_qc_table.empty and "qc_pass" in domain_qc_table.columns)
            else domain_qc_table.copy()
        )

        membership_df = build_domain_membership_table(accepted_domains, spot_ids=spot_ids)
        domain_graph_df = build_domain_graph_table(
            domains=accepted_domains,
            spot_edges=payload["spot_edges"],
            cfg=cfg.adjacency,
        )
        metadomains_df = _build_metadomains_table(
            domains_df=domains_df,
            rsp_score_table=payload["rsp_score_table"],
            sample_id=sample_id,
        )
        rsmd_lookup = metadomains_df.set_index("rsp_id")["rsmd_id"].astype(str).to_dict()
        for table in (domains_df, domain_qc_table):
            if "rsmd_id" in table.columns:
                continue
            if "rsp_seed_id" in table.columns:
                table.insert(0, "rsmd_id", table["rsp_seed_id"].astype(str).map(rsmd_lookup).fillna(""))
            else:
                table.insert(0, "rsmd_id", pd.Series(dtype=str))

        write_parquet(metadomains_df, tmp_bundle / "metadomains.parquet")
        write_parquet(domains_df, tmp_bundle / "domains.parquet")
        write_parquet(membership_df, tmp_bundle / "domain_spot_membership.parquet")
        write_parquet(domain_graph_df, tmp_bundle / "domain_graph.parquet")

        domain_rsp_map = domains_df.loc[:, ["rsmd_id", "domain_id", "domain_key", "rsp_seed_id", "qc_pass"]].copy()
        write_parquet(domain_rsp_map, tmp_bundle / "domain_rsp_map.parquet")

        rsp_summary_table = build_rsp_metadomain_summary_table(
            rsp_summary_rows=rsp_summary_rows,
            domains=domains_all,
        )
        write_parquet(domain_qc_table, tmp_bundle / "qc_tables" / "domain_qc.parquet")
        write_parquet(rsp_summary_table, tmp_bundle / "qc_tables" / "rsp_domain_summary.parquet")
        write_parquet(payload["rsp_score_table"], tmp_bundle / "qc_tables" / "rsp_score_table.parquet")

        acceptance_gates = {
            "flow_graph_mode": str(cfg.potential.flow_graph_mode),
            "spatial_graph_k": int(cfg.potential.spatial_graph_k),
            "spatial_graph_max_distance_factor": float(cfg.potential.spatial_graph_max_distance_factor),
            "min_domain_spots": int(cfg.filter.min_domain_spots)
            if cfg.filter.min_domain_spots is not None
            else None,
            "min_domain_spots_frac": float(cfg.filter.min_domain_spots_frac),
            "min_domain_internal_density": float(cfg.filter.min_domain_internal_density),
            "filtration_activation_threshold_quantile": float(cfg.filtration.activation_threshold_quantile),
            "smoothing_alpha": float(cfg.potential.smoothing_alpha),
            "smoothing_steps": int(cfg.potential.smoothing_steps),
            "active_floor_abs": float(cfg.potential.active_floor_abs),
            "active_floor_scale_factor": float(cfg.potential.active_floor_scale_factor),
            "rsp_score_filter_enabled": bool(cfg.input.use_rsp_score_filter),
            "rsp_score_filter_min": float(cfg.input.min_rsp_score),
            "domain_reliability_enabled": bool(cfg.domain_reliability.enabled),
            "domain_reliability_rsp_score_exponent": float(cfg.domain_reliability.rsp_score_exponent),
            "domain_reliability_prominence_exponent": float(cfg.domain_reliability.prominence_exponent),
            "domain_reliability_density_exponent": float(cfg.domain_reliability.density_exponent),
            "domain_reliability_prominence_scale_quantile": float(cfg.domain_reliability.prominence_scale_quantile),
            "domain_reliability_density_scale": float(cfg.domain_reliability.density_scale),
            "domain_reliability_min_node": float(cfg.domain_reliability.min_node_reliability),
            "domain_reliability_max_node": float(cfg.domain_reliability.max_node_reliability),
        }
        qc_report = build_qc_report(
            sample_id=sample_id,
            n_spots=n_spots,
            n_rsps=n_rsps,
            segmentation_summary=segmentation_summary,
            domains=domains_all,
            rsp_summary_table=rsp_summary_table,
            acceptance_gates=acceptance_gates,
        )
        qc_report["final_domain_stage"] = final_stage
        qc_report["rsp_score_summary"] = rsp_score_summary
        qc_report["rsp_qc_selection"] = rsp_qc_selection_summary
        qc_report["domain_reliability"] = domain_reliability_summary
        write_json(tmp_bundle / "qc_report.json", qc_report)

        rsmd_meta = {
            "sample_id": sample_id,
            "rsmd_backend": "rsp_activation_low_quantile_components",
            "rsp_input_schema": "rspbundle.v1",
            "definition": (
                "Each RSMD is instantiated from one Regional Specificity Program "
                "and represented by connected RSD carrier domains obtained after "
                "weak-link pruning with a global activation quantile threshold."
            ),
            "n_spots": n_spots,
            "n_rsps": n_rsps,
            "metadomain_count": int(metadomains_df.shape[0]),
            "spot_order_source": payload["spot_order_source"],
            "spot_ids_hash": hash_array(spot_ids.astype(str)),
            "candidate_domain_count": int(len(domains_all)),
            "accepted_domain_count": int(len(accepted_domains)),
            "flow_graph_mode": str(payload.get("flow_graph_mode", str(cfg.potential.flow_graph_mode))),
            "graph_source": str(payload.get("graph_source", "unknown")),
            "spatial_graph_k": int(cfg.potential.spatial_graph_k),
            "spatial_graph_max_distance_factor": float(cfg.potential.spatial_graph_max_distance_factor),
            "accepted_rsp_count": int(
                np.sum(rsp_summary_table["domain_count_pass"].to_numpy(dtype=np.int32) > 0)
            )
            if ("domain_count_pass" in rsp_summary_table.columns and not rsp_summary_table.empty)
            else 0,
            "domain_graph_edge_count": int(domain_graph_df.shape[0]),
            "domain_adjacency_mode": cfg.adjacency.mode,
            "rsp_score_summary": rsp_score_summary,
            "rsp_qc_selection": rsp_qc_selection_summary,
            "domain_reliability": domain_reliability_summary,
            "filtration": segmentation_summary.get("filtration", {}),
            "final_domain_stage": final_stage,
        }
        write_json(tmp_bundle / "rsmd_meta.json", rsmd_meta)

        rsp_manifest_path = rsp_out_path / cfg.input.rsp_manifest_relpath
        rsp_path = rsp_out_path / cfg.input.rsp_relpath
        activation_path = rsp_out_path / cfg.input.rsp_activation_relpath
        rsp_qc_path = payload["rsp_qc_path"]
        rss_manifest_path = payload["rss_out_path"] / cfg.input.rss_manifest_relpath

        manifest = {
            "schema_version": cfg.schema_version,
            "rsmd_backend": "rsp_activation_low_quantile_components",
            "rsp_input_schema": "rspbundle.v1",
            "created_at": iso_now(),
            "sample_id": sample_id,
            "code_version": get_code_version(
                repo_root=Path(__file__).resolve().parents[1],
                override=cfg.code_version_override,
            ),
            "random_seed": cfg.random_seed,
            "inputs": {
                "rsp_out_path": str(rsp_out_path.resolve()),
                "rss_out_path": str(payload["rss_out_path"].resolve()),
                "rsp_manifest_path": str(rsp_manifest_path.resolve()),
                "rsp_path": str(rsp_path.resolve()),
                "rsp_activation_path": str(activation_path.resolve()),
                "rsp_qc_path": str(rsp_qc_path.resolve()) if rsp_qc_path.exists() else None,
                "rss_manifest_path": str(rss_manifest_path.resolve()),
                "microdomain_idx_path": str(payload["microdomain_idx_path"].resolve()),
                "microdomain_meta_path": str(payload["microdomain_meta_path"].resolve()),
                "rsp_manifest_hash": hash_file(rsp_manifest_path),
                "rsp_hash": hash_file(rsp_path),
                "rsp_activation_hash": hash_file(activation_path),
                "rsp_qc_hash": hash_file(rsp_qc_path) if rsp_qc_path.exists() else None,
                "rss_manifest_hash": hash_file(rss_manifest_path),
                "microdomain_idx_hash": hash_file(payload["microdomain_idx_path"]),
                "n_spots": n_spots,
                "n_rsps": n_rsps,
                "spot_order_hash": hash_array(spot_ids.astype(str)),
                "rsp_score_summary": rsp_score_summary,
                "rsp_qc_selection": rsp_qc_selection_summary,
                "domain_reliability": domain_reliability_summary,
                "filtration": segmentation_summary.get("filtration", {}),
                "final_domain_stage": final_stage,
            },
            "params": asdict(cfg),
            "outputs": {
                "metadomains": "metadomains.parquet",
                "domains": "domains.parquet",
                "domain_spot_membership": "domain_spot_membership.parquet",
                "domain_graph": "domain_graph.parquet",
                "domain_rsp_map": "domain_rsp_map.parquet",
                "rsmd_meta": "rsmd_meta.json",
                "qc_report": "qc_report.json",
                "qc_domain_table": "qc_tables/domain_qc.parquet",
                "qc_rsp_domain_summary": "qc_tables/rsp_domain_summary.parquet",
                "qc_rsp_score_table": "qc_tables/rsp_score_table.parquet",
            },
            "timestamps": {"finished_at": iso_now()},
        }
        write_json(tmp_bundle / "manifest.json", manifest)

        promote_bundle(tmp_bundle, final_bundle)
        logger.info("RSMD bundle generated at %s", final_bundle)
        return final_bundle
    except Exception:
        if tmp_bundle.exists():
            shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
