from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/regal_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/regal_mplconfig")

import numpy as np
import pandas as pd
import scanpy as sc

try:
    from .schema import (
        LatentConfig,
        NeighborsConfig,
        PreprocessConfig,
        QCConfig,
        RSSConfig,
        RSSPipelineConfig,
    )
    from .bundle_io import (
        ensure_bundle_dirs,
        get_code_version,
        hash_file,
        iso_now,
        promote_bundle,
        set_random_seed,
        write_json,
        write_parquet,
    )
    from .neighbors import build_neighbors, build_spatial_candidates
    from .metrics import compute_neighbor_qc, compute_stability_qc
    from .latent import compute_latent, prepare_expression_and_latent_input, resolve_spot_ids
    from .rss_compute import compute_and_sparsify_rss
except ImportError:
    from RSS.schema import (
        LatentConfig,
        NeighborsConfig,
        PreprocessConfig,
        QCConfig,
        RSSConfig,
        RSSPipelineConfig,
    )
    from RSS.bundle_io import (
        ensure_bundle_dirs,
        get_code_version,
        hash_file,
        iso_now,
        promote_bundle,
        set_random_seed,
        write_json,
        write_parquet,
    )
    from RSS.neighbors import build_neighbors, build_spatial_candidates
    from RSS.metrics import compute_neighbor_qc, compute_stability_qc
    from RSS.latent import (
        compute_latent,
        prepare_expression_and_latent_input,
        resolve_spot_ids,
    )
    from RSS.rss_compute import compute_and_sparsify_rss

logger = logging.getLogger(__name__)

RSS_OUTPUT_DIR = "rss_out"

__all__ = [
    "PreprocessConfig",
    "LatentConfig",
    "NeighborsConfig",
    "RSSConfig",
    "QCConfig",
    "RSSPipelineConfig",
    "RSS_OUTPUT_DIR",
    "run_rss_pipeline",
]


def run_rss_pipeline(
    h5ad_path: str | Path,
    out_root: str | Path,
    sample_id: str,
    config: RSSPipelineConfig | None = None,
) -> Path:
    cfg = config or RSSPipelineConfig()
    set_random_seed(cfg.random_seed)

    h5ad_path = Path(h5ad_path)
    out_root = Path(out_root)
    sample_dir = out_root / sample_id
    final_bundle = sample_dir / RSS_OUTPUT_DIR
    tmp_bundle = sample_dir / f"{RSS_OUTPUT_DIR}.__tmp__{int(time.time())}_{os.getpid()}"

    if tmp_bundle.exists():
        shutil.rmtree(tmp_bundle)

    ensure_bundle_dirs(tmp_bundle)

    try:
        adata = sc.read_h5ad(h5ad_path)
        n_spots_raw = int(adata.n_obs)
        n_genes_raw = int(adata.n_vars)
        adata.var_names_make_unique()
        sc.pp.filter_genes(adata, min_cells=cfg.preprocess.min_cells)

        spot_ids, spot_id_field, spot_id_generated = resolve_spot_ids(
            adata,
            cfg.preprocess.spot_id_field,
        )
        expression, latent_input, hvg_mask = prepare_expression_and_latent_input(
            adata,
            cfg.preprocess,
        )
        gene_names = adata.var_names.to_numpy().astype(str)

        latent, latent_meta = compute_latent(adata, latent_input, cfg.latent, cfg.random_seed)
        np.save(tmp_bundle / "latent" / "latent.embedding.npy", latent)

        if "spatial" not in adata.obsm:
            raise ValueError("Missing adata.obsm['spatial']; required for MicroDomain construction.")

        candidate_idx, candidate_dist, spatial_adj = build_spatial_candidates(
            adata.obsm["spatial"],
            cfg.neighbors,
        )

        microdomain_idx, microdomain_sim, microdomain_weight, microdomain_meta, connectivity_sets = build_neighbors(
            latent,
            candidate_idx,
            candidate_dist,
            spatial_adj,
            cfg.neighbors,
        )
        np.save(tmp_bundle / "microdomains" / "microdomain.idx.npy", microdomain_idx)
        np.save(tmp_bundle / "microdomains" / "microdomain.sim.npy", microdomain_sim)
        np.save(tmp_bundle / "microdomains" / "microdomain.weight.npy", microdomain_weight)

        rss_payload = compute_and_sparsify_rss(
            expression=expression,
            gene_names=gene_names,
            spot_ids=spot_ids,
            neighbor_idx=microdomain_idx,
            rss_cfg=cfg.rss,
            remove_mt=cfg.preprocess.remove_mt_genes,
            top_n_for_qc=cfg.qc.bootstrap_top_n,
            neighbor_weight=microdomain_weight,
        )
        write_parquet(
            rss_payload["rss_sparse_df"],
            tmp_bundle / "rss" / "rss_sparse.parquet",
        )

        neighbor_qc_payload = compute_neighbor_qc(
            coords=adata.obsm["spatial"],
            neighbor_idx=microdomain_idx,
            neighbor_sim=microdomain_sim,
            connectivity_sets=connectivity_sets,
            radius=cfg.neighbors.spatial_radius,
            return_per_spot=True,
        )
        neighbor_qc = neighbor_qc_payload["summary"]
        neighbor_per_spot = neighbor_qc_payload["per_spot"]

        stability_qc = compute_stability_qc(
            latent=latent,
            expression=expression,
            base_neighbor_idx=microdomain_idx,
            base_top_sets=rss_payload["top_sets"],
            candidate_idx=candidate_idx,
            candidate_dist=candidate_dist,
            spatial_adj=spatial_adj,
            neighbors_cfg=cfg.neighbors,
            rss_cfg=cfg.rss,
            top_ns=cfg.qc.bootstrap_top_n,
            repeats=cfg.qc.bootstrap_repeats,
            noise_std=cfg.qc.bootstrap_latent_noise_std,
            seed=cfg.random_seed,
            remove_mt=cfg.preprocess.remove_mt_genes,
            gene_names=gene_names,
            return_per_spot=True,
        )
        stability_per_spot = stability_qc.pop("per_spot")

        microdomain_per_spot_df = pd.DataFrame(
            {
                "spot_id": spot_ids,
                "microdomain_locality": neighbor_per_spot["neighbor_locality"],
                "microdomain_homogeneity_mean": neighbor_per_spot["neighbor_homogeneity_mean"],
                "microdomain_homogeneity_var": neighbor_per_spot["neighbor_homogeneity_var"],
                "microdomain_connectivity": neighbor_per_spot["neighbor_connectivity"],
                "within_radius_ratio": neighbor_per_spot["within_radius_ratio"],
            }
        )
        write_parquet(
            microdomain_per_spot_df,
            tmp_bundle / "qc" / "qc_tables" / "microdomain_per_spot.parquet",
        )

        stability_table_payload = {
            "spot_id": spot_ids,
            "microdomain_stability_jaccard": stability_per_spot["neighbor_stability_jaccard"],
        }
        for top_n, values in stability_per_spot["rss_topN_stability_jaccard"].items():
            stability_table_payload[f"rss_top{top_n}_stability_jaccard"] = values
        stability_per_spot_df = pd.DataFrame(stability_table_payload)
        write_parquet(
            stability_per_spot_df,
            tmp_bundle / "qc" / "qc_tables" / "stability_per_spot.parquet",
        )

        qc_report = {
            "sample_id": sample_id,
            "summary": {
                "n_spots": int(adata.n_obs),
                "n_genes": int(expression.shape[1]),
                "bootstrap_repeats": cfg.qc.bootstrap_repeats,
            },
            "microdomain_quality": neighbor_qc,
            "microdomain_stability_jaccard": stability_qc["neighbor_stability_jaccard"],
            "rss_topN_stability_jaccard": stability_qc["rss_topN_stability_jaccard"],
            "rss_distribution": rss_payload["distribution"],
            "acceptance": {
                "microdomain_stability_p50_ge": cfg.qc.neighbor_stability_gate_p50,
                "microdomain_stability_p50": stability_qc["neighbor_stability_jaccard"][
                    "quantiles"
                ]["p50"],
                "microdomain_stability_pass": stability_qc["neighbor_stability_jaccard"][
                    "quantiles"
                ]["p50"]
                >= cfg.qc.neighbor_stability_gate_p50,
                "rss_top100_stability_p50_ge": cfg.qc.rss_stability_gate_p50,
                "rss_top100_stability_p50": stability_qc["rss_topN_stability_jaccard"]
                .get("100", {})
                .get("quantiles", {})
                .get("p50", 0.0),
                "rss_top100_stability_pass": stability_qc["rss_topN_stability_jaccard"]
                .get("100", {})
                .get("quantiles", {})
                .get("p50", 0.0)
                >= cfg.qc.rss_stability_gate_p50,
            },
        }

        latent_meta.update(
            {
                "shape": [int(latent.shape[0]), int(latent.shape[1])],
                "hvg_used": bool(cfg.preprocess.use_hvg_for_latent),
                "hvg_count": int(np.count_nonzero(hvg_mask))
                if hvg_mask is not None
                else int(expression.shape[1]),
                "random_seed": cfg.random_seed,
            }
        )
        microdomain_meta.update(
            {
                "shape": [int(microdomain_idx.shape[0]), int(microdomain_idx.shape[1])],
                "weight_shape": [
                    int(microdomain_weight.shape[0]),
                    int(microdomain_weight.shape[1]),
                ],
                "spot_id_field": spot_id_field,
                "spot_id_generated": spot_id_generated,
                "semantics": (
                    "spot-anchored weighted MicroDomains used internally for RSS calculation"
                ),
            }
        )

        rss_meta = {
            "score_name": "Regional Specificity Score",
            "score_abbreviation": "RSS",
            "comparison": "microdomain_vs_global",
            "projection": cfg.rss.projection,
            "sparsify_rule": cfg.rss.sparsify_rule,
            "top_m": int(cfg.rss.top_m),
            "score_col": "rss_score",
            "rank_col": "rss_rank_in_spot",
            "raw_col": "rss_raw",
            "support_col": "rss_support",
            "sparse": {
                "path": "rss/rss_sparse.parquet",
            },
            "remove_mt_genes": cfg.preprocess.remove_mt_genes,
            "spot_count": int(adata.n_obs),
            "gene_count": int(expression.shape[1]),
            "rss_sparse_rows": int(rss_payload["rss_sparse_df"].shape[0]),
            "rss_columns": list(rss_payload["rss_sparse_df"].columns),
            "distribution": rss_payload["distribution"],
        }

        write_json(tmp_bundle / "latent" / "latent_meta.json", latent_meta)
        write_json(tmp_bundle / "microdomains" / "microdomain_meta.json", microdomain_meta)
        write_json(tmp_bundle / "rss" / "rss_meta.json", rss_meta)
        write_json(tmp_bundle / "qc" / "qc_report.json", qc_report)

        manifest = {
            "schema_version": cfg.schema_version,
            "created_at": iso_now(),
            "sample_id": sample_id,
            "code_version": get_code_version(
                repo_root=Path(__file__).resolve().parents[1],
                override=cfg.code_version_override,
            ),
            "random_seed": cfg.random_seed,
            "inputs": {
                "h5ad_path": str(h5ad_path.resolve()),
                "n_spots": n_spots_raw,
                "n_genes": n_genes_raw,
                "gene_id_type": cfg.preprocess.gene_id_type,
                "spot_id_field": spot_id_field,
                "input_hash": hash_file(h5ad_path),
            },
            "params": asdict(cfg),
            "outputs": {
                "latent": "latent/latent.embedding.npy",
                "latent_meta": "latent/latent_meta.json",
                "microdomain_idx": "microdomains/microdomain.idx.npy",
                "microdomain_sim": "microdomains/microdomain.sim.npy",
                "microdomain_weight": "microdomains/microdomain.weight.npy",
                "microdomain_meta": "microdomains/microdomain_meta.json",
                "rss_sparse": "rss/rss_sparse.parquet",
                "rss_meta": "rss/rss_meta.json",
                "qc_report": "qc/qc_report.json",
                "qc_microdomain_per_spot": "qc/qc_tables/microdomain_per_spot.parquet",
                "qc_stability_per_spot": "qc/qc_tables/stability_per_spot.parquet",
            },
            "timestamps": {
                "finished_at": iso_now(),
            },
        }
        write_json(tmp_bundle / "manifest.json", manifest)

        promote_bundle(tmp_bundle, final_bundle)
        logger.info("RSS bundle generated at %s", final_bundle)
        return final_bundle
    except Exception:
        if tmp_bundle.exists():
            shutil.rmtree(tmp_bundle, ignore_errors=True)
        raise
