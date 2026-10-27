#!/usr/bin/env python
"""Fit vanilla NMF and save raw weights plus REGAL-style spatial active regions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


RUNTIME_CACHE = Path(tempfile.gettempdir()) / "nmf_runtime_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import NMF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RSP.common import build_spatial_neighbors, _activation_noise_floor, _filter_active_components
from RSP.io import _trim_activation_for_output
from RSP.schema import RSPDiscoveryConfig, RSPScoringConfig, RSPStabilityConfig
from RSP.stability import _prepare_candidate_for_stability


DEFAULT_INPUT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/NMF"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <NMF root>/K<k>).",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-genes", type=int, default=50)
    parser.add_argument("--min-detection-rate", type=float, default=0.03)
    parser.add_argument("--expression-cutoff-99q", type=float, default=1.0)
    parser.add_argument("--n-top-genes", type=int, default=6000)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    if args.k < 2:
        parser.error("--k must be at least 2.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.min_genes < 1:
        parser.error("--min-genes must be positive.")
    if not 0 < args.min_detection_rate <= 1:
        parser.error("--min-detection-rate must be in (0, 1].")
    if args.expression_cutoff_99q < 0:
        parser.error("--expression-cutoff-99q must be non-negative.")
    if args.n_top_genes < 2:
        parser.error("--n-top-genes must be at least 2.")
    if args.target_sum <= 0:
        parser.error("--target-sum must be positive.")
    if args.max_iter < 1 or args.tol <= 0:
        parser.error("--max-iter and --tol must be positive.")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"K{args.k}"
    return args


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "unknown"


def validate_raw_counts(adata: sc.AnnData) -> None:
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError("NMF requires at least two spots and two genes.")
    values = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X)
    if values.size and (not np.isfinite(values).all() or np.min(values) < 0):
        raise ValueError("Expression matrix contains non-finite or negative values.")
    if values.size and not np.allclose(values, np.rint(values), atol=1e-8):
        raise ValueError("adata.X is not integer-like; STAMP-style preprocessing expects raw counts.")


def filter_genes_like_stamp(
    adata: sc.AnnData,
    min_detection_rate: float,
    expression_cutoff_99q: float,
) -> None:
    """Reproduce sctm.pp.filter_genes without installing the STAMP stack."""
    n_spots = adata.n_obs
    min_cells = round(n_spots * min_detection_rate)
    if sp.issparse(adata.X):
        matrix = adata.X.tocsr(copy=True)
        matrix.eliminate_zeros()
        detected = np.asarray(matrix.getnnz(axis=0)).ravel()
        dense = matrix.toarray()
    else:
        dense = np.asarray(adata.X)
        detected = np.count_nonzero(dense, axis=0)

    keep = detected >= min_cells
    if expression_cutoff_99q > 0:
        keep &= np.quantile(dense, q=0.99, axis=0) > expression_cutoff_99q
    del dense
    adata._inplace_subset_var(keep)


def prepare_input(adata: sc.AnnData, args: argparse.Namespace) -> sc.AnnData:
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    filter_genes_like_stamp(
        adata,
        min_detection_rate=args.min_detection_rate,
        expression_cutoff_99q=args.expression_cutoff_99q,
    )
    if adata.n_vars <= args.k:
        raise ValueError("Too few genes remain after STAMP-style filtering.")

    n_top_genes = min(args.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
    )
    return adata[:, adata.var["highly_variable"].to_numpy()].copy()


def validate_outputs(
    weights: pd.DataFrame,
    loadings: pd.DataFrame,
    n_spots: int,
    n_genes: int,
    k: int,
) -> None:
    if weights.shape != (n_spots, k):
        raise RuntimeError(f"Unexpected spot-weight shape: {weights.shape}.")
    if loadings.shape != (n_genes, k):
        raise RuntimeError(f"Unexpected gene-loading shape: {loadings.shape}.")
    if not np.isfinite(weights.to_numpy()).all() or not np.isfinite(loadings.to_numpy()).all():
        raise RuntimeError("NMF returned non-finite output values.")
    if (weights.to_numpy() < 0).any() or (loadings.to_numpy() < 0).any():
        raise RuntimeError("NMF returned negative output values.")


def spatial_activation(
    weights: pd.DataFrame, spatial: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reuse RSP spatial cleanup, but use only the base coverage threshold.

    NMF deliberately skips the adaptive elbow threshold; RSP stays unchanged.

    This transfers only per-component spatial postprocessing. It does not
    apply RSS, RSP gene selection, cross-run stability or component merging.
    Neither counts, gene loadings nor the fitted NMF weights are changed.
    """
    spatial = np.asarray(spatial, dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[0] != len(weights) or spatial.shape[1] < 2:
        raise ValueError("Spatial postprocessing requires spot-aligned coordinates with shape (n_spots, >=2).")
    if not np.isfinite(spatial[:, :2]).all():
        raise ValueError("Spatial coordinates must be finite.")
    discovery = RSPDiscoveryConfig()
    stability, scoring = RSPStabilityConfig(), RSPScoringConfig()
    candidate_neighbors = build_spatial_neighbors(spatial, discovery.moran_knn)
    output_neighbors = build_spatial_neighbors(spatial, discovery.spatial_knn)
    candidates = []
    for component in weights.columns:
        raw = weights[component].to_numpy(dtype=np.float32, copy=True)
        candidate = {"rsp_id": str(component), "activation_raw": raw, "activation": raw.copy()}
        _prepare_candidate_for_stability(candidate, stability, candidate_neighbors)
        candidates.append(candidate)
    clean = np.column_stack([c["activation_clean"] for c in candidates])
    tables = []
    for candidate in candidates:
        values = _trim_activation_for_output(candidate["activation_clean"], scoring, output_neighbors)
        peak = float(values.max()) if values.size else 0.0
        threshold = max(scoring.activation_active_min_abs,
                        scoring.activation_active_min_peak_fraction * peak,
                        _activation_noise_floor(values, scoring.activation_active_quantile))
        mask = _filter_active_components(
            (values > 0) & (values >= threshold), output_neighbors,
            min_component_size=scoring.activation_active_min_component_size,
        )
        candidate["activation"] = values
        candidate["activation_active_mask"] = mask
        order = np.argsort(-values)
        ranks = np.empty(len(values), dtype=np.int32)
        ranks[order] = np.arange(1, len(values) + 1)
        active_order = order[mask[order]]
        active_ranks = np.zeros(len(values), dtype=np.int32)
        active_ranks[active_order] = np.arange(1, len(active_order) + 1)
        tables.append(pd.DataFrame({
            "component_id": str(candidate["rsp_id"]),
            "spot_id": weights.index.to_numpy(dtype=str),
            "activation_raw": candidate["activation_raw"], "activation": values,
            "rank_in_spot": ranks, "activation_active": mask,
            "activation_active_rank": active_ranks,
            "activation_active_threshold": threshold,
        }))
    table = pd.concat(tables, ignore_index=True)
    activation = np.column_stack([c["activation"] for c in candidates])
    active = np.column_stack([c["activation_active_mask"] for c in candidates])
    assert activation.shape == weights.shape and active.shape == weights.shape
    assert active.dtype == bool and np.all(activation[active] > 0)
    assert np.isfinite(activation).all() and np.all(activation >= 0)
    assert np.all(activation <= weights.to_numpy(dtype=np.float32))
    parameters = {
        "method": "RSP_cleanup_with_base_threshold_only",
        "activation_active_elbow_enabled": False,
        "active_threshold_rule": "max(min_abs, min_peak_fraction * peak, positive_quantile)",
        "spatial_knn": discovery.spatial_knn, "moran_knn": discovery.moran_knn,
        "candidate_cleanup": {k: v for k, v in asdict(stability).items() if k.startswith("activation_cleanup_")},
        "output_and_active_region": {k: v for k, v in asdict(scoring).items()
                                    if k.startswith(("activation_trim_", "activation_active_"))},
        "uses_truth": False,
        "coverage_field": "activation_active",
        "raw_weights_obsm": "NMF_weights", "activation_obsm": "NMF_activation",
        "coverage_obsm": "NMF_activation_active",
    }
    return table, clean, activation, active, parameters


def run_nmf(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input h5ad does not exist: {args.input}")

    adata = sc.read_h5ad(args.input)
    n_spots_input, n_genes_input = adata.shape
    validate_raw_counts(adata)
    model_adata = prepare_input(adata, args)
    if "spatial" not in model_adata.obsm:
        raise ValueError("Missing obsm['spatial']; required to save spatial NMF coverage.")
    sc.pp.normalize_total(model_adata, target_sum=args.target_sum)
    sc.pp.log1p(model_adata)
    matrix = model_adata.X.tocsr() if sp.issparse(model_adata.X) else np.asarray(model_adata.X)
    matrix = matrix.astype(np.float64, copy=False)

    print(
        f"Prepared {model_adata.n_obs} spots and {model_adata.n_vars} HVGs; "
        f"fitting vanilla NMF K={args.k} after normalize_total + log1p."
    )
    model = NMF(
        n_components=args.k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        tol=args.tol,
        max_iter=args.max_iter,
        random_state=args.seed,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        shuffle=False,
    )
    weight_values = model.fit_transform(matrix)
    loading_values = model.components_.T
    names = [f"NMF_{index:02d}" for index in range(1, args.k + 1)]
    weights = pd.DataFrame(weight_values, index=model_adata.obs_names, columns=names)
    loadings = pd.DataFrame(loading_values, index=model_adata.var_names, columns=names)
    validate_outputs(weights, loadings, model_adata.n_obs, model_adata.n_vars, args.k)
    activation_table, clean, activation, active, spatial_parameters = spatial_activation(
        weights, model_adata.obsm["spatial"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "nmf_spot_weights.csv"
    loadings_path = args.output_dir / "nmf_gene_loadings.csv"
    h5ad_path = args.output_dir / "nmf.h5ad"
    model_path = args.output_dir / "nmf_model.joblib"
    metadata_path = args.output_dir / "run_metadata.json"
    activation_path = args.output_dir / "nmf_activation.csv"

    weights.rename_axis("spot_id").to_csv(weights_path)
    loadings.rename_axis("gene").to_csv(loadings_path)
    activation_table.to_csv(activation_path, index=False)
    model_adata.obsm["NMF_weights"] = weights.to_numpy()
    model_adata.obsm["NMF_activation_clean"] = clean
    model_adata.obsm["NMF_activation"] = activation
    model_adata.obsm["NMF_activation_active"] = active
    model_adata.varm["NMF_loadings"] = loadings.to_numpy()
    model_adata.uns["NMF_component_names"] = np.asarray(names, dtype=str)
    model_adata.uns["NMF_spatial_postprocessing"] = spatial_parameters
    model_adata.write_h5ad(h5ad_path)
    joblib.dump(model, model_path)

    metadata = {
        "input": str(args.input.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "vanilla_nmf_after_stamp_style_preprocessing",
        "python": platform.python_version(),
        "scanpy": package_version("scanpy"),
        "scikit_learn": package_version("scikit-learn"),
        "scikit_misc": package_version("scikit-misc"),
        "k": args.k,
        "seed": args.seed,
        "n_spots_input": n_spots_input,
        "n_genes_input": n_genes_input,
        "n_spots_analyzed": model_adata.n_obs,
        "n_genes_analyzed": model_adata.n_vars,
        "min_genes": args.min_genes,
        "min_detection_rate": args.min_detection_rate,
        "expression_cutoff_99q": args.expression_cutoff_99q,
        "hvg_flavor": "seurat_v3",
        "n_top_genes": args.n_top_genes,
        "matrix_transform": "normalize_total_then_log1p_after_hvg_selection",
        "normalization_target_sum": args.target_sum,
        "nmf": {
            "implementation": "sklearn.decomposition.NMF",
            "init": "nndsvda",
            "solver": "cd",
            "beta_loss": "frobenius",
            "tol": args.tol,
            "max_iter": args.max_iter,
            "n_iter": int(model.n_iter_),
            "reconstruction_error": float(model.reconstruction_err_),
            "regularization": "none",
        },
        "spatial_information_used": True,
        "spatial_information_used_for_fitting": False,
        "spatial_information_used_for_postprocessing": True,
        "spatial_postprocessing": spatial_parameters,
        "outputs": {
            "spot_weights": str(weights_path),
            "gene_loadings": str(loadings_path),
            "h5ad": str(h5ad_path),
            "model": str(model_path),
            "activation": str(activation_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Saved NMF K={args.k} outputs to {args.output_dir}; "
        f"iterations={model.n_iter_}, reconstruction_error={model.reconstruction_err_:.3f}."
    )


def main() -> None:
    run_nmf(parse_args())


if __name__ == "__main__":
    main()
