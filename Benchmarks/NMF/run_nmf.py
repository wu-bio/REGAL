#!/usr/bin/env python
"""Run vanilla NMF after STAMP-style filtering and standard log normalization."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
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


def run_nmf(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input h5ad does not exist: {args.input}")

    adata = sc.read_h5ad(args.input)
    n_spots_input, n_genes_input = adata.shape
    validate_raw_counts(adata)
    model_adata = prepare_input(adata, args)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "nmf_spot_weights.csv"
    loadings_path = args.output_dir / "nmf_gene_loadings.csv"
    h5ad_path = args.output_dir / "nmf.h5ad"
    model_path = args.output_dir / "nmf_model.joblib"
    metadata_path = args.output_dir / "run_metadata.json"

    weights.rename_axis("spot_id").to_csv(weights_path)
    loadings.rename_axis("gene").to_csv(loadings_path)
    model_adata.obsm["NMF_weights"] = weights.to_numpy()
    model_adata.varm["NMF_loadings"] = loadings.to_numpy()
    model_adata.uns["NMF_component_names"] = np.asarray(names, dtype=str)
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
        "spatial_information_used": False,
        "outputs": {
            "spot_weights": str(weights_path),
            "gene_loadings": str(loadings_path),
            "h5ad": str(h5ad_path),
            "model": str(model_path),
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
