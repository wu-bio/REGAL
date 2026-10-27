#!/usr/bin/env python
"""Run official STAMP and save raw topics plus spatial active regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


RUNTIME_CACHE = Path(tempfile.gettempdir()) / "stamp_runtime_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
import torch

import sctm
from sctm.seed import seed_everything
from sctm.stamp import STAMP


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
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/STAMP"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn STAMP spot-topic proportions and gene modules using the "
            "official breast-cancer tutorial workflow."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <STAMP root>/K<k>).",
    )
    parser.add_argument("--spatial-key", default="spatial")

    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of topics (default: 20, as used in the official BREAST tutorial).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-genes", type=int, default=50)
    parser.add_argument("--min-detection-rate", type=float, default=0.03)
    parser.add_argument("--expression-cutoff-99q", type=float, default=1.0)
    parser.add_argument("--n-top-genes", type=int, default=6000)
    parser.add_argument("--n-layers", type=int, default=1)

    parser.add_argument("--max-epochs", type=int, default=800)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Training device; auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable STAMP's default early stopping.",
    )
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
    if args.n_layers < 1:
        parser.error("--n-layers must be at least 1 for spatial STAMP.")
    if args.max_epochs < 1 or args.min_epochs < 0:
        parser.error("Epoch counts must be non-negative and --max-epochs must be positive.")
    if args.min_epochs >= args.max_epochs and not args.no_early_stop:
        parser.error("--min-epochs must be smaller than --max-epochs when early stopping is enabled.")
    if args.learning_rate <= 0 or args.batch_size < 1 or args.patience < 1:
        parser.error("Learning rate, batch size, and patience must be positive.")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"K{args.k}"
    return args


def package_version(name: str) -> str:
    try:
        return version(name)
    except Exception:
        return "unknown"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda:0"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return requested


def validate_raw_counts(adata: sc.AnnData, spatial_key: str) -> None:
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError("STAMP requires at least two spots and two genes.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}].")
    coordinates = np.asarray(adata.obsm[spatial_key])
    if coordinates.ndim != 2 or coordinates.shape[0] != adata.n_obs or coordinates.shape[1] < 2:
        raise ValueError(f"Invalid spatial coordinate matrix: {coordinates.shape}.")
    if not np.isfinite(coordinates[:, :2]).all():
        raise ValueError("Spatial coordinates contain non-finite values.")

    values = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X)
    if values.size and (not np.isfinite(values).all() or np.min(values) < 0):
        raise ValueError("Expression matrix contains non-finite or negative values.")
    if values.size and not np.allclose(values, np.rint(values), atol=1e-8):
        raise ValueError("adata.X is not integer-like; STAMP must receive raw counts.")


def prepare_input(adata: sc.AnnData, args: argparse.Namespace) -> sc.AnnData:
    adata.var_names_make_unique()
    if args.spatial_key != "spatial":
        adata.obsm["spatial"] = np.asarray(adata.obsm[args.spatial_key]).copy()

    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sctm.pp.filter_genes(
        adata,
        min_cutoff=args.min_detection_rate,
        expression_cutoff_99q=args.expression_cutoff_99q,
    )
    if adata.n_vars <= args.k:
        raise ValueError("Too few genes remain after native STAMP filtering.")

    n_top_genes = min(args.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
    )
    sq.gr.spatial_neighbors(adata)
    if "spatial_connectivities" not in adata.obsp or adata.obsp["spatial_connectivities"].nnz == 0:
        raise RuntimeError("Squidpy produced an empty spatial graph.")
    return adata[:, adata.var["highly_variable"].to_numpy()].copy()


def validate_outputs(
    topics: pd.DataFrame,
    modules: pd.DataFrame,
    n_spots: int,
    n_genes: int,
    k: int,
) -> None:
    if topics.shape != (n_spots, k):
        raise RuntimeError(f"Unexpected spot-topic shape: {topics.shape}.")
    if modules.shape != (n_genes, k):
        raise RuntimeError(f"Unexpected gene-module shape: {modules.shape}.")
    if not np.isfinite(topics.to_numpy()).all() or not np.isfinite(modules.to_numpy()).all():
        raise RuntimeError("STAMP returned non-finite output values.")
    if (topics.to_numpy() < 0).any():
        raise RuntimeError("STAMP returned negative topic proportions.")
    if not np.allclose(topics.sum(axis=1).to_numpy(), 1.0, atol=1e-5):
        raise RuntimeError("STAMP spot-topic proportions do not sum to one.")


def spatial_activation(
    weights: pd.DataFrame, spatial: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reuse RSP spatial cleanup, but use only the base coverage threshold.

    STAMP deliberately skips the adaptive elbow threshold; RSP stays unchanged.

    This transfers only per-component spatial postprocessing. It does not
    apply RSS, RSP gene selection, cross-run stability or component merging.
    Neither counts, gene loadings nor the fitted STAMP weights are changed.
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
        "raw_weights_obsm": "STAMP_topics", "activation_obsm": "STAMP_activation",
        "coverage_obsm": "STAMP_activation_active",
    }
    return table, clean, activation, active, parameters


def run_stamp(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input h5ad does not exist: {args.input}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing STAMP outputs: {args.output_dir}")

    device = resolve_device(args.device)
    seed_everything(args.seed)
    adata = sc.read_h5ad(args.input)
    n_spots_input, n_genes_input = adata.shape
    validate_raw_counts(adata, args.spatial_key)
    model_adata = prepare_input(adata, args)

    print(
        f"Prepared {model_adata.n_obs} spots and {model_adata.n_vars} HVGs; "
        f"training STAMP K={args.k} on {device}."
    )
    model = STAMP(model_adata, n_topics=args.k, n_layers=args.n_layers)
    model.train(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        learning_rate=args.learning_rate,
        device=device,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        early_stop=not args.no_early_stop,
        patience=args.patience,
    )
    topics = model.get_cell_by_topic(device=device)
    modules = model.get_feature_by_topic(device="cpu")
    validate_outputs(topics, modules, model_adata.n_obs, model_adata.n_vars, args.k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topics_path = args.output_dir / "stamp_spot_topics.csv"
    modules_path = args.output_dir / "stamp_gene_modules.csv"
    h5ad_path = args.output_dir / "stamp.h5ad"
    model_path = args.output_dir / "stamp_model_params.pt"
    metadata_path = args.output_dir / "run_metadata.json"

    activation_path = args.output_dir / "stamp_activation.csv"
    aligned_topics = topics.loc[model_adata.obs_names]
    activation_table, clean, activation, active, spatial_parameters = spatial_activation(
        aligned_topics, np.asarray(model_adata.obsm["spatial"]),
    )
    activation_table.to_csv(activation_path, index=False)
    topics.rename_axis("spot_id").to_csv(topics_path)
    modules.rename_axis("gene").to_csv(modules_path)
    model_adata.obsm["STAMP_topics"] = topics.loc[model_adata.obs_names].to_numpy()
    model_adata.obsm["STAMP_activation_clean"] = clean
    model_adata.obsm["STAMP_activation"] = activation
    model_adata.obsm["STAMP_activation_active"] = active
    model_adata.varm["STAMP_gene_modules"] = modules.loc[model_adata.var_names, topics.columns].to_numpy()
    model_adata.uns["STAMP_spatial_postprocessing"] = spatial_parameters
    model_adata.uns["STAMP_topic_names"] = topics.columns.to_numpy(dtype=str)
    model_adata.write_h5ad(h5ad_path)
    model.save(model_path)

    metadata = {
        "input": str(args.input.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/JinmiaoChenLab/scTM",
        "workflow": "official_STAMP_breast_cancer_tutorial",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sctm_distribution_version": package_version("sctm"),
        "torch": str(torch.__version__),
        "scanpy": package_version("scanpy"),
        "squidpy": package_version("squidpy"),
        "device": device,
        "spatial_key": args.spatial_key,
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
        "spatial_graph": "squidpy.gr.spatial_neighbors defaults",
        "n_layers": args.n_layers,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "early_stop": not args.no_early_stop,
        "patience": args.patience,
        "spatial_postprocessing": spatial_parameters,
        "outputs": {
            "activation": str(activation_path),
            "spot_topics": str(topics_path),
            "gene_modules": str(modules_path),
            "h5ad": str(h5ad_path),
            "model_parameters": str(model_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved STAMP K={args.k} outputs to {args.output_dir}.")


def main() -> None:
    run_stamp(parse_args())


if __name__ == "__main__":
    main()
