#!/usr/bin/env python
"""Run the official STAGATE_pyG model through unsupervised spatial partitioning."""

from __future__ import annotations

import argparse
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "stagate_runtime_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_CACHE / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import scanpy as sc
import torch
import STAGATE_pyG


STAGATE_SOURCE_COMMIT = "ae1158ca8cf1eb6bb8ee198298552d44c9ac21db"
DEFAULT_INPUT = Path("/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad")
DEFAULT_OUTPUT = Path("/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/STAGATE/stagate.h5ad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate STAGATE spot embeddings and unsupervised spatial domains from an h5ad file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input h5ad containing counts and spatial coordinates (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output h5ad with STAGATE embedding and domains (default: %(default)s).",
    )
    parser.add_argument("--spatial-key", default="spatial", help="Key in adata.obsm containing spatial coordinates.")
    parser.add_argument("--n-top-genes", type=int, default=3000, help="Number of highly variable genes.")
    parser.add_argument("--rad-cutoff", type=float, default=300.0, help="Radius cutoff for the STAGATE spatial graph.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        default=(512, 30),
        metavar=("HIDDEN", "LATENT"),
        help="Hidden and latent dimensions of the STAGATE encoder.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of STAGATE training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Adam weight decay.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors used to cluster the STAGATE embedding.")
    parser.add_argument("--resolution", type=float, default=0.5, help="Leiden resolution for spatial domains.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for STAGATE and Leiden clustering.")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu", help="Torch training device.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available. Use --device cpu on this Mac.")
    return torch.device(device_name)


def validate_input(adata: sc.AnnData, spatial_key: str, n_top_genes: int) -> None:
    if adata.n_obs < 2:
        raise ValueError("STAGATE requires at least two spots.")
    if adata.n_vars < 2:
        raise ValueError("STAGATE requires at least two genes.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}].")

    spatial = np.asarray(adata.obsm[spatial_key])
    if spatial.ndim != 2 or spatial.shape[0] != adata.n_obs or spatial.shape[1] < 2:
        raise ValueError(f"adata.obsm[{spatial_key!r}] must have shape (n_spots, >=2); got {spatial.shape}.")
    if not np.isfinite(spatial[:, :2]).all():
        raise ValueError(f"adata.obsm[{spatial_key!r}] contains non-finite coordinates.")
    if n_top_genes < 2:
        raise ValueError("--n-top-genes must be at least 2.")


def package_version(package: str) -> str:
    try:
        return version(package)
    except Exception:
        return "unknown"


def run_stagate(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input h5ad does not exist: {args.input}")

    adata = sc.read_h5ad(args.input)
    validate_input(adata, args.spatial_key, args.n_top_genes)
    adata.var_names_make_unique()

    if args.spatial_key != "spatial":
        adata.obsm["spatial"] = np.asarray(adata.obsm[args.spatial_key]).copy()

    n_top_genes = min(args.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=args.rad_cutoff)
    spatial_net = adata.uns.get("Spatial_Net")
    if spatial_net is None or len(spatial_net) == 0:
        raise RuntimeError(
            f"STAGATE spatial graph has no edges at rad_cutoff={args.rad_cutoff}. "
            "Check the coordinate scale and increase --rad-cutoff."
        )
    device = resolve_device(args.device)
    adata = STAGATE_pyG.train_STAGATE(
        adata,
        hidden_dims=list(args.hidden_dims),
        n_epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        random_seed=args.seed,
        device=device,
    )

    embedding = np.asarray(adata.obsm["STAGATE"])
    if embedding.shape != (adata.n_obs, args.hidden_dims[1]) or not np.isfinite(embedding).all():
        raise RuntimeError(f"Invalid STAGATE embedding with shape {embedding.shape}.")

    n_neighbors = min(args.n_neighbors, adata.n_obs - 1)
    if n_neighbors < 2:
        raise ValueError("At least three spots are required for domain clustering.")
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="STAGATE",
        key_added="STAGATE_neighbors",
        random_state=args.seed,
    )
    sc.tl.leiden(
        adata,
        resolution=args.resolution,
        key_added="STAGATE_domain",
        neighbors_key="STAGATE_neighbors",
        random_state=args.seed,
        flavor="igraph",
        n_iterations=2,
    )

    adata.uns["STAGATE_benchmark"] = {
        "input": str(args.input.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/RucDongLab/STAGATE_pyG",
        "source_commit": STAGATE_SOURCE_COMMIT,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_geometric": package_version("torch-geometric"),
        "torch_sparse": package_version("torch-sparse"),
        "torch_scatter": package_version("torch-scatter"),
        "scanpy": package_version("scanpy"),
        "device": str(device),
        "spatial_key": args.spatial_key,
        "n_top_genes": n_top_genes,
        "rad_cutoff": args.rad_cutoff,
        "hidden_dims": list(args.hidden_dims),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "n_neighbors": n_neighbors,
        "clustering": "leiden_igraph",
        "resolution": args.resolution,
        "seed": args.seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    n_domains = adata.obs["STAGATE_domain"].nunique()
    print(f"Saved {adata.n_obs} spots, {embedding.shape[1]} latent dimensions, and {n_domains} domains to {args.output}")


def main() -> None:
    run_stagate(parse_args())


if __name__ == "__main__":
    main()
