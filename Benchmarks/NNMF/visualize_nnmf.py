#!/usr/bin/env python
"""Visualize native NNMF signature weights and gene loadings."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "nnmf_visualization_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
)
DEFAULT_RESULT_DIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/NNMF/K24"
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize NNMF signature outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_RESULT_DIR / "nnmf_weights.csv")
    parser.add_argument("--signatures", type=Path, default=DEFAULT_RESULT_DIR / "nnmf_signatures.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR / "vis")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--top-genes", type=int, default=3)
    parser.add_argument("--point-size", type=float, default=7.0)
    args = parser.parse_args()
    if args.top_genes < 1 or args.point_size <= 0:
        parser.error("--top-genes and --point-size must be positive.")
    return args


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_aligned(
    input_path: Path, weights_path: Path, signatures_path: Path, spatial_key: str
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(input_path, backed="r")
    weights = pd.read_csv(weights_path, index_col=0)
    signatures = pd.read_csv(signatures_path, index_col=0)
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}].")
    missing = adata.obs_names.difference(weights.index)
    if len(missing):
        raise ValueError(f"NNMF weights are missing {len(missing)} spots.")
    if set(weights.columns) != set(signatures.columns):
        raise ValueError("NNMF weight and signature component names do not match.")
    weights = weights.loc[adata.obs_names, signatures.columns]
    coordinates = np.asarray(adata.obsm[spatial_key])[:, :2]
    return adata, weights, signatures, coordinates


def top_gene_table(signatures: pd.DataFrame, n_top: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for component in signatures.columns:
        ranked = signatures[component].sort_values(ascending=False).head(n_top)
        for rank, (gene, loading) in enumerate(ranked.items(), start=1):
            records.append(
                {"component": component, "rank": rank, "gene": gene, "loading": float(loading)}
            )
    return pd.DataFrame(records)


def plot_weight_maps(
    coordinates: np.ndarray,
    weights: pd.DataFrame,
    top_genes: pd.DataFrame,
    point_size: float,
) -> mpl.figure.Figure:
    ncols = min(6, weights.shape[1])
    nrows = int(np.ceil(weights.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 2.05 * nrows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= weights.shape[1]:
            continue
        component = weights.columns[index]
        values = weights[component].to_numpy(dtype=float)
        vmax = max(float(np.nanquantile(values, 0.99)), 1e-12)
        genes = top_genes.loc[top_genes["component"] == component, "gene"].astype(str).tolist()
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=values, s=point_size, cmap="magma",
            vmin=0, vmax=vmax, linewidths=0, rasterized=True
        )
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_title(f"{component}\n{', '.join(genes[:2])}", fontsize=6.7, pad=1.5)
    fig.suptitle("NNMF signature weights (each panel scaled to its 99th percentile)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), pad=0.25)
    return fig


def plot_loading_heatmap(signatures: pd.DataFrame, top_genes: pd.DataFrame) -> mpl.figure.Figure:
    genes = list(dict.fromkeys(top_genes["gene"].astype(str)))
    matrix = signatures.loc[genes].to_numpy(dtype=float)
    row_max = np.nanmax(np.abs(matrix), axis=1, keepdims=True)
    matrix = np.divide(matrix, row_max, out=np.zeros_like(matrix), where=row_max > 0)
    fig, axis = plt.subplots(figsize=(max(7.0, 0.31 * signatures.shape[1]), max(4.5, 0.15 * len(genes))))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(np.arange(signatures.shape[1]), signatures.columns, rotation=90)
    axis.set_yticks(np.arange(len(genes)), genes)
    axis.set(xlabel="NNMF signature", ylabel="Top-loading genes")
    axis.set_title("NNMF gene loadings (row-max scaled)", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.015, label="Relative loading")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    for path in (args.input, args.weights, args.signatures):
        if not path.is_file():
            raise FileNotFoundError(path)
    configure_style()
    adata, weights, signatures, coordinates = load_aligned(
        args.input, args.weights, args.signatures, args.spatial_key
    )
    top_genes = top_gene_table(signatures, args.top_genes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        plot_weight_maps(coordinates, weights, top_genes, args.point_size),
        args.output_dir / "nnmf_signature_maps",
    )
    save_figure(
        plot_loading_heatmap(signatures, top_genes),
        args.output_dir / "nnmf_top_gene_loadings",
    )
    top_genes.to_csv(args.output_dir / "nnmf_top_genes.csv", index=False)
    if adata.isbacked:
        adata.file.close()
    print(f"Saved NNMF visualizations for {weights.shape[1]} signatures to {args.output_dir}.")


if __name__ == "__main__":
    main()
