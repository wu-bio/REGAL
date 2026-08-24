#!/usr/bin/env python
"""Visualize native SPARK-X gene-level results on the BREAST section."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

RUNTIME_CACHE = Path(tempfile.gettempdir()) / "sparkx_visualization_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE))

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_INPUT = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1.h5ad"
)
DEFAULT_RESULT_DIR = Path(
    "/Users/wuyang/Documents/SC-ST data/BREAST/ST/Breast_BlockA_Section1/SPARKX"
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
            "pdf.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SPARK-X significance and top SVG spatial maps.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULT_DIR / "sparkx_results.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR / "vis")
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--point-size", type=float, default=7.0)
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be positive.")
    if not 0 < args.fdr < 1:
        parser.error("--fdr must be in (0, 1).")
    if args.point_size <= 0:
        parser.error("--point-size must be positive.")
    return args


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate_inputs(adata: ad.AnnData, results: pd.DataFrame, spatial_key: str) -> None:
    required = {"gene", "combined_pvalue", "adjusted_pvalue", "total_counts", "detected_spots"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"SPARK-X results are missing columns: {sorted(missing)}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}].")
    coordinates = np.asarray(adata.obsm[spatial_key])
    if coordinates.shape[0] != adata.n_obs or coordinates.shape[1] < 2:
        raise ValueError(f"Invalid spatial coordinates: {coordinates.shape}.")


def select_top_genes(results: pd.DataFrame, adata: ad.AnnData, top_n: int) -> pd.DataFrame:
    ranked = results.copy()
    for column in ("combined_pvalue", "adjusted_pvalue", "total_counts", "detected_spots"):
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    ranked = ranked.loc[ranked["gene"].astype(str).isin(adata.var_names)].copy()
    ranked = ranked.sort_values(
        ["adjusted_pvalue", "combined_pvalue", "total_counts", "gene"],
        ascending=[True, True, False, True],
    )
    selected = ranked.drop_duplicates("gene").head(top_n).copy()
    if selected.empty:
        raise ValueError("No ranked SPARK-X genes were found in the input h5ad.")
    return selected


def plot_significance_summary(results: pd.DataFrame, fdr: float) -> mpl.figure.Figure:
    adjusted = pd.to_numeric(results["adjusted_pvalue"], errors="coerce").to_numpy(dtype=float)
    adjusted = adjusted[np.isfinite(adjusted)]
    adjusted.sort()
    floor = np.nextafter(0.0, 1.0)
    score = np.minimum(-np.log10(np.maximum(adjusted, floor)), 50.0)
    significant = int(np.count_nonzero(adjusted <= fdr))
    nonsignificant = int(adjusted.size - significant)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), gridspec_kw={"width_ratios": [2.1, 1.0]})
    axes[0].plot(np.arange(1, score.size + 1), score, color="#355F8D", linewidth=1.2)
    axes[0].axhline(-np.log10(fdr), color="#C95D63", linewidth=0.9, linestyle="--")
    axes[0].set(xlabel="Ranked genes", ylabel=r"$-\log_{10}$(adjusted P), capped at 50")
    axes[0].set_title("SPARK-X ranked spatial significance", loc="left", fontweight="bold")

    axes[1].bar(
        ["FDR ≤ threshold", "Not significant"],
        [significant, nonsignificant],
        color=["#4C956C", "#C9CED6"],
        width=0.7,
    )
    axes[1].set_ylabel("Genes")
    axes[1].set_title(f"FDR threshold = {fdr:g}", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_gene_maps(
    adata: ad.AnnData,
    selected: pd.DataFrame,
    spatial_key: str,
    point_size: float,
) -> mpl.figure.Figure:
    genes = selected["gene"].astype(str).tolist()
    expression = adata[:, genes].X
    if sp.issparse(expression):
        expression = expression.toarray()
    expression = np.log1p(np.asarray(expression, dtype=float))
    coordinates = np.asarray(adata.obsm[spatial_key])[:, :2]

    ncols = min(4, len(genes))
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.15 * nrows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= len(genes):
            continue
        values = expression[:, index]
        positive = values[values > 0]
        vmax = float(np.quantile(positive, 0.99)) if positive.size else 1.0
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            s=point_size,
            cmap="magma",
            vmin=0,
            vmax=max(vmax, 1e-12),
            linewidths=0,
            rasterized=True,
        )
        axis.invert_yaxis()
        axis.set_aspect("equal")
        adjusted = float(selected.iloc[index]["adjusted_pvalue"])
        p_label = "0" if adjusted == 0 else f"{adjusted:.1e}"
        axis.set_title(f"{genes[index]}  |  adj. P={p_label}", fontsize=7.5, pad=2)
    fig.suptitle("Top-ranked SPARK-X genes: raw log1p expression", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=0.35)
    return fig


def main() -> None:
    args = parse_args()
    if not args.input.is_file() or not args.results.is_file():
        raise FileNotFoundError("Input h5ad or SPARK-X results file does not exist.")
    configure_style()
    adata = ad.read_h5ad(args.input)
    results = pd.read_csv(args.results)
    validate_inputs(adata, results, args.spatial_key)
    selected = select_top_genes(results, adata, args.top_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_figure(plot_significance_summary(results, args.fdr), args.output_dir / "sparkx_significance_summary")
    save_figure(
        plot_gene_maps(adata, selected, args.spatial_key, args.point_size),
        args.output_dir / "sparkx_top_gene_maps",
    )
    selected.to_csv(args.output_dir / "sparkx_visualized_genes.csv", index=False)
    print(f"Saved SPARK-X visualizations for {len(selected)} genes to {args.output_dir}.")


if __name__ == "__main__":
    main()
